# -*- coding: utf-8 -*-
"""
reddit_sentiment_job_v2.py
Standalone worker script for Reddit ETL + sentiment, mirroring the
YouTube template's structure and conventions so it can run locally,
from Airflow, or via GitHub Actions.

Key traits:
- No secrets in code. Pull from ENV first, then GCP Secret Manager.
- GCP upload helper (gs://apidatabase/...)
- Deterministic UTC window (e.g., collect back to a start date)
- Robust retries, polite rate limiting
- NLTK VADER sentiment (same analyzer as YouTube job)
- CLI args for flexible use in GitHub / Airflow
"""
import os, json, time, math, argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable
from datetime import datetime, timezone, timedelta

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from google.cloud import storage, secretmanager
from google.oauth2 import service_account
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# 3rd‑party Reddit API
import praw
from praw.models import MoreComments

# === [1] GCP Setup ===
PROJECT_ID = os.getenv("PROJECT_ID", "ba882-qstba-group7-fall2025")
BUCKET_NAME = os.getenv("BUCKET_NAME", "apidatabase")
OUTPUT_PREFIX_SUBMISSIONS = os.getenv("OUTPUT_PREFIX_SUBMISSIONS", "reddit/submissions")
OUTPUT_PREFIX_COMMENTS    = os.getenv("OUTPUT_PREFIX_COMMENTS",    "reddit/comments")
OUTPUT_PREFIX_SUMMARY     = os.getenv("OUTPUT_PREFIX_SUMMARY",     "reddit/summary")

# === [2] Credentials (GCP) ===
GCP_KEY_JSON = os.getenv("GCP_KEYFILE_DICT")
if GCP_KEY_JSON:
    creds_info = json.loads(GCP_KEY_JSON)
    GCP_CREDS = service_account.Credentials.from_service_account_info(creds_info)
    print("✅ Using GCP credentials from environment (GCP_KEYFILE_DICT).")
else:
    GCP_CREDS = None  # ADC fallback (useful on GCE/Cloud Run/Airflow with Workload Identity)
    print("⚠️ Falling back to ADC for GCP credentials.")

# === [3] Reddit API Secrets ===
# Priority: ENV → Secret Manager (projects/<id>/secrets/reddit-*)
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "ba882-research via GitHub Actions")

if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET):
    print("ℹ️ Loading Reddit secrets from Secret Manager...")
    sm_client = secretmanager.SecretManagerServiceClient(credentials=GCP_CREDS)
    def _from_secret(name: str) -> str:
        res = sm_client.access_secret_version(request={"name": name})
        return res.payload.data.decode("utf-8")
    base = f"projects/{PROJECT_ID}/secrets"
    REDDIT_CLIENT_ID     = REDDIT_CLIENT_ID     or _from_secret(f"{base}/reddit-client-id/versions/latest")
    REDDIT_CLIENT_SECRET = REDDIT_CLIENT_SECRET or _from_secret(f"{base}/reddit-client-secret/versions/latest")
    # user agent is not secret, keep env default

# === [4] Initialize Reddit API ===
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    ratelimit_seconds=5,
)
print("✅ Reddit client initialized.")

# === [5] NLP Setup (same as YouTube job) ===
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
SIA = SentimentIntensityAnalyzer()

# === [6] Helpers ===
NOW_UTC = datetime.now(timezone.utc)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _age_days(published_utc: float) -> float:
    try:
        ts = datetime.fromtimestamp(published_utc, tz=timezone.utc)
        return max((NOW_UTC - ts).total_seconds() / 86400.0, 0.04)
    except Exception:
        return None

def safe_author(obj) -> Optional[str]:
    try:
        return obj.author.name if obj.author else None
    except Exception:
        return None

def upload_to_gcs(local_path: str, gcs_path: str):
    client = storage.Client(credentials=GCP_CREDS, project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"☁️ Uploaded: gs://{BUCKET_NAME}/{gcs_path}")

# === [7] Data Classes / Config ===
@dataclass
class QueryBundle:
    artist: str
    song: str
    keywords: List[str]

# === [8] Core: Fetch Submissions ===
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
def _search_in_subreddit(subreddit: str, query: str, limit: int, time_filter: str, sort: str):
    return list(reddit.subreddit(subreddit).search(query=query, sort=sort, time_filter=time_filter, limit=limit))

def submission_to_row(sub, bundle: QueryBundle, query: str) -> Dict[str, Any]:
    return {
        "bundle_artist": bundle.artist,
        "bundle_song":   bundle.song,
        "bundle_query":  query,
        "subreddit":     str(sub.subreddit),
        "submission_id": sub.id,
        "title":         sub.title,
        "selftext":      (sub.selftext or ""),
        "author":        safe_author(sub),
        "score":         getattr(sub, "score", None),
        "upvote_ratio":  getattr(sub, "upvote_ratio", None),
        "num_comments":  getattr(sub, "num_comments", None),
        "created_utc":   float(getattr(sub, "created_utc", 0.0) or 0.0),
        "created_iso":   datetime.fromtimestamp(getattr(sub, "created_utc", 0.0) or 0.0, tz=timezone.utc).isoformat() if getattr(sub, "created_utc", None) else None,
        "url":           getattr(sub, "url", None),
        "permalink":     f"https://www.reddit.com{sub.permalink}",
        "collected_at":  now_iso(),
    }

def fetch_submissions(bundles: List[QueryBundle], subreddits: List[str], *,
                      limit_per_query: int = 50, time_filter: str = "year", sort: str = "relevance",
                      earliest_ts: Optional[int] = None) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    seen = set()
    for b in bundles:
        for sr in subreddits:
            for q in b.keywords:
                try:
                    results = _search_in_subreddit(sr, q, limit_per_query, time_filter, sort)
                    for sub in results:
                        if earliest_ts and getattr(sub, "created_utc", 0) and sub.created_utc < earliest_ts:
                            continue
                        key = (sub.id)
                        if key in seen:
                            continue
                        seen.add(key)
                        row = submission_to_row(sub, b, q)
                        records.append(row)
                    time.sleep(1.0)  # polite pacing
                except Exception as e:
                    print(f"❌ search error [{sr}] '{q}': {e}")
                    continue
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df.sort_values(["created_utc"], ascending=False, inplace=True)
        df.drop_duplicates(subset=["submission_id"], inplace=True)
    print(f"🧵 collected submissions: {len(df)}")
    return df

# === [9] Core: Expand Comments ===
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20))
def _load_submission(submission_id: str):
    return reddit.submission(id=submission_id)

def comment_to_row(c, sub_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **{k: sub_row[k] for k in ["bundle_artist", "bundle_song", "bundle_query", "subreddit", "submission_id", "title", "permalink"]},
        "comment_id":  c.id,
        "parent_id":   getattr(c, "parent_id", None),
        "author":      safe_author(c),
        "score":       getattr(c, "score", None),
        "created_utc": float(getattr(c, "created_utc", 0.0) or 0.0),
        "created_iso": datetime.fromtimestamp(getattr(c, "created_utc", 0.0) or 0.0, tz=timezone.utc).isoformat() if getattr(c, "created_utc", None) else None,
        "body":        getattr(c, "body", ""),
        "collected_at": now_iso(),
    }

def fetch_comments_for(df_subs: pd.DataFrame, *, limit_per_submission: Optional[int] = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, sub_row in df_subs.iterrows():
        try:
            sub = _load_submission(sub_row["submission_id"])
            sub.comments.replace_more(limit=0)
            counter = 0
            for c in sub.comments.list():
                if isinstance(c, MoreComments):
                    continue
                rows.append(comment_to_row(c, sub_row))
                counter += 1
                if limit_per_submission and counter >= limit_per_submission:
                    break
            time.sleep(0.7)
        except Exception as e:
            print(f"❌ comments error for {sub_row['submission_id']}: {e}")
            continue
    df = pd.DataFrame.from_records(rows)
    print(f"💬 collected comments: {len(df)}")
    return df

# === [10] Sentiment ===

def add_sentiment(df: pd.DataFrame, text_col: str = "body") -> pd.DataFrame:
    if df.empty:
        return df
    scores = df[text_col].fillna("").astype(str).apply(SIA.polarity_scores)
    df["compound"] = scores.apply(lambda s: s["compound"]).astype(float)
    df["sentiment"] = pd.cut(df["compound"], bins=[-1.0, -0.05, 0.05, 1.0], labels=["negative", "neutral", "positive"], include_lowest=True)
    return df

# === [11] Persist ===

def save_local_then_gcs(df: pd.DataFrame, local_path: str, gcs_prefix: str):
    if df.empty:
        print(f"⚠️ Nothing to save for {local_path}.")
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_csv(local_path, index=False)
    print(f"💾 Saved local: {local_path}")
    date_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    gcs_path = f"{gcs_prefix}/dt={date_tag}/part-000.csv"
    upload_to_gcs(local_path, gcs_path)

# === [12] CLI / Main ===

def parse_args():
    p = argparse.ArgumentParser(description="Reddit ETL + Sentiment (YouTube-style)")
    p.add_argument("--bundles_json", type=str, default=None,
                   help="Path to JSON file containing a list of {artist, song, keywords[]} dicts.")
    p.add_argument("--subreddits", type=str, default="all",
                   help="Comma-separated subreddit names (e.g., music,hiphopheads). Use 'all' for site‑wide search.")
    p.add_argument("--start_date", type=str, default=None,
                   help="Earliest UTC date (YYYY-MM-DD). If provided, submissions older than this will be skipped.")
    p.add_argument("--limit_per_query", type=int, default=50)
    p.add_argument("--limit_comments_per_submission", type=int, default=None)
    p.add_argument("--time_filter", type=str, default="year", choices=["hour","day","week","month","year","all"])
    p.add_argument("--sort", type=str, default="relevance", choices=["relevance","hot","top","new","comments"])
    p.add_argument("--outdir", type=str, default="./out")
    return p.parse_args()

DEFAULT_BUNDLES = [
    {"artist": "Taylor Swift", "song": "The Fate of Ophelia", "keywords": [
        '"Taylor Swift" "The Fate of Ophelia"', 'Taylor Swift Fate of Ophelia song']},
    {"artist": "Tyla", "song": "CHANEL", "keywords": [
        '"Tyla" "CHANEL"', 'Tyla CHANEL song']},
    {"artist": "Tame Impala", "song": "Dracula", "keywords": [
        '"Tame Impala" "Dracula"', 'Tame Impala Dracula']},
]

if __name__ == "__main__":
    args = parse_args()

    # Resolve bundles
    if args.bundles_json and os.path.exists(args.bundles_json):
        with open(args.bundles_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = DEFAULT_BUNDLES
    bundles = [QueryBundle(**b) for b in raw]

    subreddits = [s.strip() for s in args.subreddits.split(",") if s.strip()] if args.subreddits != "all" else ["all"]

    earliest_ts = None
    if args.start_date:
        earliest_ts = int(datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    # 1) submissions
    df_subs = fetch_submissions(
        bundles, subreddits,
        limit_per_query=args.limit_per_query,
        time_filter=args.time_filter,
        sort=args.sort,
        earliest_ts=earliest_ts,
    )

    # 2) comments
    df_coms = fetch_comments_for(df_subs, limit_per_submission=args.limit_comments_per_submission)

    # 3) sentiment
    df_coms = add_sentiment(df_coms, text_col="body")

    # 4) summary (per bundle / sentiment)
    if not df_coms.empty:
        summary = (
            df_coms.groupby(["bundle_artist","bundle_song","sentiment"], dropna=False)
                    .size().unstack(fill_value=0).reset_index()
        )
    else:
        summary = pd.DataFrame()

    # 5) save local + GCS
    os.makedirs(args.outdir, exist_ok=True)
    sub_local = os.path.join(args.outdir, "reddit_submissions.csv")
    com_local = os.path.join(args.outdir, "reddit_comments.csv")
    sum_local = os.path.join(args.outdir, "reddit_summary.csv")

    save_local_then_gcs(df_subs, sub_local, OUTPUT_PREFIX_SUBMISSIONS)
    save_local_then_gcs(df_coms, com_local, OUTPUT_PREFIX_COMMENTS)
    save_local_then_gcs(summary, sum_local, OUTPUT_PREFIX_SUMMARY)

    # Log a tiny report
    print("\n===== RUN SUMMARY =====")
    print(f"Submissions: {len(df_subs):,}")
    print(f"Comments:    {len(df_coms):,}")
    if not summary.empty:
        print("Per‑song sentiment counts:\n", summary.to_string(index=False))
    print("✅ Completed job successfully.")

