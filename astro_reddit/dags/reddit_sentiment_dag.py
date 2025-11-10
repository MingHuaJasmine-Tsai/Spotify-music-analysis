# reddit_sentiment_dag.py
# Airflow DAG: fetch Reddit posts/comments for given songs, run VADER sentiment,
# and upload CSVs to GCS under:
#   gs://apidatabase/reddit/comments/<song_slug>/YYYYMMDD_HHMM.csv
#   gs://apidatabase/reddit/summary/<song_slug>/YYYYMMDD_HHMM.csv
#
# Uses existing Airflow Connection "gcp_conn" (keyfile_dict) and
# Secret Manager secret "reddit-api-key" that stores PRAW credentials in JSON:
# {
#   "client_id": "...",
#   "client_secret": "...",
#   "user_agent": "your-app/1.0"
# }
#
# Rate-limit friendly: sleeps between API calls; retries handled by Airflow.

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import json
import time
import typing as t

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import praw
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.hooks.base import BaseHook

from google.cloud import secretmanager, storage
from google.oauth2 import service_account

# ---- GCP / Airflow config (match your existing setup) ----
GCP_CONN_ID = "gcp_conn"
PROJECT_ID = "ba882-qstba-group7-fall2025"
BUCKET_NAME = "apidatabase"
SECRET_ID = "reddit-api-key"  # do NOT create new; reuse existing

# Where to put results in the bucket
REDDIT_PREFIX = "reddit"
COMMENTS_DIR = f"{REDDIT_PREFIX}/comments"
SUMMARY_DIR = f"{REDDIT_PREFIX}/summary"

# What to collect
SONGS: list[dict[str, str]] = [
    {"artist": "Taylor Swift", "title": "The Fate of Ophelia"},
    {"artist": "Ed Sheeran", "title": "Camera"},
    {"artist": "Olivia Rodrigo", "title": "Obsessed"},  # example; edit as needed
]
SUBREDDITS = ["Music", "popheads", "TaylorSwift", "EdSheeran"]  # safe defaults
POSTS_PER_SONG = 30        # number of posts to pull per song
COMMENTS_PER_POST = 50     # up to N comments per post
SLEEP_BETWEEN_CALLS = 1.0  # gentle rate-limiting

# ---- Helpers ----
def _slug(s: str) -> str:
    return (
        s.lower()
        .replace("/", " ")
        .replace("\\", " ")
        .replace("&", " and ")
        .replace("  ", " ")
        .strip()
        .replace(" ", "_")
    )

def _gcp_creds():
    conn = BaseHook.get_connection(GCP_CONN_ID)
    info = conn.extra_dejson.get("extra__google_cloud_platform__keyfile_dict")
    if not info:
        raise RuntimeError("Missing keyfile_dict in GCP connection extras.")
    return service_account.Credentials.from_service_account_info(info)

def _read_reddit_secret(creds):
    client = secretmanager.SecretManagerServiceClient(credentials=creds)
    name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID}/versions/latest"
    payload = client.access_secret_version(request={"name": name}).payload.data
    cfg = json.loads(payload.decode("utf-8"))
    required = {"client_id", "client_secret", "user_agent"}
    if not required.issubset(cfg):
        raise RuntimeError("reddit-api-key secret must contain client_id, client_secret, user_agent.")
    return cfg

def _gcs_upload_df(df: pd.DataFrame, blob_path: str, creds) -> str:
    client = storage.Client(credentials=creds, project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    # write CSV to memory to avoid tmp files
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    blob.upload_from_string(buf.getvalue(), content_type="text/csv")
    uri = f"gs://{BUCKET_NAME}/{blob_path}"
    print(f"Uploaded -> {uri}")
    return uri

# ---- Main job ----
def run_reddit_sentiment(**_):
    # set up creds + clients
    gcp_creds = _gcp_creds()
    praw_cfg = _read_reddit_secret(gcp_creds)
    reddit = praw.Reddit(
        client_id=praw_cfg["client_id"],
        client_secret=praw_cfg["client_secret"],
        user_agent=praw_cfg["user_agent"],
        check_for_async=False,  # avoid event loop warnings in Airflow
    )

    # make sure VADER is available
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()

    ts = datetime.now(timezone.utc)
    stamp = ts.strftime("%Y%m%d_%H%M")

    for s in SONGS:
        query = f'{s["artist"]} {s["title"]}'
        song_slug = _slug(query)
        print(f"Searching posts for: {query}")

        # Collect posts across several subreddits (de-duplicate by id)
        posts_rows: list[dict[str, t.Any]] = []
        seen_ids: set[str] = set()

        for sub in SUBREDDITS:
            try:
                subreddit = reddit.subreddit(sub)
                # limit keeps API light; .search respects PRAW rate limits
                for submission in subreddit.search(query, sort="new", limit=POSTS_PER_SONG):
                    if submission.id in seen_ids:
                        continue
                    seen_ids.add(submission.id)
                    posts_rows.append(
                        {
                            "song": query,
                            "song_slug": song_slug,
                            "subreddit": sub,
                            "post_id": submission.id,
                            "title": submission.title or "",
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                            "created_utc": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
                            "permalink": f"https://reddit.com{submission.permalink}",
                        }
                    )
                time.sleep(SLEEP_BETWEEN_CALLS)
            except Exception as e:
                print(f"[warn] subreddit {sub} search failed: {e}")
                time.sleep(SLEEP_BETWEEN_CALLS)

        if not posts_rows:
            print(f"No posts found for: {query}")
            continue

        posts_df = pd.DataFrame(posts_rows).drop_duplicates(subset=["post_id"])

        # Pull comments for each post (lightweight, top-level)
        comments_rows: list[dict[str, t.Any]] = []
        for _, row in posts_df.iterrows():
            pid = row["post_id"]
            try:
                subm = reddit.submission(id=pid)
                subm.comments.replace_more(limit=0)
                count = 0
                for c in subm.comments:
                    if count >= COMMENTS_PER_POST:
                        break
                    txt = c.body or ""
                    score = sia.polarity_scores(txt)["compound"]
                    comments_rows.append(
                        {
                            "song": row["song"],
                            "song_slug": song_slug,
                            "post_id": pid,
                            "comment_id": c.id,
                            "body": txt,
                            "sentiment": score,
                            "created_utc": datetime.fromtimestamp(c.created_utc, tz=timezone.utc).isoformat(),
                            "permalink": f"https://reddit.com{c.permalink}" if hasattr(c, "permalink") else "",
                        }
                    )
                    count += 1
                time.sleep(SLEEP_BETWEEN_CALLS)
            except Exception as e:
                print(f"[warn] comments fetch failed for post {pid}: {e}")
                time.sleep(SLEEP_BETWEEN_CALLS)

        comments_df = pd.DataFrame(comments_rows)

        # Build per-song summary (simple aggregates)
        if not comments_df.empty:
            agg = comments_df.groupby("song_slug")["sentiment"].agg(["count", "mean", "median"]).reset_index()
            agg = agg.rename(columns={"count": "n_comments", "mean": "sent_mean", "median": "sent_median"})
            agg["as_of_utc"] = ts.isoformat()
        else:
            agg = pd.DataFrame(
                [{"song_slug": song_slug, "n_comments": 0, "sent_mean": float("nan"),
                  "sent_median": float("nan"), "as_of_utc": ts.isoformat()}]
            )

        # Upload to GCS under the requested layout
        comments_path = f"{COMMENTS_DIR}/{song_slug}/{stamp}.csv"
        summary_path = f"{SUMMARY_DIR}/{song_slug}/{stamp}.csv"
        _gcs_upload_df(comments_df, comments_path, gcp_creds)
        _gcs_upload_df(agg, summary_path, gcp_creds)

        print(f"✅ Done: {query} -> {comments_path} / {summary_path}")

# ---- DAG ----
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="reddit_sentiment_dag",
    default_args=default_args,
    description="Reddit ETL + VADER sentiment to GCS (comments & summary by song).",
    schedule="0 1 * * *",              # same cadence as your YouTube DAG
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=["reddit", "etl", "sentiment"],
) as dag:
    run = PythonOperator(
        task_id="run_reddit_sentiment",
        python_callable=run_reddit_sentiment,
    )
