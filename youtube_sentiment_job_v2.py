# youtube_sentiment_job_v2.py
# -*- coding: utf-8 -*-
"""youtube_sentiment_job_v2.py
GCP-ready: reads API key from Secret Manager, uploads outputs to GCS (apidatabase).
Supports backfill (Oct → today) and daily run (Airflow)
"""

import os
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import secretmanager, storage
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# === [1] GCP Config ===
PROJECT_ID = "ba882-qstba-group7-fall2025"
BUCKET_NAME = "apidatabase"
OUTPUT_PREFIX_SUMMARY = "youtube/summary"
OUTPUT_PREFIX_COMMENTS = "youtube/comments"

# === [2] Secret Manager Helper ===
def get_secret(secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    return client.access_secret_version(name=name).payload.data.decode("utf-8")

# === [3] Load YouTube API Key ===
YOUTUBE_API_KEY = get_secret("youtube-api-key")

# === [4] Init YouTube API client ===
from googleapiclient.discovery_cache.base import Cache
class NoCache(Cache):
    def get(self, url): return None
    def set(self, url, content): pass

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache=NoCache())
print("✅ YouTube API client initialized.")

# === [5] Setup Sentiment Analyzer ===
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
SIA = SentimentIntensityAnalyzer()

# === [6] Helper functions ===
NOW_UTC = datetime.now(timezone.utc)

def safe_int(x):
    try: return int(x)
    except: return None

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def _age_days(published_at):
    ts = pd.to_datetime(published_at, utc=True, errors="coerce")
    return max((NOW_UTC - ts).total_seconds() / 86400, 0.04)

def upload_to_gcs(local_path: str, gcs_path: str):
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"☁️ Uploaded to GCS: gs://{BUCKET_NAME}/{gcs_path}")

# === [7] YouTube API Core ===
def search_video_id(artist, title, region="US", category_id="10"):
    q = f"{artist} {title}"
    try:
        req = youtube.search().list(
            q=q, part="id,snippet", type="video",
            maxResults=5, regionCode=region,
            videoCategoryId=category_id, order="relevance"
        )
        resp = req.execute()
    except HttpError as e:
        print("❌ Search error:", e)
        return None, None
    items = resp.get("items", [])
    if not items:
        print(f"⚠️ No search results for {artist} - {title}")
        return None, None
    first = items[0]
    return first["id"]["videoId"], first["snippet"]

def get_video_stats(video_id):
    try:
        resp = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    except HttpError as e:
        print("❌ Video stats error:", e)
        return None
    if not resp.get("items"):
        return None
    return resp["items"][0]

def fetch_comments_all(video_id: str, order="time", since=None, until=None):
    out, page_token, total_fetched = [], None, 0
    while True:
        try:
            resp = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                order=order, textFormat="plainText", pageToken=page_token
            ).execute()
        except HttpError as e:
            print(f"❌ Failed to fetch comments for {video_id}: {e}")
            break
        items = resp.get("items", [])
        if not items: break

        for item in items:
            top = item["snippet"]["topLevelComment"]["snippet"]
            published_at = pd.to_datetime(top.get("publishedAt"), utc=True, errors="coerce")

            # Filter by date range if specified
            if since and published_at < since: 
                continue
            if until and published_at > until:
                continue

            out.append({
                "video_id": video_id,
                "author": top.get("authorDisplayName"),
                "text": top.get("textDisplay"),
                "like_count": safe_int(top.get("likeCount")),
                "published_at": published_at,
            })
            total_fetched += 1
        page_token = resp.get("nextPageToken")
        if not page_token: break
        time.sleep(0.15)
    print(f"💬 Total comments fetched for {video_id}: {total_fetched}")
    return out

def analyze_sentiment_overall(df: pd.DataFrame):
    if df.empty:
        return {"pos": 0, "neg": 0, "neu": 0, "mean_compound": 0.0}
    scores = df["text"].fillna("").apply(SIA.polarity_scores)
    df["compound"] = scores.apply(lambda s: s["compound"])
    df["label"] = df["compound"].apply(
        lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")
    )
    summary = df["label"].value_counts().to_dict()
    return {
        "pos": summary.get("positive", 0),
        "neg": summary.get("negative", 0),
        "neu": summary.get("neutral", 0),
        "mean_compound": round(df["compound"].mean(), 4),
    }

# === [8] Main Job ===
def run_sentiment_job(targets, start_date, end_date):
    summary_records = []
    today_str = datetime.now().strftime("%Y%m%d")

    for t in targets:
        artist, title = t["artist"], t["title"]
        print(f"\n=== Processing {artist} – {title} ===")

        vid, _ = search_video_id(artist, title)
        if not vid: continue

        vmeta = get_video_stats(vid)
        if not vmeta: continue

        snip = vmeta.get("snippet", {})
        stats = vmeta.get("statistics", {})
        published_at = snip.get("publishedAt")
        views = safe_int(stats.get("viewCount"))
        likes = safe_int(stats.get("likeCount"))
        age_days = _age_days(published_at)

        comments = fetch_comments_all(vid, since=start_date, until=end_date)
        df_comments = pd.DataFrame(comments)

        senti = analyze_sentiment_overall(df_comments)
        total_comments = len(df_comments)
        avg_rate = round(total_comments / max((end_date - start_date).days, 1), 4)

        if not df_comments.empty:
            local_comments = f"comments_{artist.replace(' ', '_')}_{today_str}.csv"
            df_comments.to_csv(local_comments, index=False)
            upload_to_gcs(local_comments, f"{OUTPUT_PREFIX_COMMENTS}/{local_comments}")

        summary_records.append({
            "fetch_time": now_iso(),
            "video_id": vid,
            "title": snip.get("title"),
            "channel": snip.get("channelTitle"),
            "published_at": published_at,
            "views": views,
            "likes": likes,
            "age_days": round(age_days, 2),
            "comment_count": total_comments,
            "avg_comment_rate": avg_rate,
            "pos_comments": senti["pos"],
            "neu_comments": senti["neu"],
            "neg_comments": senti["neg"],
            "mean_compound": senti["mean_compound"],
        })

    df_summary = pd.DataFrame(summary_records)
    local_summary = f"youtube_summary_{today_str}.csv"
    df_summary.to_csv(local_summary, index=False)
    upload_to_gcs(local_summary, f"{OUTPUT_PREFIX_SUMMARY}/{local_summary}")

    print("\n✅ Job completed successfully — all outputs saved to GCS.")

# === [9] Run mode ===
if __name__ == "__main__":
    TARGET_SONGS = [
        {"artist": "Taylor Swift", "title": "The Fate of Ophelia"},
        {"artist": "Ed Sheeran", "title": "Camera"},
        {"artist": "Doja Cat", "title": "Gorgeous"},
        {"artist": "Demi Lovato", "title": "Kiss"},
        {"artist": "Anne-Marie", "title": "DEPRESSED"},
        {"artist": "Madison Beer", "title": "bittersweet"},
        {"artist": "Cardi B feat. Kehlani", "title": "Safe"},
        {"artist": "Grimes", "title": "Artificial Angels"},
        {"artist": "Tyla", "title": "CHANEL"},
        {"artist": "Tame Impala", "title": "Dracula"},
    ]

    mode = os.getenv("MODE", "backfill")  # or "daily"
    print(f"▶️ Running YouTube sentiment job in mode: {mode}")

    if mode == "backfill":
        start = datetime(2025, 10, 1, tzinfo=timezone.utc)
        end = datetime.now(timezone.utc)
        run_sentiment_job(TARGET_SONGS, start, end)
    else:
        today = datetime.now(timezone.utc)
        start = today - timedelta(days=1)
        run_sentiment_job(TARGET_SONGS, start, today)
