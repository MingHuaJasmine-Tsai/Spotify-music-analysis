# -*- coding: utf-8 -*-
"""
build_daily_song_summary.py
Airflow callable function for daily data backfill, integrated with GCS and Secret Manager.
"""

import os
import time
import random
from datetime import datetime, timezone, timedelta
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.discovery_cache.base import Cache
from googleapiclient.errors import HttpError
from google.cloud import storage # <-- 修正 NameError: name 'storage' is not defined
from google.cloud import secretmanager
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# =========================
# GCP CONFIG
# =========================
PROJECT_ID = "ba882-qstba-group7-fall2025"
BUCKET_NAME = "apidatabase"

OUTPUT_PREFIX_SUMMARY = "youtube/summary"
OUTPUT_PREFIX_COMMENTS = "youtube/comments"

# =========================
# API KEY LOADING
# =========================
def get_youtube_api_key():
    """Fetches YouTube API Key from Secret Manager or Environment."""
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

    if not YOUTUBE_API_KEY:
        try:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{PROJECT_ID}/secrets/youtube-api-key/versions/latest"
            YOUTUBE_API_KEY = client.access_secret_version(name=name).payload.data.decode("utf-8")
        except Exception as e:
            print(f"Error accessing Secret Manager: {e}")
            raise ValueError("YOUTUBE_API_KEY not found in env or Secret Manager")
            
    return YOUTUBE_API_KEY

# =========================
# GLOBALS & INIT
# =========================
# Initialize once outside main logic if possible, but safely inside a class/function for Airflow
class NoCache(Cache):
    def get(self, url): return None
    def set(self, url, content): pass

YOUTUBE = None
SIA = None
NOW_UTC = datetime.now(timezone.utc)

def initialize_globals():
    """Initializes API client and NLP tools."""
    global YOUTUBE, SIA
    
    # 1. Init YouTube Client
    try:
        api_key = get_youtube_api_key()
        YOUTUBE = build("youtube", "v3", developerKey=api_key, cache=NoCache())
        print("YouTube API client initialized.")
    except Exception as e:
        print(f"Failed to initialize YouTube Client: {e}")
        raise

    # 2. Init Sentiment Analyzer
    try:
        # Check and download NLTK data if necessary
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            print("Downloading NLTK VADER lexicon...")
            nltk.download("vader_lexicon")
        SIA = SentimentIntensityAnalyzer()
        print("Sentiment Analyzer initialized.")
    except Exception as e:
        print(f"Failed to initialize Sentiment Analyzer: {e}")
        # Note: We won't raise for NLP tools unless they are critical, but here they are.
        raise

# =========================
# HELPER FUNCTIONS
# =========================
def safe_int(x):
    try: return int(x)
    except Exception: return None

def _age_days(published_at):
    ts = pd.to_datetime(published_at, utc=True, errors="coerce")
    return max((NOW_UTC - ts).total_seconds() / 86400, 0.04)

def upload_to_gcs(local_path: str, gcs_path: str):
    """Uploads file to GCS and cleans up local file."""
    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded to GCS: gs://{BUCKET_NAME}/{gcs_path}")
    except Exception as e:
        print(f"GCS upload failed for {local_path}: {e}")
        # Depending on requirements, you might re-raise or just log
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

def yt_api_call(request_func, *args, **kwargs):
    """Handles API calls with retry logic for quota errors."""
    retry = 0
    while True:
        try:
            request = request_func(*args, **kwargs)
            return request.execute()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in (403, 429):
                print(f"Quota hit (HTTP {status}). Waiting 100s...")
                time.sleep(100) 
                retry += 1
                if retry > 2: raise 
                continue
            raise

# =========================
# YOUTUBE SEARCH & FETCH
# =========================
def search_video_id(artist, title):
    # Costs 100 units! Use sparingly.
    q = f"{artist} {title} official music video"
    try:
        resp = yt_api_call(YOUTUBE.search().list, q=q, part="id,snippet", type="video", maxResults=1, order="relevance")
        items = resp.get("items", [])
        if items:
            return items[0]["id"]["videoId"], items[0]["snippet"]
    except Exception as e:
        print(f"Search failed for {title}: {e}")
    return None, None

def get_video_stats(video_id):
    # Costs 1 unit
    resp = yt_api_call(YOUTUBE.videos().list, part="snippet,statistics", id=video_id)
    if not resp.get("items"): return None
    return resp["items"][0]

def fetch_comments_in_window(video_id, start_dt, end_dt):
    # Costs 1 unit per page
    out = []
    page_token = None
    
    while True:
        try:
            # We assume YOUTUBE is initialized globally
            resp = yt_api_call(
                YOUTUBE.commentThreads().list,
                part="snippet",
                videoId=video_id,
                maxResults=100,
                order="time",
                textFormat="plainText",
                pageToken=page_token,
            )
        except Exception:
            break

        items = resp.get("items", [])
        if not items: break

        for item in items:
            top = item["snippet"]["topLevelComment"]["snippet"]
            published_at = pd.to_datetime(top["publishedAt"], utc=True)

            if published_at > end_dt: continue
            if published_at < start_dt: return out 

            out.append({
                "video_id": video_id,
                "author": top.get("authorDisplayName"),
                "text": top.get("textDisplay"),
                "like_count": safe_int(top.get("likeCount")),
                "published_at": published_at,
            })

        page_token = resp.get("nextPageToken")
        if not page_token: break
        
    return out

def analyze_sentiment_overall(df):
    if df.empty:
        return {"pos": 0, "neg": 0, "neu": 0, "mean_compound": 0.0}
    scores = df["text"].fillna("").apply(SIA.polarity_scores) # SIA is initialized globally
    df["compound"] = scores.apply(lambda s: s["compound"])
    summary = df["compound"].apply(lambda x: "positive" if x >= 0.05 else ("negative" if x <= -0.05 else "neutral")).value_counts().to_dict()
    return {
        "pos": summary.get("positive", 0),
        "neg": summary.get("negative", 0),
        "neu": summary.get("neutral", 0),
        "mean_compound": round(df["compound"].mean(), 4),
    }

# =========================
# AIRFLOW MAIN CALLABLE
# =========================
def main():
    """
    The main function called by the Airflow PythonOperator.
    This performs the entire backfill job optimized for quota.
    """
    initialize_globals()
    
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
    
    # 1. PRE-FETCH VIDEO IDs (Cost: ~1000 units TOTAL)
    print("--- Phase 1: Pre-fetching Video IDs to save quota ---")
    video_map = {}
    for t in TARGET_SONGS:
        print(f"Finding ID for: {t['artist']} - {t['title']}")
        vid, snip = search_video_id(t['artist'], t['title'])
        if vid:
            video_map[t['title']] = {"id": vid, "snip": snip}
        else:
            print(f"Could not find video for {t['title']}")
    
    print(f"--- Phase 2: Starting Daily Loop for {len(video_map)} videos ---")

    # 2. START DATE LOOP
    # Airflow usually injects context, but since this is a backfill, we hardcode the range.
    start_date = datetime(2025, 11, 1, tzinfo=timezone.utc)
    end_date = datetime(2025, 12, 6, tzinfo=timezone.utc)
    current_date = start_date

    while current_date <= end_date:
        process_date_str = current_date.strftime("%Y%m%d")
        print(f"\nProcessing Date: {process_date_str}")

        window_start = current_date
        window_end = current_date + timedelta(days=1) - timedelta(microseconds=1)
        summary_records = []

        for t in TARGET_SONGS:
            title = t["title"]
            if title not in video_map: continue
            
            vid_data = video_map[title]
            vid = vid_data["id"]
            
            # Fetch stats (1 unit)
            vmeta = get_video_stats(vid)
            if not vmeta: continue
            stats = vmeta["statistics"]
            snip = vmeta["snippet"]

            # Fetch comments (1 unit per page)
            comments = fetch_comments_in_window(vid, window_start, window_end)
            df_comments = pd.DataFrame(comments)
            
            # Save Comments
            if not df_comments.empty:
                fname = t["artist"].replace(" ", "_")
                local_comments = f"comments_{fname}_{process_date_str}.csv"
                df_comments.to_csv(local_comments, index=False)
                upload_to_gcs(local_comments, f"{OUTPUT_PREFIX_COMMENTS}/{local_comments}")

            # Summarize
            senti = analyze_sentiment_overall(df_comments)
            summary_records.append({
                "fetch_time": window_end.isoformat(),
                "video_id": vid,
                "title": snip.get("title"),
                "channel": snip.get("channelTitle"),
                "published_at": snip.get("publishedAt"),
                "views": safe_int(stats.get("viewCount")),
                "likes": safe_int(stats.get("likeCount")),
                "age_days": round(_age_days(snip.get("publishedAt")), 2),
                "comment_count": len(df_comments),
                "avg_comment_rate": len(df_comments), # Simplified, should be rate/day if more days were covered
                "pos_comments": senti["pos"],
                "neu_comments": senti["neu"],
                "neg_comments": senti["neg"],
                "mean_compound": senti["mean_compound"],
            })

        # Save Summary
        if summary_records:
            df_summary = pd.DataFrame(summary_records)
            local_summary = f"youtube_summary_{process_date_str}.csv"
            df_summary.to_csv(local_summary, index=False)
            upload_to_gcs(local_summary, f"{OUTPUT_PREFIX_SUMMARY}/{local_summary}")
            print(f"Finished {process_date_str}")

        current_date += timedelta(days=1)
        time.sleep(1) # Small pause between days to be gentle on resources

    print("\nBackfill completed successfully.")

if __name__ == "__main__":
    # If run locally, outside Airflow
    main()
