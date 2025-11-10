
"""
YouTube Sentiment DAG
---------------------
Fetches YouTube video stats for selected artists and uploads the summary CSV to GCS.

Project: BA882-QSTBA-Group7-Fall2025
Author: <Your Name>
Last updated: 2025-11-10
"""

from datetime import datetime, timedelta, timezone
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.hooks.base import BaseHook

from google.cloud import secretmanager, storage
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# ===== GCP CONFIG =====
GCP_CONN_ID = "gcp_conn"
SECRET_ID = "youtube-api-key"
PROJECT_ID = "ba882-qstba-group7-fall2025"
BUCKET_NAME = "apidatabase"
OUTPUT_PREFIX_SUMMARY = "youtube/summary"


# ===== HELPERS =====
def _get_gcp_creds():
    """Retrieve GCP service account credentials from Airflow Connection."""
    conn = BaseHook.get_connection(GCP_CONN_ID)
    info = conn.extra_dejson.get("extra__google_cloud_platform__keyfile_dict")
    if not info:
        raise RuntimeError("Missing keyfile_dict in GCP connection extras.")
    return service_account.Credentials.from_service_account_info(info)


def _get_youtube_key(creds):
    """Access YouTube API key from GCP Secret Manager."""
    sm = secretmanager.SecretManagerServiceClient(credentials=creds)
    name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID}/versions/latest"
    return sm.access_secret_version(request={"name": name}).payload.data.decode("utf-8")


def _upload_to_gcs(local_path, remote_path, creds):
    """Upload a local file to a GCS bucket."""
    client = storage.Client(credentials=creds, project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(local_path)
    print(f"☁️ Uploaded: gs://{BUCKET_NAME}/{remote_path}")


# ===== MAIN TASK =====
def run_youtube_sentiment(**_):
    """Fetch YouTube video data and upload summary CSV to GCS."""
    creds = _get_gcp_creds()
    api_key = _get_youtube_key(creds)
    print("✅ Got API key")

    youtube = build("youtube", "v3", developerKey=api_key)

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()

    # Example songs — can be replaced with dynamic input or extended list
    songs = [
        {"artist": "Taylor Swift", "title": "The Fate of Ophelia"},
        {"artist": "Ed Sheeran", "title": "Camera"},
    ]

    records = []
    today_str = datetime.now().strftime("%Y%m%d")

    for s in songs:
        q = f"{s['artist']} {s['title']}"
        print(f"🎵 Searching: {q}")

        try:
            res = youtube.search().list(
                q=q, part="id,snippet", type="video", maxResults=1
            ).execute()
            vid = res["items"][0]["id"]["videoId"]
        except Exception as e:
            print(f"❌ Failed to search: {e}")
            continue

        try:
            vmeta = youtube.videos().list(part="snippet,statistics", id=vid).execute()
        except HttpError as e:
            print(f"❌ Stats error: {e}")
            continue

        snip = vmeta["items"][0]["snippet"]
        stats = vmeta["items"][0]["statistics"]
        title = snip.get("title")
        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))

        record = {
            "artist": s["artist"],
            "title": title,
            "views": views,
            "likes": likes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        records.append(record)

    df = pd.DataFrame(records)
    local_csv = f"/tmp/youtube_summary_{today_str}.csv"
    df.to_csv(local_csv, index=False)

    remote_path = f"{OUTPUT_PREFIX_SUMMARY}/youtube_summary_{today_str}.csv"
    _upload_to_gcs(local_csv, remote_path, creds)
    print("✅ Job complete and uploaded.")


# ===== DAG DEFINITION =====
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="youtube_sentiment_dag",
    default_args=default_args,
    description="YouTube Sentiment ETL DAG",
    schedule="0 1 * * *",  # runs daily at 8 PM EST
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=["youtube", "etl"],
) as dag:
    run = PythonOperator(
        task_id="run_youtube_sentiment",
        python_callable=run_youtube_sentiment,
    )
