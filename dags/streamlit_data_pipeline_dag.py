"""
Streamlit Data Pipeline DAG - Self-Contained Version
----------------------------------------------------
Automated pipeline to prepare data for Streamlit dashboard deployment.
All processing logic is embedded - no external dependencies required.

This DAG:
1. Processes raw YouTube and Reddit data from local gcs_downloads/ directory
2. Cleans and consolidates data into parquet files
3. Merges Reddit data into YouTube summary
4. Uploads processed files to GCS Silver Layer (cleaned/)

Project: BA882-QSTBA-Group7-Fall2025
Author: Team 7
Last updated: 2025-12-08
Version: 2.0 (Self-Contained)
"""

from datetime import datetime, timedelta
from airflow import DAG
try:
    from airflow.operators.python import PythonOperator
except ImportError:
    # Fallback for older Airflow versions
    from airflow.providers.standard.operators.python import PythonOperator
from airflow.hooks.base import BaseHook

from google.cloud import storage
from google.oauth2 import service_account
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

# ===== CONFIG =====
GCP_CONN_ID = "gcp_conn"
PROJECT_ID = "ba882-qstba-group7-fall2025"
BUCKET_NAME = "apidatabase"

# Paths configuration
# In Astronomer/Airflow, DAG files are typically in the repo root
# We'll use the DAG file's location to find project root
DAG_FILE = Path(__file__)
DAG_DIR = DAG_FILE.parent
# Project root is the directory containing dags/ folder
ROOT_DIR = DAG_DIR.parent if DAG_DIR.name == "dags" else DAG_DIR

# ===== EMBEDDED PROCESSING LOGIC =====

# YouTube Column Aliases
SUMMARY_COLUMN_ALIASES: Dict[str, str] = {
    "artist": "channel",
    "channel": "channel",
    "title": "title",
    "track_name": "title",
    "views": "views",
    "view_count": "views",
    "likes": "likes",
    "like_count": "likes",
    "comment_count": "comment_count",
    "comments": "comment_count",
    "pos_comments": "pos_comments",
    "neu_comments": "neu_comments",
    "neg_comments": "neg_comments",
    "mean_compound": "mean_compound",
    "compound": "mean_compound",
    "fetch_time": "fetch_time",
    "timestamp": "fetch_time",
    "snapshot_ts": "fetch_time",
    "published_at": "published_at",
    "publish_time": "published_at",
    "video_id": "video_id",
    "id": "video_id",
}

SUMMARY_NUMERIC_COLUMNS = [
    "views", "likes", "comment_count", "pos_comments",
    "neu_comments", "neg_comments", "mean_compound",
]

SUMMARY_DEFAULT_COLUMNS = [
    "snapshot_date", "video_id", "title", "channel", "published_at",
    "views", "likes", "comment_count", "pos_comments", "neu_comments",
    "neg_comments", "mean_compound", "fetch_time",
]

COMMENTS_COLUMN_ALIASES: Dict[str, str] = {
    "artist": "artist",
    "channel": "artist",
    "video_id": "video_id",
    "id": "video_id",
    "author": "author",
    "user": "author",
    "text": "text",
    "body": "text",
    "comment": "text",
    "like_count": "like_count",
    "likes": "like_count",
    "published_at": "published_at",
    "timestamp": "published_at",
    "compound": "compound",
    "sentiment": "label",
    "label": "label",
}

COMMENTS_DEFAULT_COLUMNS = [
    "snapshot_date", "artist", "video_id", "author", "text",
    "like_count", "published_at", "compound", "label",
]

# Reddit Default Columns
REDDIT_SUMMARY_COLUMNS = [
    "snapshot_date", "submission_id", "subreddit", "artist", "song",
    "title", "author", "score", "num_comments", "pos_comments",
    "neu_comments", "neg_comments", "mean_compound", "created_utc",
    "fetch_time", "permalink",
]

REDDIT_SUMMARY_NUMERIC = [
    "score", "num_comments", "pos_comments", "neu_comments",
    "neg_comments", "mean_compound",
]

REDDIT_COMMENTS_COLUMNS = [
    "snapshot_date", "submission_id", "comment_id", "artist", "song",
    "author", "body", "score", "created_utc", "compound", "label", "permalink",
]


# ===== HELPER FUNCTIONS =====

def _get_gcp_creds():
    """Retrieve GCP service account credentials from Airflow Connection."""
    conn = BaseHook.get_connection(GCP_CONN_ID)
    # Try different possible key locations in connection extras
    info = conn.extra_dejson.get("extra__google_cloud_platform__keyfile_dict")
    if not info:
        info = conn.extra_dejson.get("keyfile_dict")
    if not info:
        raise RuntimeError(
            f"Missing keyfile_dict in GCP connection '{GCP_CONN_ID}' extras. "
            "Please configure the connection with Service Account JSON credentials."
        )
    return service_account.Credentials.from_service_account_info(info)


def _upload_to_gcs(local_path: str, gcs_path: str, creds):
    """Upload a local file to GCS."""
    client = storage.Client(credentials=creds, project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"   ✅ Uploaded: gs://{BUCKET_NAME}/{gcs_path}")


def _extract_snapshot_date(path: Path) -> pd.Timestamp:
    """Extract date from filename (YYYYMMDD format)."""
    match = pd.Series([path.stem]).str.extract(r"(\d{8})")[0]
    if pd.isna(match.iloc[0]):
        return pd.NaT
    return pd.to_datetime(match.iloc[0], format="%Y%m%d")


def _normalise_columns(df: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    """Normalize column names using aliases."""
    renamed = {col: aliases[col] for col in df.columns if col in aliases}
    return df.rename(columns=renamed)


def _to_datetime(series: pd.Series, utc: bool = True) -> pd.Series:
    """Convert series to datetime."""
    return pd.to_datetime(series, errors="coerce", utc=utc).dt.tz_convert(None)


# ===== YOUTUBE PROCESSING FUNCTIONS =====

def _load_youtube_summary_frames(raw_summary_dir: Path) -> List[pd.DataFrame]:
    """Load and normalize YouTube summary CSV files."""
    frames: List[pd.DataFrame] = []
    if not raw_summary_dir.exists():
        return frames

    for csv_path in sorted(raw_summary_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df = _normalise_columns(df, SUMMARY_COLUMN_ALIASES)
            df["snapshot_date"] = _extract_snapshot_date(csv_path)

            for col in SUMMARY_NUMERIC_COLUMNS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "fetch_time" in df.columns:
                df["fetch_time"] = pd.to_datetime(df["fetch_time"], errors="coerce")
            if "published_at" in df.columns:
                df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

            if "comment_count" not in df.columns:
                if {"pos_comments", "neu_comments", "neg_comments"} <= set(df.columns):
                    df["comment_count"] = (
                        df["pos_comments"].fillna(0) +
                        df["neu_comments"].fillna(0) +
                        df["neg_comments"].fillna(0)
                    )
                else:
                    df["comment_count"] = pd.NA

            for required in SUMMARY_DEFAULT_COLUMNS:
                if required not in df.columns:
                    df[required] = pd.NA

            frames.append(df[SUMMARY_DEFAULT_COLUMNS])
        except Exception as e:
            print(f"   ⚠️  Error processing {csv_path.name}: {e}")
            continue

    return frames


def _clean_youtube_summary(raw_summary_dir: Path) -> pd.DataFrame:
    """Clean YouTube summary data."""
    frames = _load_youtube_summary_frames(raw_summary_dir)
    if not frames:
        return pd.DataFrame(columns=SUMMARY_DEFAULT_COLUMNS)

    summary = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["snapshot_date", "video_id", "channel", "title"], keep="last")
    )
    summary["snapshot_date"] = pd.to_datetime(summary["snapshot_date"], errors="coerce")

    for col in SUMMARY_NUMERIC_COLUMNS:
        summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0)

    summary.sort_values(["snapshot_date", "channel", "title"], inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary


def _load_youtube_comment_frames(raw_comments_dir: Path) -> List[pd.DataFrame]:
    """Load and normalize YouTube comment CSV files."""
    frames: List[pd.DataFrame] = []
    if not raw_comments_dir.exists():
        return frames

    for csv_path in sorted(raw_comments_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df = _normalise_columns(df, COMMENTS_COLUMN_ALIASES)
            df["snapshot_date"] = _extract_snapshot_date(csv_path)

            if "artist" not in df.columns:
                parts = csv_path.stem.split("_")
                if len(parts) > 2:
                    df["artist"] = " ".join(parts[1:-1]).replace("-", " ").replace(".", "")
                else:
                    df["artist"] = pd.NA

            df["like_count"] = pd.to_numeric(df.get("like_count"), errors="coerce").fillna(0).astype(int)
            if "published_at" in df.columns:
                df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

            for required in COMMENTS_DEFAULT_COLUMNS:
                if required not in df.columns:
                    df[required] = pd.NA

            frames.append(df[COMMENTS_DEFAULT_COLUMNS])
        except Exception as e:
            print(f"   ⚠️  Error processing {csv_path.name}: {e}")
            continue

    return frames


def _clean_youtube_comments(raw_comments_dir: Path) -> pd.DataFrame:
    """Clean YouTube comments data."""
    frames = _load_youtube_comment_frames(raw_comments_dir)
    if not frames:
        return pd.DataFrame(columns=COMMENTS_DEFAULT_COLUMNS)

    comments = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(
            subset=["snapshot_date", "video_id", "author", "published_at", "text"],
            keep="last",
        )
    )
    comments["snapshot_date"] = pd.to_datetime(comments["snapshot_date"], errors="coerce")
    comments.sort_values(["snapshot_date", "artist", "published_at"], inplace=True)
    comments.reset_index(drop=True, inplace=True)
    return comments


# ===== REDDIT PROCESSING FUNCTIONS =====

def _load_reddit_summary_frames(raw_summary_dir: Path) -> List[pd.DataFrame]:
    """Load and normalize Reddit summary CSV files."""
    frames: List[pd.DataFrame] = []
    if not raw_summary_dir.exists():
        return frames

    for csv_path in sorted(raw_summary_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            # Unify artist/song columns
            artist_cols = [col for col in df.columns if col.startswith("artist")]
            song_cols = [col for col in df.columns if col.startswith("song")]
            if artist_cols:
                df["artist"] = (
                    df[artist_cols]
                    .replace({np.nan: None, "": None})
                    .bfill(axis=1)
                    .iloc[:, 0]
                )
            if song_cols:
                df["song"] = (
                    df[song_cols]
                    .replace({np.nan: None, "": None})
                    .bfill(axis=1)
                    .iloc[:, 0]
                )

            df["fetch_time"] = _to_datetime(df.get("fetch_time"))
            df["created_utc"] = _to_datetime(df.get("created_utc"))
            df["snapshot_date"] = df["fetch_time"].dt.normalize()

            for col in REDDIT_SUMMARY_NUMERIC:
                df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

            for required in REDDIT_SUMMARY_COLUMNS:
                if required not in df.columns:
                    df[required] = pd.NA

            frames.append(df[REDDIT_SUMMARY_COLUMNS])
        except Exception as e:
            print(f"   ⚠️  Error processing {csv_path.name}: {e}")
            continue

    return frames


def _clean_reddit_summary(raw_summary_dir: Path) -> pd.DataFrame:
    """Clean Reddit summary data."""
    frames = _load_reddit_summary_frames(raw_summary_dir)
    if not frames:
        return pd.DataFrame(columns=REDDIT_SUMMARY_COLUMNS)

    summary = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["snapshot_date", "submission_id"], keep="last")
        .sort_values(["snapshot_date", "artist", "song", "subreddit"])
    )
    summary.reset_index(drop=True, inplace=True)
    return summary


def _load_reddit_comment_frames(raw_comments_dir: Path) -> List[pd.DataFrame]:
    """Load and normalize Reddit comment CSV files."""
    frames: List[pd.DataFrame] = []
    if not raw_comments_dir.exists():
        return frames

    for csv_path in sorted(raw_comments_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            df["created_utc"] = _to_datetime(df.get("created_utc"))
            df["snapshot_date"] = df["created_utc"].dt.normalize()
            df["score"] = pd.to_numeric(df.get("score"), errors="coerce").fillna(0).astype(int)
            df["compound"] = pd.to_numeric(df.get("compound"), errors="coerce")

            for required in REDDIT_COMMENTS_COLUMNS:
                if required not in df.columns:
                    df[required] = pd.NA

            frames.append(df[REDDIT_COMMENTS_COLUMNS])
        except Exception as e:
            print(f"   ⚠️  Error processing {csv_path.name}: {e}")
            continue

    return frames


def _clean_reddit_comments(raw_comments_dir: Path) -> pd.DataFrame:
    """Clean Reddit comments data."""
    frames = _load_reddit_comment_frames(raw_comments_dir)
    if not frames:
        return pd.DataFrame(columns=REDDIT_COMMENTS_COLUMNS)

    comments = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(
            subset=["snapshot_date", "submission_id", "comment_id", "author", "created_utc"],
            keep="last",
        )
        .sort_values(["snapshot_date", "artist", "created_utc"])
    )
    comments.reset_index(drop=True, inplace=True)
    return comments


# ===== DAG TASKS =====

def process_youtube_data(**context):
    """
    Process YouTube data: clean and generate parquet files.
    """
    print("=" * 80)
    print("Processing YouTube Data for Streamlit")
    print("=" * 80)

    creds = _get_gcp_creds()
    processed_dir = ROOT_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_summary_dir = ROOT_DIR / "gcs_downloads" / "summary"
    raw_comments_dir = ROOT_DIR / "gcs_downloads" / "comments"

    print(f"\n📁 Project root: {ROOT_DIR}")
    print(f"📁 Raw summary dir: {raw_summary_dir} (exists: {raw_summary_dir.exists()})")
    print(f"📁 Raw comments dir: {raw_comments_dir} (exists: {raw_comments_dir.exists()})")

    print("\n🔄 Processing YouTube data...")

    # Clean data
    summary = _clean_youtube_summary(raw_summary_dir)
    comments = _clean_youtube_comments(raw_comments_dir)

    # Save parquet files
    summary_path = processed_dir / "youtube_summary.parquet"
    comments_path = processed_dir / "youtube_comments.parquet"

    summary.to_parquet(summary_path, index=False)
    comments.to_parquet(comments_path, index=False)

    print(f"✅ YouTube processing complete!")
    print(f"   Summary: {len(summary)} rows -> {summary_path}")
    print(f"   Comments: {len(comments)} rows -> {comments_path}")

    # Upload comments to Silver Layer
    execution_date = context['ds_nodash']
    if comments_path.exists() and len(comments) > 0:
        csv_comments_path = comments_path.with_suffix('.csv')
        comments.to_csv(csv_comments_path, index=False)

        gcs_comments_path = f"cleaned/all_comments_{execution_date}.csv"
        _upload_to_gcs(str(csv_comments_path), gcs_comments_path, creds)
        print(f"   ✅ Uploaded to Silver Layer: {gcs_comments_path}")

    return "YouTube data processed successfully"


def process_reddit_data(**context):
    """
    Process Reddit data: clean and generate parquet files.
    """
    print("=" * 80)
    print("Processing Reddit Data for Streamlit")
    print("=" * 80)

    creds = _get_gcp_creds()
    processed_dir = ROOT_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_summary_dir = ROOT_DIR / "gcs_downloads" / "reddit" / "summary"
    raw_comments_dir = ROOT_DIR / "gcs_downloads" / "reddit" / "comments"

    print(f"\n📁 Project root: {ROOT_DIR}")
    print(f"📁 Raw summary dir: {raw_summary_dir} (exists: {raw_summary_dir.exists()})")
    print(f"📁 Raw comments dir: {raw_comments_dir} (exists: {raw_comments_dir.exists()})")

    print("\n🔄 Processing Reddit data...")

    # Clean data
    summary = _clean_reddit_summary(raw_summary_dir)
    comments = _clean_reddit_comments(raw_comments_dir)

    # Save parquet files
    summary_path = processed_dir / "reddit_summary.parquet"
    comments_path = processed_dir / "reddit_comments.parquet"

    summary.to_parquet(summary_path, index=False)
    comments.to_parquet(comments_path, index=False)

    print(f"✅ Reddit processing complete!")
    print(f"   Summary: {len(summary)} rows -> {summary_path}")
    print(f"   Comments: {len(comments)} rows -> {comments_path}")

    return "Reddit data processed successfully"


def merge_reddit_into_summary(**context):
    """
    Merge Reddit data into YouTube summary to create unified daily_song_summary.
    This creates the final Silver Layer summary file with reddit_comment_count.
    """
    print("=" * 80)
    print("Merging Reddit Data into Summary")
    print("=" * 80)

    creds = _get_gcp_creds()
    execution_date = context['ds_nodash']
    processed_dir = ROOT_DIR / "data" / "processed"

    # Load YouTube summary
    youtube_summary_path = processed_dir / "youtube_summary.parquet"
    if not youtube_summary_path.exists():
        raise FileNotFoundError(f"YouTube summary not found: {youtube_summary_path}")

    print(f"\n📥 Loading YouTube summary...")
    df_youtube = pd.read_parquet(youtube_summary_path)
    print(f"   YouTube summary: {len(df_youtube)} rows")

    # Standardize YouTube columns
    if 'channel' in df_youtube.columns and 'artist' not in df_youtube.columns:
        df_youtube['artist'] = df_youtube['channel']
    if 'title' in df_youtube.columns and 'song' not in df_youtube.columns:
        df_youtube['song'] = df_youtube['title']

    # Load Reddit summary
    reddit_summary_path = processed_dir / "reddit_summary.parquet"
    df_reddit = None
    if reddit_summary_path.exists():
        print(f"\n📥 Loading Reddit summary...")
        df_reddit = pd.read_parquet(reddit_summary_path)
        print(f"   Reddit summary: {len(df_reddit)} rows")

        # Fix missing artist/song in Reddit data
        if 'artist' in df_reddit.columns:
            if 'artist_x' in df_reddit.columns:
                df_reddit['artist'] = df_reddit['artist'].fillna(df_reddit['artist_x'])
            if 'artist_y' in df_reddit.columns:
                df_reddit['artist'] = df_reddit['artist'].fillna(df_reddit['artist_y'])

        if 'song' in df_reddit.columns:
            if 'song_x' in df_reddit.columns:
                df_reddit['song'] = df_reddit['song'].fillna(df_reddit['song_x'])
            if 'song_y' in df_reddit.columns:
                df_reddit['song'] = df_reddit['song'].fillna(df_reddit['song_y'])

        # Filter out rows without artist
        df_reddit = df_reddit[df_reddit['artist'].notna()].copy()

        if len(df_reddit) > 0:
            # Aggregate Reddit comments by artist, song, and date
            print(f"\n🔄 Aggregating Reddit comments...")
            reddit_agg = df_reddit.groupby(['snapshot_date', 'artist', 'song']).agg({
                'num_comments': 'sum'
            }).reset_index()
            reddit_agg.rename(columns={'num_comments': 'reddit_comment_count'}, inplace=True)
            print(f"   Reddit aggregated: {len(reddit_agg)} unique artist/song/date combinations")
            print(f"   Total Reddit comments: {reddit_agg['reddit_comment_count'].sum():,.0f}")
        else:
            print(f"   ⚠️  No valid Reddit data after filtering")
            reddit_agg = pd.DataFrame(columns=['snapshot_date', 'artist', 'song', 'reddit_comment_count'])
    else:
        print(f"\n⚠️  Reddit summary not found: {reddit_summary_path}")
        reddit_agg = pd.DataFrame(columns=['snapshot_date', 'artist', 'song', 'reddit_comment_count'])

    # Merge Reddit data into YouTube summary
    print(f"\n🔗 Merging Reddit data into YouTube summary...")

    # Ensure snapshot_date is datetime for both
    df_youtube['snapshot_date'] = pd.to_datetime(df_youtube['snapshot_date'], errors='coerce')
    if len(reddit_agg) > 0:
        reddit_agg['snapshot_date'] = pd.to_datetime(reddit_agg['snapshot_date'], errors='coerce')

        # Normalize artist names for better matching (case-insensitive, strip whitespace)
        df_youtube['artist_normalized'] = df_youtube['artist'].astype(str).str.strip().str.lower()
        reddit_agg['artist_normalized'] = reddit_agg['artist'].astype(str).str.strip().str.lower()

        df_youtube['song_normalized'] = df_youtube['song'].astype(str).str.strip().str.lower()
        reddit_agg['song_normalized'] = reddit_agg['song'].astype(str).str.strip().str.lower()

        # Merge on normalized keys
        df_merged = df_youtube.merge(
            reddit_agg[['snapshot_date', 'artist_normalized', 'song_normalized', 'reddit_comment_count']],
            on=['snapshot_date', 'artist_normalized', 'song_normalized'],
            how='left'
        )

        # Drop normalized columns
        df_merged = df_merged.drop(columns=['artist_normalized', 'song_normalized'])

        # Fill missing reddit_comment_count with 0
        df_merged['reddit_comment_count'] = df_merged['reddit_comment_count'].fillna(0).astype(int)

        print(f"   Merged summary: {len(df_merged)} rows")
        print(f"   Rows with Reddit data: {(df_merged['reddit_comment_count'] > 0).sum()}")
        print(f"   Total Reddit comments in merged: {df_merged['reddit_comment_count'].sum():,.0f}")
    else:
        # No Reddit data, just add column with 0
        df_merged = df_youtube.copy()
        df_merged['reddit_comment_count'] = 0
        print(f"   No Reddit data to merge, added reddit_comment_count=0")

    # Standardize column names for Silver Layer
    if 'snapshot_date' in df_merged.columns:
        df_merged['date'] = df_merged['snapshot_date'].dt.date

    # Rename YouTube columns to match expected format
    column_mapping = {
        'views': 'youtube_views',
        'likes': 'youtube_likes',
        'comment_count': 'youtube_comment_count',
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df_merged.columns and new_col not in df_merged.columns:
            df_merged[new_col] = df_merged[old_col]

    # Calculate youtube_pos_ratio if not present
    if 'youtube_pos_ratio' not in df_merged.columns:
        if 'pos_comments' in df_merged.columns and 'youtube_comment_count' in df_merged.columns:
            df_merged['youtube_pos_ratio'] = (
                df_merged['pos_comments'] / df_merged['youtube_comment_count'].replace(0, 1)
            ).fillna(0)
        else:
            df_merged['youtube_pos_ratio'] = 0.0

    # Ensure numeric columns are numeric
    numeric_cols = ['youtube_views', 'youtube_likes', 'youtube_comment_count',
                    'pos_comments', 'youtube_pos_ratio', 'reddit_comment_count']
    for col in numeric_cols:
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0)

    # Select final columns for Silver Layer (keep all important columns)
    final_columns = [
        'date', 'snapshot_date', 'artist', 'song', 'channel', 'title',
        'youtube_views', 'youtube_likes', 'youtube_comment_count',
        'pos_comments', 'neu_comments', 'neg_comments',
        'youtube_pos_ratio', 'reddit_comment_count',
        'mean_compound', 'video_id', 'published_at', 'fetch_time'
    ]

    # Keep only columns that exist
    available_columns = [col for col in final_columns if col in df_merged.columns]
    df_final = df_merged[available_columns].copy()

    # Save merged summary to CSV
    csv_summary_path = processed_dir / f"daily_song_summary_{execution_date}.csv"
    df_final.to_csv(csv_summary_path, index=False)
    print(f"\n💾 Saved merged summary: {csv_summary_path}")
    print(f"   Rows: {len(df_final)}")
    print(f"   Columns: {list(df_final.columns)}")

    # Upload to Silver Layer
    gcs_summary_path = f"cleaned/daily_song_summary_{execution_date}.csv"
    _upload_to_gcs(str(csv_summary_path), gcs_summary_path, creds)
    print(f"   ✅ Uploaded to Silver Layer: gs://{BUCKET_NAME}/{gcs_summary_path}")

    return "Reddit data merged successfully"


def verify_streamlit_data(**context):
    """
    Verify that all required data files exist for Streamlit dashboard.
    """
    print("=" * 80)
    print("Verifying Streamlit Data Files")
    print("=" * 80)

    creds = _get_gcp_creds()
    execution_date = context['ds_nodash']

    from google.cloud import storage
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_info(
        BaseHook.get_connection(GCP_CONN_ID).extra_dejson.get("keyfile_dict") or
        BaseHook.get_connection(GCP_CONN_ID).extra_dejson.get("extra__google_cloud_platform__keyfile_dict")
    )
    client = storage.Client(credentials=credentials, project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    required_files = [
        f"cleaned/daily_song_summary_{execution_date}.csv",
        f"cleaned/all_comments_{execution_date}.csv",
    ]

    all_exist = True
    for file_path in required_files:
        blob = bucket.blob(file_path)
        if blob.exists():
            size = blob.size
            print(f"✅ {file_path} exists ({size:,} bytes)")
        else:
            print(f"❌ {file_path} NOT FOUND")
            all_exist = False

    if all_exist:
        print("\n✅ All required Streamlit data files verified in GCS!")
        return "All files verified"
    else:
        raise ValueError("Missing required data files in GCS for Streamlit dashboard")


# ===== DAG DEFINITION =====
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="streamlit_data_pipeline",
    default_args=default_args,
    description="Process data and prepare files for Streamlit dashboard deployment (Self-Contained)",
    schedule="0 2 * * *",  # Run daily at 2 AM UTC (9 PM EST previous day)
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=["streamlit", "data-pipeline", "etl", "self-contained"],
    max_active_runs=1,
) as dag:

    # Task 1: Process YouTube data
    process_youtube = PythonOperator(
        task_id="process_youtube_data",
        python_callable=process_youtube_data,
    )

    # Task 2: Process Reddit data
    process_reddit = PythonOperator(
        task_id="process_reddit_data",
        python_callable=process_reddit_data,
    )

    # Task 3: Merge Reddit data into summary
    merge_summary = PythonOperator(
        task_id="merge_reddit_into_summary",
        python_callable=merge_reddit_into_summary,
    )

    # Task 4: Verify all data files exist in GCS
    verify_data = PythonOperator(
        task_id="verify_streamlit_data",
        python_callable=verify_streamlit_data,
    )

    # Set task dependencies: process both in parallel, then merge, then verify
    [process_youtube, process_reddit] >> merge_summary >> verify_data
