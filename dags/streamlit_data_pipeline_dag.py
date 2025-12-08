"""
Streamlit Data Pipeline DAG
----------------------------
Automated pipeline to prepare data for Streamlit dashboard deployment.

This DAG:
1. Processes raw YouTube and Reddit data from GCS
2. Cleans and consolidates data into parquet files
3. Prepares data files for Streamlit dashboard

Project: BA882-QSTBA-Group7-Fall2025
Author: Team 7
Last updated: 2025-12-08
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
import sys
from pathlib import Path
import pandas as pd

# ===== CONFIG =====
GCP_CONN_ID = "gcp_conn"
PROJECT_ID = "ba882-qstba-group7-fall2025"
BUCKET_NAME = "apidatabase"
GCS_DATA_PREFIX = "youtube"  # Path in GCS where raw data is stored

# Paths configuration
# In Astronomer/Airflow, DAG files are typically in the repo root
# We'll use the DAG file's location to find project root
DAG_FILE = Path(__file__)
DAG_DIR = DAG_FILE.parent
# Project root is the directory containing dags/ folder
ROOT_DIR = DAG_DIR.parent if DAG_DIR.name == "dags" else DAG_DIR


# ===== HELPER FUNCTIONS =====
def _get_gcp_creds():
    """Retrieve GCP service account credentials from Airflow Connection."""
    conn = BaseHook.get_connection(GCP_CONN_ID)
    info = conn.extra_dejson.get("extra__google_cloud_platform__keyfile_dict")
    if not info:
        raise RuntimeError("Missing keyfile_dict in GCP connection extras.")
    return service_account.Credentials.from_service_account_info(info)


def _download_from_gcs(gcs_path: str, local_path: str, creds):
    """Download a file from GCS to local path."""
    client = storage.Client(credentials=creds, project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    
    local_file = Path(local_path)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    
    blob.download_to_filename(str(local_file))
    print(f"✅ Downloaded: gs://{BUCKET_NAME}/{gcs_path} -> {local_path}")


def _upload_to_gcs(local_path: str, gcs_path: str, creds):
    """Upload a local file to GCS."""
    client = storage.Client(credentials=creds, project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"✅ Uploaded: {local_path} -> gs://{BUCKET_NAME}/{gcs_path}")


# ===== TASKS =====
def process_youtube_data(**context):
    """
    Process YouTube data: download from GCS, clean, and generate parquet files.
    """
    print("=" * 80)
    print("Processing YouTube Data for Streamlit")
    print("=" * 80)
    
    creds = _get_gcp_creds()
    
    # Import the cleaning script
    sys.path.insert(0, str(ROOT_DIR / "src" / "processing"))
    from clean_youtube import Paths, run
    
    paths = Paths(root=ROOT_DIR)
    
    # Note: Raw data should already be in gcs_downloads/ directory
    # The cleaning scripts will read from there
    print("\n📥 Data should be in gcs_downloads/ directory...")
    
    # Process data
    print("\n🔄 Processing YouTube data...")
    run(paths)
    
    print(f"\n✅ YouTube processing complete!")
    print(f"   Summary: {paths.processed_summary_path}")
    print(f"   Comments: {paths.processed_comments_path}")
    
    # Upload processed files to Silver Layer (cleaned/) with date suffix
    print("\n☁️ Uploading processed files to Silver Layer (cleaned/)...")
    execution_date = context['ds_nodash']  # Format: YYYYMMDD
    
    # Note: YouTube summary will be merged with Reddit in merge_reddit_into_summary task
    # Don't upload YouTube summary separately here
    if paths.processed_summary_path.exists():
        print(f"   ℹ️  YouTube summary processed (will be merged with Reddit in next step)")
    
    if paths.processed_comments_path.exists():
        # Convert parquet to CSV for Silver Layer compatibility
        df_comments = pd.read_parquet(paths.processed_comments_path)
        csv_comments_path = paths.processed_comments_path.with_suffix('.csv')
        df_comments.to_csv(csv_comments_path, index=False)
        
        # Upload to cleaned/ with date suffix
        gcs_comments_path = f"cleaned/all_comments_{execution_date}.csv"
        _upload_to_gcs(str(csv_comments_path), gcs_comments_path, creds)
        print(f"   ✅ Uploaded to Silver Layer: {gcs_comments_path}")
    
    return "YouTube data processed successfully"


def process_reddit_data(**context):
    """
    Process Reddit data: download from GCS, clean, and generate parquet files.
    """
    print("=" * 80)
    print("Processing Reddit Data for Streamlit")
    print("=" * 80)
    
    creds = _get_gcp_creds()
    
    # Import the cleaning script
    sys.path.insert(0, str(ROOT_DIR / "src" / "processing"))
    from clean_reddit import Paths, run
    
    paths = Paths(root=ROOT_DIR)
    
    # Process data
    print("\n🔄 Processing Reddit data...")
    run(paths)
    
    print(f"\n✅ Reddit processing complete!")
    print(f"   Summary: {paths.processed_summary_path}")
    print(f"   Comments: {paths.processed_comments_path}")
    
    # Upload processed files to Silver Layer (cleaned/) with date suffix
    print("\n☁️ Uploading processed files to Silver Layer (cleaned/)...")
    execution_date = context['ds_nodash']  # Format: YYYYMMDD
    
    # Note: Reddit summary will be merged with YouTube in merge_reddit_into_summary task
    # Don't upload Reddit summary separately here
    if paths.processed_summary_path.exists():
        print(f"   ℹ️  Reddit summary processed (will be merged with YouTube in next step)")
    
    if paths.processed_comments_path.exists():
        # Convert parquet to CSV for Silver Layer compatibility
        df_comments = pd.read_parquet(paths.processed_comments_path)
        csv_comments_path = paths.processed_comments_path.with_suffix('.csv')
        df_comments.to_csv(csv_comments_path, index=False)
        
        # Upload to cleaned/ with date suffix
        gcs_comments_path = f"cleaned/all_comments_{execution_date}.csv"
        _upload_to_gcs(str(csv_comments_path), gcs_comments_path, creds)
        print(f"   ✅ Uploaded to Silver Layer: {gcs_comments_path}")
    
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
    execution_date = context['ds_nodash']  # Format: YYYYMMDD
    
    processed_dir = ROOT_DIR / "data" / "processed"
    
    # Load YouTube summary
    youtube_summary_path = processed_dir / "youtube_summary.parquet"
    if not youtube_summary_path.exists():
        raise FileNotFoundError(f"YouTube summary not found: {youtube_summary_path}")
    
    print(f"\n📥 Loading YouTube summary...")
    df_youtube = pd.read_parquet(youtube_summary_path)
    print(f"   YouTube summary: {len(df_youtube)} rows")
    
    # Standardize YouTube columns
    # YouTube uses 'channel' but we need 'artist' for merging
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
            # Fill NaN artist from artist_x or artist_y
            if 'artist_x' in df_reddit.columns:
                df_reddit['artist'] = df_reddit['artist'].fillna(df_reddit['artist_x'])
            if 'artist_y' in df_reddit.columns:
                df_reddit['artist'] = df_reddit['artist'].fillna(df_reddit['artist_y'])
        
        if 'song' in df_reddit.columns:
            # Fill NaN song from song_x or song_y
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
    # Ensure we have: date, artist, song, youtube_views, youtube_likes, youtube_comment_count, etc.
    if 'snapshot_date' in df_merged.columns:
        df_merged['date'] = df_merged['snapshot_date'].dt.date
    
    # Rename YouTube columns to match expected format
    column_mapping = {
        'views': 'youtube_views',
        'likes': 'youtube_likes',
        'comment_count': 'youtube_comment_count',
        'channel': 'artist',  # Keep both for compatibility
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df_merged.columns and new_col not in df_merged.columns:
            df_merged[new_col] = df_merged[old_col]
    
    # Select final columns for Silver Layer
    final_columns = [
        'date', 'artist', 'song',
        'youtube_views', 'youtube_likes', 'youtube_comment_count',
        'pos_comments', 'youtube_pos_ratio', 'reddit_comment_count'
    ]
    
    # Keep only columns that exist
    available_columns = [col for col in final_columns if col in df_merged.columns]
    df_final = df_merged[available_columns].copy()
    
    # Calculate youtube_pos_ratio if not present
    if 'youtube_pos_ratio' not in df_final.columns:
        if 'pos_comments' in df_final.columns and 'youtube_comment_count' in df_final.columns:
            df_final['youtube_pos_ratio'] = (
                df_final['pos_comments'] / df_final['youtube_comment_count'].replace(0, 1)
            ).fillna(0)
        else:
            df_final['youtube_pos_ratio'] = 0.0
    
    # Ensure numeric columns are numeric
    numeric_cols = ['youtube_views', 'youtube_likes', 'youtube_comment_count', 
                    'pos_comments', 'youtube_pos_ratio', 'reddit_comment_count']
    for col in numeric_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)
    
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
    
    processed_dir = ROOT_DIR / "data" / "processed"
    required_files = [
        "youtube_summary.parquet",
        "youtube_comments.parquet",
    ]
    
    all_exist = True
    for file_name in required_files:
        file_path = processed_dir / file_name
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        
        if exists:
            size = file_path.stat().st_size
            print(f"{status} {file_name}: {size:,} bytes")
        else:
            print(f"{status} {file_name}: NOT FOUND")
            all_exist = False
    
    # Reddit files are optional
    optional_files = [
        "reddit_summary.parquet",
        "reddit_comments.parquet",
    ]
    for file_name in optional_files:
        file_path = processed_dir / file_name
        exists = file_path.exists()
        status = "✅" if exists else "⚠️"
        if exists:
            size = file_path.stat().st_size
            print(f"{status} {file_name}: {size:,} bytes (optional)")
        else:
            print(f"{status} {file_name}: NOT FOUND (optional)")
    
    if all_exist:
        print("\n✅ All required Streamlit data files verified!")
        return "All files verified"
    else:
        raise ValueError("Missing required data files for Streamlit dashboard")


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
    description="Process data and prepare files for Streamlit dashboard deployment",
    schedule="0 2 * * *",  # Run daily at 2 AM UTC (9 PM EST previous day)
    start_date=datetime(2025, 11, 1),
    catchup=False,
    tags=["streamlit", "data-pipeline", "etl"],
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
    
    # Task 4: Verify all data files exist
    verify_data = PythonOperator(
        task_id="verify_streamlit_data",
        python_callable=verify_streamlit_data,
    )
    
    # Optional: Commit and push processed data to GitHub (to trigger Streamlit Cloud redeploy)
    # Uncomment the following task if you want to auto-commit data files to GitHub
    # commit_to_github = PythonOperator(
    #     task_id="commit_data_to_github",
    #     python_callable=commit_data_to_github,
    # )
    
    # Set task dependencies: process both in parallel, then merge, then verify
    [process_youtube, process_reddit] >> merge_summary >> verify_data
    # Uncomment if using GitHub commit task:
    # verify_data >> commit_to_github

