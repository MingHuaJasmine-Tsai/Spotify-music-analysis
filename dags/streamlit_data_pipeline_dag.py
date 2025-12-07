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
Last updated: 2025-12-02
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from airflow.models import Variable

from google.cloud import storage
from google.oauth2 import service_account
import sys
from pathlib import Path

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
    
    # Download latest CSV files from GCS if needed
    print("\n📥 Checking for new data in GCS...")
    # TODO: Add logic to download latest files from GCS
    
    # Process data
    print("\n🔄 Processing YouTube data...")
    run(paths)
    
    print(f"\n✅ YouTube processing complete!")
    print(f"   Summary: {paths.processed_summary_path}")
    print(f"   Comments: {paths.processed_comments_path}")
    
    # Upload processed files to Silver Layer (cleaned/) with date suffix
    print("\n☁️ Uploading processed files to Silver Layer (cleaned/)...")
    execution_date = context['ds_nodash']  # Format: YYYYMMDD
    
    if paths.processed_summary_path.exists():
        # Convert parquet to CSV for Silver Layer compatibility
        import pandas as pd
        df_summary = pd.read_parquet(paths.processed_summary_path)
        csv_summary_path = paths.processed_summary_path.with_suffix('.csv')
        df_summary.to_csv(csv_summary_path, index=False)
        
        # Upload to cleaned/ with date suffix
        gcs_summary_path = f"cleaned/daily_song_summary_{execution_date}.csv"
        _upload_to_gcs(str(csv_summary_path), gcs_summary_path, creds)
        print(f"   ✅ Uploaded to Silver Layer: {gcs_summary_path}")
    
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
    
    if paths.processed_summary_path.exists():
        # Convert parquet to CSV for Silver Layer compatibility
        import pandas as pd
        df_summary = pd.read_parquet(paths.processed_summary_path)
        csv_summary_path = paths.processed_summary_path.with_suffix('.csv')
        df_summary.to_csv(csv_summary_path, index=False)
        
        # Upload to cleaned/ with date suffix (Reddit data merged into main summary)
        # Note: Reddit data might be merged with YouTube data in Silver Layer
        print(f"   ℹ️  Reddit summary processed (may be merged with YouTube data)")
    
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
        "reddit_summary.parquet",
        "reddit_comments.parquet",
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
    
    if all_exist:
        print("\n✅ All Streamlit data files verified!")
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
    
    # Task 3: Verify all data files exist
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
    
    # Set task dependencies: process both in parallel, then verify
    [process_youtube, process_reddit] >> verify_data
    # Uncomment if using GitHub commit task:
    # verify_data >> commit_to_github

