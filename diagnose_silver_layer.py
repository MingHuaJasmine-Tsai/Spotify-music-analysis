#!/usr/bin/env python3
"""
Diagnostic script to check Silver Layer data in GCS.
This script helps identify data issues like missing artists, duplicates, etc.
"""

import sys
from pathlib import Path
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
import io

# Configuration
PROJECT_ID = "ba882-qstba-group7-fall2025"
BUCKET_NAME = "apidatabase"
GCS_CLEANED_PREFIX = "cleaned"
SILVER_SUMMARY_PATTERN = "daily_song_summary_"

def get_gcs_client():
    """Initialize GCS client."""
    try:
        client = storage.Client(project=PROJECT_ID)
        return client
    except Exception as e:
        print(f"❌ Error initializing GCS client: {e}")
        return None

def get_latest_silver_file(bucket, prefix: str, pattern: str):
    """Find the latest file matching the pattern."""
    import re
    try:
        blobs = list(bucket.list_blobs(prefix=prefix))
        matching_files = [
            b for b in blobs 
            if pattern in b.name and b.name.endswith('.csv')
        ]
        
        if matching_files:
            def get_date_from_filename(blob):
                filename = blob.name.split('/')[-1]
                match = re.search(r'(\d{8})', filename)
                if match:
                    return match.group(1)
                return blob.time_created.strftime('%Y%m%d%H%M%S')
            
            matching_files.sort(key=get_date_from_filename, reverse=True)
            return matching_files[0]
        return None
    except Exception as e:
        print(f"⚠️ Error finding latest file: {e}")
        return None

def diagnose_silver_layer():
    """Run diagnostics on Silver Layer data."""
    print("=" * 80)
    print("Silver Layer Data Diagnostics")
    print("=" * 80)
    
    client = get_gcs_client()
    if client is None:
        return
    
    bucket = client.bucket(BUCKET_NAME)
    
    # Find latest summary file
    print("\n📁 Searching for latest Silver Layer summary file...")
    latest_blob = get_latest_silver_file(
        bucket,
        f"{GCS_CLEANED_PREFIX}/",
        SILVER_SUMMARY_PATTERN
    )
    
    if not latest_blob:
        print("❌ No Silver Layer summary file found!")
        return
    
    print(f"✅ Found: {latest_blob.name}")
    print(f"   Size: {latest_blob.size:,} bytes")
    print(f"   Created: {latest_blob.time_created}")
    
    # Load data
    print("\n📊 Loading data...")
    try:
        content = latest_blob.download_as_text()
        df = pd.read_csv(io.StringIO(content))
        print(f"✅ Loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Basic info
    print("\n" + "=" * 80)
    print("Basic Data Info")
    print("=" * 80)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Check for artist column
    if "artist" not in df.columns:
        print("\n❌ ERROR: 'artist' column not found!")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Artist analysis
    print("\n" + "=" * 80)
    print("Artist Analysis")
    print("=" * 80)
    
    unique_artists = df["artist"].dropna().unique()
    print(f"Unique artists: {len(unique_artists)}")
    print(f"\nAll artists:")
    for i, artist in enumerate(sorted(unique_artists), 1):
        count = len(df[df["artist"] == artist])
        print(f"  {i:2d}. {artist:30s} ({count:4d} rows)")
    
    # Check for duplicates
    print("\n" + "=" * 80)
    print("Duplicate Check")
    print("=" * 80)
    
    if "snapshot_date" in df.columns and "song" in df.columns:
        duplicates = df.duplicated(subset=["snapshot_date", "artist", "song"], keep=False)
        dup_count = duplicates.sum()
        print(f"Duplicate rows (same date/artist/song): {dup_count}")
        if dup_count > 0:
            print("\n⚠️ Found duplicates:")
            print(df[duplicates][["snapshot_date", "artist", "song"]].head(10))
    else:
        print("⚠️ Cannot check duplicates - missing date or song column")
    
    # Date analysis
    if "snapshot_date" in df.columns or "date" in df.columns:
        date_col = "snapshot_date" if "snapshot_date" in df.columns else "date"
        print("\n" + "=" * 80)
        print("Date Analysis")
        print("=" * 80)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        unique_dates = df[date_col].dropna().unique()
        print(f"Unique dates: {len(unique_dates)}")
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        print(f"\nRows per date:")
        for date in sorted(unique_dates):
            count = len(df[df[date_col] == date])
            print(f"  {date.date()}: {count:4d} rows")
    
    # Data quality checks
    print("\n" + "=" * 80)
    print("Data Quality Checks")
    print("=" * 80)
    
    # Check for missing values
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col:30s}: {missing[col]:4d} ({missing[col]/len(df)*100:.1f}%)")
    
    # Check numeric columns
    numeric_cols = ["youtube_views", "youtube_likes", "youtube_comment_count", 
                    "reddit_comment_count", "youtube_pos_ratio"]
    print("\nNumeric column statistics:")
    for col in numeric_cols:
        if col in df.columns:
            print(f"\n  {col}:")
            print(f"    Min: {df[col].min():,.0f}")
            print(f"    Max: {df[col].max():,.0f}")
            print(f"    Mean: {df[col].mean():,.2f}")
            print(f"    Non-zero: {(df[col] != 0).sum():,} rows")
    
    # Sample data
    print("\n" + "=" * 80)
    print("Sample Data (first 10 rows)")
    print("=" * 80)
    print(df.head(10).to_string())
    
    print("\n" + "=" * 80)
    print("Diagnostics Complete")
    print("=" * 80)

if __name__ == "__main__":
    diagnose_silver_layer()

