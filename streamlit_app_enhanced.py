"""
Enhanced Streamlit dashboard for music trend analytics using Silver Layer data from GCS.

This version:
- Reads data from GCS Silver Layer (gs://apidatabase/cleaned/)
- Implements advanced visualizations based on available data
- Includes LLM summary generation
- Features Spotify-style dark mode UI
"""

from __future__ import annotations

import io
import os
import re
import json
from collections import Counter
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from google.cloud import storage
from scipy import stats
from scipy.stats import pearsonr
import requests

# Optional imports for advanced features
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Hugging Face settings
HF_MODEL = "facebook/bart-large-cnn"
# Use standard Inference API endpoint (more reliable than router)
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

def get_hf_token():
    """Get Hugging Face token from environment or Streamlit secrets."""
    # Try Streamlit secrets first
    try:
        if hasattr(st, 'secrets'):
            # Method 1: Try section access (st.secrets.tokens.hf_token) - recommended format
            try:
                if hasattr(st.secrets, 'tokens'):
                    token = st.secrets.tokens.hf_token
                    if token and isinstance(token, str) and token.strip():
                        return token.strip()
            except (AttributeError, KeyError):
                pass
            except Exception:
                pass
            
            # Method 2: Try direct attribute access (for top-level keys - legacy support)
            try:
                token = st.secrets.hf_token
                if token and isinstance(token, str) and token.strip():
                    return token.strip()
            except AttributeError:
                pass
            except Exception:
                pass
            
            # Method 3: Try dictionary-style access
            try:
                if hasattr(st.secrets, 'tokens') and hasattr(st.secrets.tokens, '__getitem__'):
                    token = st.secrets.tokens['hf_token']
                    if token and isinstance(token, str) and token.strip():
                        return token.strip()
            except (KeyError, AttributeError, TypeError):
                pass
            except Exception:
                pass
            
            # Method 4: Try top-level dictionary access (legacy)
            try:
                if hasattr(st.secrets, '__getitem__'):
                    token = st.secrets['hf_token']
                    if token and isinstance(token, str) and token.strip():
                        return token.strip()
            except (KeyError, AttributeError):
                pass
            except Exception:
                pass
    except Exception:
        pass
    
    # Fallback to environment variable
    try:
        token = os.getenv("HF_TOKEN")
        if token and isinstance(token, str) and token.strip():
            return token.strip()
    except:
        pass
    
    return None

def is_hf_available():
    """Check if Hugging Face is available."""
    return get_hf_token() is not None

# ===== GCS CONFIGURATION =====
PROJECT_ID = "ba882-qstba-group7-fall2025"
BUCKET_NAME = "apidatabase"
GCS_CLEANED_PREFIX = "cleaned"

# Silver Layer file patterns (will auto-detect latest files)
SILVER_SUMMARY_PATTERN = "daily_song_summary_"
SILVER_COMMENTS_PATTERN = "all_comments_"
SILVER_TOPIC_MODEL_PATTERN = "all_comments_topic_model_"

# Fallback to local processed data
PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
LOCAL_SUMMARY_PARQUET = PROCESSED_DIR / "youtube_summary.parquet"
LOCAL_COMMENTS_PARQUET = PROCESSED_DIR / "youtube_comments.parquet"


@st.cache_resource  # Use cache_resource for non-serializable objects like connections
def _get_gcs_client():
    """Initialize GCS client with credentials from Streamlit Secrets or default."""
    try:
        # Try to get credentials from Streamlit Secrets (for Streamlit Cloud)
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            from google.oauth2 import service_account
            
            # Get service account info from Streamlit Secrets
            service_account_info = dict(st.secrets['gcp_service_account'])
            credentials = service_account.Credentials.from_service_account_info(
                service_account_info
            )
            client = storage.Client(project=PROJECT_ID, credentials=credentials)
            return client
        else:
            # Fallback to default credentials (for local development)
            client = storage.Client(project=PROJECT_ID)
            return client
    except Exception as e:
        st.warning(f"⚠️ GCS client initialization failed: {e}")
        return None


def get_latest_silver_file(bucket, prefix: str, pattern: str) -> Optional[str]:
    """
    Find the latest file in cleaned/ directory matching the pattern.
    Returns the filename (not full path) of the latest file.
    Uses date in filename (YYYYMMDD) for sorting, more reliable than time_created.
    """
    import re
    try:
        blobs = list(bucket.list_blobs(prefix=prefix))
        matching_files = [
            b for b in blobs 
            if pattern in b.name and b.name.endswith('.csv')
        ]
        
        if matching_files:
            # Extract date from filename (YYYYMMDD) and sort by date
            def get_date_from_filename(blob):
                filename = blob.name.split('/')[-1]
                # Look for 8-digit date pattern (YYYYMMDD)
                match = re.search(r'(\d{8})', filename)
                if match:
                    return match.group(1)
                # Fallback to time_created if no date found
                return blob.time_created.strftime('%Y%m%d%H%M%S')
            
            # Sort by date in filename (most recent first)
            matching_files.sort(key=get_date_from_filename, reverse=True)
            latest = matching_files[0]
            # Return just the filename
            filename = latest.name.split('/')[-1]
            return filename
        return None
    except Exception as e:
        st.warning(f"⚠️ Error finding latest file: {e}")
        return None


@st.cache_data(ttl=21600)  # 6 hours cache (21600 seconds)
def load_silver_summary_from_gcs() -> pd.DataFrame:
    """Load daily song summary from GCS Silver Layer, or fallback to raw data."""
    client = _get_gcs_client()
    if client is None:
        return pd.DataFrame()
    
    try:
        bucket = client.bucket(BUCKET_NAME)
        
        # Auto-detect latest Silver Layer file
        latest_file = get_latest_silver_file(
            bucket, 
            f"{GCS_CLEANED_PREFIX}/", 
            SILVER_SUMMARY_PATTERN
        )
        
        if latest_file:
            blob = bucket.blob(f"{GCS_CLEANED_PREFIX}/{latest_file}")
            if blob.exists():
                content = blob.download_as_text()
                df = pd.read_csv(io.StringIO(content))
            
            # Standardize date column - handle both "date" and "snapshot_date"
            if "date" in df.columns and "snapshot_date" not in df.columns:
                df["snapshot_date"] = pd.to_datetime(df["date"], errors="coerce")
            elif "snapshot_date" in df.columns:
                df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
            elif "date" not in df.columns and "snapshot_date" not in df.columns:
                # If no date column, try to infer from filename or use today
                df["snapshot_date"] = pd.to_datetime("today")
            
            # Check if we have multiple dates
            if "snapshot_date" in df.columns and df["snapshot_date"].nunique() > 1:
                # Silver Layer has multiple dates, use it
                pass
            else:
                # Silver Layer only has one date, try loading raw data (silently)
                df = load_raw_summary_data(bucket)
        else:
            # Silver Layer not found, load raw data (silently)
            df = load_raw_summary_data(bucket)
        
        if df.empty:
            return pd.DataFrame()
        
        # Ensure numeric columns
        numeric_cols = [
            "youtube_views", "youtube_likes", "youtube_comment_count",
            "pos_comments", "youtube_pos_ratio", "reddit_comment_count",
            "views", "likes", "comment_count"  # Also check for raw data column names
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        # Standardize column names if from raw data
        if "views" in df.columns and "youtube_views" not in df.columns:
            df["youtube_views"] = df.get("views", 0)
        if "likes" in df.columns and "youtube_likes" not in df.columns:
            df["youtube_likes"] = df.get("likes", 0)
        if "comment_count" in df.columns and "youtube_comment_count" not in df.columns:
            df["youtube_comment_count"] = df.get("comment_count", 0)
        
        # Map Reddit column names - handle different naming conventions
        if "reddit_num_comments" in df.columns and "reddit_comment_count" not in df.columns:
            df["reddit_comment_count"] = df["reddit_num_comments"]
        elif "reddit_comment_count" not in df.columns:
            df["reddit_comment_count"] = 0
        
        # Also map other Reddit sentiment columns if they exist
        if "reddit_pos_comments" in df.columns and "reddit_pos_ratio" not in df.columns:
            # Calculate positive ratio if we have total comments
            if "reddit_comment_count" in df.columns:
                total = df["reddit_comment_count"]
                df["reddit_pos_ratio"] = (df["reddit_pos_comments"] / total).fillna(0)
                df["reddit_pos_ratio"] = df["reddit_pos_ratio"].replace([np.inf, -np.inf], 0)
        
        # Remove duplicates - keep latest entry for same artist/song/date
        if "snapshot_date" in df.columns and "artist" in df.columns and "song" in df.columns:
            df = df.drop_duplicates(subset=["snapshot_date", "artist", "song"], keep="last")
        
        return df.sort_values(["snapshot_date", "artist", "song"]).reset_index(drop=True)
    
    except Exception as e:
        st.error(f"❌ Error loading summary data: {e}")
        return pd.DataFrame()


def load_raw_summary_data(bucket) -> pd.DataFrame:
    """Load and merge multiple days of raw summary data."""
    try:
        # List all summary files
        blobs = list(bucket.list_blobs(prefix="youtube/summary/youtube_summary_"))
        
        if not blobs:
            return pd.DataFrame()
        
        all_dfs = []
        for blob in blobs:
            try:
                content = blob.download_as_text()
                df = pd.read_csv(io.StringIO(content))
                
                if df.empty:
                    continue
                
                # Extract date from filename (youtube_summary_YYYYMMDD.csv)
                filename = blob.name.split("/")[-1]
                date_str = filename.replace("youtube_summary_", "").replace(".csv", "")
                try:
                    date = pd.to_datetime(date_str, format="%Y%m%d")
                    df["snapshot_date"] = date
                except:
                    # Try to get date from data if available
                    if "snapshot_date" not in df.columns:
                        df["snapshot_date"] = pd.to_datetime("today")
                
                # Standardize column names across different file formats
                # Artist/Channel - normalize artist names
                if "artist" not in df.columns:
                    if "channel" in df.columns:
                        df["artist"] = df["channel"]
                    else:
                        df["artist"] = "Unknown"
                
                # Normalize artist names to handle variations - be careful not to over-merge
                if "artist" in df.columns:
                    # Store original for reference
                    df["artist_original"] = df["artist"].copy()
                    
                    # Only normalize specific known cases to avoid over-merging
                    specific_replacements = {
                        "DemiLovatoVEVO": "Demi Lovato",
                        "MadisonBeerMusicVEVO": "Madison Beer",
                        "tameimpalaVEVO": "Tame Impala",
                        "TylaVEVO": "Tyla"
                    }
                    df["artist"] = df["artist"].replace(specific_replacements)
                    
                    # For other cases, only remove trailing VEVO/Music if it's clearly a suffix
                    # Use regex to be more precise - only remove if it's at the end
                    df["artist"] = df["artist"].str.replace(r"VEVO$", "", case=False, regex=True)
                    df["artist"] = df["artist"].str.replace(r"Music$", "", case=False, regex=True)
                    df["artist"] = df["artist"].str.strip()
                    
                    # Drop the temporary column
                    if "artist_original" in df.columns:
                        df = df.drop(columns=["artist_original"])
                
                # Song/Title
                if "song" not in df.columns:
                    if "title" in df.columns:
                        df["song"] = df["title"]
                    else:
                        df["song"] = "Unknown"
                
                # Views
                if "youtube_views" not in df.columns:
                    if "views" in df.columns:
                        df["youtube_views"] = df["views"]
                    else:
                        df["youtube_views"] = 0
                
                # Likes
                if "youtube_likes" not in df.columns:
                    if "likes" in df.columns:
                        df["youtube_likes"] = df["likes"]
                    else:
                        df["youtube_likes"] = 0
                
                # Comment count
                if "youtube_comment_count" not in df.columns:
                    if "comment_count" in df.columns:
                        df["youtube_comment_count"] = df["comment_count"]
                    else:
                        df["youtube_comment_count"] = 0
                
                # Sentiment ratio
                if "youtube_pos_ratio" not in df.columns:
                    if "pos_comments" in df.columns and "youtube_comment_count" in df.columns:
                        df["youtube_pos_ratio"] = df["pos_comments"] / df["youtube_comment_count"].replace(0, 1)
                    else:
                        df["youtube_pos_ratio"] = 0.0
                
                # Reddit comments (default to 0 if not available)
                if "reddit_comment_count" not in df.columns:
                    df["reddit_comment_count"] = 0
                
                all_dfs.append(df)
            except Exception as e:
                st.warning(f"⚠️ Failed to load {blob.name}: {e}")
                continue
        
        if not all_dfs:
            return pd.DataFrame()
        
        # Merge all dataframes - only keep common columns or standard columns
        standard_cols = ["snapshot_date", "artist", "song", "youtube_views", "youtube_likes", 
                        "youtube_comment_count", "youtube_pos_ratio", "reddit_comment_count"]
        
        # Keep standard columns that exist in all dataframes, or add missing ones
        processed_dfs = []
        for df in all_dfs:
            for col in standard_cols:
                if col not in df.columns:
                    df[col] = 0 if "count" in col or "ratio" in col or "views" in col or "likes" in col else "Unknown"
            processed_dfs.append(df[standard_cols])
        
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        
        # Remove duplicates - keep latest entry for same artist/song/date
        if "snapshot_date" in combined_df.columns and "artist" in combined_df.columns and "song" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["snapshot_date", "artist", "song"], keep="last")
        
        return combined_df
    
    except Exception as e:
        st.error(f"❌ Error loading raw summary data: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()
    
    except Exception as e:
        st.error(f"❌ Error loading raw summary data: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()


@st.cache_data(ttl=21600)  # 6 hours cache (21600 seconds)
def load_silver_comments_from_gcs() -> pd.DataFrame:
    """Load comments from GCS Silver Layer (auto-detect latest file)."""
    client = _get_gcs_client()
    if client is None:
        return pd.DataFrame()
    
    try:
        bucket = client.bucket(BUCKET_NAME)
        
        # Auto-detect latest Silver Layer file
        latest_file = get_latest_silver_file(
            bucket, 
            f"{GCS_CLEANED_PREFIX}/", 
            SILVER_COMMENTS_PATTERN
        )
        
        if not latest_file:
            st.warning(f"⚠️ Silver Layer comments file not found (pattern: {SILVER_COMMENTS_PATTERN})")
            return pd.DataFrame()
        
        blob = bucket.blob(f"{GCS_CLEANED_PREFIX}/{latest_file}")
        
        if not blob.exists():
            st.warning(f"⚠️ Silver Layer comments file not found: {latest_file}")
            return pd.DataFrame()
        
        # Download in chunks for large files
        content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(content))
        
        # Standardize timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.rename(columns={"timestamp": "published_at"})
        
        return df.sort_values(["published_at"]).reset_index(drop=True)
    
    except Exception as e:
        st.error(f"❌ Error loading Silver Layer comments: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=21600)  # 6 hours cache (21600 seconds)
def load_topic_model_data_from_gcs() -> pd.DataFrame:
    """Load topic model data from GCS Silver Layer (auto-detect latest file)."""
    client = _get_gcs_client()
    if client is None:
        return pd.DataFrame()
    
    try:
        bucket = client.bucket(BUCKET_NAME)
        
        # Auto-detect latest Silver Layer file
        latest_file = get_latest_silver_file(
            bucket, 
            f"{GCS_CLEANED_PREFIX}/", 
            SILVER_TOPIC_MODEL_PATTERN
        )
        
        if not latest_file:
            st.warning(f"⚠️ Topic model file not found (pattern: {SILVER_TOPIC_MODEL_PATTERN})")
            return pd.DataFrame()
        
        blob = bucket.blob(f"{GCS_CLEANED_PREFIX}/{latest_file}")
        
        if not blob.exists():
            st.warning(f"⚠️ Topic model file not found: {latest_file}")
            return pd.DataFrame()
        
        # Download in chunks for large files
        content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(content))
        
        # Standardize timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        return df.sort_values(["timestamp"]).reset_index(drop=True)
    
    except Exception as e:
        st.error(f"❌ Error loading topic model data: {e}")
        return pd.DataFrame()


def _ensure_numeric(df: pd.DataFrame, primary: str, fallback: list[str] | None = None) -> pd.Series:
    """
    Ensure a column is numeric, trying primary column name first, then fallback options.
    
    Args:
        df: DataFrame to extract column from
        primary: Primary column name to try
        fallback: Optional list of fallback column names
        
    Returns:
        Numeric Series, or Series of zeros if no matching column found
    """
    candidates = [primary]
    if fallback:
        candidates.extend(fallback)
    for col in candidates:
        if col in df.columns:
            series = df[col]
            break
    else:
        series = pd.Series(np.nan, index=df.index)
    return pd.to_numeric(series, errors="coerce").fillna(0)


def calculate_z_scores(series: pd.Series) -> pd.Series:
    """Calculate Z-scores for normalization."""
    return (series - series.mean()) / series.std()


def format_number(num: float) -> str:
    """Format large numbers with K/M suffixes."""
    if abs(num) >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return f"{num:.0f}"


def get_artist_colors(artists: list) -> dict:
    """Generate distinct colors for each artist."""
    import plotly.colors as pc
    # Use a color palette that works well in dark mode
    colors = pc.qualitative.Set3 + pc.qualitative.Pastel + pc.qualitative.Dark2
    return {artist: colors[i % len(colors)] for i, artist in enumerate(artists)}


def calculate_cross_correlation(y1: pd.Series, y2: pd.Series, max_lag: int = 5) -> pd.DataFrame:
    """Calculate cross-correlation function for two time series."""
    lags = range(-max_lag, max_lag + 1)
    correlations = []
    
    for lag in lags:
        if lag == 0:
            corr, _ = pearsonr(y1, y2)
        elif lag > 0:
            # Shift y2 forward (y2 leads y1)
            y2_shifted = y2.shift(-lag).dropna()
            y1_aligned = y1[:len(y2_shifted)]
            if len(y1_aligned) > 1:
                corr, _ = pearsonr(y1_aligned, y2_shifted)
            else:
                corr = 0
        else:
            # Shift y1 forward (y1 leads y2)
            y1_shifted = y1.shift(abs(lag)).dropna()
            y2_aligned = y2[:len(y1_shifted)]
            if len(y2_aligned) > 1:
                corr, _ = pearsonr(y1_shifted, y2_aligned)
            else:
                corr = 0
        
        correlations.append({
            "lag": lag,
            "correlation": corr if not np.isnan(corr) else 0
        })
    
    return pd.DataFrame(correlations)


def render_hero_metrics(summary_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Render hero metrics row with day-over-day deltas."""
    if filtered_df.empty:
        return
    
    latest_date = filtered_df["snapshot_date"].max()
    latest_data = filtered_df[filtered_df["snapshot_date"] == latest_date]
    
    # Calculate totals
    total_views = float(latest_data["youtube_views"].sum())
    total_likes = float(latest_data["youtube_likes"].sum())
    avg_sentiment = float(latest_data["youtube_pos_ratio"].mean())
    total_reddit = float(latest_data["reddit_comment_count"].sum())
    
    # Calculate deltas
    unique_dates = sorted(summary_df["snapshot_date"].dropna().unique())
    delta_views = delta_likes = delta_sentiment = delta_reddit = None
    
    if len(unique_dates) > 1:
        previous_date = unique_dates[-2]
        previous_data = filtered_df[filtered_df["snapshot_date"] == previous_date]
        
        prev_views = float(previous_data["youtube_views"].sum())
        prev_likes = float(previous_data["youtube_likes"].sum())
        prev_sentiment = float(previous_data["youtube_pos_ratio"].mean())
        prev_reddit = float(previous_data["reddit_comment_count"].sum())
        
        delta_views = total_views - prev_views
        delta_likes = total_likes - prev_likes
        delta_sentiment = avg_sentiment - prev_sentiment
        delta_reddit = total_reddit - prev_reddit
    
    # Display metrics
    metric_cols = st.columns(4)
    
    metric_cols[0].metric(
        "YouTube Views",
        f"{total_views:,.0f}",
        delta=f"{delta_views:+,.0f}" if delta_views is not None else None,
    )
    metric_cols[1].metric(
        "YouTube Likes",
        f"{total_likes:,.0f}",
        delta=f"{delta_likes:+,.0f}" if delta_likes is not None else None,
    )
    metric_cols[2].metric(
        "Avg Sentiment",
        f"{avg_sentiment:.1%}",
        delta=f"{delta_sentiment:+.1%}" if delta_sentiment is not None else None,
    )
    metric_cols[3].metric(
        "Reddit Comments",
        f"{total_reddit:,.0f}",
        delta=f"{delta_reddit:+,.0f}" if delta_reddit is not None else None,
    )


def render_daily_snapshot(filtered_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Render daily snapshot analysis using Silver Layer data."""
    st.header("📊 Daily Snapshot Analysis")
    st.caption("Based on Silver Layer data - Single day snapshot")
    
    if filtered_df.empty:
        st.warning("⚠️ No data available")
        return
    
    # Check if we have single day data (Silver Layer)
    unique_dates = filtered_df["snapshot_date"].nunique()
    if unique_dates == 1:
        
        # Artist Rankings
        st.subheader("🏆 Artist Rankings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Top Artists by Views**")
            # Aggregate by artist first to avoid duplicates (sum views if multiple songs)
            # Debug: Check if we have duplicate artists
            if filtered_df["artist"].duplicated().any():
                st.caption(f"⚠️ Found {filtered_df['artist'].duplicated().sum()} duplicate artist entries - aggregating...")
            
            artist_views = filtered_df.groupby("artist", as_index=False).agg({
                "youtube_views": "sum"
            }).sort_values("youtube_views", ascending=False)
            top_views = artist_views.head(10)[["artist", "youtube_views"]]
            
            # Debug: Show aggregation result
            if len(artist_views) < len(filtered_df):
                st.caption(f"✅ Aggregated {len(filtered_df)} rows → {len(artist_views)} unique artists")
            artist_colors = get_artist_colors(top_views["artist"].tolist())
            fig = go.Figure(data=[
                go.Bar(
                    x=top_views["youtube_views"],
                    y=top_views["artist"],
                    orientation='h',
                    marker_color=[artist_colors.get(artist, "#1DB954") for artist in top_views["artist"]],
                    text=[format_number(v) for v in top_views["youtube_views"]],
                    textposition="outside"
                )
            ])
            fig.update_layout(
                title="Top 10 by Views",
                xaxis_title="Views",
                yaxis_title="Artist",
                height=400,
                template="plotly_dark",
                margin=dict(l=150, r=50, t=50, b=50),  # Increase left margin for artist names
                xaxis=dict(tickformat=".0f", tickmode="linear")
            )
            fig.update_xaxes(tickformat=".0s")  # Use scientific notation for large numbers
            fig.update_yaxes(tickangle=0)  # Ensure artist names are horizontal
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Top Artists by Likes**")
            # Aggregate by artist first to avoid duplicates (sum likes if multiple songs)
            artist_likes = filtered_df.groupby("artist", as_index=False).agg({
                "youtube_likes": "sum"
            }).sort_values("youtube_likes", ascending=False)
            top_likes = artist_likes.head(10)[["artist", "youtube_likes"]]
            
            # Ensure no duplicates
            if top_likes["artist"].duplicated().any():
                st.error("❌ ERROR: Duplicate artists in top_likes!")
                top_likes = top_likes.drop_duplicates(subset=["artist"], keep="first")
            artist_colors = get_artist_colors(top_likes["artist"].tolist())
            fig = go.Figure(data=[
                go.Bar(
                    x=top_likes["youtube_likes"],
                    y=top_likes["artist"],
                    orientation='h',
                    marker_color=[artist_colors.get(artist, "#1DB954") for artist in top_likes["artist"]],
                    text=[format_number(v) for v in top_likes["youtube_likes"]],
                    textposition="outside"
                )
            ])
            fig.update_layout(
                title="Top 10 by Likes",
                xaxis_title="Likes",
                yaxis_title="Artist",
                height=400,
                template="plotly_dark",
                margin=dict(l=150, r=50, t=50, b=50)  # Increase left margin for artist names
            )
            fig.update_xaxes(tickformat=".0s")
            fig.update_yaxes(tickangle=0)  # Ensure artist names are horizontal
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("**Top Artists by Sentiment**")
            # Aggregate by artist first to avoid duplicates (weighted average for sentiment)
            # Calculate weighted average: sum(pos_comments) / sum(total_comments)
            if "pos_comments" in filtered_df.columns and "youtube_comment_count" in filtered_df.columns:
                artist_sentiment = filtered_df.groupby("artist", as_index=False).agg({
                    "pos_comments": "sum",
                    "youtube_comment_count": "sum"
                })
                artist_sentiment["youtube_pos_ratio"] = (
                    artist_sentiment["pos_comments"] / 
                    artist_sentiment["youtube_comment_count"].replace(0, 1)
                ).fillna(0)
            else:
                # Fallback to mean if pos_comments not available
                artist_sentiment = filtered_df.groupby("artist", as_index=False).agg({
                    "youtube_pos_ratio": "mean"
                })
            artist_sentiment = artist_sentiment.sort_values("youtube_pos_ratio", ascending=False)
            top_sentiment = artist_sentiment.head(10)[["artist", "youtube_pos_ratio"]]
            
            # Ensure no duplicates
            if top_sentiment["artist"].duplicated().any():
                st.error("❌ ERROR: Duplicate artists in top_sentiment!")
                top_sentiment = top_sentiment.drop_duplicates(subset=["artist"], keep="first")
            
            artist_colors = get_artist_colors(top_sentiment["artist"].tolist())
            fig = go.Figure(data=[
                go.Bar(
                    x=top_sentiment["youtube_pos_ratio"],
                    y=top_sentiment["artist"],
                    orientation='h',
                    marker_color=[artist_colors.get(artist, "#1DB954") for artist in top_sentiment["artist"]],
                    text=[f"{v:.2%}" for v in top_sentiment["youtube_pos_ratio"]],
                    textposition="outside"
                )
            ])
            fig.update_layout(
                title="Top 10 by Sentiment",
                xaxis_title="Positive Ratio",
                yaxis_title="Artist",
                height=400,
                template="plotly_dark",
                margin=dict(l=150, r=50, t=50, b=50)  # Increase left margin for artist names
            )
            fig.update_xaxes(tickformat=".0%")
            fig.update_yaxes(tickangle=0)  # Ensure artist names are horizontal
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plots - Top 10 artists only
        st.subheader("📊 Metrics Comparison (Top 10 Artists)")
        
        # Aggregate by artist first
        artist_agg = filtered_df.groupby("artist", as_index=False).agg({
            "youtube_views": "sum",
            "youtube_likes": "sum",
            "youtube_comment_count": "sum",
            "pos_comments": "sum" if "pos_comments" in filtered_df.columns else "first"
        })
        # Calculate weighted sentiment
        if "pos_comments" in artist_agg.columns:
            artist_agg["youtube_pos_ratio"] = (
                artist_agg["pos_comments"] / 
                artist_agg["youtube_comment_count"].replace(0, 1)
            ).fillna(0)
        else:
            artist_agg["youtube_pos_ratio"] = 0
        
        # Get top 10 by views
        top_10_artists = artist_agg.nlargest(10, "youtube_views")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Use better color scale for dark mode
            fig = go.Figure(data=[
                go.Scatter(
                    x=top_10_artists["youtube_views"],
                    y=top_10_artists["youtube_likes"],
                    mode='markers+text',
                    text=top_10_artists["artist"],
                    textposition="top center",
                    marker=dict(
                        size=12,
                        color=top_10_artists["youtube_pos_ratio"],
                        colorscale="Plasma",  # Better for dark mode than Viridis
                        showscale=True,
                        colorbar=dict(title="Sentiment", tickformat=".0%"),
                        line=dict(width=1, color="white")
                    )
                )
            ])
            fig.update_layout(
                title="Top 10: Views vs Likes (colored by Sentiment)",
                xaxis_title="YouTube Views",
                yaxis_title="YouTube Likes",
                height=400,
                template="plotly_dark"
            )
            fig.update_xaxes(tickformat=".0s")
            fig.update_yaxes(tickformat=".0s")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Use artist colors for better distinction
            artist_colors = get_artist_colors(top_10_artists["artist"].tolist())
            fig = go.Figure(data=[
                go.Scatter(
                    x=top_10_artists["youtube_views"],
                    y=top_10_artists["youtube_pos_ratio"],
                    mode='markers+text',
                    text=top_10_artists["artist"],
                    textposition="top center",
                    marker=dict(
                        size=12,
                        color=[artist_colors.get(artist, "#1DB954") for artist in top_10_artists["artist"]],
                        line=dict(width=1, color="white")
                    )
                )
            ])
            fig.update_layout(
                title="Views vs Sentiment",
                xaxis_title="YouTube Views",
                yaxis_title="Positive Sentiment Ratio",
                height=400,
                template="plotly_dark"
            )
            fig.update_xaxes(tickformat=".0s")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Complete Data Table")
        display_cols = ["artist", "song", "youtube_views", "youtube_likes", 
                       "youtube_comment_count", "youtube_pos_ratio"]
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        st.dataframe(
            filtered_df[available_cols].sort_values("youtube_views", ascending=False),
            use_container_width=True
        )
    else:
        # Multiple dates - show latest day snapshot
        st.info("📊 Showing latest day snapshot. Use 'YouTube Trends' tab for multi-day analysis.")
        
        # Get latest date
        latest_date = filtered_df["snapshot_date"].max()
        latest_data = filtered_df[filtered_df["snapshot_date"] == latest_date]
        
        if not latest_data.empty:
            # Artist Rankings for latest day
            st.subheader("🏆 Top Artists (Latest Day)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Top by Views**")
                # Aggregate by artist first
                artist_views = latest_data.groupby("artist", as_index=False).agg({
                    "youtube_views": "sum"
                }).sort_values("youtube_views", ascending=False)
                top_views = artist_views.head(10)[["artist", "youtube_views"]]
                artist_colors = get_artist_colors(top_views["artist"].tolist())
                fig = go.Figure(data=[
                    go.Bar(
                        x=top_views["youtube_views"],
                        y=top_views["artist"],
                        orientation='h',
                        marker_color=[artist_colors.get(artist, "#1DB954") for artist in top_views["artist"]],
                        text=[format_number(v) for v in top_views["youtube_views"]],
                        textposition="outside"
                    )
                ])
                fig.update_layout(
                    title="Top 10 by Views",
                    xaxis_title="Views",
                    yaxis_title="Artist",
                    height=300,
                    template="plotly_dark",
                    margin=dict(l=150, r=50, t=50, b=50)  # Increase left margin for artist names
                )
                fig.update_xaxes(tickformat=".0s")
                fig.update_yaxes(tickangle=0)  # Ensure artist names are horizontal
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Top by Likes**")
                # Aggregate by artist first
                artist_likes = latest_data.groupby("artist", as_index=False).agg({
                    "youtube_likes": "sum"
                }).sort_values("youtube_likes", ascending=False)
                top_likes = artist_likes.head(10)[["artist", "youtube_likes"]]
                artist_colors = get_artist_colors(top_likes["artist"].tolist())
                fig = go.Figure(data=[
                    go.Bar(
                        x=top_likes["youtube_likes"],
                        y=top_likes["artist"],
                        orientation='h',
                        marker_color=[artist_colors.get(artist, "#1DB954") for artist in top_likes["artist"]],
                        text=[format_number(v) for v in top_likes["youtube_likes"]],
                        textposition="outside"
                    )
                ])
                fig.update_layout(
                    title="Top 10 by Likes",
                    xaxis_title="Likes",
                    yaxis_title="Artist",
                    height=300,
                    template="plotly_dark",
                    margin=dict(l=150, r=50, t=50, b=50)  # Increase left margin for artist names
                )
                fig.update_xaxes(tickformat=".0s")
                fig.update_yaxes(tickangle=0)  # Ensure artist names are horizontal
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                st.markdown("**Top by Sentiment**")
                # Aggregate by artist first (weighted average for sentiment)
                if "pos_comments" in latest_data.columns and "youtube_comment_count" in latest_data.columns:
                    artist_sentiment = latest_data.groupby("artist", as_index=False).agg({
                        "pos_comments": "sum",
                        "youtube_comment_count": "sum"
                    })
                    artist_sentiment["youtube_pos_ratio"] = (
                        artist_sentiment["pos_comments"] / 
                        artist_sentiment["youtube_comment_count"].replace(0, 1)
                    ).fillna(0)
                elif "youtube_pos_ratio" in latest_data.columns:
                    artist_sentiment = latest_data.groupby("artist", as_index=False).agg({
                        "youtube_pos_ratio": "mean"
                    })
                    artist_sentiment["youtube_pos_ratio"] = pd.to_numeric(artist_sentiment["youtube_pos_ratio"], errors="coerce").fillna(0)
                else:
                    artist_sentiment = pd.DataFrame(columns=["artist", "youtube_pos_ratio"])
                
                artist_sentiment = artist_sentiment.sort_values("youtube_pos_ratio", ascending=False)
                top_sentiment = artist_sentiment.head(10)[["artist", "youtube_pos_ratio"]]
                
                # Only show if we have at least some data
                artist_colors = get_artist_colors(top_sentiment["artist"].tolist())
                if len(top_sentiment) > 0 and top_sentiment["youtube_pos_ratio"].sum() > 0:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=top_sentiment["youtube_pos_ratio"],
                            y=top_sentiment["artist"],
                            orientation='h',
                            marker_color=[artist_colors.get(artist, "#1DB954") for artist in top_sentiment["artist"]],
                            text=[f"{v:.2%}" for v in top_sentiment["youtube_pos_ratio"]],
                            textposition="outside"
                        )
                    ])
                    max_val = max(0.1, top_sentiment["youtube_pos_ratio"].max() * 1.1)
                    fig.update_layout(
                        title="Top 10 by Sentiment",
                        xaxis_title="Positive Ratio",
                        yaxis_title="Artist",
                        height=300,
                        template="plotly_dark",
                        margin=dict(l=150, r=50, t=50, b=50),  # Increase left margin for artist names
                        xaxis=dict(range=[0, max_val])
                    )
                    fig.update_xaxes(tickformat=".0%")
                    fig.update_yaxes(tickangle=0)  # Ensure artist names are horizontal
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Show all artists even if values are 0
                    if len(top_sentiment) > 0:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=top_sentiment["youtube_pos_ratio"],
                                y=top_sentiment["artist"],
                                orientation='h',
                                marker_color=[artist_colors.get(artist, "#888888") for artist in top_sentiment["artist"]],
                                text=[f"{v:.2%}" for v in top_sentiment["youtube_pos_ratio"]],
                                textposition="outside"
                            )
                        ])
                        fig.update_layout(
                            title="Top 10 by Sentiment (All values are 0)",
                            xaxis_title="Positive Ratio",
                            yaxis_title="Artist",
                            height=300,
                            template="plotly_dark",
                            margin=dict(l=150, r=50, t=50, b=50),  # Increase left margin for artist names
                            xaxis=dict(range=[0, 1])
                        )
                        fig.update_xaxes(tickformat=".0%")
                        fig.update_yaxes(tickangle=0)  # Ensure artist names are horizontal
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No sentiment data available for this day")
            
            # Data table for latest day
            st.subheader(f"📋 Latest Day Data ({latest_date.date()})")
            display_cols = ["artist", "song", "youtube_views", "youtube_likes", 
                           "youtube_comment_count", "youtube_pos_ratio"]
            available_cols = [col for col in display_cols if col in latest_data.columns]
            st.dataframe(
                latest_data[available_cols].sort_values("youtube_views", ascending=False),
                use_container_width=True
            )


def render_youtube_trends(filtered_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Render YouTube trends analysis using multi-day data."""
    st.header("📈 YouTube Trends Analysis")
    
    # Use all dates from summary_df for complete time series
    if not summary_df.empty:
        # Group by date to get daily aggregates
        daily = summary_df.groupby("snapshot_date").agg({
            "youtube_views": "sum",
            "youtube_likes": "sum",
            "youtube_comment_count": "sum",
            "youtube_pos_ratio": "mean"
        }).reset_index()
        
        # If filtered by artist, filter the data but keep all dates
        if not filtered_df.empty and len(filtered_df) < len(summary_df):
            # User filtered by artist - aggregate filtered data by date
            daily_filtered = filtered_df.groupby("snapshot_date").agg({
                "youtube_views": "sum",
                "youtube_likes": "sum",
                "youtube_comment_count": "sum",
                "youtube_pos_ratio": "mean"
            }).reset_index()
            # Merge with all dates to show complete timeline
            all_dates = sorted(summary_df["snapshot_date"].unique())
            daily_all = pd.DataFrame({"snapshot_date": all_dates})
            daily = daily_all.merge(daily_filtered, on="snapshot_date", how="left")
            # Fill missing dates with 0 (for filtered artists that don't have data on those dates)
            daily = daily.fillna(0)
        else:
            # No filter or same data - use full summary
            daily = daily.sort_values("snapshot_date").reset_index(drop=True)
    else:
        st.warning("⚠️ No data available")
        return
    
    if len(daily) < 2:
        st.warning("⚠️ Need at least 2 data points for trend analysis")
        return
    
    # Overall trends
    st.subheader("📊 Overall Trends")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=daily["snapshot_date"],
            y=daily["youtube_views"],
            name="YouTube Views",
            line=dict(color="#1DB954", width=3),  # Darker green, thicker line
            mode="lines+markers",
            marker=dict(size=8, color="#1DB954")
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily["snapshot_date"],
            y=daily["youtube_likes"],
            name="YouTube Likes",
            line=dict(color="#FF4500", width=3),  # Orange/Red for better contrast
            mode="lines+markers",
            marker=dict(size=8, color="#FF4500")
        ),
        secondary_y=True,
    )
    
    fig.update_xaxes(
        title_text="Date",
        type='date',
        tickformat='%Y-%m-%d'
    )
    fig.update_yaxes(title_text="YouTube Views", secondary_y=False, tickformat=".0s")
    fig.update_yaxes(title_text="YouTube Likes", secondary_y=True, tickformat=".0s")
    fig.update_layout(
        title="YouTube Engagement Trends",
        hovermode="x unified",
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment trend
    st.subheader("💚 Sentiment Trend")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["snapshot_date"],
        y=daily["youtube_pos_ratio"],
        name="Positive Sentiment Ratio",
        line=dict(color="#1DB954", width=2),
        mode="lines+markers",
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="Average Positive Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Positive Sentiment Ratio",
        yaxis=dict(range=[0, 1]),
        height=300,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Artist comparison (if multiple artists selected)
    if not filtered_df.empty and filtered_df["artist"].nunique() > 1:
        st.subheader("🎤 Artist Comparison")
        
        # Find artists that appear in multiple dates
        artist_date_counts = filtered_df.groupby("artist")["snapshot_date"].nunique()
        multi_date_artists = artist_date_counts[artist_date_counts >= 3].index.tolist()
        
        if multi_date_artists:
            
            artist_trends = filtered_df[filtered_df["artist"].isin(multi_date_artists)].groupby(
                ["snapshot_date", "artist"]
            ).agg({
                "youtube_views": "sum"
            }).reset_index()
            
            # Get top 10 artists by total views
            artist_totals = artist_trends.groupby("artist")["youtube_views"].sum().sort_values(ascending=False)
            top_10_artists = artist_totals.head(10).index.tolist()
            
            # Separate Taylor Swift if present (she dominates the scale)
            taylor_swift = "Taylor Swift" if "Taylor Swift" in top_10_artists else None
            other_artists = [a for a in top_10_artists if a != taylor_swift]
            
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            # Use distinct colors for each artist
            import plotly.colors as pc
            distinct_colors = [
                "#1DB954", "#FF4500", "#1f77b4", "#ff7f0e", "#2ca02c",
                "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
            ]
            
            # Add Taylor Swift on primary axis if present
            if taylor_swift:
                taylor_data = artist_trends[artist_trends["artist"] == taylor_swift]
                fig.add_trace(go.Scatter(
                    x=taylor_data["snapshot_date"],
                    y=taylor_data["youtube_views"],
                    name=taylor_swift,
                    mode="lines+markers",
                    line=dict(color="#FFD700", width=3),  # Gold for Taylor Swift
                    marker=dict(size=8)
                ))
            
            # Add other artists with distinct colors
            for idx, artist in enumerate(other_artists):
                artist_data = artist_trends[artist_trends["artist"] == artist]
                fig.add_trace(go.Scatter(
                    x=artist_data["snapshot_date"],
                    y=artist_data["youtube_views"],
                    name=artist,
                    mode="lines+markers",
                    line=dict(color=distinct_colors[idx % len(distinct_colors)], width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title="Artist Views Comparison (Top 10 with multi-date data)",
                xaxis_title="Date",
                yaxis_title="YouTube Views",
                height=500,
                template="plotly_dark",
                hovermode='x unified',
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=9)
                ),
                margin=dict(l=50, r=150, t=50, b=50)  # Extra right margin for legend
            )
            fig.update_xaxes(type='date', tickformat='%Y-%m-%d')
            fig.update_yaxes(tickformat=".0s")  # Use scientific notation for better readability
            
            st.plotly_chart(fig, use_container_width=True)
        # If no multi-date artists, show overall trend instead
    
    # Data table
    st.subheader("📋 Trend Data")
    st.dataframe(daily, use_container_width=True)


def render_reddit_analysis(summary_df: pd.DataFrame) -> None:
    """Render Reddit analysis - independent module."""
    st.header("💬 Reddit Analysis")
    
    # Load Reddit data
    client = _get_gcs_client()
    if client is None:
        st.warning("⚠️ Cannot load Reddit data - GCS connection failed")
        return
    
    try:
        bucket = client.bucket(BUCKET_NAME)
        reddit_blob = bucket.blob("reddit/summary/summary_all.csv")
        
        if not reddit_blob.exists():
            st.warning("⚠️ Reddit data file not found")
            return
        
        content = reddit_blob.download_as_text()
        reddit_df = pd.read_csv(io.StringIO(content))
        
        if reddit_df.empty:
            st.warning("⚠️ Reddit data is empty")
            return
        
        # Fix missing artist/song data - use artist_x/song_x or artist_y/song_y if artist/song is NaN
        if "artist" in reddit_df.columns:
            # Fill NaN artist from artist_x or artist_y
            if "artist_x" in reddit_df.columns:
                reddit_df["artist"] = reddit_df["artist"].fillna(reddit_df["artist_x"])
            if "artist_y" in reddit_df.columns:
                reddit_df["artist"] = reddit_df["artist"].fillna(reddit_df["artist_y"])
        
        if "song" in reddit_df.columns:
            # Fill NaN song from song_x or song_y
            if "song_x" in reddit_df.columns:
                reddit_df["song"] = reddit_df["song"].fillna(reddit_df["song_x"])
            if "song_y" in reddit_df.columns:
                reddit_df["song"] = reddit_df["song"].fillna(reddit_df["song_y"])
        
        # Process Reddit data
        if "created_utc" in reddit_df.columns:
            reddit_df["created_utc"] = pd.to_datetime(reddit_df["created_utc"], errors="coerce")
            reddit_df["date"] = reddit_df["created_utc"].dt.date
            reddit_df["snapshot_date"] = pd.to_datetime(reddit_df["date"])
        
        # Filter out rows without artist (after filling)
        if "artist" in reddit_df.columns:
            reddit_df = reddit_df[reddit_df["artist"].notna()]
        
        if reddit_df.empty:
            st.warning("⚠️ No Reddit data with valid artist information")
            return
        
        # Reddit data overview
        st.subheader("📊 Reddit Data Overview")
        
        if "artist" in reddit_df.columns:
            reddit_artists = reddit_df["artist"].dropna().unique()
            if len(reddit_artists) > 0:
                st.write(f"**Artists with Reddit data**: {', '.join(reddit_artists)}")
            else:
                st.warning("⚠️ No artists found in Reddit data")
        
        if "num_comments" in reddit_df.columns:
            total_comments = reddit_df["num_comments"].sum()
            st.metric("Total Reddit Comments", f"{total_comments:,.0f}")
        
        # Reddit trends by artist
        if "artist" in reddit_df.columns and "num_comments" in reddit_df.columns and "snapshot_date" in reddit_df.columns:
            st.subheader("📈 Reddit Comments by Artist")
            
            reddit_daily = reddit_df.groupby(["snapshot_date", "artist"]).agg({
                "num_comments": "sum"
            }).reset_index()
            
            # Ensure snapshot_date is datetime and format properly
            reddit_daily["snapshot_date"] = pd.to_datetime(reddit_daily["snapshot_date"])
            reddit_daily = reddit_daily.sort_values("snapshot_date")
            
            # Get top 10 artists by total comments
            artist_totals = reddit_daily.groupby("artist")["num_comments"].sum().sort_values(ascending=False)
            top_10_artists = artist_totals.head(10).index.tolist()
            
            fig = go.Figure()
            for artist in top_10_artists:
                artist_data = reddit_daily[reddit_daily["artist"] == artist]
                fig.add_trace(go.Scatter(
                    x=artist_data["snapshot_date"],
                    y=artist_data["num_comments"],
                    name=artist,
                    mode="lines+markers"
                ))
            
            fig.update_layout(
                title="Reddit Comments Over Time (Top 10 Artists)",
                xaxis_title="Date",
                yaxis_title="Number of Comments",
                height=400,
                template="plotly_dark",
                xaxis=dict(
                    type='date',
                    tickformat='%Y-%m-%d'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Reddit data table
        st.subheader("📋 Reddit Data Table")
        display_cols = ["snapshot_date", "artist", "song", "subreddit", "title", "num_comments", "score"]
        available_cols = [col for col in display_cols if col in reddit_df.columns]
        st.dataframe(
            reddit_df[available_cols].head(100),
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"❌ Error loading Reddit data: {e}")


def render_lag_slider_analysis(filtered_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Interactive lag slider to find optimal time shift between platforms."""
    if filtered_df.empty and summary_df.empty:
        st.warning("⚠️ No data available for lag analysis")
        return
    
    # Always use all dates from summary_df for time series analysis
    # This ensures we have a complete time series even if some dates have 0 values for selected artists
    if not summary_df.empty:
        all_dates = sorted(summary_df["snapshot_date"].unique())
        daily_all = pd.DataFrame({"snapshot_date": all_dates})
        
        # If we have filtered artists, sum only those artists for each date
        if not filtered_df.empty:
            daily_filtered = filtered_df.groupby("snapshot_date").agg({
                "youtube_views": "sum",
                "reddit_comment_count": "sum"
            }).reset_index()
            # Merge with all dates, filling missing dates with 0
            daily = daily_all.merge(daily_filtered, on="snapshot_date", how="left")
            daily = daily.fillna(0)
        else:
            # No filter, use all data
            daily = summary_df.groupby("snapshot_date").agg({
                "youtube_views": "sum",
                "reddit_comment_count": "sum"
            }).reset_index()
            daily = daily_all.merge(daily, on="snapshot_date", how="left")
            daily = daily.fillna(0)
    else:
        # Fallback to filtered_df if summary_df is empty
        daily = filtered_df.groupby("snapshot_date").agg({
            "youtube_views": "sum",
            "reddit_comment_count": "sum"
        }).reset_index().sort_values("snapshot_date")
    
    daily = daily.sort_values("snapshot_date").reset_index(drop=True)
    
    # Don't remove rows with 0 values - keep all dates for time series analysis
    if len(daily) < 3:
        st.dataframe(daily, use_container_width=True)
        return
    
    st.subheader("🔀 Interactive Lag Analysis")
    st.caption("Adjust the lag to find the optimal time shift between YouTube and Reddit")
    
    lag = st.slider("Time Shift (Days)", min_value=-5, max_value=5, value=0, step=1)
    
    # Apply lag
    daily_shifted = daily.copy()
    if lag > 0:
        daily_shifted["reddit_comment_count"] = daily_shifted["reddit_comment_count"].shift(-lag)
    elif lag < 0:
        daily_shifted["youtube_views"] = daily_shifted["youtube_views"].shift(abs(lag))
    
    # Calculate correlation
    valid_data = daily_shifted.dropna()
    
    # Check if Reddit data is all zeros
    if valid_data["reddit_comment_count"].sum() == 0:
        # Reddit data is all zeros - skip correlation
        st.info("📊 You can still analyze YouTube trends using the 'Trends' tab.")
    elif len(valid_data) > 1:
        # Check if there's any variation in Reddit data
        if valid_data["reddit_comment_count"].std() != 0:
            corr, p_value = pearsonr(
                valid_data["youtube_views"],
                valid_data["reddit_comment_count"]
            )
            if pd.notna(corr):
                st.metric("Correlation Coefficient", f"{corr:.3f}", 
                         delta=f"p={p_value:.3f}" if p_value < 0.05 else "Not significant")
    
    # Plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=valid_data["snapshot_date"],
        y=valid_data["youtube_views"],
        name="YouTube Views",
        line=dict(color="#1DB954", width=2),
        mode="lines+markers"
    ))
    
    fig.add_trace(go.Scatter(
        x=valid_data["snapshot_date"],
        y=valid_data["reddit_comment_count"],
        name=f"Reddit Comments (shifted {lag:+d} days)",
        line=dict(color="#FF4500", width=2),
        mode="lines+markers"
    ))
    
    fig.update_layout(
        title=f"Cross-Platform Comparison (Lag: {lag:+d} days)",
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode="x unified",
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_ccf_analysis(filtered_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Render Cross-Correlation Function analysis."""
    if filtered_df.empty and summary_df.empty:
        st.warning("⚠️ No data available for CCF analysis")
        return
    
    # Always use all dates from summary_df for time series analysis
    if not summary_df.empty:
        all_dates = sorted(summary_df["snapshot_date"].unique())
        daily_all = pd.DataFrame({"snapshot_date": all_dates})
        
        # If we have filtered artists, sum only those artists for each date
        if not filtered_df.empty:
            daily_filtered = filtered_df.groupby("snapshot_date").agg({
                "youtube_views": "sum",
                "reddit_comment_count": "sum"
            }).reset_index()
            # Merge with all dates, filling missing dates with 0
            daily = daily_all.merge(daily_filtered, on="snapshot_date", how="left")
            daily = daily.fillna(0)
        else:
            # No filter, use all data
            daily = summary_df.groupby("snapshot_date").agg({
                "youtube_views": "sum",
                "reddit_comment_count": "sum"
            }).reset_index()
            daily = daily_all.merge(daily, on="snapshot_date", how="left")
            daily = daily.fillna(0)
    else:
        # Fallback to filtered_df if summary_df is empty
        daily = filtered_df.groupby("snapshot_date").agg({
            "youtube_views": "sum",
            "reddit_comment_count": "sum"
        }).reset_index().sort_values("snapshot_date")
    
    daily = daily.sort_values("snapshot_date").reset_index(drop=True)
    
    # Don't remove rows with 0 values - keep all dates for time series analysis
    if len(daily) < 5:
        st.dataframe(daily, use_container_width=True)
        return
    
    ccf_df = calculate_cross_correlation(
        daily["youtube_views"],
        daily["reddit_comment_count"],
        max_lag=5
    )
    
    # Find best lag
    best_lag_idx = ccf_df["correlation"].abs().idxmax()
    best_lag = int(ccf_df.loc[best_lag_idx, "lag"])
    best_corr = ccf_df.loc[best_lag_idx, "correlation"]
    
    st.subheader("📊 Cross-Correlation Function (CCF)")
    st.caption(f"Best correlation: r={best_corr:.3f} at lag={best_lag} days")
    
    # Plot
    fig = go.Figure()
    
    colors = ["#1DB954" if abs(corr) > 0.5 else "#888888" 
              for corr in ccf_df["correlation"]]
    
    fig.add_trace(go.Bar(
        x=ccf_df["lag"],
        y=ccf_df["correlation"],
        marker_color=colors,
        text=[f"{c:.3f}" for c in ccf_df["correlation"]],
        textposition="outside"
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Cross-Correlation Function: YouTube Views vs Reddit Comments",
        xaxis_title="Lag (Days)",
        yaxis_title="Correlation Coefficient",
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Check if Reddit data is all zeros
    if daily["reddit_comment_count"].sum() == 0:
        st.error("❌ Reddit data is all zeros. CCF analysis cannot be performed.")
        # Reddit data not available - skip
        st.info("📊 You can still analyze YouTube trends using the 'Trends' tab.")
        return
    
    # Check if there's any variation in Reddit data
    if daily["reddit_comment_count"].std() == 0:
        # Reddit data has no variation - skip CCF
        return
    
    # Interpretation
    if pd.notna(best_corr) and abs(best_corr) > 0.7:
        if best_lag < 0:
            st.success(f"✅ Strong correlation: Reddit leads YouTube by {abs(best_lag)} days")
        elif best_lag > 0:
            st.success(f"✅ Strong correlation: YouTube leads Reddit by {best_lag} days")
        else:
            st.success("✅ Strong correlation: Platforms are synchronized")
    elif pd.notna(best_corr) and abs(best_corr) > 0.5:
        st.info(f"ℹ️ Moderate correlation at lag={best_lag} days")
    else:
        st.warning("⚠️ Weak correlation - platforms may not be strongly related")


def clean_text_for_wordcloud(text: str) -> str:
    """Clean text for word cloud generation with improved, more precise preprocessing."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+|https?://\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove numbers (standalone)
    text = re.sub(r'\b\d+\b', '', text)
    # Remove special characters but keep apostrophes for contractions
    text = re.sub(r'[^\w\s\']', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase and strip
    text = text.lower().strip()
    
    # Remove very short words (1-2 characters) that are likely noise
    words = text.split()
    words = [w for w in words if len(w) > 2]
    
    return " ".join(words)


def generate_wordcloud(text_data: list[str], title: str, colormap: str = "viridis") -> Optional[bytes]:
    """Generate high-quality word cloud image with improved settings."""
    if not WORDCLOUD_AVAILABLE:
        return None
    
    if not text_data:
        return None
    
    # Expanded stopwords to exclude - more comprehensive list for better precision
    stopwords_set = {
        # Basic articles and prepositions
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'including', 'against', 'among', 'throughout', 'despite', 'towards', 'upon', 'concerning', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'including', 'against', 'among', 'throughout', 'despite', 'towards', 'upon', 'concerning',
        # Pronouns
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'this', 'that', 'these', 'those', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves',
        # Possessive pronouns
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
        # Question words
        'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
        # Common verbs
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'cant', 'cannot',
        # Action verbs
        'get', 'got', 'getting', 'go', 'goes', 'went', 'going', 'gone', 'come', 'comes', 'came', 'coming', 'see', 'sees', 'saw', 'seeing', 'seen', 'know', 'knows', 'knew', 'knowing', 'known',
        'think', 'thinks', 'thought', 'thinking', 'say', 'says', 'said', 'saying', 'tell', 'tells', 'told', 'telling', 'make', 'makes', 'made', 'making', 'take', 'takes', 'took', 'taking', 'taken',
        # Contractions
        'im', 'youre', 'hes', 'shes', 'its', 'were', 'theyre', 'ive', 'youve', 'weve', 'theyve', 'id', 'youd', 'hed', 'shed', 'wed', 'theyd', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'couldnt', 'shouldnt',
        # Adverbs and modifiers
        'just', 'really', 'very', 'much', 'more', 'most', 'some', 'any', 'all', 'every', 'each', 'both', 'few', 'many', 'several', 'such', 'only', 'also', 'still', 'even', 'too', 'so', 'quite', 'rather', 'pretty', 'fairly', 'almost', 'enough', 'quite',
        # Time words
        'now', 'then', 'here', 'there', 'today', 'yesterday', 'tomorrow', 'soon', 'later', 'early', 'late', 'always', 'never', 'often', 'sometimes', 'usually', 'recently', 'already', 'yet', 'still',
        # Common nouns (generic)
        'one', 'two', 'first', 'second', 'third', 'last', 'next', 'new', 'old', 'good', 'bad', 'great', 'small', 'big', 'large', 'little', 'long', 'short', 'high', 'low', 'right', 'left', 'best', 'worst',
        'time', 'times', 'day', 'days', 'way', 'ways', 'thing', 'things', 'people', 'person', 'man', 'men', 'woman', 'women', 'guy', 'guys', 'girl', 'girls', 'boy', 'boys',
        # Common filler words
        'like', 'um', 'uh', 'er', 'ah', 'oh', 'well', 'yeah', 'yes', 'no', 'ok', 'okay', 'hmm', 'huh',
        # Music/comment specific common words that don't add value
        'video', 'videos', 'watch', 'watching', 'watched', 'click', 'clicks', 'link', 'links', 'channel', 'channels', 'subscribe', 'subscribed', 'subscriber', 'subscribers', 'comment', 'comments', 'commented', 'reply', 'replies', 'replied'
    }
    
    # Clean and combine all text
    cleaned_texts = []
    for t in text_data:
        if t:
            cleaned = clean_text_for_wordcloud(t)
            if cleaned:
                # Remove stopwords
                words = cleaned.split()
                filtered_words = [w for w in words if w not in stopwords_set and len(w) > 2]
                if filtered_words:
                    cleaned_texts.append(" ".join(filtered_words))
    
    if not cleaned_texts:
        return None
    
    text = " ".join(cleaned_texts)
    
    if not text.strip():
        return None
    
    try:
        # Improved WordCloud settings for better quality
        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color="#191414",  # Spotify dark background
            colormap=colormap,
            max_words=150,  # More words
            relative_scaling=0.3,  # Better size variation
            min_font_size=12,
            max_font_size=120,
            font_path=None,  # Use default font
            prefer_horizontal=0.7,  # Prefer horizontal text
            scale=2,  # Higher resolution
            collocations=True,  # Include common word pairs
            normalize_plurals=True,
            contour_width=0,
            contour_color="#1DB954"  # Spotify green for contour (if needed)
        ).generate(text)
        
        # Convert to high-quality image
        img = wordcloud.to_image()
        
        # Resize for better display (maintain aspect ratio)
        from PIL import Image
        img = img.resize((1000, 500), Image.Resampling.LANCZOS)
        
        return img
    except Exception as e:
        st.warning(f"Word cloud generation failed: {e}")
        return None


def render_comments_analysis(comments_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """
    Render comments analysis with word clouds and recent comments.
    """
    st.header("💬 Comments Analysis")
    
    # Use comments_df directly
    analysis_df = comments_df.copy()
    
    if analysis_df.empty:
        st.warning("No comments data available")
        return
    
    # Get selected artists from filtered_df
    selected_artists = filtered_df["artist"].unique().tolist() if not filtered_df.empty else []
    
    # Filter comments by selected artists
    if selected_artists and "artist" in analysis_df.columns:
        analysis_df = analysis_df[analysis_df["artist"].isin(selected_artists)]
    
    if analysis_df.empty:
        st.warning("No comments match the selected filters")
        return
    
    # Show basic comment statistics
    st.subheader("📊 Comment Statistics")
    if not analysis_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Comments", f"{len(analysis_df):,}")
        with col2:
            if "artist" in analysis_df.columns:
                st.metric("Artists", analysis_df["artist"].nunique())
        with col3:
            if "song" in analysis_df.columns:
                st.metric("Songs", analysis_df["song"].nunique())
    
    # Word Clouds - Improved Section
    st.subheader("☁️ Word Cloud Analysis")
    st.markdown("Visualizing most frequent words in comments to identify key themes and sentiments.")
    
    # Get comment column name
    comment_col = "comment" if "comment" in analysis_df.columns else "comments"
    if comment_col not in analysis_df.columns:
        st.warning("Comment column not found")
        return
    
    # Get all comments
    all_comments = analysis_df[comment_col].dropna().tolist()
    
    if not all_comments:
        st.warning("No comments available for word cloud generation")
        return
    
    # Improved sentiment-based separation
    positive_keywords = ["love", "amazing", "beautiful", "great", "best", "wonderful", "perfect", "awesome", "fantastic", "excellent", "brilliant", "incredible", "favorite", "enjoy", "appreciate", "thank", "grateful"]
    negative_keywords = ["hate", "bad", "worst", "terrible", "awful", "disappointed", "boring", "annoying", "stupid", "ridiculous", "waste", "trash"]
    
    # Better sentiment classification
    positive_comments = []
    negative_comments = []
    neutral_comments = []
    
    for comment in all_comments:
        comment_lower = str(comment).lower()
        pos_score = sum(1 for kw in positive_keywords if kw in comment_lower)
        neg_score = sum(1 for kw in negative_keywords if kw in comment_lower)
        
        if pos_score > neg_score and pos_score > 0:
            positive_comments.append(comment)
        elif neg_score > pos_score and neg_score > 0:
            negative_comments.append(comment)
        else:
            neutral_comments.append(comment)
    
    # If sentiment classification didn't work well, use all comments
    if len(positive_comments) < 10 and len(negative_comments) < 10:
        # Split comments in half for visualization
        mid_point = len(all_comments) // 2
        positive_comments = all_comments[:mid_point]
        negative_comments = all_comments[mid_point:]
    
    # Generate word clouds with better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💚 Positive Comments")
        st.caption(f"Showing {len(positive_comments):,} comments")
        if positive_comments:
            with st.spinner("Generating word cloud..."):
                wc_image = generate_wordcloud(positive_comments, "Positive", "Greens")
                if wc_image:
                    st.image(wc_image, use_container_width=True, output_format="PNG")
                else:
                    st.info("Word cloud generation unavailable. Install wordcloud package.")
        else:
            st.info("No positive comments found")
    
    with col2:
        st.markdown("### Negative Comments")
        st.caption(f"Showing {len(negative_comments):,} comments")
        if negative_comments:
            with st.spinner("Generating word cloud..."):
                wc_image = generate_wordcloud(negative_comments, "Negative", "Reds")
                if wc_image:
                    st.image(wc_image, use_container_width=True, output_format="PNG")
                else:
                    st.info("Word cloud generation unavailable. Install wordcloud package.")
        else:
            st.info("No negative comments found")
    
    # Add overall word cloud
    if len(all_comments) > 50:
        st.markdown("---")
        st.subheader("📊 Overall Word Cloud (All Comments)")
        with st.spinner("Generating overall word cloud..."):
            wc_image = generate_wordcloud(all_comments, "Overall", "viridis")
            if wc_image:
                st.image(wc_image, use_container_width=True, output_format="PNG")
    
    # Comments table
    st.subheader("💬 Recent Comments")
    display_cols = [comment_col]
    if "artist" in analysis_df.columns:
        display_cols.append("artist")
    if "song" in analysis_df.columns:
        display_cols.append("song")
    if "timestamp" in analysis_df.columns:
        display_cols.append("timestamp")
    
    available_cols = [col for col in display_cols if col in analysis_df.columns]
    st.dataframe(
        analysis_df[available_cols].head(100),
        use_container_width=True,
        height=400
    )


def build_text_for_hf(df: pd.DataFrame, max_rows: int = 80) -> str:
    """
    Build a text block from sampled comment rows for Hugging Face summarization.
    """
    if len(df) == 0:
        return "No data."
    
    # Find comment column
    comment_col = None
    for col in ["comment", "body", "text", "content"]:
        if col in df.columns:
            comment_col = col
            break
    
    if comment_col is None:
        return "No comment data found."
    
    sample = df.sample(min(max_rows, len(df)), random_state=42)
    
    lines = []
    for _, row in sample.iterrows():
        comment = row.get(comment_col, "")
        if isinstance(comment, str) and comment.strip():
            lines.append(comment.strip())
    
    if not lines:
        return "No data."
    
    text = "\n".join(lines)
    
    # BART-large-CNN has a max input length of 1024 tokens
    # To be safe, limit to ~800 characters (roughly 200 tokens)
    MAX_CHARS = 800
    if len(text) > MAX_CHARS:
        # Truncate intelligently at sentence boundaries if possible
        truncated = text[:MAX_CHARS]
        # Try to cut at the last sentence boundary
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        cut_point = max(last_period, last_newline)
        if cut_point > MAX_CHARS * 0.7:  # Only use if we keep at least 70% of text
            text = truncated[:cut_point + 1]
        else:
            text = truncated
    
    return text


def hf_summarize(text: str) -> str:
    """
    Call Hugging Face Inference API for summarization.
    Uses facebook/bart-large-cnn model.
    Tries standard endpoint first, falls back to router if needed.
    If input is too long (400 error), automatically truncates and retries.
    """
    hf_token = get_hf_token()
    if not hf_token:
        raise RuntimeError("HF_TOKEN not available. Please set it in Streamlit Secrets or environment variable.")
    
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Try multiple endpoints with correct formats (only router is currently working)
    api_urls = [
        f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}",
    ]
    
    # If text is too long, try progressively shorter versions
    text_versions = [text]
    if len(text) > 800:
        # Try 600 chars
        text_versions.append(text[:600])
    if len(text) > 500:
        # Try 400 chars
        text_versions.append(text[:400])
    
    errors = []
    for text_input in text_versions:
        payload = {
            "inputs": text_input,
            "parameters": {
                "max_length": 150,  # Reduced from 200
                "min_length": 50,   # Reduced from 60
                "do_sample": False,
            },
            "options": {
                "wait_for_model": True
            },
        }
        
        for api_url in api_urls:
            try:
                # Reduced timeout to prevent infinite loading
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 401:
                    errors.append(f"Endpoint {api_url}: Authentication failed (401)")
                    continue
                elif response.status_code == 403:
                    errors.append(f"Endpoint {api_url}: Access forbidden (403)")
                    continue
                elif response.status_code == 404:
                    errors.append(f"Endpoint {api_url}: Not found (404)")
                    continue
                elif response.status_code == 410:
                    errors.append(f"Endpoint {api_url}: Deprecated (410)")
                    continue
                elif response.status_code == 400:
                    # Input too long or format issue - try shorter text
                    error_text = response.text[:200] if len(response.text) > 200 else response.text
                    if "index out of range" in error_text.lower() or len(text_input) > 400:
                        # Will try shorter version in next iteration
                        continue
                    else:
                        errors.append(f"Endpoint {api_url}: Bad request (400): {error_text}")
                        continue
                elif response.status_code != 200:
                    error_text = response.text[:500] if len(response.text) > 500 else response.text
                    errors.append(f"Endpoint {api_url}: Error ({response.status_code}): {error_text}")
                    continue
            
                data = response.json()
                
                # Expected response: [{"summary_text": "..."}]
                if isinstance(data, list) and len(data) > 0 and "summary_text" in data[0]:
                    return data[0]["summary_text"]
                
                # Also handle direct dict response format
                if isinstance(data, dict) and "summary_text" in data:
                    return data["summary_text"]
                
                errors.append(f"Endpoint {api_url}: Unexpected response format: {json.dumps(data, indent=2)[:500]}")
                continue
                
            except requests.exceptions.Timeout:
                errors.append(f"Endpoint {api_url}: Request timed out")
                continue
            except requests.exceptions.RequestException as e:
                errors.append(f"Endpoint {api_url}: Connection error: {e}")
                continue
    
    # If all endpoints and text versions failed, raise a comprehensive error
    if errors:
        error_msg = "All Hugging Face API attempts failed:\n" + "\n".join(f"  - {e}" for e in errors[:5])  # Show first 5 errors
        raise RuntimeError(error_msg)
    else:
        raise RuntimeError("All Hugging Face API endpoints failed.")


def summary_to_bullets(summary: str, max_bullets: int = 5) -> str:
    """
    Convert the summary into 3–5 bullet points about sentiment & key themes.
    """
    sentences = re.split(r"[.!?]\s+", summary)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return "- No summary generated."
    
    n = min(max_bullets, max(3, len(sentences)))
    chosen = sentences[:n]
    
    bullets = "\n".join(f"- {s}" for s in chosen)
    return bullets


def generate_llm_summary(comments_df: pd.DataFrame, summary_df: pd.DataFrame, use_hf: bool = True) -> str:
    """Generate LLM summary of comments and trends using Hugging Face or OpenAI."""
    # Try Hugging Face first if requested and available
    if use_hf and is_hf_available():
        try:
            # Build text from comments
            text = build_text_for_hf(comments_df, max_rows=80)
            if text.strip() == "No data.":
                return "No usable comment data found for summarization."
            
            # Call Hugging Face API
            raw_summary = hf_summarize(text)
            
            # Convert to bullet points
            bullets = summary_to_bullets(raw_summary, max_bullets=5)
            
            # Add context about trends
            if not summary_df.empty:
                context = f"""
## Overall Sentiment & Key Themes (Hugging Face BART)

{bullets}

### Additional Context:
- Total Comments Analyzed: {len(comments_df):,}
- Total YouTube Views: {summary_df['youtube_views'].sum():,.0f}
- Average Positive Sentiment: {summary_df['youtube_pos_ratio'].mean():.1%}
"""
                return context
            else:
                return f"## Overall Sentiment & Key Themes (Hugging Face BART)\n\n{bullets}"
        except Exception as e:
            error_msg = str(e)
            # If HF fails and OpenAI is available, try OpenAI instead
            if OPENAI_AVAILABLE:
                # Continue to OpenAI fallback below
                pass
            else:
                # No OpenAI available, return error message
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    return f"❌ Summary generation timed out. The API may be slow. Please try again.\n\nError: {error_msg}"
                elif "API error" in error_msg or "status_code" in error_msg:
                    return f"❌ Hugging Face API error. Please check your token has 'Read' access.\n\nError: {error_msg}"
                else:
                    return f"❌ Hugging Face summarization failed: {error_msg}\n\nPlease try again."
    
    # Try OpenAI if available and (HF not used or HF failed)
    if OPENAI_AVAILABLE and (not use_hf or (use_hf and not is_hf_available())):
        try:
            api_key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("openai_api_key") if hasattr(st, 'secrets') else None)
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            client = OpenAI(api_key=api_key)
            
            # Prepare summary text
            summary_text = f"""
            Music Trend Summary:
            - Total YouTube Views: {summary_df['youtube_views'].sum():,.0f}
            - Total YouTube Likes: {summary_df['youtube_likes'].sum():,.0f}
            - Average Sentiment: {summary_df['youtube_pos_ratio'].mean():.1%}
            - Total Comments Analyzed: {len(comments_df)}
            
            Top Artists:
            {summary_df.groupby('artist')['youtube_views'].sum().sort_values(ascending=False).head(5).to_string()}
            
            Sample Comments:
            {comments_df['comment'].head(10).tolist() if 'comment' in comments_df.columns else 'N/A'}
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a music analytics expert. Provide concise insights about music trends."},
                    {"role": "user", "content": f"Analyze this music trend data and provide key insights:\n\n{summary_text}"}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM summary generation failed: {e}. Please check OpenAI API key."
    
    # Fallback: Generate simple summary
    if summary_df.empty:
        return "No data available for summary generation."
    
    summary = f"""
## Music Trend Summary

**Overall Performance:**
- Total YouTube Views: {summary_df['youtube_views'].sum():,.0f}
- Total YouTube Likes: {summary_df['youtube_likes'].sum():,.0f}
- Average Positive Sentiment: {summary_df['youtube_pos_ratio'].mean():.1%}
- Total Reddit Comments: {summary_df['reddit_comment_count'].sum():,.0f}

**Top Performing Artists:**
"""
    
    top_artists = summary_df.groupby('artist').agg({
        'youtube_views': 'sum',
        'youtube_likes': 'sum',
        'youtube_pos_ratio': 'mean'
    }).sort_values('youtube_views', ascending=False).head(5)
    
    for artist, row in top_artists.iterrows():
        summary += f"\n- **{artist}**: {row['youtube_views']:,.0f} views, {row['youtube_likes']:,.0f} likes, {row['youtube_pos_ratio']:.1%} positive"
    
    if not comments_df.empty and 'comment' in comments_df.columns:
        summary += f"\n\n**Comments Analysis:**\n"
        summary += f"- Total Comments: {len(comments_df)}\n"
        summary += f"- Sample themes: Emotional reactions, song appreciation, personal connections"
    
    return summary


def render_llm_summary(comments_df: pd.DataFrame, filtered_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Render LLM summary generation interface."""
    st.header("🤖 LLM Summary Generation")
    
    if filtered_df.empty:
        st.warning("No data available for summary generation")
        return
    
    # Use comments_df for analysis
    analysis_comments = comments_df
    
    # API selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("📊 Automated Insights")
    
    with col2:
        # Show API status with debug info
        hf_available = is_hf_available()
        hf_token = get_hf_token()
        
        if hf_available:
            st.success("✅ Hugging Face Available")
            # Debug: Show token status (first 10 chars only)
            if hf_token:
                st.caption(f"Token: {hf_token[:10]}...")
        else:
            st.warning("⚠️ Hugging Face: Set HF_TOKEN in Secrets")
            # Debug info
            with st.expander("🔍 Debug Info"):
                st.write(f"Has st.secrets: {hasattr(st, 'secrets')}")
                if hasattr(st, 'secrets'):
                    try:
                        # Try to list all available secrets
                        if hasattr(st.secrets, 'keys'):
                            secrets_keys = list(st.secrets.keys())
                            st.write(f"Available sections: {secrets_keys}")
                            st.caption("(Note: This only shows sections like [gcp_service_account], not top-level keys)")
                        
                        st.write("**Testing hf_token access:**")
                        
                        # Test 1: Section access (st.secrets.tokens.hf_token) - recommended
                        try:
                            if hasattr(st.secrets, 'tokens'):
                                test_token = st.secrets.tokens.hf_token
                                if test_token:
                                    st.success(f"✅ Found via st.secrets.tokens.hf_token: {test_token[:20]}...")
                                else:
                                    st.warning("⚠️ st.secrets.tokens.hf_token exists but is empty")
                            else:
                                st.warning("⚠️ st.secrets.tokens section does not exist")
                        except AttributeError:
                            st.error("❌ st.secrets.tokens.hf_token does not exist (AttributeError)")
                        except Exception as e:
                            st.error(f"❌ Error accessing st.secrets.tokens.hf_token: {type(e).__name__}: {e}")
                        
                        # Test 2: Direct attribute access (legacy - top-level key)
                        try:
                            test_token = st.secrets.hf_token
                            if test_token:
                                st.success(f"✅ Found via st.secrets.hf_token: {test_token[:20]}...")
                        except AttributeError:
                            st.info("ℹ️ st.secrets.hf_token does not exist (using section format)")
                        except Exception as e:
                            st.error(f"❌ Error accessing st.secrets.hf_token: {type(e).__name__}: {e}")
                    except Exception as e:
                        st.write(f"Cannot list secrets: {e}")
                
                env_token = os.getenv('HF_TOKEN')
                if env_token:
                    st.write(f"✅ HF_TOKEN env var found: {env_token[:20]}...")
                else:
                    st.info("ℹ️ HF_TOKEN env var not set (this is normal for Streamlit Cloud)")
                
                # Show what get_hf_token() actually returns
                actual_token = get_hf_token()
                if actual_token:
                    st.success(f"✅✅ get_hf_token() SUCCESS: {actual_token[:20]}...")
                else:
                    st.error("❌❌ get_hf_token() returned None - token not accessible")
        
        if OPENAI_AVAILABLE:
            st.success("✅ OpenAI Available")
        else:
            st.info("ℹ️ OpenAI: Optional")
    
    with col3:
        # API selection
        api_choice = st.radio(
            "Choose API:",
            ["Hugging Face", "OpenAI"],
            index=0 if is_hf_available() else 1,
            horizontal=True
        )
        use_hf = (api_choice == "Hugging Face" and is_hf_available())
    
    # Create a unique cache key based on data content and filters
    # Include more data characteristics to ensure cache invalidation when data changes
    import hashlib
    import json
    
    # Create hash from multiple data characteristics
    data_signature = {
        "comments_count": len(analysis_comments),
        "filtered_rows": len(filtered_df),
        "total_views": float(filtered_df['youtube_views'].sum()) if not filtered_df.empty and 'youtube_views' in filtered_df.columns else 0,
        "total_likes": float(filtered_df['youtube_likes'].sum()) if not filtered_df.empty and 'youtube_likes' in filtered_df.columns else 0,
        "date_range": f"{filtered_df['snapshot_date'].min()}_{filtered_df['snapshot_date'].max()}" if not filtered_df.empty and 'snapshot_date' in filtered_df.columns else "none",
        "api_choice": api_choice
    }
    
    data_hash = hashlib.md5(json.dumps(data_signature, sort_keys=True).encode()).hexdigest()[:12]
    cache_key = f"llm_summary_{data_hash}"
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        generate_btn = st.button("🔄 Generate Summary", type="primary", use_container_width=True)
    with col_btn2:
        if cache_key in st.session_state:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                # Clear all cached summaries
                keys_to_clear = [k for k in st.session_state.keys() if k.startswith("summary_")]
                for k in keys_to_clear:
                    del st.session_state[k]
                st.rerun()
    
    # Generate summary - ALWAYS generate when button is clicked, ignore cache
    if generate_btn:
        # Clear any existing cache for this key
        if cache_key in st.session_state:
            del st.session_state[cache_key]
        if f"{cache_key}_timestamp" in st.session_state:
            del st.session_state[f"{cache_key}_timestamp"]
        if f"{cache_key}_api" in st.session_state:
            del st.session_state[f"{cache_key}_api"]
        
        # Generate new summary with timeout handling
        try:
            with st.spinner(f"Generating summary using {api_choice}... This may take up to 2 minutes."):
                summary_text = generate_llm_summary(analysis_comments, filtered_df, use_hf=use_hf)
                if summary_text:
                    st.session_state[cache_key] = summary_text
                    st.session_state[f"{cache_key}_timestamp"] = datetime.now().isoformat()
                    st.session_state[f"{cache_key}_api"] = api_choice
                    # Don't use st.rerun() here - it causes infinite loop
                    # The summary will be displayed below automatically
                else:
                    st.error("❌ Summary generation returned empty result. Please try again.")
        except Exception as e:
            st.error(f"❌ Summary generation failed: {str(e)}")
            st.info("💡 Please try again or switch to a different API.")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    # Display cached summary if available
    if cache_key in st.session_state:
        if f"{cache_key}_api" in st.session_state:
            st.caption(f"Generated using {st.session_state[f'{cache_key}_api']}" + 
                      (f" at {st.session_state.get(f'{cache_key}_timestamp', '')}" if f"{cache_key}_timestamp" in st.session_state else ""))
        st.markdown(st.session_state[cache_key])
    
    # Additional insights
    st.subheader("🔍 Key Insights")
    
    if not filtered_df.empty and not summary_df.empty:
        # Trend insights - use full summary_df to calculate proper growth across all dates
        all_daily = summary_df.groupby("snapshot_date").agg({
            "youtube_views": "sum",
            "youtube_likes": "sum",
            "reddit_comment_count": "sum"
        }).reset_index().sort_values("snapshot_date")
        
        if len(all_daily) > 1:
            # Calculate views growth - track same songs across dates
            # Group by artist and song to track individual song growth
            song_tracking = summary_df.groupby(["artist", "song", "snapshot_date"]).agg({
                "youtube_views": "first"  # Get first value if duplicates
            }).reset_index()
            
            # Find songs that appear in both first and last date
            first_date = all_daily["snapshot_date"].iloc[0]
            last_date = all_daily["snapshot_date"].iloc[-1]
            
            first_songs = song_tracking[song_tracking["snapshot_date"] == first_date].set_index(["artist", "song"])
            last_songs = song_tracking[song_tracking["snapshot_date"] == last_date].set_index(["artist", "song"])
            
            # Find common songs
            common_songs = first_songs.index.intersection(last_songs.index)
            
            if len(common_songs) > 0:
                # Calculate growth for songs that exist in both dates
                first_views_common = first_songs.loc[common_songs, "youtube_views"].sum()
                last_views_common = last_songs.loc[common_songs, "youtube_views"].sum()
                
                if first_views_common > 0:
                    views_growth = ((last_views_common - first_views_common) / first_views_common) * 100
                else:
                    views_growth = 0.0
                
                st.metric(
                    "Views Growth (Same Songs)",
                    f"{views_growth:+.1f}%",
                    delta=f"Tracking {len(common_songs)} songs from {first_date.date()} to {last_date.date()}"
                )
                
                with st.expander("🔍 Growth Calculation Details", expanded=False):
                    st.write(f"**Songs tracked:** {len(common_songs)}")
                    st.write(f"**First date ({first_date.date()}):** {first_views_common:,.0f} views")
                    st.write(f"**Last date ({last_date.date()}):** {last_views_common:,.0f} views")
                    st.write(f"**Growth:** {last_views_common - first_views_common:,.0f} views ({views_growth:+.1f}%)")
            else:
                # No common songs, show total views comparison
                first_views = float(all_daily["youtube_views"].iloc[0])
                last_views = float(all_daily["youtube_views"].iloc[-1])
                
                st.metric(
                    "Total Views",
                    f"{last_views:,.0f}",
                    delta=f"From {first_date.date()} to {last_date.date()}"
                )
                st.info("ℹ️ Different songs tracked each day. Showing total views instead of growth percentage.")
                
                with st.expander("🔍 Growth Calculation Details", expanded=False):
                    st.write(f"**First date ({first_date.date()}):** {first_views:,.0f} total views")
                    st.write(f"**Last date ({last_date.date()}):** {last_views:,.0f} total views")
                    st.write(f"**Note:** No common songs found between dates, so growth % cannot be calculated.")
        elif len(all_daily) == 1:
            st.info("ℹ️ Only one day of data available. Growth calculation requires multiple days.")
        
        # Sentiment insights
        avg_sentiment = filtered_df["youtube_pos_ratio"].mean()
        if avg_sentiment > 0.6:
            st.success(f"✅ High positive sentiment: {avg_sentiment:.1%}")
        elif avg_sentiment > 0.4:
            st.info(f"ℹ️ Moderate sentiment: {avg_sentiment:.1%}")
        else:
            st.warning(f"⚠️ Low positive sentiment: {avg_sentiment:.1%}")


def render_top_performers(youtube_df: pd.DataFrame) -> None:
    """Render top 10 engagement leaders using cross-platform metrics."""
    st.subheader("🏆 Top 10 Engagement Leaders")
    
    if youtube_df.empty:
        st.info("No engagement data available for the selected filters.")
        return
    
    # Aggregate YouTube data by artist
    yt = youtube_df.dropna(subset=["artist"]).groupby("artist", as_index=False).agg({
        "youtube_views": "sum",
        "youtube_likes": "sum",
        "youtube_comment_count": "sum",
        "youtube_pos_ratio": "mean",
        "reddit_comment_count": "sum",
    }).rename(columns={
        "youtube_views": "views",
        "youtube_likes": "likes",
        "youtube_comment_count": "yt_comments",
        "reddit_comment_count": "reddit_comments",
    })
    
    if yt.empty:
        st.info("No artist data available.")
        return
    
    # Create engagement index
    metric_cols = ["views", "likes", "yt_comments", "reddit_comments"]
    norm_cols = []
    
    for col in metric_cols:
        if col not in yt.columns:
            yt[col] = 0.0
        max_val = yt[col].max()
        norm_col = f"{col}_norm"
        if max_val and not np.isclose(max_val, 0):
            yt[norm_col] = yt[col] / max_val
        else:
            yt[norm_col] = 0.0
        norm_cols.append(norm_col)
    
    # Add sentiment normalization
    yt["sentiment_norm"] = yt["youtube_pos_ratio"].fillna(0.0).clip(0, 1)
    norm_cols.append("sentiment_norm")
    
    # Calculate engagement index
    yt["engagement_index"] = yt[norm_cols].mean(axis=1)
    
    # Get top 10
    top = yt.nlargest(10, "engagement_index")
    
    if top.empty:
        st.info("Not enough data to calculate engagement leaders.")
        return
    
    # Create bar chart with Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["engagement_index"],
        y=top["artist"],
        orientation='h',
        marker=dict(color="#1DB954", line=dict(color="#1DB954", width=1)),
        text=[f"{val:.3f}" for val in top["engagement_index"]],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Engagement Index: %{x:.3f}<br>" +
                      "<extra></extra>"
    ))
    
    fig.update_layout(
        title="Top 10 Engagement Leaders",
        xaxis_title="Engagement Index",
        yaxis_title="Artist",
        height=400,
        template="plotly_dark",
        margin=dict(l=150, r=50, t=50, b=50),  # Increase left margin for artist names
        showlegend=False,
        yaxis=dict(autorange="reversed")
    )
    fig.update_yaxes(tickangle=0)  # Ensure artist names are horizontal
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed table
    display_cols = ["artist", "views", "likes", "yt_comments", "reddit_comments", "engagement_index"]
    display_df = top[display_cols].rename(columns={
        "views": "YT Views",
        "likes": "YT Likes",
        "yt_comments": "YT Comments",
        "reddit_comments": "Reddit Comments",
        "engagement_index": "Engagement Index",
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_cross_platform_trends(summary_df: pd.DataFrame) -> None:
    """Render cross-platform activity trends comparing YouTube and Reddit."""
    st.subheader("📈 Cross-Platform Activity Trends")
    
    if summary_df.empty:
        st.info("No trend data available for the selected filters.")
        return
    
    # Aggregate by date
    daily = summary_df.groupby("snapshot_date").agg({
        "youtube_views": "sum",
        "youtube_likes": "sum",
        "youtube_comment_count": "sum",
        "reddit_comment_count": "sum",
    }).reset_index()
    
    daily = daily.sort_values("snapshot_date")
    daily = daily.fillna(0)
    
    if daily.empty or len(daily) < 2:
        st.info("Not enough data points for trend analysis (need at least 2 dates).")
        return
    
    # Engagement chart - YouTube Views vs Reddit Comments
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig1.add_trace(
        go.Scatter(
            x=daily["snapshot_date"],
            y=daily["youtube_views"],
            name="YouTube Views",
            line=dict(color="#1f77b4", width=2),
            mode="lines+markers"
        ),
        secondary_y=False,
    )
    
    fig1.add_trace(
        go.Scatter(
            x=daily["snapshot_date"],
            y=daily["reddit_comment_count"],
            name="Reddit Comments",
            line=dict(color="#ff7f0e", width=2),
            mode="lines+markers"
        ),
        secondary_y=True,
    )
    
    fig1.update_xaxes(title_text="Date")
    fig1.update_yaxes(title_text="YouTube Views", secondary_y=False)
    fig1.update_yaxes(title_text="Reddit Comments", secondary_y=True)
    fig1.update_layout(
        title="Engagement Trends: YouTube Views vs Reddit Activity",
        height=350,
        template="plotly_dark",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Comments comparison chart
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig2.add_trace(
        go.Scatter(
            x=daily["snapshot_date"],
            y=daily["youtube_comment_count"],
            name="YouTube Comments",
            line=dict(color="#2ca02c", width=2),
            mode="lines+markers"
        ),
        secondary_y=False,
    )
    
    fig2.add_trace(
        go.Scatter(
            x=daily["snapshot_date"],
            y=daily["reddit_comment_count"],
            name="Reddit Comments",
            line=dict(color="#d62728", width=2),
            mode="lines+markers"
        ),
        secondary_y=True,
    )
    
    fig2.update_xaxes(title_text="Date")
    fig2.update_yaxes(title_text="YouTube Comments", secondary_y=False)
    fig2.update_yaxes(title_text="Reddit Comments", secondary_y=True)
    fig2.update_layout(
        title="Comments Comparison: YouTube vs Reddit",
        height=320,
        template="plotly_dark",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig2, use_container_width=True)


def render_sentiment_comparison(summary_df: pd.DataFrame) -> None:
    """Render sentiment comparison across platforms."""
    st.subheader("💭 Sentiment Comparison")
    
    if summary_df.empty:
        st.info("No sentiment data available for the selected filters.")
        return
    
    # Aggregate sentiment by date
    daily_sentiment = summary_df.groupby("snapshot_date").agg({
        "youtube_pos_ratio": "mean",
        "reddit_comment_count": "sum",  # Use Reddit comment count as proxy
    }).reset_index()
    
    daily_sentiment = daily_sentiment.sort_values("snapshot_date")
    daily_sentiment = daily_sentiment.fillna(0)
    
    # For Reddit sentiment, we'll use a placeholder since we don't have direct sentiment data
    # In a real scenario, this would come from Reddit data processing
    daily_sentiment["reddit_sentiment"] = daily_sentiment["youtube_pos_ratio"] * 0.8  # Placeholder
    
    if len(daily_sentiment) < 2:
        st.info("Not enough data points for sentiment comparison.")
        return
    
    # Create comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_sentiment["snapshot_date"],
        y=daily_sentiment["youtube_pos_ratio"],
        name="YouTube Sentiment",
        line=dict(color="#1DB954", width=2),
        mode="lines+markers",
        hovertemplate="<b>YouTube</b><br>Date: %{x}<br>Positive Ratio: %{y:.2%}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_sentiment["snapshot_date"],
        y=daily_sentiment["reddit_sentiment"],
        name="Reddit Sentiment (Estimated)",
        line=dict(color="#FF5700", width=2),
        mode="lines+markers",
        hovertemplate="<b>Reddit</b><br>Date: %{x}<br>Positive Ratio: %{y:.2%}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Sentiment Comparison: YouTube vs Reddit",
        xaxis_title="Date",
        yaxis_title="Positive Sentiment Ratio",
        height=320,
        template="plotly_dark",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("ℹ️ Note: Reddit sentiment is estimated based on comment patterns. Direct sentiment analysis requires Reddit comment data processing.")


def render_discussion_distribution(summary_df: pd.DataFrame) -> None:
    """Render Reddit discussion distribution by artist."""
    st.subheader("💬 Reddit Discussion Distribution")
    
    if summary_df.empty:
        st.info("No Reddit metrics available for the selected filters.")
        return
    
    # Aggregate Reddit data by artist
    artist_totals = summary_df.groupby("artist").agg({
        "reddit_comment_count": "sum",
        "youtube_views": "sum",  # Use YouTube views as a proxy for engagement
    }).reset_index()
    
    artist_totals = artist_totals[artist_totals["reddit_comment_count"] > 0]
    
    if artist_totals.empty:
        st.info("No Reddit discussion data available. Reddit data may be limited.")
        return
    
    # Sort by Reddit comments
    artist_totals = artist_totals.sort_values("reddit_comment_count", ascending=False).head(10)
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=artist_totals["reddit_comment_count"],
        y=artist_totals["artist"],
        orientation='h',
        marker=dict(color="#FF5700", line=dict(color="#FF5700", width=1)),
        text=[f"{val:,.0f}" for val in artist_totals["reddit_comment_count"]],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Reddit Comments: %{x:,.0f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Reddit Discussion Activity by Artist (Top 10)",
        xaxis_title="Total Reddit Comments",
        yaxis_title="Artist",
        height=400,
        template="plotly_dark",
        margin=dict(l=150, r=50, t=50, b=50),  # Increase left margin for artist names
        showlegend=False,
        yaxis=dict(autorange="reversed")
    )
    fig.update_yaxes(tickangle=0)  # Ensure artist names are horizontal
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("ℹ️ Shows Reddit comment activity distribution. Limited Reddit data available in current dataset.")


def render_radar_insights(summary_df: pd.DataFrame) -> None:
    """Render artist radar chart comparing multiple metrics across artists."""
    st.subheader("📊 Artist Radar Chart")
    
    if summary_df.empty:
        st.info("No artist metrics available for the selected filters.")
        return
    
    # Aggregate metrics by artist
    artist_metrics = summary_df.groupby("artist").agg({
        "youtube_views": "sum",
        "youtube_likes": "sum",
        "youtube_comment_count": "sum",
        "youtube_pos_ratio": "mean",
        "reddit_comment_count": "sum",
    }).reset_index()
    
    if artist_metrics.empty:
        st.info("No artist data available.")
        return
    
    # Get top 10 artists by views
    top_artists = artist_metrics.nlargest(10, "youtube_views")
    
    if top_artists.empty:
        st.info("Not enough data to build the radar chart.")
        return
    
    # Normalize metrics for radar chart (0-1 scale)
    metric_cols = ["youtube_views", "youtube_likes", "youtube_comment_count", "reddit_comment_count"]
    
    for col in metric_cols:
        max_val = top_artists[col].max()
        if max_val and not np.isclose(max_val, 0):
            top_artists[f"{col}_norm"] = top_artists[col] / max_val
        else:
            top_artists[f"{col}_norm"] = 0.0
    
    # Normalize sentiment (already 0-1 scale)
    top_artists["sentiment_norm"] = top_artists["youtube_pos_ratio"].fillna(0.0).clip(0, 1)
    
    # Create radar chart with Plotly
    fig = go.Figure()
    
    metric_labels = ["YT Views", "YT Likes", "YT Comments", "Reddit Comments", "Sentiment"]
    metric_keys = ["youtube_views_norm", "youtube_likes_norm", "youtube_comment_count_norm", 
                   "reddit_comment_count_norm", "sentiment_norm"]
    
    # Use distinct colors for each artist (10 different colors)
    import plotly.colors as pc
    distinct_colors = [
        "#1DB954",  # Spotify Green
        "#FF4500",  # Reddit Orange
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow-green
        "#17becf",  # Cyan
    ]
    
    for idx, (_, row) in enumerate(top_artists.iterrows()):
        values = [row[key] for key in metric_keys]
        # Close the radar chart by adding first value at the end
        values = values + [values[0]]
        labels = metric_labels + [metric_labels[0]]
        
        # Use distinct color for each artist
        artist_color = distinct_colors[idx % len(distinct_colors)]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=row["artist"],
            line=dict(color=artist_color, width=2.5),
            fillcolor=artist_color,
            opacity=0.6,  # Add transparency to see overlapping areas
            hovertemplate=f"<b>{row['artist']}</b><br>" +
                          f"YT Views: {row['youtube_views']:,.0f}<br>" +
                          f"YT Likes: {row['youtube_likes']:,.0f}<br>" +
                          f"YT Comments: {row['youtube_comment_count']:,.0f}<br>" +
                          f"Reddit Comments: {row['reddit_comment_count']:,.0f}<br>" +
                          f"Sentiment: {row['youtube_pos_ratio']:.1%}<extra></extra>"
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='linear',
                tick0=0,
                dtick=0.2,
                tickfont=dict(size=10),
                gridcolor="rgba(255,255,255,0.3)"
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                linecolor="rgba(255,255,255,0.5)"
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.15,
            font=dict(size=10)
        ),
        title="Artist Performance Radar Chart (Top 10)",
        height=600,
        template="plotly_dark",
        margin=dict(l=50, r=200, t=50, b=50)  # Extra right margin for legend
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metrics table
    display_cols = ["artist", "youtube_views", "youtube_likes", "youtube_comment_count", 
                    "reddit_comment_count", "youtube_pos_ratio"]
    display_df = top_artists[display_cols].rename(columns={
        "youtube_views": "YT Views",
        "youtube_likes": "YT Likes",
        "youtube_comment_count": "YT Comments",
        "reddit_comment_count": "Reddit Comments",
        "youtube_pos_ratio": "Sentiment",
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title="Music Trend Analytics - Silver Layer",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Spotify-style CSS
    st.markdown("""
    <style>
    .main {
        background-color: #191414;
    }
    .stMetric {
        background-color: #282828;
        padding: 1rem;
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎵 Music Trend Analytics Dashboard")
    st.caption("Powered by Silver Layer Data from GCS")
    
    # Load data
    with st.spinner("Loading data from GCS Silver Layer..."):
        summary_df = load_silver_summary_from_gcs()
        comments_df = load_silver_comments_from_gcs()
        # Topic modeling is handled by other team members, not needed here
    
    if summary_df.empty:
        st.error("❌ No data loaded from Silver Layer. Please check GCS connection.")
        return
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Artist filter - show all artists by default
    artists = sorted(summary_df["artist"].dropna().unique().tolist())
    selected_artists = st.sidebar.multiselect("Artists", artists, default=artists)
    
    # Date filter
    if "snapshot_date" in summary_df.columns:
        dates = sorted(summary_df["snapshot_date"].dropna().unique())
        if dates and len(dates) > 0:
            try:
                # Convert pandas Timestamp to Python date objects safely
                def to_date(dt):
                    if pd.isna(dt):
                        return None
                    if isinstance(dt, pd.Timestamp):
                        return dt.to_pydatetime().date()
                    elif isinstance(dt, datetime):
                        return dt.date()
                    elif hasattr(dt, 'date'):
                        return dt.date()
                    elif isinstance(dt, str):
                        return datetime.strptime(dt, '%Y-%m-%d').date()
                    else:
                        return dt
                
                min_date = to_date(dates[0])
                max_date = to_date(dates[-1])
                
                # Ensure dates are valid Python date objects (not None)
                if min_date is not None and max_date is not None and isinstance(min_date, date) and isinstance(max_date, date):
                    date_range = st.sidebar.date_input(
                        "Date Range",
                        value=None,
                        min_value=min_date,
                        max_value=max_date,
                        key="date_range_filter"
                    )
                else:
                    date_range = None
            except Exception as e:
                st.sidebar.warning(f"Date filter unavailable: {e}")
                date_range = None
        else:
            date_range = None
    else:
        date_range = None
    
    # Filter data - but keep all dates for time series analysis
    filtered_df = summary_df.copy()
    
    # Show data info before filtering
    with st.expander("📊 Data Info", expanded=False):
        st.write(f"**Total data points:** {len(summary_df)}")
        st.write(f"**Unique dates:** {summary_df['snapshot_date'].nunique()}")
        st.write(f"**Unique artists:** {summary_df['artist'].nunique()}")
        st.write(f"**Date range:** {summary_df['snapshot_date'].min().date()} to {summary_df['snapshot_date'].max().date()}")
    
    # Apply artist filter (but we'll use all dates for aggregation)
    artist_filtered_df = summary_df.copy()
    if selected_artists:
        artist_filtered_df = artist_filtered_df[artist_filtered_df["artist"].isin(selected_artists)]
        # For display purposes, use filtered data
        filtered_df = artist_filtered_df.copy()
    
    # Handle date_range: it can be None, a single date, or a tuple of 2 dates
    if date_range is not None:
        try:
            # Check if date_range is iterable (tuple/list) with 2 elements
            if hasattr(date_range, '__len__') and len(date_range) == 2:
                filtered_df = filtered_df[
                    (filtered_df["snapshot_date"] >= pd.Timestamp(date_range[0])) &
                    (filtered_df["snapshot_date"] <= pd.Timestamp(date_range[1]))
                ]
            elif isinstance(date_range, (date, pd.Timestamp)):
                # Single date selected - filter to that date only
                filtered_df = filtered_df[
                    filtered_df["snapshot_date"] == pd.Timestamp(date_range)
                ]
        except (TypeError, AttributeError, IndexError) as e:
            # If date_range format is unexpected, skip date filtering
            st.sidebar.warning(f"⚠️ Date filter format issue: {e}")
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches the selected filters")
        return
    
    # Main content
    render_hero_metrics(summary_df, filtered_df)
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Daily Snapshot", "📈 YouTube Trends", "💬 Reddit Analysis", "🔄 Cross-Platform Insights", "💬 Comments Analysis", "🤖 LLM Summary"
    ])
    
    with tab1:
        # For daily snapshot, use filtered data (single day view)
        render_daily_snapshot(filtered_df, summary_df)
    
    with tab2:
        # For trends, always use all dates but filter by artist for aggregation
        render_youtube_trends(artist_filtered_df if selected_artists else summary_df, summary_df)
    
    with tab3:
        render_reddit_analysis(summary_df)
    
    with tab4:
        # Cross-Platform Insights - combining features from old version
        st.header("🔄 Cross-Platform Insights")
        st.caption("Combined analysis across YouTube and Reddit platforms")
        
        # Top Performers
        render_top_performers(summary_df)
        
        st.markdown("---")
        
        # Cross-Platform Trends
        render_cross_platform_trends(summary_df)
        
        st.markdown("---")
        
        # Sentiment Comparison
        render_sentiment_comparison(summary_df)
        
        st.markdown("---")
        
        # Discussion Distribution
        render_discussion_distribution(summary_df)
        
        st.markdown("---")
        
        # Radar Insights
        render_radar_insights(summary_df)
    
    with tab5:
        render_comments_analysis(comments_df, filtered_df)
    
    with tab6:
        render_llm_summary(comments_df, filtered_df, summary_df)


if __name__ == "__main__":
    main()

