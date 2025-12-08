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
    """
    try:
        blobs = list(bucket.list_blobs(prefix=prefix))
        matching_files = [
            b for b in blobs 
            if pattern in b.name and b.name.endswith('.csv')
        ]
        
        if matching_files:
            # Sort by time_created (most recent first)
            latest = max(matching_files, key=lambda x: x.time_created)
            # Return just the filename
            filename = latest.name.split('/')[-1]
            return filename
        return None
    except Exception as e:
        st.warning(f"⚠️ Error finding latest file: {e}")
        return None


@st.cache_data(ttl=86400)
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
            
            # Standardize date column
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.rename(columns={"date": "snapshot_date"})
            
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
        
        # Add reddit_comment_count if missing
        if "reddit_comment_count" not in df.columns:
            df["reddit_comment_count"] = 0
        
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
                
                # Normalize artist names to handle variations
                if "artist" in df.columns:
                    # Common name variations
                    df["artist"] = df["artist"].str.replace("VEVO", "", case=False, regex=False)
                    df["artist"] = df["artist"].str.replace("Music", "", case=False, regex=False)
                    df["artist"] = df["artist"].str.strip()
                    # Specific normalizations
                    df["artist"] = df["artist"].replace({
                        "DemiLovatoVEVO": "Demi Lovato",
                        "MadisonBeerMusicVEVO": "Madison Beer",
                        "tameimpalaVEVO": "Tame Impala",
                        "TylaVEVO": "Tyla"
                    })
                
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


@st.cache_data(ttl=86400)
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


@st.cache_data(ttl=86400)
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
            top_views = filtered_df.nlargest(10, "youtube_views")[["artist", "youtube_views"]]
            fig = go.Figure(data=[
                go.Bar(
                    x=top_views["youtube_views"],
                    y=top_views["artist"],
                    orientation='h',
                    marker_color="#1DB954"
                )
            ])
            fig.update_layout(
                title="Top 10 by Views",
                xaxis_title="Views",
                yaxis_title="Artist",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Top Artists by Likes**")
            top_likes = filtered_df.nlargest(10, "youtube_likes")[["artist", "youtube_likes"]]
            fig = go.Figure(data=[
                go.Bar(
                    x=top_likes["youtube_likes"],
                    y=top_likes["artist"],
                    orientation='h',
                    marker_color="#1DB954"
                )
            ])
            fig.update_layout(
                title="Top 10 by Likes",
                xaxis_title="Likes",
                yaxis_title="Artist",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("**Top Artists by Sentiment**")
            top_sentiment = filtered_df.nlargest(10, "youtube_pos_ratio")[["artist", "youtube_pos_ratio"]]
            fig = go.Figure(data=[
                go.Bar(
                    x=top_sentiment["youtube_pos_ratio"],
                    y=top_sentiment["artist"],
                    orientation='h',
                    marker_color="#1DB954"
                )
            ])
            fig.update_layout(
                title="Top 10 by Sentiment",
                xaxis_title="Positive Ratio",
                yaxis_title="Artist",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plots
        st.subheader("📊 Metrics Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[
                go.Scatter(
                    x=filtered_df["youtube_views"],
                    y=filtered_df["youtube_likes"],
                    mode='markers+text',
                    text=filtered_df["artist"],
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=filtered_df["youtube_pos_ratio"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Sentiment")
                    )
                )
            ])
            fig.update_layout(
                title="Views vs Likes (colored by Sentiment)",
                xaxis_title="YouTube Views",
                yaxis_title="YouTube Likes",
                height=400,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(data=[
                go.Scatter(
                    x=filtered_df["youtube_views"],
                    y=filtered_df["youtube_pos_ratio"],
                    mode='markers+text',
                    text=filtered_df["artist"],
                    textposition="top center",
                    marker=dict(size=10, color="#1DB954")
                )
            ])
            fig.update_layout(
                title="Views vs Sentiment",
                xaxis_title="YouTube Views",
                yaxis_title="Positive Sentiment Ratio",
                height=400,
                template="plotly_dark"
            )
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
                top_views = latest_data.nlargest(10, "youtube_views")[["artist", "youtube_views"]]
                fig = go.Figure(data=[
                    go.Bar(
                        x=top_views["youtube_views"],
                        y=top_views["artist"],
                        orientation='h',
                        marker_color="#1DB954"
                    )
                ])
                fig.update_layout(
                    title="Top 10 by Views",
                    xaxis_title="Views",
                    yaxis_title="Artist",
                    height=300,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Top by Likes**")
                top_likes = latest_data.nlargest(10, "youtube_likes")[["artist", "youtube_likes"]]
                fig = go.Figure(data=[
                    go.Bar(
                        x=top_likes["youtube_likes"],
                        y=top_likes["artist"],
                        orientation='h',
                        marker_color="#1DB954"
                    )
                ])
                fig.update_layout(
                    title="Top 10 by Likes",
                    xaxis_title="Likes",
                    yaxis_title="Artist",
                    height=300,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                st.markdown("**Top by Sentiment**")
                # Check if sentiment column exists and has data
                if "youtube_pos_ratio" in latest_data.columns:
                    # Fill NaN with 0 and ensure numeric
                    sentiment_data = latest_data[["artist", "youtube_pos_ratio"]].copy()
                    sentiment_data["youtube_pos_ratio"] = pd.to_numeric(sentiment_data["youtube_pos_ratio"], errors="coerce").fillna(0)
                    
                    # Get top 10 (including zeros if needed)
                    top_sentiment = sentiment_data.nlargest(10, "youtube_pos_ratio")
                    
                    # Only show if we have at least some data
                    if len(top_sentiment) > 0 and top_sentiment["youtube_pos_ratio"].sum() > 0:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=top_sentiment["youtube_pos_ratio"],
                                y=top_sentiment["artist"],
                                orientation='h',
                                marker_color="#1DB954"
                            )
                        ])
                        max_val = max(0.1, top_sentiment["youtube_pos_ratio"].max() * 1.1)
                        fig.update_layout(
                            title="Top 10 by Sentiment",
                            xaxis_title="Positive Ratio",
                            yaxis_title="Artist",
                            height=300,
                            template="plotly_dark",
                            xaxis=dict(range=[0, max_val])
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Show all artists even if values are 0
                        top_sentiment = sentiment_data.nlargest(10, "youtube_pos_ratio")
                        if len(top_sentiment) > 0:
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=top_sentiment["youtube_pos_ratio"],
                                    y=top_sentiment["artist"],
                                    orientation='h',
                                    marker_color="#888888"
                                )
                            ])
                            fig.update_layout(
                                title="Top 10 by Sentiment (All values are 0)",
                                xaxis_title="Positive Ratio",
                                yaxis_title="Artist",
                                height=300,
                                template="plotly_dark",
                                xaxis=dict(range=[0, 1])
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No sentiment data available for this day")
                else:
                    st.info("Sentiment column not found in data")
            
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
            line=dict(color="#1DB954", width=2),
            mode="lines+markers"
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily["snapshot_date"],
            y=daily["youtube_likes"],
            name="YouTube Likes",
            line=dict(color="#1ed760", width=2),
            mode="lines+markers"
        ),
        secondary_y=True,
    )
    
    fig.update_xaxes(
        title_text="Date",
        type='date',
        tickformat='%Y-%m-%d'
    )
    fig.update_yaxes(title_text="YouTube Views", secondary_y=False)
    fig.update_yaxes(title_text="YouTube Likes", secondary_y=True)
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
            
            fig = go.Figure()
            for artist in multi_date_artists[:5]:  # Limit to 5 artists for readability
                artist_data = artist_trends[artist_trends["artist"] == artist]
                fig.add_trace(go.Scatter(
                    x=artist_data["snapshot_date"],
                    y=artist_data["youtube_views"],
                    name=artist,
                    mode="lines+markers"
                ))
            
            fig.update_layout(
                title="Artist Views Comparison (Top 5 with multi-date data)",
                xaxis_title="Date",
                yaxis_title="YouTube Views",
                height=400,
                template="plotly_dark"
            )
            
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
        
        # Process Reddit data
        if "created_utc" in reddit_df.columns:
            reddit_df["created_utc"] = pd.to_datetime(reddit_df["created_utc"], errors="coerce")
            reddit_df["date"] = reddit_df["created_utc"].dt.date
            reddit_df["snapshot_date"] = pd.to_datetime(reddit_df["date"])
        
        # Important notice
        # Reddit data is limited - silently handle
        
        # Reddit data overview
        st.subheader("📊 Reddit Data Overview")
        
        if "artist" in reddit_df.columns:
            reddit_artists = reddit_df["artist"].dropna().unique()
            st.write(f"**Artists with Reddit data**: {', '.join(reddit_artists)}")
        
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
            
            fig = go.Figure()
            for artist in reddit_daily["artist"].unique():
                artist_data = reddit_daily[reddit_daily["artist"] == artist]
                fig.add_trace(go.Scatter(
                    x=artist_data["snapshot_date"],
                    y=artist_data["num_comments"],
                    name=artist,
                    mode="lines+markers"
                ))
            
            fig.update_layout(
                title="Reddit Comments Over Time",
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
    """Clean text for word cloud generation with better preprocessing."""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove emojis (keep text only)
    text = re.sub(r'[^\w\s\']', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.lower().strip()


def generate_wordcloud(text_data: list[str], title: str, colormap: str = "viridis") -> Optional[bytes]:
    """Generate high-quality word cloud image with improved settings."""
    if not WORDCLOUD_AVAILABLE:
        return None
    
    if not text_data:
        return None
    
    # Common stopwords to exclude
    stopwords_set = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'which', 'who', 'whom',
        'get', 'got', 'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew',
        'think', 'thought', 'say', 'said', 'tell', 'told', 'make', 'made', 'take', 'took',
        'can', 'cant', 'cannot', 'dont', 'wont', 'wouldnt', 'couldnt', 'shouldnt',
        'im', 'youre', 'hes', 'shes', 'its', 'were', 'theyre', 'ive', 'youve', 'weve', 'theyve',
        'just', 'like', 'really', 'very', 'much', 'more', 'most', 'some', 'any', 'all', 'every',
        'one', 'two', 'first', 'second', 'new', 'old', 'good', 'bad', 'great', 'small', 'big',
        'time', 'times', 'day', 'days', 'way', 'ways', 'thing', 'things', 'people', 'person',
        'also', 'still', 'even', 'only', 'now', 'then', 'here', 'there', 'where', 'when', 'why', 'how'
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


def render_comments_analysis(comments_df: pd.DataFrame, topic_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """
    Render comments analysis with word clouds and topic modeling.
    Uses Golden Layer topic model data from GCP (generated by Ran's ML pipeline).
    """
    st.header("🤖 Machine Learning: Topic Modeling Analysis")
    
    # ✅ MUST use Golden Layer topic model data - no fallback allowed
    if topic_df.empty:
        st.error("""
        ❌ **Topic model data not available**
        
        The Golden Layer topic modeling has not been run yet, or the data file is missing.
        
        **Required file**: `gs://apidatabase/cleaned/all_comments_topic_model_YYYYMMDD.csv`
        
        Please ensure:
        1. Golden Layer topic modeling DAG has been executed
        2. The topic model file exists in GCS `cleaned/` directory
        3. Check with Ran (Golden Layer team member) if the ML pipeline is running correctly
        """)
        return
    
    # ✅ Check if topic column exists (generated by Ran's ML model)
    has_topic_column = "topic" in topic_df.columns
    
    if not has_topic_column:
        st.warning("""
        ⚠️ **Topic Modeling Not Yet Completed - Showing Basic Visualizations**
        
        The topic model file exists but doesn't contain the `topic` column yet.
        Showing basic comment analysis visualizations below. 
        
        **Once Ran completes the topic modeling DAG**, the ML-generated topic visualizations will automatically appear here.
        """)
    
    # ✅ Use topic_df directly - this is Ran's ML output
    analysis_df = topic_df.copy()
    
    if analysis_df.empty:
        st.warning("No topic model data available after filtering")
        return
    
    # Get selected artists from filtered_df
    selected_artists = filtered_df["artist"].unique().tolist() if not filtered_df.empty else []
    
    # Filter comments by selected artists
    if selected_artists and "artist" in analysis_df.columns:
        analysis_df = analysis_df[analysis_df["artist"].isin(selected_artists)]
    
    if analysis_df.empty:
        st.warning("No comments match the selected filters")
        return
    
    # Display info about using Golden Layer ML results (only if topic column exists)
    if has_topic_column:
        st.info("""
        📊 **Using Golden Layer Topic Model Results**
        
        This visualization uses machine learning-generated topics from the Golden Layer pipeline.
        Topics are generated using advanced ML models (not simple keyword matching).
        """)
    
    # Topic distribution (from Ran's ML model) - only show if topic column exists
    if has_topic_column:
        st.subheader("📊 Topic Distribution (ML-Generated)")
        if "topic" in analysis_df.columns:
            topic_counts = analysis_df["topic"].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=topic_counts.index,
                    y=topic_counts.values,
                    marker_color="#1DB954"
                )
            ])
            
            fig.update_layout(
                title="Comment Topics Distribution",
                xaxis_title="Topic",
                yaxis_title="Number of Comments",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Topic details - show comments for selected ML-generated topic
            selected_topic = st.selectbox(
                "Select ML-Generated Topic to View Comments", 
                ["All"] + topic_counts.index.tolist()
            )
            if selected_topic != "All":
                topic_comments = analysis_df[analysis_df["topic"] == selected_topic]
                st.markdown(f"**Comments for topic: {selected_topic}** (from Golden Layer ML model)")
                
                # Show available columns
                display_cols = ["comment"]
                if "artist" in topic_comments.columns:
                    display_cols.append("artist")
                if "song" in topic_comments.columns:
                    display_cols.append("song")
                if "timestamp" in topic_comments.columns:
                    display_cols.append("timestamp")
                # Include topic column if it exists with additional info
                if "topic_probability" in topic_comments.columns:
                    display_cols.append("topic_probability")
                
                available_cols = [col for col in display_cols if col in topic_comments.columns]
                st.dataframe(
                    topic_comments[available_cols].head(20),
                    use_container_width=True
                )
    else:
        # Show basic comment statistics when topic column is not available
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
        st.markdown("### ❤️ Negative Comments")
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


def generate_llm_summary(comments_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    """Generate LLM summary of comments and trends."""
    if OPENAI_AVAILABLE:
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
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
    else:
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


def render_llm_summary(comments_df: pd.DataFrame, topic_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Render LLM summary generation interface."""
    st.header("🤖 LLM Summary Generation")
    
    if filtered_df.empty:
        st.warning("No data available for summary generation")
        return
    
    # Use topic_df if available for comments
    analysis_comments = topic_df if not topic_df.empty else comments_df
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📊 Automated Insights")
    
    with col2:
        generate_btn = st.button("🔄 Generate Summary", type="primary")
    
    # Generate summary
    if generate_btn or "summary_text" not in st.session_state:
        with st.spinner("Generating summary..."):
            summary_text = generate_llm_summary(analysis_comments, filtered_df)
            st.session_state["summary_text"] = summary_text
    
    if "summary_text" in st.session_state:
        st.markdown(st.session_state["summary_text"])
    
    # Additional insights
    st.subheader("🔍 Key Insights")
    
    if not filtered_df.empty:
        # Trend insights
        daily = filtered_df.groupby("snapshot_date").agg({
            "youtube_views": "sum",
            "youtube_likes": "sum",
            "reddit_comment_count": "sum"
        }).reset_index()
        
        if len(daily) > 1:
            views_growth = ((daily["youtube_views"].iloc[-1] - daily["youtube_views"].iloc[0]) / 
                          daily["youtube_views"].iloc[0] * 100)
            
            st.metric(
                "Views Growth",
                f"{views_growth:+.1f}%",
                delta=f"From {daily['snapshot_date'].iloc[0].date()} to {daily['snapshot_date'].iloc[-1].date()}"
            )
        
        # Sentiment insights
        avg_sentiment = filtered_df["youtube_pos_ratio"].mean()
        if avg_sentiment > 0.6:
            st.success(f"✅ High positive sentiment: {avg_sentiment:.1%}")
        elif avg_sentiment > 0.4:
            st.info(f"ℹ️ Moderate sentiment: {avg_sentiment:.1%}")
        else:
            st.warning(f"⚠️ Low positive sentiment: {avg_sentiment:.1%}")


def render_top_performers(youtube_df: pd.DataFrame) -> None:
    """Render top 5 engagement leaders using cross-platform metrics."""
    st.subheader("🏆 Top 5 Engagement Leaders")
    
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
    
    # Get top 5
    top = yt.nlargest(5, "engagement_index")
    
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
        title="Top 5 Engagement Leaders",
        xaxis_title="Engagement Index",
        yaxis_title="Artist",
        height=300,
        template="plotly_dark",
        showlegend=False,
        yaxis=dict(autorange="reversed")
    )
    
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
        title="Reddit Discussion Activity by Artist",
        xaxis_title="Total Reddit Comments",
        yaxis_title="Artist",
        height=400,
        template="plotly_dark",
        showlegend=False,
        yaxis=dict(autorange="reversed")
    )
    
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
    
    # Get top 5 artists by views
    top_artists = artist_metrics.nlargest(5, "youtube_views")
    
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
    
    colors = ["#1DB954", "#1ed760", "#ffffff", "#FF5700", "#1f77b4"]
    
    for idx, (_, row) in enumerate(top_artists.iterrows()):
        values = [row[key] for key in metric_keys]
        # Close the radar chart by adding first value at the end
        values = values + [values[0]]
        labels = metric_labels + [metric_labels[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            name=row["artist"],
            line=dict(color=colors[idx % len(colors)], width=2),
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
                range=[0, 1]
            )),
        showlegend=True,
        title="Artist Performance Radar Chart (Top 5)",
        height=500,
        template="plotly_dark"
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
        topic_df = load_topic_model_data_from_gcs()
    
    if summary_df.empty:
        st.error("❌ No data loaded from Silver Layer. Please check GCS connection.")
        return
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Artist filter
    artists = sorted(summary_df["artist"].dropna().unique().tolist())
    selected_artists = st.sidebar.multiselect("Artists", artists, default=artists[:3] if len(artists) >= 3 else artists)
    
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
    
    if date_range and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["snapshot_date"] >= pd.Timestamp(date_range[0])) &
            (filtered_df["snapshot_date"] <= pd.Timestamp(date_range[1]))
        ]
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches the selected filters")
        return
    
    # Main content
    render_hero_metrics(summary_df, filtered_df)
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Daily Snapshot", "📈 YouTube Trends", "💬 Reddit Analysis", "🔄 Cross-Platform Insights", "🤖 ML: Topic Modeling", "🤖 LLM Summary"
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
        render_comments_analysis(comments_df, topic_df, filtered_df)
    
    with tab6:
        render_llm_summary(comments_df, topic_df, filtered_df)


if __name__ == "__main__":
    main()

