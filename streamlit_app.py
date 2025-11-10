"""
Streamlit dashboard for daily YouTube performance and sentiment trends.

The app expects cleaned parquet datasets produced by `src/processing/clean_youtube.py`
to live under:
  - ./data/processed/youtube_summary.parquet
  - ./data/processed/youtube_comments.parquet

Run the cleaning script after syncing raw CSVs from GCS to refresh these tables.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
SUMMARY_PARQUET = PROCESSED_DIR / "youtube_summary.parquet"
COMMENTS_PARQUET = PROCESSED_DIR / "youtube_comments.parquet"


@st.cache_data(show_spinner=False)
def load_summary_data() -> pd.DataFrame:
    """Load cleaned summary parquet dataset."""
    if not SUMMARY_PARQUET.exists():
        return pd.DataFrame()

    df = pd.read_parquet(SUMMARY_PARQUET)
    if df.empty:
        return df

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    required_numeric = ["views", "likes", "comment_count", "pos_comments", "neu_comments", "neg_comments", "mean_compound"]
    for col in required_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df.sort_values(["snapshot_date", "channel", "title"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_comments_data() -> pd.DataFrame:
    """Load cleaned comments parquet dataset."""
    if not COMMENTS_PARQUET.exists():
        return pd.DataFrame()

    df = pd.read_parquet(COMMENTS_PARQUET)
    if df.empty:
        return df

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    if "like_count" in df.columns:
        df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    return df.sort_values(["snapshot_date", "artist", "published_at"]).reset_index(drop=True)


def render_kpis(summary_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    latest_snapshot = filtered_df["snapshot_date"].max()
    latest_data = filtered_df[filtered_df["snapshot_date"] == latest_snapshot]

    total_views = float(latest_data["views"].sum())
    total_likes = float(latest_data["likes"].sum())
    avg_sentiment = latest_data["mean_compound"].mean()

    st.subheader("Latest Snapshot KPIs")
    kpi_cols = st.columns(3)

    delta_views = delta_likes = delta_sentiment = None
    unique_dates = sorted(summary_df["snapshot_date"].dropna().unique())
    if len(unique_dates) > 1:
        previous_snapshot = unique_dates[-2]
        previous_data = filtered_df[filtered_df["snapshot_date"] == previous_snapshot]
        prev_views = float(previous_data["views"].sum())
        prev_likes = float(previous_data["likes"].sum())
        prev_sentiment = previous_data["mean_compound"].mean()

        delta_views = total_views - prev_views if pd.notna(prev_views) else None
        delta_likes = total_likes - prev_likes if pd.notna(prev_likes) else None
        if pd.notna(avg_sentiment) and pd.notna(prev_sentiment):
            delta_sentiment = avg_sentiment - prev_sentiment

    kpi_cols[0].metric(
        "Total Views",
        f"{total_views:,.0f}",
        delta=f"{delta_views:,.0f}" if delta_views is not None else None,
    )
    kpi_cols[1].metric(
        "Total Likes",
        f"{total_likes:,.0f}",
        delta=f"{delta_likes:,.0f}" if delta_likes is not None else None,
    )
    kpi_cols[2].metric(
        "Average Sentiment",
        f"{avg_sentiment:.3f}" if pd.notna(avg_sentiment) else "N/A",
        delta=f"{delta_sentiment:+.3f}" if delta_sentiment is not None else None,
    )


def render_view_trends(filtered_df: pd.DataFrame) -> None:
    trend_metrics = filtered_df.melt(
        id_vars=["channel", "title", "snapshot_date"],
        value_vars=["views", "likes", "comment_count"],
        var_name="metric",
        value_name="value",
    )
    chart = (
        alt.Chart(trend_metrics.dropna())
        .mark_line(point=True)
        .encode(
            x="snapshot_date:T",
            y=alt.Y("value:Q", title="Count"),
            color="metric:N",
            tooltip=["snapshot_date:T", "metric:N", "value:Q", "title:N", "channel:N"],
        )
        .properties(height=320)
    )
    st.subheader("Engagement Trend")
    st.altair_chart(chart, width="stretch")


def render_sentiment_trends(filtered_df: pd.DataFrame) -> None:
    sentiment_trend = (
        filtered_df.groupby(["snapshot_date"])
        .agg(
            pos_comments=("pos_comments", "sum"),
            neu_comments=("neu_comments", "sum"),
            neg_comments=("neg_comments", "sum"),
            mean_compound=("mean_compound", "mean"),
        )
        .reset_index()
    )

    long_sentiment = sentiment_trend.melt(
        id_vars=["snapshot_date", "mean_compound"],
        value_vars=["pos_comments", "neu_comments", "neg_comments"],
        var_name="sentiment",
        value_name="count",
    )

    sentiment_chart = (
        alt.Chart(long_sentiment)
        .mark_area(opacity=0.6)
        .encode(
            x="snapshot_date:T",
            y=alt.Y("count:Q", title="Comment Volume"),
            color=alt.Color("sentiment:N", scale=alt.Scale(scheme="set1")),
            tooltip=["snapshot_date:T", "sentiment:N", "count:Q"],
        )
        .properties(height=280)
    )

    compound_chart = (
        alt.Chart(sentiment_trend)
        .mark_line(point=True, color="#333")
        .encode(
            x="snapshot_date:T",
            y=alt.Y("mean_compound:Q", title="Average Compound Sentiment"),
            tooltip=["snapshot_date:T", "mean_compound:Q"],
        )
        .properties(height=200)
    )

    st.subheader("Sentiment Trend")
    st.altair_chart(sentiment_chart, width="stretch")
    st.altair_chart(compound_chart, width="stretch")


def render_comments_view(comments_df: pd.DataFrame, selected_artist: str, max_comments: int) -> None:
    if comments_df.empty:
        st.info("No comment data available for the current filters.")
        return

    st.subheader("Recent Comments")
    recent_comments = (
        comments_df.sort_values("published_at", ascending=False)
        .head(max_comments)
        .loc[:, ["published_at", "author", "text", "compound", "label"]]
    )
    st.dataframe(recent_comments, use_container_width=True)

    sentiment_counts = comments_df["label"].value_counts(normalize=True)
    if sentiment_counts.empty:
        return

    sentiment_distribution = sentiment_counts.rename_axis("sentiment").reset_index(name="share")

    bar_chart = (
        alt.Chart(sentiment_distribution)
        .mark_bar()
        .encode(
            x=alt.X("sentiment:N", title="Sentiment"),
            y=alt.Y("share:Q", title="Share", axis=alt.Axis(format="%")),
            tooltip=["sentiment:N", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=200)
    )

    st.altair_chart(bar_chart, width="stretch")
    if selected_artist:
        st.caption(f"Showing latest {max_comments} comments for {selected_artist}.")


def main() -> None:
    st.set_page_config(page_title="YouTube Trend Dashboard", layout="wide")
    st.title("YouTube Daily Trend Dashboard")
    st.caption("Data refreshed via automated pipeline and synced from Google Cloud Storage.")

    summary_df = load_summary_data()
    comments_df = load_comments_data()

    if summary_df.empty:
        if not SUMMARY_PARQUET.exists():
            st.warning(
                "No processed summary data found. Run `python src/processing/clean_youtube.py` "
                "after syncing raw CSVs."
            )
        else:
            st.warning("Processed summary dataset is empty.")
        return

    available_dates = summary_df["snapshot_date"].dropna()
    if available_dates.empty:
        st.warning("Summary data is missing snapshot dates.")
        return

    channels = summary_df["channel"].dropna().unique()
    st.sidebar.header("Filters")
    selected_channels = st.sidebar.multiselect(
        "Artists / Channels",
        options=sorted(channels),
        default=sorted(channels),
    )
    if not selected_channels:
        st.info("Select at least one artist/channel to display data.")
        return

    date_range = st.sidebar.slider(
        "Snapshot Date Range",
        min_value=available_dates.min().date(),
        max_value=available_dates.max().date(),
        value=(
            available_dates.min().date(),
            available_dates.max().date(),
        ),
    )
    max_comments = st.sidebar.slider("Comments to Display", min_value=10, max_value=200, value=50, step=10)

    filtered_summary = summary_df[
        summary_df["channel"].isin(selected_channels)
        & summary_df["snapshot_date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
    ]

    if filtered_summary.empty:
        st.warning("No data for the selected filters.")
        return

    render_kpis(summary_df, filtered_summary)
    render_view_trends(filtered_summary)
    render_sentiment_trends(filtered_summary)

    if not comments_df.empty:
        selected_video_ids = filtered_summary["video_id"].unique()
        filtered_comments = comments_df[
            comments_df["video_id"].isin(selected_video_ids)
            & comments_df["snapshot_date"].between(
                pd.to_datetime(date_range[0]),
                pd.to_datetime(date_range[1]),
            )
        ]
    else:
        filtered_comments = pd.DataFrame()

    render_comments_view(filtered_comments, ", ".join(selected_channels), max_comments)


if __name__ == "__main__":
    main()

