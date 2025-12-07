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
import numpy as np
import pandas as pd
import streamlit as st


def _ensure_numeric(df: pd.DataFrame, primary: str, fallback: list[str] | None = None) -> pd.Series:
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


PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
SUMMARY_PARQUET = PROCESSED_DIR / "youtube_summary.parquet"
COMMENTS_PARQUET = PROCESSED_DIR / "youtube_comments.parquet"
REDDIT_SUMMARY_PARQUET = PROCESSED_DIR / "reddit_summary.parquet"
REDDIT_COMMENTS_PARQUET = PROCESSED_DIR / "reddit_comments.parquet"


REDDIT_SUMMARY_COLUMNS = [
    "snapshot_date",
    "artist",
    "song",
    "submission_id",
    "subreddit",
    "title",
    "score",
    "comment_count",
    "pos_comments",
    "neu_comments",
    "neg_comments",
    "avg_compound",
    "fetch_time",
    "created_utc",
]

REDDIT_COMMENT_COLUMNS = [
    "snapshot_date",
    "artist",
    "song",
    "submission_id",
    "comment_id",
    "author",
    "score",
    "compound",
    "label",
    "created_utc",
    "permalink",
]


@st.cache_data(show_spinner=False)
def load_summary_data() -> pd.DataFrame:
    """Load cleaned summary parquet dataset."""
    if not SUMMARY_PARQUET.exists():
        return pd.DataFrame()

    df = pd.read_parquet(SUMMARY_PARQUET)
    if df.empty:
        return pd.DataFrame()

    df["snapshot_date"] = pd.to_datetime(df.get("snapshot_date"), errors="coerce")
    if "fetch_time" in df.columns:
        df["fetch_time"] = pd.to_datetime(df.get("fetch_time"), errors="coerce")
    if "timestamp" in df.columns and "fetch_time" not in df.columns:
        df["fetch_time"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df["published_at"] = pd.to_datetime(df.get("published_at"), errors="coerce")

    required_numeric = [
        "views",
        "likes",
        "comment_count",
        "pos_comments",
        "neu_comments",
        "neg_comments",
        "mean_compound",
    ]
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


@st.cache_data(show_spinner=False)
def load_reddit_summary() -> pd.DataFrame:
    if not REDDIT_SUMMARY_PARQUET.exists():
        return pd.DataFrame(columns=REDDIT_SUMMARY_COLUMNS)

    df = pd.read_parquet(REDDIT_SUMMARY_PARQUET)
    if df.empty:
        return pd.DataFrame(columns=REDDIT_SUMMARY_COLUMNS)

    df["snapshot_date"] = pd.to_datetime(df.get("snapshot_date"), errors="coerce")
    df["created_utc"] = pd.to_datetime(df.get("created_utc"), errors="coerce")
    df["fetch_time"] = pd.to_datetime(df.get("fetch_time"), errors="coerce")

    numeric_cols = [
        "post_count",
        "comment_count",
        "score",
        "avg_score",
        "avg_upvote_ratio",
        "pos_comments",
        "neu_comments",
        "neg_comments",
        "avg_compound",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "artist" not in df.columns and "subreddit" in df.columns:
        df["artist"] = df["subreddit"]
    elif "artist" not in df.columns:
        artist_like_cols = [col for col in df.columns if col.lower().startswith("artist")]
        if artist_like_cols:
            df["artist"] = (
                df[artist_like_cols]
                .replace({np.nan: None, "": None})
                .bfill(axis=1)
                .iloc[:, 0]
            )
        else:
            df["artist"] = "Unknown"

    missing_cols = [col for col in REDDIT_SUMMARY_COLUMNS if col not in df.columns]
    for col in missing_cols:
        df[col] = pd.NA

    return df[REDDIT_SUMMARY_COLUMNS].sort_values(["snapshot_date", "artist"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_reddit_comments() -> pd.DataFrame:
    if not REDDIT_COMMENTS_PARQUET.exists():
        return pd.DataFrame(columns=REDDIT_COMMENT_COLUMNS)

    df = pd.read_parquet(REDDIT_COMMENTS_PARQUET)
    if df.empty:
        return pd.DataFrame(columns=REDDIT_COMMENT_COLUMNS)

    df["snapshot_date"] = pd.to_datetime(df.get("snapshot_date"), errors="coerce")
    df["created_utc"] = pd.to_datetime(df.get("created_utc"), errors="coerce")
    numeric_cols = ["score", "compound"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "artist" not in df.columns:
        artist_like_cols = [col for col in df.columns if col.lower().startswith("artist")]
        if artist_like_cols:
            df["artist"] = (
                df[artist_like_cols]
                .replace({np.nan: None, "": None})
                .bfill(axis=1)
                .iloc[:, 0]
            )
        else:
            df["artist"] = "Unknown"

    missing_cols = [col for col in REDDIT_COMMENT_COLUMNS if col not in df.columns]
    for col in missing_cols:
        df[col] = pd.NA

    return df[REDDIT_COMMENT_COLUMNS].sort_values(["snapshot_date", "artist", "created_utc"]).reset_index(drop=True)


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


def render_top_performers(youtube_df: pd.DataFrame, reddit_df: pd.DataFrame) -> None:
    st.subheader("Top 5 Engagement Leaders")
    if youtube_df.empty and reddit_df.empty:
        st.info("No engagement data available for the selected filters.")
        return

    yt = (
        youtube_df.dropna(subset=["channel"])
        .assign(
            views=lambda d: _ensure_numeric(d, "views"),
            likes=lambda d: _ensure_numeric(d, "likes"),
            yt_comments=lambda d: _ensure_numeric(d, "comment_count", ["num_comments"]),
        )
        .groupby("channel", as_index=False)
        .agg(
            views=("views", "sum"),
            likes=("likes", "sum"),
            yt_comments=("yt_comments", "sum"),
        )
        .rename(columns={"channel": "artist"})
    )

    reddit = (
        reddit_df.dropna(subset=["artist"])
        .assign(
            reddit_posts=lambda d: _ensure_numeric(d, "post_count"),
            reddit_comments=lambda d: _ensure_numeric(d, "comment_count", ["num_comments"]),
            reddit_score=lambda d: _ensure_numeric(d, "score"),
        )
        .groupby("artist", as_index=False)
        .agg(
            reddit_posts=("reddit_posts", "sum"),
            reddit_comments=("reddit_comments", "sum"),
            reddit_score=("reddit_score", "sum"),
        )
    )

    combined = pd.merge(yt, reddit, on="artist", how="outer").fillna(0)
    if combined.empty:
        st.info("No overlapping artist data for the current filters.")
        return

    metric_cols = ["views", "likes", "yt_comments", "reddit_posts", "reddit_comments", "reddit_score"]
    norm_cols: list[str] = []
    for col in metric_cols:
        if col not in combined:
            combined[col] = 0.0
        max_val = combined[col].max()
        norm_col = f"{col}_norm"
        if max_val and not np.isclose(max_val, 0):
            combined[norm_col] = combined[col] / max_val
        else:
            combined[norm_col] = 0.0
        norm_cols.append(norm_col)

    combined["reddit_sentiment"] = combined.get("reddit_sentiment", 0.0)
    combined["sentiment_norm"] = ((combined["reddit_sentiment"].fillna(0.0) + 1) / 2).clip(0, 1)
    norm_cols.append("sentiment_norm")
    combined["engagement_index"] = combined[norm_cols].mean(axis=1)

    top = combined.sort_values("engagement_index", ascending=False).head(5)

    chart = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X("engagement_index:Q", title="Engagement Index"),
            y=alt.Y("artist:N", sort="-x"),
            tooltip=[
                alt.Tooltip("artist:N", title="Artist"),
                alt.Tooltip("views:Q", title="YouTube Views", format=",.0f"),
                alt.Tooltip("likes:Q", title="YouTube Likes", format=",.0f"),
                alt.Tooltip("yt_comments:Q", title="YouTube Comments", format=",.0f"),
                alt.Tooltip("reddit_posts:Q", title="Reddit Posts", format=",.0f"),
                alt.Tooltip("reddit_comments:Q", title="Reddit Comments", format=",.0f"),
                alt.Tooltip("reddit_score:Q", title="Reddit Score", format=",.0f"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, width="stretch")

    view_cols = [
        "artist",
        "views",
        "likes",
        "yt_comments",
        "reddit_posts",
        "reddit_comments",
        "reddit_score",
        "engagement_index",
    ]
    st.dataframe(
        top[view_cols].rename(
            columns={
                "views": "YT Views",
                "likes": "YT Likes",
                "yt_comments": "YT Comments",
                "reddit_posts": "Reddit Posts",
                "reddit_comments": "Reddit Comments",
                "reddit_score": "Reddit Score",
                "engagement_index": "Engagement Index",
            }
        ),
        width="stretch",
    )


def render_cross_platform_trends(youtube_df: pd.DataFrame, reddit_df: pd.DataFrame) -> None:
    st.subheader("Cross-Platform Activity")
    if youtube_df.empty and reddit_df.empty:
        st.info("No trend data available for the selected filters.")
        return

    yt_numeric = youtube_df.copy()
    yt_numeric["views"] = _ensure_numeric(yt_numeric, "views")
    yt_numeric["likes"] = _ensure_numeric(yt_numeric, "likes")
    yt_numeric["yt_comments"] = _ensure_numeric(yt_numeric, "comment_count", ["num_comments"])

    yt_daily = (
        yt_numeric.groupby("snapshot_date")
        .agg(
            views=("views", "sum"),
            likes=("likes", "sum"),
            yt_comments=("yt_comments", "sum"),
        )
        .reset_index()
    )

    reddit_numeric = reddit_df.copy()
    if reddit_numeric.empty:
        reddit_daily = pd.DataFrame(columns=["snapshot_date", "reddit_posts", "reddit_comments", "reddit_score"])
    else:
        reddit_numeric["reddit_comments"] = _ensure_numeric(reddit_numeric, "comment_count", ["num_comments"])
        reddit_numeric["reddit_score"] = _ensure_numeric(reddit_numeric, "score")

        if "submission_id" in reddit_numeric.columns:
            reddit_posts = (
                reddit_numeric.groupby("snapshot_date")["submission_id"].nunique().reset_index(name="reddit_posts")
            )
        else:
            reddit_posts = reddit_numeric.groupby("snapshot_date").size().reset_index(name="reddit_posts")

        reddit_daily = (
            reddit_numeric.groupby("snapshot_date")
            .agg(
                reddit_comments=("reddit_comments", "sum"),
                reddit_score=("reddit_score", "sum"),
            )
            .reset_index()
        )
        reddit_daily = pd.merge(reddit_daily, reddit_posts, on="snapshot_date", how="left")

    daily = pd.merge(yt_daily, reddit_daily, on="snapshot_date", how="outer").fillna(0)
    daily.sort_values("snapshot_date", inplace=True)

    if daily.empty:
        st.info("No trend data available for the selected filters.")
        return

    engagement_chart = (
        alt.layer(
            alt.Chart(daily)
            .mark_line(point=True, color="#1f77b4")
            .encode(
                x="snapshot_date:T",
                y=alt.Y("views:Q", title="YouTube Views"),
                tooltip=["snapshot_date:T", alt.Tooltip("views:Q", format=",.0f")],
            ),
            alt.Chart(daily)
            .mark_line(point=True, color="#ff7f0e")
            .encode(
                x="snapshot_date:T",
                y=alt.Y("reddit_posts:Q", title="Reddit Posts"),
                tooltip=["snapshot_date:T", alt.Tooltip("reddit_posts:Q", format=",.0f")],
            ),
        )
        .resolve_scale(y="independent")
        .properties(height=320)
    )
    st.altair_chart(engagement_chart, width="stretch")

    comments_chart = (
        alt.layer(
            alt.Chart(daily)
            .mark_line(point=True, color="#2ca02c")
            .encode(
                x="snapshot_date:T",
                y=alt.Y("yt_comments:Q", title="YouTube Comments"),
                tooltip=["snapshot_date:T", alt.Tooltip("yt_comments:Q", format=",.0f")],
            ),
            alt.Chart(daily)
            .mark_line(point=True, color="#d62728")
            .encode(
                x="snapshot_date:T",
                y=alt.Y("reddit_comments:Q", title="Reddit Comments"),
                tooltip=["snapshot_date:T", alt.Tooltip("reddit_comments:Q", format=",.0f")],
            ),
        )
        .resolve_scale(y="independent")
        .properties(height=280)
    )
    st.altair_chart(comments_chart, width="stretch")


def render_sentiment_comparison(youtube_df: pd.DataFrame, reddit_df: pd.DataFrame) -> None:
    st.subheader("Sentiment Comparison")
    if youtube_df.empty and reddit_df.empty:
        st.info("No sentiment data available for the selected filters.")
        return

    yt_df = youtube_df.copy()
    yt_df["yt_sentiment"] = _ensure_numeric(yt_df, "mean_compound", ["avg_compound", "compound"])
    yt_sentiment = yt_df.groupby("snapshot_date")["yt_sentiment"].mean().reset_index(name="YouTube")

    reddit_df = reddit_df.copy()
    reddit_df["reddit_sentiment"] = _ensure_numeric(
        reddit_df, "mean_compound", ["avg_compound", "compound"]
    )
    reddit_sentiment = reddit_df.groupby("snapshot_date")["reddit_sentiment"].mean().reset_index(name="Reddit")

    sentiment = (
        pd.merge(yt_sentiment, reddit_sentiment, on="snapshot_date", how="outer")
        .sort_values("snapshot_date")
        .ffill()
    )

    if sentiment[["YouTube", "Reddit"]].dropna(how="all").empty:
        st.info("No sentiment data available for the selected filters.")
        return

    sentiment_long = sentiment.melt(
        id_vars="snapshot_date", value_name="sentiment", var_name="platform"
    )
    chart = (
        alt.Chart(sentiment_long.dropna())
        .mark_line(point=True)
        .encode(
            x="snapshot_date:T",
            y=alt.Y("sentiment:Q", title="Average Compound Sentiment"),
            color=alt.Color("platform:N", title="Platform"),
            tooltip=["snapshot_date:T", "platform:N", alt.Tooltip("sentiment:Q", format=".2f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")


def render_radar_insights(youtube_df: pd.DataFrame, reddit_df: pd.DataFrame) -> None:
    st.subheader("Artist Radar")
    if youtube_df.empty and reddit_df.empty:
        st.info("No artist metrics available for the selected filters.")
        return

    yt_df = youtube_df.copy()
    yt_df["views"] = _ensure_numeric(yt_df, "views")
    yt_df["likes"] = _ensure_numeric(yt_df, "likes")
    yt_df["yt_comments"] = _ensure_numeric(yt_df, "comment_count", ["num_comments"])
    yt_df["yt_sentiment"] = _ensure_numeric(yt_df, "mean_compound", ["avg_compound", "compound"])

    yt_artist = (
        yt_df.dropna(subset=["channel"])
        .groupby("channel", as_index=False)
        .agg(
            views=("views", "sum"),
            likes=("likes", "sum"),
            yt_comments=("yt_comments", "sum"),
            yt_sentiment=("yt_sentiment", "mean"),
        )
        .rename(columns={"channel": "artist"})
    )

    reddit_df = reddit_df.copy()
    if reddit_df.empty:
        reddit_artist = pd.DataFrame(columns=["artist", "reddit_posts", "reddit_comments", "reddit_sentiment"])
    else:
        reddit_df["reddit_comments"] = _ensure_numeric(reddit_df, "comment_count", ["num_comments"])
        reddit_df["reddit_sentiment"] = _ensure_numeric(
            reddit_df, "mean_compound", ["avg_compound", "compound"]
        )

        if "submission_id" in reddit_df.columns:
            reddit_posts = (
                reddit_df.dropna(subset=["artist"])
                .groupby("artist")["submission_id"].nunique()
                .reset_index(name="reddit_posts")
            )
        else:
            reddit_posts = (
                reddit_df.dropna(subset=["artist"])
                .groupby("artist")
                .size()
                .reset_index(name="reddit_posts")
            )

        reddit_artist = (
            reddit_df.dropna(subset=["artist"])
            .groupby("artist", as_index=False)
            .agg(
                reddit_comments=("reddit_comments", "sum"),
                reddit_sentiment=("reddit_sentiment", "mean"),
            )
        )
        reddit_artist = pd.merge(reddit_artist, reddit_posts, on="artist", how="left").fillna(0)

    metrics = pd.merge(yt_artist, reddit_artist, on="artist", how="outer").fillna(0.0)
    if metrics.empty:
        st.info("No artist metrics available for the selected filters.")
        return

    metrics["sentiment_blend"] = (
        metrics[["yt_sentiment", "reddit_sentiment"]]
        .replace({0.0: np.nan})
        .mean(axis=1, skipna=True)
        .fillna(0.0)
    )

    value_cols = [
        "views",
        "likes",
        "yt_comments",
        "reddit_posts",
        "reddit_comments",
        "sentiment_blend",
    ]
    radar_metrics = metrics.sort_values("views", ascending=False).head(5).copy()
    if radar_metrics.empty:
        st.info("Not enough data to build the radar chart.")
        return

    normalized_values: dict[str, pd.Series] = {}
    for col in value_cols[:-1]:
        max_val = radar_metrics[col].max()
        if max_val and not np.isclose(max_val, 0):
            normalized_values[col] = radar_metrics[col] / max_val
        else:
            normalized_values[col] = pd.Series(0.0, index=radar_metrics.index)
    normalized_values["sentiment"] = ((radar_metrics["sentiment_blend"] + 1) / 2).clip(0, 1)

    radar_norm = pd.DataFrame(normalized_values)
    radar_norm["artist"] = radar_metrics["artist"].values

    tooltip_base = radar_metrics[
        ["artist", "views", "likes", "yt_comments", "reddit_posts", "reddit_comments", "sentiment_blend"]
    ].rename(columns={"sentiment_blend": "sentiment"})

    radar_long = radar_norm.melt(id_vars="artist", var_name="metric", value_name="value")
    tooltip_long = tooltip_base.melt(id_vars="artist", var_name="metric", value_name="raw_value")
    metric_labels = {
        "views": "YT Views",
        "likes": "YT Likes",
        "yt_comments": "YT Comments",
        "reddit_posts": "Reddit Posts",
        "reddit_comments": "Reddit Comments",
        "sentiment": "Avg Sentiment",
    }
    radar_long["metric"] = radar_long["metric"].map(metric_labels)
    tooltip_long["metric"] = tooltip_long["metric"].map(metric_labels)
    radar_data = radar_long.merge(tooltip_long, on=["artist", "metric"], how="left")

    radar_chart = (
        alt.Chart(radar_data)
        .mark_line(point=True)
        .encode(
            theta=alt.Theta("metric:N", sort=None, title=None),
            radius=alt.Radius("value:Q", scale=alt.Scale(domain=[0, 1]), title=""),
            color=alt.Color("artist:N", title="Artist"),
            tooltip=[
                alt.Tooltip("artist:N", title="Artist"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("raw_value:Q", title="Value", format=",.2f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(radar_chart, width="stretch")


def render_discussion_distribution(reddit_df: pd.DataFrame) -> None:
    st.subheader("Reddit Discussion Mix")
    if reddit_df.empty:
        st.info("No Reddit metrics available for the selected filters.")
        return

    reddit_df = reddit_df.copy()
    reddit_df["reddit_comments"] = _ensure_numeric(reddit_df, "comment_count", ["num_comments"])
    reddit_df["reddit_score"] = _ensure_numeric(reddit_df, "score")

    subreddit_totals = (
        reddit_df.groupby("artist")
        .agg(
            reddit_comments=("reddit_comments", "sum"),
            reddit_posts=("submission_id", "nunique"),
            reddit_score=("reddit_score", "sum"),
        )
        .reset_index()
    )

    if subreddit_totals.empty:
        st.info("No Reddit discussion data for the selected filters.")
        return

    chart = (
        alt.Chart(subreddit_totals)
        .transform_fold(["reddit_comments", "reddit_posts", "reddit_score"], as_=["metric", "value"])
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title="Volume"),
            y=alt.Y("artist:N", sort="-x", title="Artist"),
            color=alt.Color("metric:N", title=""),
            tooltip=[
                alt.Tooltip("artist:N", title="Artist"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Volume", format=",.0f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def render_reddit_comments_view(
    comments_df: pd.DataFrame, selected_artist: str, max_comments: int
) -> None:
    if comments_df.empty:
        st.info("No Reddit comments available for the current filters.")
        return

    comments_df = comments_df.copy()
    comments_df["created_utc"] = pd.to_datetime(comments_df.get("created_utc"), errors="coerce")

    text_column = None
    for candidate in ("body", "text", "comment", "message"):
        if candidate in comments_df.columns:
            text_column = candidate
            break
    if text_column is None:
        object_cols = [col for col in comments_df.columns if comments_df[col].dtype == "object"]
        if object_cols:
            text_column = object_cols[0]
            comments_df[text_column] = comments_df[text_column].astype(str)
        else:
            text_column = "comment_text"
            comments_df[text_column] = ""

    display_columns = ["created_utc", "author", text_column, "score", "compound", "label"]
    display_columns = [col for col in display_columns if col in comments_df.columns]

    st.subheader("Recent Reddit Comments")
    recent = (
        comments_df.sort_values("created_utc", ascending=False)
        .head(max_comments)
        .loc[:, display_columns]
        .rename(columns={text_column: "comment", "created_utc": "created_at"})
    )
    st.dataframe(recent, width="stretch")

    if "label" in comments_df.columns:
        sentiment_counts = comments_df["label"].value_counts(normalize=True)
        if not sentiment_counts.empty:
            sentiment_distribution = (
                sentiment_counts.rename_axis("sentiment").reset_index(name="share")
            )
            chart = (
                alt.Chart(sentiment_distribution)
                .mark_bar()
                .encode(
                    x=alt.X("sentiment:N", title="Sentiment"),
                    y=alt.Y("share:Q", title="Share", axis=alt.Axis(format="%")),
                    tooltip=["sentiment:N", alt.Tooltip("share:Q", format=".1%")],
                )
                .properties(height=200)
            )
            st.altair_chart(chart, width="stretch")

    if selected_artist:
        st.caption(f"Showing latest {max_comments} Reddit comments for {selected_artist}.")


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
    st.dataframe(recent_comments, width="stretch")

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
    try:
        st.set_page_config(page_title="Multi-Platform Trend Dashboard", layout="wide")
    except Exception:
        pass  # Already set
        
    # Always show this to confirm app is running
    st.title("🎵 Daily Music Trend Dashboard")
    st.caption("YouTube & Reddit data refreshed via automated pipeline.")
    
    st.markdown("---")
    st.info("ℹ️ Application is loading...")

    # Debug: Show file paths
    with st.expander("🔍 Debug Info", expanded=False):
        st.write("**Data file paths:**")
        st.write(f"- SUMMARY_PARQUET: {SUMMARY_PARQUET}")
        st.write(f"- Exists: {SUMMARY_PARQUET.exists()}")
        st.write(f"- COMMENTS_PARQUET: {COMMENTS_PARQUET}")
        st.write(f"- Exists: {COMMENTS_PARQUET.exists()}")
        st.write(f"- REDDIT_SUMMARY_PARQUET: {REDDIT_SUMMARY_PARQUET}")
        st.write(f"- Exists: {REDDIT_SUMMARY_PARQUET.exists()}")
        st.write(f"- REDDIT_COMMENTS_PARQUET: {REDDIT_COMMENTS_PARQUET}")
        st.write(f"- Exists: {REDDIT_COMMENTS_PARQUET.exists()}")
    
    summary_df = load_summary_data()
    comments_df = load_comments_data()
    reddit_summary_df = load_reddit_summary()
    reddit_comments_df = load_reddit_comments()

    # Debug: Show data status
    with st.expander("🔍 Debug Info", expanded=False):
        st.write("**Data loaded:**")
        st.write(f"- YouTube summary: {len(summary_df)} rows")
        st.write(f"- YouTube comments: {len(comments_df)} rows")
        st.write(f"- Reddit summary: {len(reddit_summary_df)} rows")
        st.write(f"- Reddit comments: {len(reddit_comments_df)} rows")
        
        # Show column info
        if not summary_df.empty:
            st.write(f"- YouTube columns: {list(summary_df.columns)[:5]}...")
            if "channel" in summary_df.columns:
                st.write(f"- YouTube channels: {summary_df['channel'].dropna().unique()[:3]}")
        if not reddit_summary_df.empty:
            st.write(f"- Reddit columns: {list(reddit_summary_df.columns)[:5]}...")
            if "artist" in reddit_summary_df.columns:
                st.write(f"- Reddit artists: {reddit_summary_df['artist'].dropna().unique()[:3]}")

    if summary_df.empty and reddit_summary_df.empty:
        st.error("❌ No processed data found!")
        st.info("Expected data files:")
        st.code(f"""
- {SUMMARY_PARQUET}
- {COMMENTS_PARQUET}
- {REDDIT_SUMMARY_PARQUET}
- {REDDIT_COMMENTS_PARQUET}
        """)
        st.warning("Please run the cleaning scripts after syncing raw CSVs from GCS.")
        return

    date_series = []
    if not summary_df.empty:
        if "snapshot_date" in summary_df.columns:
            date_series.append(summary_df["snapshot_date"].dropna())
    if not reddit_summary_df.empty:
        if "snapshot_date" in reddit_summary_df.columns:
            date_series.append(reddit_summary_df["snapshot_date"].dropna())
    if date_series:
        all_dates = pd.concat(date_series)
    else:
        st.warning("⚠️ No snapshot dates available in the processed datasets.")
        st.info("Debug: Checking date columns...")
        with st.expander("Debug Date Info"):
            if not summary_df.empty:
                st.write(f"YouTube columns: {list(summary_df.columns)}")
                if "snapshot_date" in summary_df.columns:
                    st.write(f"YouTube dates: {summary_df['snapshot_date'].head()}")
            if not reddit_summary_df.empty:
                st.write(f"Reddit columns: {list(reddit_summary_df.columns)}")
                if "snapshot_date" in reddit_summary_df.columns:
                    st.write(f"Reddit dates: {reddit_summary_df['snapshot_date'].head()}")
        # Don't return - continue to show what we can
        st.stop()

    yt_channels = []
    if not summary_df.empty:
        if "channel" in summary_df.columns:
            yt_channels = summary_df["channel"].dropna().unique().tolist()
        elif "artist" in summary_df.columns:
            yt_channels = summary_df["artist"].dropna().unique().tolist()
    
    reddit_artists = []
    if not reddit_summary_df.empty:
        if "artist" in reddit_summary_df.columns:
            reddit_artists = reddit_summary_df["artist"].dropna().unique().tolist()
        elif "subreddit" in reddit_summary_df.columns:
            reddit_artists = reddit_summary_df["subreddit"].dropna().unique().tolist()
    
    all_artists = sorted(set(yt_channels) | set(reddit_artists))
    
    # Debug: Show what we found
    with st.expander("🔍 Debug Info", expanded=False):
        st.write(f"**Artists found:** {len(all_artists)}")
        if all_artists:
            st.write(f"First 5: {all_artists[:5]}")
    
    if not all_artists:
        st.error("❌ No artist/channel identifiers found!")
        st.info("Available columns:")
        if not summary_df.empty:
            st.write(f"YouTube: {list(summary_df.columns)}")
        if not reddit_summary_df.empty:
            st.write(f"Reddit: {list(reddit_summary_df.columns)}")
        st.stop()

    st.sidebar.header("Filters")
    selected_artists = st.sidebar.multiselect(
        "Artists / Channels",
        options=all_artists,
        default=all_artists,
    )
    if not selected_artists:
        st.info("请选择至少一位艺人/频道以查看图表。")
        return

    min_date = all_dates.min().date()
    max_date = all_dates.max().date()
    start_date, end_date = st.sidebar.slider(
        "Snapshot Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )
    date_start = pd.to_datetime(start_date)
    date_end = pd.to_datetime(end_date)

    max_comments = st.sidebar.slider("Comments to Display", min_value=10, max_value=200, value=50, step=10)

    filtered_summary = summary_df[
        (summary_df["channel"].isin(selected_artists))
        & (summary_df["snapshot_date"].between(date_start, date_end))
    ]
    filtered_reddit_summary = reddit_summary_df[
        (reddit_summary_df["artist"].isin(selected_artists))
        & (reddit_summary_df["snapshot_date"].between(date_start, date_end))
    ]

    filtered_youtube_comments = comments_df[
        (comments_df["artist"].isin(selected_artists))
        & (comments_df["snapshot_date"].between(date_start, date_end))
    ] if not comments_df.empty else pd.DataFrame()

    filtered_reddit_comments = reddit_comments_df[
        (reddit_comments_df["artist"].isin(selected_artists))
        & (reddit_comments_df["snapshot_date"].between(date_start, date_end))
    ] if not reddit_comments_df.empty else pd.DataFrame()

    selected_label = ", ".join(selected_artists)

    tabs = st.tabs([
        "YouTube Overview",
        "Cross-Platform",
        "Community Insights",
        "Comments",
    ])

    with tabs[0]:
        if filtered_summary.empty and filtered_reddit_summary.empty:
            st.info("No YouTube or Reddit engagement data for the selected filters.")
        else:
            if not filtered_summary.empty:
                render_kpis(summary_df, filtered_summary)
            render_top_performers(filtered_summary, filtered_reddit_summary)
            if not filtered_summary.empty:
                render_view_trends(filtered_summary)
                render_sentiment_trends(filtered_summary)

    with tabs[1]:
        render_cross_platform_trends(filtered_summary, filtered_reddit_summary)
        render_sentiment_comparison(filtered_summary, filtered_reddit_summary)

    with tabs[2]:
        render_radar_insights(filtered_summary, filtered_reddit_summary)
        render_discussion_distribution(filtered_reddit_summary)

    with tabs[3]:
        render_comments_view(filtered_youtube_comments, selected_label, max_comments)
        render_reddit_comments_view(filtered_reddit_comments, selected_label, max_comments)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error("❌ Application Error!")
        st.exception(e)
        st.code(traceback.format_exc())

