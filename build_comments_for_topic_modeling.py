import io
import re
from datetime import datetime, timezone

import pandas as pd
from google.cloud import storage


BUCKET_NAME = "apidatabase"

YOUTUBE_COMMENTS_PREFIX = "youtube/comments/"
YOUTUBE_SUMMARY_PREFIX = "youtube/summary/"
REDDIT_COMMENTS_PREFIX = "reddit/comments/"

OUTPUT_PREFIX = "cleaned/"


# ---------- Helpers ----------

def _list_csv_blobs(client, prefix: str):
    bucket = client.bucket(BUCKET_NAME)
    blobs = client.list_blobs(bucket_or_name=bucket, prefix=prefix)
    return [b for b in blobs if b.name.endswith(".csv")]


def _read_csv_blob(client, blob):
    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data))


def normalize_song_from_title(title: str) -> str:
    """Extract clean song name from a YouTube title."""
    if not isinstance(title, str):
        return None
    parts = title.split(" - ", 1)
    if len(parts) == 2:
        candidate = parts[1]
    else:
        candidate = parts[0]
    # Remove trailing "(Official ...)" or "[Official ...]"
    candidate = re.sub(r"\s*[\(\[].*?[\)\]]\s*$", "", candidate)
    return candidate.strip()


def normalize_artist_from_title_or_channel(title: str, channel: str) -> str:
    """Prefer artist from 'Artist - Song', fallback to channel name."""
    artist = None
    if isinstance(title, str) and " - " in title:
        artist = title.split(" - ", 1)[0].strip()
    if artist:
        return artist
    if isinstance(channel, str):
        return channel.strip()
    return None


# ---------- YouTube comments ----------

def load_youtube_comments_for_topic(client) -> pd.DataFrame:
    comment_blobs = _list_csv_blobs(client, YOUTUBE_COMMENTS_PREFIX)
    if not comment_blobs:
        print("No YouTube comments CSV found.")
        return pd.DataFrame()

    comment_dfs = [_read_csv_blob(client, b) for b in comment_blobs]
    comments_raw = pd.concat(comment_dfs, ignore_index=True)
    print(f"YouTube comments rows loaded: {len(comments_raw)}")

    required = ["video_id", "published_at"]
    for c in required:
        if c not in comments_raw.columns:
            raise ValueError(f"YouTube comments must contain '{c}' column")

    # comment text column: 'text' or 'comment'
    if "text" in comments_raw.columns:
        comment_text = comments_raw["text"]
    elif "comment" in comments_raw.columns:
        comment_text = comments_raw["comment"]
    else:
        raise ValueError("YouTube comments must contain 'text' or 'comment' column")

    # timestamp
    ts = pd.to_datetime(comments_raw["published_at"], utc=True, errors="coerce")
    ts_str = ts.dt.strftime("%Y-%m-%d %H:%M:%S")

    comments_df = pd.DataFrame(
        {
            "video_id": comments_raw["video_id"],
            "timestamp": ts_str,
            "comment": comment_text,
        }
    )

    # load YouTube summary to get title/channel -> artist/song
    summary_blobs = _list_csv_blobs(client, YOUTUBE_SUMMARY_PREFIX)
    if not summary_blobs:
        raise RuntimeError("YouTube summary CSV is required but none found.")

    summary_dfs = [_read_csv_blob(client, b) for b in summary_blobs]
    summary_raw = pd.concat(summary_dfs, ignore_index=True)

    for c in ["video_id", "title", "channel"]:
        if c not in summary_raw.columns:
            raise ValueError(f"YouTube summary must contain '{c}' column")

    artist_norm = [
        normalize_artist_from_title_or_channel(t, ch)
        for t, ch in zip(summary_raw["title"], summary_raw["channel"])
    ]
    song_norm = summary_raw["title"].apply(normalize_song_from_title)

    summary_df = pd.DataFrame(
        {
            "video_id": summary_raw["video_id"],
            "artist": artist_norm,
            "song": song_norm,
        }
    ).drop_duplicates()

    # join comments with artist/song
    merged = comments_df.merge(summary_df, on="video_id", how="left")

    merged["artist"] = merged["artist"].astype(str).str.strip()
    merged["song"] = merged["song"].astype(str).str.strip()

    merged = merged.reset_index(drop=True)
    merged["platform"] = "youtube"
    merged["SingleID"] = (
        "youtube_" + merged["video_id"].astype(str) + "_" + merged.index.astype(str)
    )
    merged["subreddit"] = None  # YouTube has no subreddit

    final = merged[
        [
            "SingleID",
            "platform",
            "artist",
            "song",
            "timestamp",
            "comment",
            "video_id",
            "subreddit",
        ]
    ]

    print(f"YouTube final comments rows: {len(final)}")
    return final


# ---------- Reddit comments ----------

def load_reddit_comments_for_topic(client) -> pd.DataFrame:
    comment_blobs = _list_csv_blobs(client, REDDIT_COMMENTS_PREFIX)
    if not comment_blobs:
        print("No Reddit comments CSV found.")
        return pd.DataFrame()

    dfs = [_read_csv_blob(client, b) for b in comment_blobs]
    raw = pd.concat(dfs, ignore_index=True)
    print(f"Reddit comments rows loaded: {len(raw)}")

    required = ["comment_id", "body", "created_utc", "artist", "song"]
    for c in required:
        if c not in raw.columns:
            raise ValueError(f"Reddit comments must contain '{c}' column")

    artist = raw["artist"].astype(str).str.strip()
    song = raw["song"].astype(str).str.strip()

    ts = pd.to_datetime(raw["created_utc"], unit="s", utc=True, errors="coerce")
    ts_str = ts.dt.strftime("%Y-%m-%d %H:%M:%S")

    subreddit = raw["subreddit"] if "subreddit" in raw.columns else None

    rd_df = pd.DataFrame(
        {
            "comment_id": raw["comment_id"],
            "platform": "reddit",
            "artist": artist,
            "song": song,
            "timestamp": ts_str,
            "comment": raw["body"],
            "video_id": None,
            "subreddit": subreddit,
        }
    )

    rd_df["SingleID"] = "reddit_" + rd_df["comment_id"].astype(str)

    final = rd_df[
        [
            "SingleID",
            "platform",
            "artist",
            "song",
            "timestamp",
            "comment",
            "video_id",
            "subreddit",
        ]
    ]

    print(f"Reddit final comments rows: {len(final)}")
    return final


# ---------- Save + main ----------

def save_comments_table(client, df: pd.DataFrame):
    if df.empty:
        print("Comments table is empty. No output saved.")
        return

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = f"{OUTPUT_PREFIX}all_comments_topic_model_{today}.csv"

    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(path)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")

    print(f"Saved comments table to gs://{BUCKET_NAME}/{path}")


def main():
    print("=== build_comments_for_topic_modeling START ===")

    client = storage.Client()

    yt_df = load_youtube_comments_for_topic(client)
    rd_df = load_reddit_comments_for_topic(client)

    merged = pd.concat([yt_df, rd_df], ignore_index=True)
    print(f"Total merged comments rows: {len(merged)}")

    save_comments_table(client, merged)
    print("=== build_comments_for_topic_modeling DONE ===")


if __name__ == "__main__":
    main()
