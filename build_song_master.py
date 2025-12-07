import io
import re
from datetime import datetime, timezone

import pandas as pd
from google.cloud import storage


BUCKET_NAME = "apidatabase"

YOUTUBE_SUMMARY_PREFIX = "youtube/summary/"
REDDIT_COMMENTS_PREFIX = "reddit/comments/"
OUTPUT_PREFIX = "cleaned/"


# ---------- Helpers ----------

def _list_csv_blobs(client, prefix: str):
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(client.list_blobs(bucket_or_name=bucket, prefix=prefix))
    return [b for b in blobs if b.name.endswith(".csv")]


def _read_csv_blob(client, blob):
    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data))


def normalize_song_from_title(title: str) -> str:
    """Extract a clean song name from a YouTube title."""
    if not isinstance(title, str):
        return None
    # Split "Artist - Song (Official ...)"
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


# ---------- YouTube summary → daily per song ----------

def load_youtube_daily_summary(client) -> pd.DataFrame:
    blobs = _list_csv_blobs(client, YOUTUBE_SUMMARY_PREFIX)
    if not blobs:
        print("No YouTube summary CSV found.")
        return pd.DataFrame()

    dfs = [_read_csv_blob(client, b) for b in blobs]
    raw = pd.concat(dfs, ignore_index=True)

    required = ["fetch_time", "title", "channel", "views", "likes", "comment_count", "pos_comments"]
    for c in required:
        if c not in raw.columns:
            raise ValueError(f"YouTube summary must contain '{c}' column")

    # date from fetch_time
    fetch_ts = pd.to_datetime(raw["fetch_time"], utc=True, errors="coerce")
    date_str = fetch_ts.dt.strftime("%Y-%m-%d")

    artists = [
        normalize_artist_from_title_or_channel(t, ch)
        for t, ch in zip(raw["title"], raw["channel"])
    ]
    songs = raw["title"].apply(normalize_song_from_title)

    df = pd.DataFrame(
        {
            "date": date_str,
            "artist": artists,
            "song": songs,
            "views": raw["views"],
            "likes": raw["likes"],
            "comment_count": raw["comment_count"],
            "pos_comments": raw["pos_comments"],
        }
    )

    # aggregate per date + artist + song
    grouped = df.groupby(["date", "artist", "song"], as_index=False).agg(
        {
            "views": "max",           # or "last", but max is usually fine
            "likes": "max",
            "comment_count": "max",
            "pos_comments": "max",
        }
    )

    # positive ratio
    grouped["youtube_pos_ratio"] = grouped["pos_comments"] / grouped["comment_count"].replace(0, pd.NA)

    grouped = grouped.rename(
        columns={
            "views": "youtube_views",
            "likes": "youtube_likes",
            "comment_count": "youtube_comment_count",
        }
    )

    return grouped


# ---------- Reddit comments → daily counts per song ----------

def load_reddit_daily_counts(client) -> pd.DataFrame:
    blobs = _list_csv_blobs(client, REDDIT_COMMENTS_PREFIX)
    if not blobs:
        print("No Reddit comments CSV found.")
        return pd.DataFrame(columns=["date", "artist", "song", "reddit_comment_count"])

    dfs = [_read_csv_blob(client, b) for b in blobs]
    raw = pd.concat(dfs, ignore_index=True)

    required = ["artist", "song", "created_utc"]
    for c in required:
        if c not in raw.columns:
            raise ValueError(f"Reddit comments must contain '{c}' column")

    artist = raw["artist"].astype(str).str.strip()
    song = raw["song"].astype(str).str.strip()

    ts = pd.to_datetime(raw["created_utc"], unit="s", utc=True, errors="coerce")
    date_str = ts.dt.strftime("%Y-%m-%d")

    df = pd.DataFrame(
        {
            "date": date_str,
            "artist": artist,
            "song": song,
        }
    )

    grouped = df.groupby(["date", "artist", "song"], as_index=False).size()
    grouped = grouped.rename(columns={"size": "reddit_comment_count"})

    return grouped


# ---------- Save + main ----------

def save_daily_summary(client, df: pd.DataFrame):
    if df.empty:
        print("Daily song summary is empty. No output.")
        return

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = f"{OUTPUT_PREFIX}daily_song_summary_{today}.csv"

    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(out_path)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")

    print(f"Saved daily song summary to gs://{BUCKET_NAME}/{out_path}")


def main():
    print("=== build_daily_song_summary START ===")
    client = storage.Client()

    yt_daily = load_youtube_daily_summary(client)
    print(f"YouTube daily rows: {len(yt_daily)}")

    rd_daily = load_reddit_daily_counts(client)
    print(f"Reddit daily rows: {len(rd_daily)}")

    # full outer join on (date, artist, song)
    merged = yt_daily.merge(
        rd_daily, on=["date", "artist", "song"], how="outer"
    ).fillna({"reddit_comment_count": 0})

    print(f"Final daily summary rows: {len(merged)}")

    save_daily_summary(client, merged)
    print("=== build_daily_song_summary DONE ===")


if __name__ == "__main__":
    main()
