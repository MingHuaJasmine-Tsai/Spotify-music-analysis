"""
Utilities for standardising daily YouTube summary and comment snapshots.

Run this after syncing raw CSVs from GCS:

    python src/processing/clean_youtube.py

The script will read everything beneath `gcs_downloads/summary` and
`gcs_downloads/comments`, perform lightweight validation / cleaning, and
persist consolidated parquet files under `data/processed/`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw_summary_dir(self) -> Path:
        return self.root / "gcs_downloads" / "summary"

    @property
    def raw_comments_dir(self) -> Path:
        return self.root / "gcs_downloads" / "comments"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def processed_summary_path(self) -> Path:
        return self.processed_dir / "youtube_summary.parquet"

    @property
    def processed_comments_path(self) -> Path:
        return self.processed_dir / "youtube_comments.parquet"


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
    "views",
    "likes",
    "comment_count",
    "pos_comments",
    "neu_comments",
    "neg_comments",
    "mean_compound",
]

SUMMARY_DEFAULT_COLUMNS = [
    "snapshot_date",
    "video_id",
    "title",
    "channel",
    "published_at",
    "views",
    "likes",
    "comment_count",
    "pos_comments",
    "neu_comments",
    "neg_comments",
    "mean_compound",
    "fetch_time",
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
    "snapshot_date",
    "artist",
    "video_id",
    "author",
    "text",
    "like_count",
    "published_at",
    "compound",
    "label",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _extract_snapshot_date(path: Path) -> pd.Timestamp:
    match = pd.Series(path.stem).str.extract(r"(\d{8})")[0]
    if pd.isna(match.iloc[0]):
        return pd.NaT
    return pd.to_datetime(match.iloc[0], format="%Y%m%d")


def _normalise_columns(df: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    renamed = {
        col: aliases[col]
        for col in df.columns
        if col in aliases
    }
    df = df.rename(columns=renamed)
    return df


def _load_summary_frames(paths: Paths) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    if not paths.raw_summary_dir.exists():
        return frames

    for csv_path in sorted(paths.raw_summary_dir.glob("*.csv")):
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
                    df["pos_comments"].fillna(0)
                    + df["neu_comments"].fillna(0)
                    + df["neg_comments"].fillna(0)
                )
            else:
                df["comment_count"] = pd.NA

        for required in SUMMARY_DEFAULT_COLUMNS:
            if required not in df.columns:
                df[required] = pd.NA

        frames.append(df[SUMMARY_DEFAULT_COLUMNS])
    return frames


def _clean_summary(paths: Paths) -> pd.DataFrame:
    frames = _load_summary_frames(paths)
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


def _load_comment_frames(paths: Paths) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    if not paths.raw_comments_dir.exists():
        return frames

    for csv_path in sorted(paths.raw_comments_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df = _normalise_columns(df, COMMENTS_COLUMN_ALIASES)
        df["snapshot_date"] = _extract_snapshot_date(csv_path)
        if "artist" not in df.columns:
            # Derive artist from filename (e.g., comments_Taylor_Swift_20251108.csv)
            parts = csv_path.stem.split("_")
            if len(parts) > 1:
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
    return frames


def _clean_comments(paths: Paths) -> pd.DataFrame:
    frames = _load_comment_frames(paths)
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


def run(paths: Paths) -> None:
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    summary = _clean_summary(paths)
    comments = _clean_comments(paths)

    summary.to_parquet(paths.processed_summary_path, index=False)
    comments.to_parquet(paths.processed_comments_path, index=False)

    print(f"Processed summary rows : {len(summary)} -> {paths.processed_summary_path}")
    print(f"Processed comments rows: {len(comments)} -> {paths.processed_comments_path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean YouTube snapshot CSVs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=_project_root(),
        help="Project root (defaults to script parents[2]).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(Paths(root=args.root.resolve()))


if __name__ == "__main__":
    main()

