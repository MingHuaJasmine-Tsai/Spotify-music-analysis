"""
Utilities for standardising daily Reddit summary and comment snapshots.

Run this after同步最新的 Reddit CSV：

    python src/processing/clean_reddit.py

脚本会读取 `gcs_downloads/reddit/summary` 与 `gcs_downloads/reddit/comments`
下的所有 CSV，做基础清洗后把合并结果写到 `data/processed/`。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw_summary_dir(self) -> Path:
        return self.root / "gcs_downloads" / "reddit" / "summary"

    @property
    def raw_comments_dir(self) -> Path:
        return self.root / "gcs_downloads" / "reddit" / "comments"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def processed_summary_path(self) -> Path:
        return self.processed_dir / "reddit_summary.parquet"

    @property
    def processed_comments_path(self) -> Path:
        return self.processed_dir / "reddit_comments.parquet"


SUMMARY_DEFAULT_COLUMNS = [
    "snapshot_date",
    "submission_id",
    "subreddit",
    "artist",
    "song",
    "title",
    "author",
    "score",
    "num_comments",
    "pos_comments",
    "neu_comments",
    "neg_comments",
    "mean_compound",
    "created_utc",
    "fetch_time",
    "permalink",
]

SUMMARY_NUMERIC_COLUMNS = [
    "score",
    "num_comments",
    "pos_comments",
    "neu_comments",
    "neg_comments",
    "mean_compound",
]

COMMENTS_DEFAULT_COLUMNS = [
    "snapshot_date",
    "submission_id",
    "comment_id",
    "artist",
    "song",
    "author",
    "body",
    "score",
    "created_utc",
    "compound",
    "label",
    "permalink",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_datetime(series: pd.Series, utc: bool = True) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=utc).dt.tz_convert(None)


def _load_summary_frames(paths: Paths) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    if not paths.raw_summary_dir.exists():
        return frames

    for csv_path in sorted(paths.raw_summary_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # 统一艺人/歌曲字段
        artist_cols = [col for col in df.columns if col.startswith("artist")]
        song_cols = [col for col in df.columns if col.startswith("song")]
        if artist_cols:
            df["artist"] = (
                df[artist_cols]
                .replace({np.nan: None, "": None})
                .bfill(axis=1)
                .iloc[:, 0]
            )
        if song_cols:
            df["song"] = (
                df[song_cols]
                .replace({np.nan: None, "": None})
                .bfill(axis=1)
                .iloc[:, 0]
            )

        df["fetch_time"] = _to_datetime(df.get("fetch_time"))
        df["created_utc"] = _to_datetime(df.get("created_utc"))
        df["snapshot_date"] = df["fetch_time"].dt.normalize()

        for col in SUMMARY_NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

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
        .drop_duplicates(subset=["snapshot_date", "submission_id"], keep="last")
        .sort_values(["snapshot_date", "artist", "song", "subreddit"])
    )

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

        df["created_utc"] = _to_datetime(df.get("created_utc"))
        df["snapshot_date"] = df["created_utc"].dt.normalize()
        df["score"] = pd.to_numeric(df.get("score"), errors="coerce").fillna(0).astype(int)
        df["compound"] = pd.to_numeric(df.get("compound"), errors="coerce")

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
            subset=["snapshot_date", "submission_id", "comment_id", "author", "created_utc"],
            keep="last",
        )
        .sort_values(["snapshot_date", "artist", "created_utc"])
    )
    comments.reset_index(drop=True, inplace=True)
    return comments


def run(paths: Paths) -> None:
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    summary = _clean_summary(paths)
    comments = _clean_comments(paths)

    summary.to_parquet(paths.processed_summary_path, index=False)
    comments.to_parquet(paths.processed_comments_path, index=False)

    print(f"Processed Reddit summary rows : {len(summary)} -> {paths.processed_summary_path}")
    print(f"Processed Reddit comments rows: {len(comments)} -> {paths.processed_comments_path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean Reddit snapshot CSVs.")
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

