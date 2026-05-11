"""Load real KuaiLive dataset from CSV files.

Schema mapping:
  live_id -> item_id, streamer_id, live_content_category -> category
  Timestamps in ms -> seconds.
  Positive behaviors: click, comment, like, gift
  Negative: negative.csv
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import config

POS_FILES = {
    "click": "click.csv",
    "comment": "comment.csv",
    "gift": "gift.csv",
    "like": "like.csv",
}


def _sample_df(df: pd.DataFrame, ratio: float, max_rows: int) -> pd.DataFrame:
    if max_rows and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    if 0.0 < ratio < 1.0:
        df = df.sample(frac=ratio, random_state=42)
    return df


def _load_csv_chunked(path: Path, ratio: float, max_rows: int) -> pd.DataFrame:
    """Load a large CSV in chunks, sampling each chunk."""
    chunk_size = 500000
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunks.append(_sample_df(chunk, ratio, 0))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _load_pos_interactions(csv_dir: Path) -> pd.DataFrame:
    """Load click, comment, gift, like CSVs and merge into one DataFrame."""
    frames = []
    for behavior, filename in POS_FILES.items():
        fp = csv_dir / filename
        if not fp.exists():
            print(f"  [WARN] {fp} not found, skipping {behavior}")
            continue
        print(f"  Loading {filename} ...")
        ratio = config.data.kualive_sample_ratio
        max_rows = config.data.kualive_max_rows_per_file
        df = _load_csv_chunked(fp, ratio, max_rows)
        df["behavior_type"] = behavior
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No KuaiLive CSV files found in {csv_dir}")
    return pd.concat(frames, ignore_index=True)


def _load_negative_interactions(csv_dir: Path) -> Optional[pd.DataFrame]:
    """Load negative.csv if available. Returns None if missing."""
    fp = csv_dir / "negative.csv"
    if not fp.exists():
        print("  [WARN] negative.csv not found")
        return None
    print(f"  Loading negative.csv ...")
    ratio = config.data.kualive_sample_ratio
    max_rows = config.data.kualive_max_rows_per_file
    df = _load_csv_chunked(fp, ratio, max_rows)
    df["behavior_type"] = "exposure"
    return df


def _load_rooms(csv_dir: Path) -> pd.DataFrame:
    """Load room.csv for category and streamer_id metadata."""
    fp = csv_dir / "room.csv"
    if not fp.exists():
        print("  [WARN] room.csv not found")
        return pd.DataFrame()
    print(f"  Loading room.csv ...")
    df = pd.read_csv(fp)
    cols = ["live_id", "streamer_id", "live_content_category"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def load_real_kualive(csv_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and merge real KuaiLive interaction data.

    Returns DataFrame with columns:
      user_id, live_id, streamer_id, behavior_type, timestamp, category
    """
    d = csv_dir or config.data.kualive_csv_dir
    print(f"Loading real KuaiLive data from: {d}")

    pos = _load_pos_interactions(d)
    neg = _load_negative_interactions(d)
    rooms = _load_rooms(d)

    if neg is not None:
        all_interactions = pd.concat([pos, neg], ignore_index=True)
    else:
        all_interactions = pos

    # Merge room metadata
    if not rooms.empty:
        all_interactions = all_interactions.merge(rooms, on="live_id", how="left")
    else:
        all_interactions["live_content_category"] = "unknown"

    all_interactions["live_content_category"] = (
        all_interactions["live_content_category"].fillna("unknown")
    )

    # Normalize timestamp: KuaiLive uses ms, convert to seconds
    all_interactions["timestamp"] = (all_interactions["timestamp"].astype("int64") // 1000).astype("int64")

    print(f"  Total interactions loaded: {len(all_interactions)}")
    print(f"  Users: {all_interactions['user_id'].nunique()}, "
          f"Lives: {all_interactions['live_id'].nunique()}")
    print(f"  Behaviors: {all_interactions['behavior_type'].value_counts().to_dict()}")

    return all_interactions


def save_kualive_parquet(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Save processed KuaiLive data as parquet for faster reload."""
    p = path or (config.data.kualive_dir / "kualive_real.parquet")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    print(f"Saved {len(df)} records to {p}")
    return p


def load_kualive_parquet(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load cached KuaiLive parquet if it exists."""
    p = path or (config.data.kualive_dir / "kualive_real.parquet")
    if p.exists():
        print(f"Loading cached KuaiLive data from {p}")
        return pd.read_parquet(p)
    return None
