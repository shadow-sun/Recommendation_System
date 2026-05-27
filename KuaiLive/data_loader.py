"""Load and preprocess KuaiLive dataset from CSV files.

Returns a DataFrame ready for training with columns:
    user_id, item_id, timestamp, behavior_type, label, category,
    popularity, avg_rating, num_ratings
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import config

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
    chunk_size = 500000
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunks.append(_sample_df(chunk, ratio, 0))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _load_pos_interactions(csv_dir: Path) -> pd.DataFrame:
    frames = []
    for behavior, filename in POS_FILES.items():
        fp = csv_dir / filename
        if not fp.exists():
            print(f"  [WARN] {fp} not found, skipping {behavior}")
            continue
        print(f"  Loading {filename} ...")
        ratio = config.data.sample_ratio
        max_rows = config.data.max_rows_per_file
        df = _load_csv_chunked(fp, ratio, max_rows)
        df["behavior_type"] = behavior
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No KuaiLive CSV files found in {csv_dir}")
    return pd.concat(frames, ignore_index=True)


def _load_negative_interactions(csv_dir: Path) -> Optional[pd.DataFrame]:
    fp = csv_dir / "negative.csv"
    if not fp.exists():
        print("  [WARN] negative.csv not found")
        return None
    print("  Loading negative.csv ...")
    ratio = config.data.sample_ratio
    max_rows = config.data.max_rows_per_file
    df = _load_csv_chunked(fp, ratio, max_rows)
    df["behavior_type"] = "exposure"
    return df


def _load_rooms(csv_dir: Path) -> pd.DataFrame:
    fp = csv_dir / "room.csv"
    if not fp.exists():
        print("  [WARN] room.csv not found")
        return pd.DataFrame()
    print("  Loading room.csv ...")
    df = pd.read_csv(fp)
    cols = ["live_id", "streamer_id", "live_content_category"]
    available = [c for c in cols if c in df.columns]
    return df[available]


def load_raw_kuailive(csv_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and merge raw KuaiLive interaction data."""
    d = csv_dir or config.data.csv_dir
    print(f"Loading raw KuaiLive data from: {d}")

    pos = _load_pos_interactions(d)
    neg = _load_negative_interactions(d)
    rooms = _load_rooms(d)

    if neg is not None:
        all_interactions = pd.concat([pos, neg], ignore_index=True)
    else:
        all_interactions = pos

    if not rooms.empty:
        all_interactions = all_interactions.merge(rooms, on="live_id", how="left")
    else:
        all_interactions["live_content_category"] = "unknown"

    all_interactions["live_content_category"] = (
        all_interactions["live_content_category"].fillna("unknown")
    )

    all_interactions["timestamp"] = (all_interactions["timestamp"].astype("int64") // 1000).astype("int64")

    print(f"  Total interactions loaded: {len(all_interactions)}")
    print(f"  Users: {all_interactions['user_id'].nunique()}, "
          f"Lives: {all_interactions['live_id'].nunique()}")
    print(f"  Behaviors: {all_interactions['behavior_type'].value_counts().to_dict()}")

    return all_interactions


def save_kuailive_parquet(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    p = path or (config.data.raw_dir / "kualive_real.parquet")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    print(f"Saved {len(df)} records to {p}")
    return p


def load_kuailive_parquet(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    p = path or (config.data.raw_dir / "kualive_real.parquet")
    if p.exists():
        print(f"Loading cached KuaiLive data from {p}")
        return pd.read_parquet(p)
    return None


def _filter_cold_start(df: pd.DataFrame) -> pd.DataFrame:
    t = config.data.min_interactions_per_user
    user_counts = df.groupby("user_id").size()
    valid_users = user_counts[user_counts >= t].index
    item_counts = df.groupby("item_id").size()
    valid_items = item_counts[item_counts >= config.data.min_interactions_per_item].index
    return df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]


def load_kuailive() -> pd.DataFrame:
    """Load KuaiLive data ready for training.

    Returns DataFrame with columns:
        user_id, item_id, timestamp, behavior_type, label, category,
        popularity, avg_rating, num_ratings
    """
    df = load_kuailive_parquet()
    if df is None:
        df = load_raw_kuailive()
        save_kuailive_parquet(df)

    df["item_id"] = df["live_id"].astype(str) if "live_id" in df.columns else df["item_id"].astype(str)
    df["user_id"] = df["user_id"].astype(str)

    if "live_content_category" in df.columns:
        df["category"] = df["live_content_category"].astype(str)
    elif "category" not in df.columns:
        df["category"] = "unknown"

    pos_behaviors = config.data.pos_behaviors
    df["label"] = df["behavior_type"].isin(pos_behaviors).astype(int)

    df = df[["user_id", "item_id", "timestamp", "behavior_type", "label", "category"]]

    pop = df.groupby("item_id").size().to_dict()
    avg = df.groupby("item_id")["label"].mean().to_dict()
    cnt = df.groupby("item_id")["label"].count().to_dict()
    df["popularity"] = df["item_id"].map(pop)
    df["avg_rating"] = df["item_id"].map(avg)
    df["num_ratings"] = df["item_id"].map(cnt)

    df = _filter_cold_start(df)
    print(f"  After cold-start filter: {len(df)} interactions, "
          f"Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")
    return df
