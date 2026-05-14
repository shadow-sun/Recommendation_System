"""Download, parse and preprocess the MovieLens 100K dataset.

Returns a DataFrame ready for training with columns:
    user_id, item_id, rating, timestamp, behavior_type, label, category,
    popularity, avg_rating, num_ratings
"""
import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np

from .config import config

COLUMNS_ITEM = [
    "item_id", "title", "release_date", "video_release_date",
    "imdb_url", "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
GENRE_COLS = COLUMNS_ITEM[5:]
COLUMNS_RATING = ["user_id", "item_id", "rating", "timestamp"]


def _download_ml100k() -> Path:
    raw_dir = config.data.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-100k.zip"

    if not zip_path.exists():
        print(f"Downloading ml-100k from {config.data.ml100k_url} ...")
        urllib.request.urlretrieve(config.data.ml100k_url, zip_path)

    extracted = raw_dir / "ml-100k" / "u.data"
    if not extracted.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    inner = raw_dir / "ml-100k"
    if inner.is_dir():
        return inner
    return raw_dir


def load_ratings(data_dir: Optional[Path] = None) -> pd.DataFrame:
    d = data_dir or _download_ml100k()
    path = d / "u.data" if (d / "u.data").exists() else d / "ml-100k" / "u.data"
    df = pd.read_csv(path, sep="\t", names=COLUMNS_RATING)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").astype("int64") // 10**9
    return df


def load_items(data_dir: Optional[Path] = None) -> pd.DataFrame:
    d = data_dir or _download_ml100k()
    path = d / "u.item" if (d / "u.item").exists() else d / "ml-100k" / "u.item"
    df = pd.read_csv(path, sep="|", encoding="latin-1", names=COLUMNS_ITEM)
    df["category"] = df[GENRE_COLS].idxmax(axis=1)
    return df[["item_id", "title", "category"]]


def load_ml100k(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ratings_df, items_df)."""
    d = data_dir or _download_ml100k()
    return load_ratings(d), load_items(d)


def _filter_cold_start(df: pd.DataFrame) -> pd.DataFrame:
    t = config.data.min_interactions_per_user
    user_counts = df.groupby("user_id").size()
    valid_users = user_counts[user_counts >= t].index
    item_counts = df.groupby("item_id").size()
    valid_items = item_counts[item_counts >= config.data.min_interactions_per_item].index
    return df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]


def load_ml100k_for_training(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load ml-100k data ready for training.

    Returns DataFrame with columns:
        user_id, item_id, rating, timestamp, behavior_type, label, category,
        popularity, avg_rating, num_ratings
    """
    ratings, items = load_ml100k(data_dir)

    df = ratings.merge(items[["item_id", "category"]], on="item_id", how="left")
    df["behavior_type"] = "rating"
    df["label"] = (df["rating"] >= config.data.pos_threshold).astype(int)

    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    df = df[["user_id", "item_id", "rating", "timestamp", "behavior_type", "label", "category"]]

    pop = df.groupby("item_id").size().rename("popularity")
    avg = df.groupby("item_id")["label"].mean().rename("avg_rating")
    cnt = df.groupby("item_id")["label"].count().rename("num_ratings")
    df = df.join(pop, on="item_id")
    df = df.join(avg, on="item_id")
    df = df.join(cnt, on="item_id")

    df = _filter_cold_start(df)
    print(f"  After cold-start filter: {len(df)} interactions, "
          f"Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")
    return df
