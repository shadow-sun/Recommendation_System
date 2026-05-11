"""Download and parse the MovieLens 100K dataset."""
import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

from src.config.settings import config


COLUMNS_USER = ["user_id", "age", "gender", "occupation", "zip_code"]
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
    """Download ml-100k.zip if not present, extract to raw/ml-100k/."""
    raw_dir = config.data.ml100k_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-100k.zip"

    if not zip_path.exists():
        print(f"Downloading ml-100k from {config.data.ml100k_url} ...")
        urllib.request.urlretrieve(config.data.ml100k_url, zip_path)

    extracted = raw_dir / "ml-100k" / "u.data"
    if not extracted.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)

    # The zip extracts to ml-100k/ml-100k/ sometimes
    inner = raw_dir / "ml-100k"
    if inner.is_dir():
        return inner
    return raw_dir


def load_ratings(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load u.data as DataFrame with columns [user_id, item_id, rating, timestamp]."""
    d = data_dir or _download_ml100k()
    path = d / "u.data" if (d / "u.data").exists() else d / "ml-100k" / "u.data"
    df = pd.read_csv(path, sep="\t", names=COLUMNS_RATING)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").astype("int64") // 10**9
    return df


def load_items(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load u.item with genres."""
    d = data_dir or _download_ml100k()
    path = d / "u.item" if (d / "u.item").exists() else d / "ml-100k" / "u.item"
    df = pd.read_csv(path, sep="|", encoding="latin-1", names=COLUMNS_ITEM)
    df["category"] = df[GENRE_COLS].idxmax(axis=1)
    return df[["item_id", "title", "category"]]


def load_users(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load u.user."""
    d = data_dir or _download_ml100k()
    path = d / "u.user" if (d / "u.user").exists() else d / "ml-100k" / "u.user"
    return pd.read_csv(path, sep="|", names=COLUMNS_USER)


def load_ml100k(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (ratings_df, items_df, users_df)."""
    d = data_dir or _download_ml100k()
    return load_ratings(d), load_items(d), load_users(d)
