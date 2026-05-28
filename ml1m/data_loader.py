"""Parse and preprocess the MovieLens 1M dataset.

Returns a DataFrame ready for retrieval training with columns:
    user_id, item_id, rating, timestamp, behavior_type, label, category,
    popularity, avg_rating, num_ratings
"""
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .config import config

COLUMNS_RATING = ["user_id", "item_id", "rating", "timestamp"]
COLUMNS_MOVIE = ["item_id", "title", "genres"]
COLUMNS_USER = ["user_id", "gender", "age", "occupation", "zip_code"]


def _resolve_raw_dir(data_dir: Optional[Path] = None) -> Path:
    raw_dir = Path(data_dir or config.data.raw_dir)
    if not (raw_dir / "ratings.dat").exists():
        raise FileNotFoundError(
            f"MovieLens 1M ratings.dat not found in {raw_dir}. "
            "Place ratings.dat, movies.dat, and users.dat under the root ml-1m folder."
        )
    return raw_dir


def load_ratings(data_dir: Optional[Path] = None) -> pd.DataFrame:
    raw_dir = _resolve_raw_dir(data_dir)
    df = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        names=COLUMNS_RATING,
        engine="python",
        encoding="latin-1",
    )
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def load_items(data_dir: Optional[Path] = None) -> pd.DataFrame:
    raw_dir = _resolve_raw_dir(data_dir)
    df = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        names=COLUMNS_MOVIE,
        engine="python",
        encoding="latin-1",
    )
    df["category"] = df["genres"].fillna("").str.split("|").str[0].fillna("")
    return df[["item_id", "title", "category", "genres"]]


def load_users(data_dir: Optional[Path] = None) -> pd.DataFrame:
    raw_dir = _resolve_raw_dir(data_dir)
    return pd.read_csv(
        raw_dir / "users.dat",
        sep="::",
        names=COLUMNS_USER,
        engine="python",
        encoding="latin-1",
    )


def load_ml1m(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = _resolve_raw_dir(data_dir)
    return load_ratings(raw_dir), load_items(raw_dir), load_users(raw_dir)


def _filter_cold_start(df: pd.DataFrame) -> pd.DataFrame:
    user_counts = df.groupby("user_id").size()
    valid_users = user_counts[user_counts >= config.data.min_interactions_per_user].index
    item_counts = df.groupby("item_id").size()
    valid_items = item_counts[item_counts >= config.data.min_interactions_per_item].index
    return df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]


def load_ml1m_for_training(data_dir: Optional[Path] = None) -> pd.DataFrame:
    ratings, items, _ = load_ml1m(data_dir)
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
    print(
        f"  After cold-start filter: {len(df)} interactions, "
        f"Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}"
    )
    return df
