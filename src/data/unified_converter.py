"""Convert ml-100k and KuaiLive data to the unified schema.

Unified columns:
    user_id, item_id, item_type, timestamp, behavior_type, label, category, source_dataset
"""
from typing import Optional, Tuple

import pandas as pd

from src.config.settings import config
from src.data.ml100k_loader import load_ml100k
from src.data.kualive_loader import load_real_kualive, load_kualive_parquet, save_kualive_parquet


def _filter_cold_start(df: pd.DataFrame) -> pd.DataFrame:
    """Remove users and items below min interaction threshold."""
    t = config.data.min_interactions_per_user
    user_counts = df.groupby("user_id").size()
    valid_users = user_counts[user_counts >= t].index
    item_counts = df.groupby("item_id").size()
    valid_items = item_counts[item_counts >= config.data.min_interactions_per_item].index
    return df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]


def _add_item_stats(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Add popularity, avg_rating, num_ratings per item."""
    pop = df.groupby("item_id").size().rename("popularity")
    avg = df.groupby("item_id")[label_col].mean().rename("avg_rating")
    cnt = df.groupby("item_id")[label_col].count().rename("num_ratings")
    df = df.join(pop, on="item_id")
    df = df.join(avg, on="item_id")
    df = df.join(cnt, on="item_id")
    return df


def convert_ml100k(
    ratings: Optional[pd.DataFrame] = None,
    items: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Convert ml-100k to unified format."""
    if ratings is None or items is None:
        ratings, items, _ = load_ml100k()

    df = ratings.merge(items[["item_id", "category"]], on="item_id", how="left")

    df["item_type"] = "movie"
    df["behavior_type"] = "rating"
    df["source_dataset"] = "ml-100k"
    df["label"] = (df["rating"] >= config.data.pos_threshold_ml100k).astype(int)

    df["user_id"] = "ml_" + df["user_id"].astype(str)
    df["item_id"] = "ml_" + df["item_id"].astype(str)

    df = df[["user_id", "item_id", "item_type", "timestamp", "behavior_type", "label", "category", "source_dataset"]]
    df = _filter_cold_start(df)
    df = _add_item_stats(df)
    return df


def convert_kualive(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Convert KuaiLive to unified format."""
    if df is None:
        df = load_kualive_parquet()
        if df is None:
            df = load_real_kualive()
            save_kualive_parquet(df)

    unified = df.copy()
    unified["item_type"] = "live_room"
    unified["source_dataset"] = "kualive"

    if "live_id" in unified.columns and "item_id" not in unified.columns:
        unified["item_id"] = unified["live_id"].astype(str)
    unified["item_id"] = unified["item_id"].astype(str)

    if "live_content_category" in unified.columns:
        unified["category"] = unified["live_content_category"].astype(str)
    elif "category" not in unified.columns:
        unified["category"] = "unknown"

    pos_behaviors = config.data.pos_behavior_kualive
    unified["label"] = unified["behavior_type"].isin(pos_behaviors).astype(int)

    unified["user_id"] = "kuai_" + unified["user_id"].astype(str)
    unified["item_id"] = "kuai_" + unified["item_id"].astype(str)

    unified = unified[
        ["user_id", "item_id", "item_type", "timestamp", "behavior_type", "label", "category", "source_dataset"]
    ]
    unified = _filter_cold_start(unified)
    unified = _add_item_stats(unified)
    return unified


def build_unified_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build and return (ml100k_unified, kualive_unified) DataFrames."""
    ml = convert_ml100k()
    kuai = convert_kualive()
    return ml, kuai


def combine_datasets(ml: pd.DataFrame, kuai: pd.DataFrame) -> pd.DataFrame:
    """Combine both datasets into one unified DataFrame sorted by timestamp."""
    combined = pd.concat([ml, kuai], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined
