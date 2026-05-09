from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


UNIFIED_COLUMNS = [
    "user_id",
    "item_id",
    "item_type",
    "timestamp",
    "behavior_type",
    "label",
    "category",
    "source_dataset",
]


@dataclass(frozen=True)
class UnifiedInteraction:
    user_id: str
    item_id: str
    item_type: str
    timestamp: int
    behavior_type: str
    label: int
    category: str
    source_dataset: str


def normalize_kuailive(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    lower_map = {c.lower(): c for c in data.columns}

    def col(*names: str, default: str | None = None) -> str | None:
        for name in names:
            if name in data.columns:
                return name
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return default

    user_col = col("user_id", "user", "uid")
    item_col = col("room_id", "streamer_id", "item_id", "live_id", "author_id")
    ts_col = col("timestamp", "time", "click_timestamp", "event_timestamp")
    behavior_col = col("behavior_type", "behavior", "action")
    category_col = col("live_content_category", "category", "live_type", default=None)

    missing = [name for name, value in {"user_id": user_col, "item_id": item_col, "timestamp": ts_col}.items() if value is None]
    if missing:
        raise ValueError(f"KuaiLive data is missing required columns: {missing}")

    behavior = data[behavior_col].astype(str).str.lower() if behavior_col else "click"
    label = behavior.isin(["click", "comment", "like", "gift", "positive", "1"]).astype(int)

    out = pd.DataFrame(
        {
            "user_id": data[user_col].astype(str),
            "item_id": data[item_col].astype(str),
            "item_type": "live_room",
            "timestamp": pd.to_numeric(data[ts_col], errors="coerce").fillna(0).astype("int64"),
            "behavior_type": behavior,
            "label": label,
            "category": data[category_col].astype(str) if category_col else "unknown",
            "source_dataset": "kuailive",
        }
    )
    return out[UNIFIED_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def normalize_ml100k(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if list(data.columns[:4]) != ["user_id", "item_id", "rating", "timestamp"]:
        data.columns = ["user_id", "item_id", "rating", "timestamp"] + list(data.columns[4:])
    ratings = pd.to_numeric(data["rating"], errors="coerce").fillna(0)
    out = pd.DataFrame(
        {
            "user_id": data["user_id"].astype(str),
            "item_id": data["item_id"].astype(str),
            "item_type": "movie",
            "timestamp": pd.to_numeric(data["timestamp"], errors="coerce").fillna(0).astype("int64"),
            "behavior_type": "rating",
            "label": (ratings >= 4).astype(int),
            "category": data["category"].astype(str) if "category" in data.columns else "movie",
            "source_dataset": "ml-100k",
        }
    )
    return out[UNIFIED_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def load_interactions(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in [".csv", ".txt"]:
            sep = "\t" if path.suffix.lower() == ".txt" else ","
            df = pd.read_csv(path, sep=sep)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported data file: {path}")

        name = path.name.lower()
        if "ml-100k" in name or name in {"u.data", "ratings.csv"}:
            frames.append(normalize_ml100k(df))
        else:
            frames.append(normalize_kuailive(df))

    if not frames:
        raise ValueError("No interaction files were provided.")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

