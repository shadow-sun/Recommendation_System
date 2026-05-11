"""PyTorch Dataset and collate functions with negative sampling."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config.settings import config


class RecommendationDataset(Dataset):
    """PyTorch Dataset for unified recommendation data.

    Returns a dict with:
        user_id_idx, item_id_idx, category_idx, user_features, item_features, label
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user_id_map: Dict[str, int],
        item_id_map: Dict[str, int],
        category_map: Dict[str, int],
        user_history: Optional[Dict[str, List[int]]] = None,
        max_seq_len: int = 50,
    ):
        self.df = df.reset_index(drop=True)
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.category_map = category_map
        self.user_history = user_history or {}
        self.max_seq_len = max_seq_len

        self.user_ids = self.df["user_id"].map(user_id_map).fillna(0).astype(np.int64).values
        self.item_ids = self.df["item_id"].map(item_id_map).fillna(0).astype(np.int64).values
        self.categories = self.df["category"].map(category_map).fillna(0).astype(np.int64).values
        self.labels = self.df["label"].astype(np.float32).values

        self._has_pop = "popularity" in self.df.columns
        self._has_avg = "avg_rating" in self.df.columns
        self._has_num = "num_ratings" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        uid = int(self.user_ids[idx])

        history = self.user_history.get(str(row["user_id"]), [])
        history = history[-self.max_seq_len:]
        history_padded = history + [0] * (self.max_seq_len - len(history))
        history_mask = [1] * len(history) + [0] * (self.max_seq_len - len(history))

        item_feats = []
        if self._has_pop:
            item_feats.append(float(row.get("popularity", 0)))
        if self._has_avg:
            item_feats.append(float(row.get("avg_rating", 0)))
        if self._has_num:
            item_feats.append(float(row.get("num_ratings", 0)))

        return {
            "user_id": uid,
            "item_id": int(self.item_ids[idx]),
            "category": int(self.categories[idx]),
            "user_history": np.array(history_padded, dtype=np.int64),
            "history_mask": np.array(history_mask, dtype=np.float32),
            "item_features": np.array(item_feats or [0.0], dtype=np.float32),
            "label": float(self.labels[idx]),
        }


class TwoTowerDataset(Dataset):
    """Dataset for Two-Tower model: returns (user_id, item_id, label) pairs
    with in-batch negatives handled by the loss function."""

    def __init__(
        self,
        df: pd.DataFrame,
        user_id_map: Dict[str, int],
        item_id_map: Dict[str, int],
        category_map: Dict[str, int],
        user_history: Optional[Dict[str, List[int]]] = None,
        max_seq_len: int = 50,
    ):
        self.df = df.reset_index(drop=True)
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.category_map = category_map
        self.user_history = user_history or {}
        self.max_seq_len = max_seq_len

        self.user_ids = self.df["user_id"].map(user_id_map).fillna(0).astype(np.int64).values
        self.item_ids = self.df["item_id"].map(item_id_map).fillna(0).astype(np.int64).values
        self.categories = self.df["category"].map(category_map).fillna(0).astype(np.int64).values
        self.labels = self.df["label"].astype(np.float32).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        uid = int(self.user_ids[idx])

        history = self.user_history.get(str(row["user_id"]), [])
        history = history[-self.max_seq_len:]
        history_padded = history + [0] * (self.max_seq_len - len(history))
        history_mask = [1] * len(history) + [0] * (self.max_seq_len - len(history))

        return {
            "user_id": uid,
            "item_id": int(self.item_ids[idx]),
            "category": int(self.categories[idx]),
            "user_history": np.array(history_padded, dtype=np.int64),
            "history_mask": np.array(history_mask, dtype=np.float32),
            "label": float(self.labels[idx]),
        }


def build_user_history(df: pd.DataFrame, item_id_map: Dict[str, int]) -> Dict[str, List[int]]:
    """Build per-user item interaction history ordered by timestamp."""
    df_sorted = df.sort_values("timestamp")
    history: Dict[str, List[int]] = {}
    for _, row in df_sorted.iterrows():
        uid = str(row["user_id"])
        iid = item_id_map.get(str(row["item_id"]), 0)
        if iid == 0:
            continue
        if uid not in history:
            history[uid] = []
        history[uid].append(iid)
    return history


def collate_fn(batch: List[dict]) -> dict:
    """Pad and stack a batch. Pads user_history to max length in batch."""
    max_hist_len = max(len(b["user_history"]) for b in batch)

    def pad_history(h, max_len):
        arr = np.array(h, dtype=np.int64)
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)))
        return arr

    def pad_mask(m, max_len):
        arr = np.array(m, dtype=np.float32)
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)))
        return arr

    def pad_features(f, target_len):
        arr = np.array(f, dtype=np.float32) if isinstance(f, (list, np.ndarray)) else np.array([f], dtype=np.float32)
        if len(arr) < target_len:
            arr = np.pad(arr, (0, target_len - len(arr)))
        return arr

    item_feats = [b.get("item_features", [0.0]) for b in batch]
    if item_feats:
        max_feat_len = max(len(f) if isinstance(f, (list, np.ndarray)) else 1 for f in item_feats)
    else:
        max_feat_len = 1

    return {
        "user_id": torch.tensor([b["user_id"] for b in batch], dtype=torch.long),
        "item_id": torch.tensor([b["item_id"] for b in batch], dtype=torch.long),
        "category": torch.tensor([b["category"] for b in batch], dtype=torch.long),
        "user_history": torch.tensor(np.array([pad_history(b["user_history"], max_hist_len) for b in batch]), dtype=torch.long),
        "history_mask": torch.tensor(np.array([pad_mask(b["history_mask"], max_hist_len) for b in batch]), dtype=torch.float32),
        "item_features": torch.tensor(np.array([pad_features(f, max_feat_len) for f in item_feats]), dtype=torch.float32),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
    }
