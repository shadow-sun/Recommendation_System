"""PyTorch Dataset and collate functions for Two-Tower training."""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import config


class TwoTowerDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        user_id_map: Dict[str, int],
        item_id_map: Dict[str, int],
        category_map: Dict[str, int],
        user_history: Optional[Dict[str, List[int]]] = None,
        max_seq_len: int = 50,
        use_causal_history: bool = False,
        positive_history_only: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.user_id_map = user_id_map
        self.item_id_map = item_id_map
        self.category_map = category_map
        self.user_history = user_history or {}
        self.max_seq_len = max_seq_len
        self.use_causal_history = use_causal_history
        self.user_ids = self.df["user_id"].map(user_id_map).fillna(0).astype(np.int64).values
        self.item_ids = self.df["item_id"].map(item_id_map).fillna(0).astype(np.int64).values
        self.categories = self.df["category"].map(category_map).fillna(0).astype(np.int64).values
        self.labels = self.df["label"].astype(np.float32).values
        self.row_histories = None
        if use_causal_history:
            self.row_histories = self._build_causal_row_histories(positive_history_only)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        uid = int(self.user_ids[idx])
        if self.row_histories is not None:
            history = self.row_histories[idx]
        else:
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

    def _build_causal_row_histories(self, positive_history_only: bool) -> List[List[int]]:
        row_histories: List[List[int]] = [[] for _ in range(len(self.df))]
        histories: Dict[str, List[int]] = {}
        df_sorted = self.df.reset_index().sort_values(["user_id", "timestamp", "index"])
        for _, row in df_sorted.iterrows():
            uid = str(row["user_id"])
            row_idx = int(row["index"])
            current = histories.setdefault(uid, [])
            row_histories[row_idx] = current[-self.max_seq_len:].copy()
            if positive_history_only and float(row.get("label", 1.0)) <= 0:
                continue
            iid = self.item_id_map.get(str(row["item_id"]), 0)
            if iid != 0:
                current.append(iid)
        return row_histories


def build_user_history(df: pd.DataFrame, item_id_map: Dict[str, int], positive_only: bool = True) -> Dict[str, List[int]]:
    df_sorted = df.sort_values("timestamp")
    history: Dict[str, List[int]] = {}
    for _, row in df_sorted.iterrows():
        if positive_only and float(row.get("label", 1.0)) <= 0:
            continue
        uid = str(row["user_id"])
        iid = item_id_map.get(str(row["item_id"]), 0)
        if iid == 0:
            continue
        if uid not in history:
            history[uid] = []
        history[uid].append(iid)
    return history


def collate_fn(batch: List[dict]) -> dict:
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

    return {
        "user_id": torch.tensor([b["user_id"] for b in batch], dtype=torch.long),
        "item_id": torch.tensor([b["item_id"] for b in batch], dtype=torch.long),
        "category": torch.tensor([b["category"] for b in batch], dtype=torch.long),
        "user_history": torch.tensor(
            np.array([pad_history(b["user_history"], max_hist_len) for b in batch]), dtype=torch.long
        ),
        "history_mask": torch.tensor(
            np.array([pad_mask(b["history_mask"], max_hist_len) for b in batch]), dtype=torch.float32
        ),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
    }
