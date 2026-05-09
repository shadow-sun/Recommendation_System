from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class DeepFMRanker:
    """Lightweight DeepFM-compatible ranker interface for a runnable prototype."""

    def __init__(self) -> None:
        self.global_ctr = 0.5
        self.user_ctr: dict[str, float] = {}
        self.item_ctr: dict[str, float] = {}
        self.category_ctr: dict[str, float] = {}
        self.fitted = False

    @staticmethod
    def _smoothed_ctr(df: pd.DataFrame, key: str, global_ctr: float, alpha: float = 8.0) -> dict[str, float]:
        grouped = df.groupby(key)["label"].agg(["sum", "count"])
        return {
            str(idx): float((row["sum"] + alpha * global_ctr) / (row["count"] + alpha))
            for idx, row in grouped.iterrows()
        }

    def fit(self, interactions: pd.DataFrame) -> "DeepFMRanker":
        data = interactions.copy()
        data["label"] = data["label"].astype(float)
        self.global_ctr = float(data["label"].mean()) if len(data) else 0.5
        self.user_ctr = self._smoothed_ctr(data, "user_id", self.global_ctr)
        self.item_ctr = self._smoothed_ctr(data, "item_id", self.global_ctr)
        self.category_ctr = self._smoothed_ctr(data, "category", self.global_ctr)
        self.fitted = True
        return self

    def predict(self, candidates: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            return np.full(len(candidates), 0.5, dtype=np.float32)
        scores = []
        for row in candidates.itertuples(index=False):
            user_score = self.user_ctr.get(str(row.user_id), self.global_ctr)
            item_score = self.item_ctr.get(str(row.item_id), self.global_ctr)
            category_score = self.category_ctr.get(str(row.category), self.global_ctr)
            recall_score = float(getattr(row, "recall_score", 0.0))
            recall_score = (recall_score + 1.0) / 2.0
            score = 0.25 * user_score + 0.35 * item_score + 0.25 * category_score + 0.15 * recall_score
            scores.append(min(1.0, max(0.0, score)))
        return np.asarray(scores, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as fh:
            pickle.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str | Path) -> "DeepFMRanker":
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        obj = cls()
        obj.__dict__.update(state)
        return obj
