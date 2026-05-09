from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class TwoTowerRecommender:
    def __init__(self, embedding_dim: int = 64, seed: int = 42) -> None:
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.user_embeddings: dict[str, np.ndarray] = {}
        self.item_embeddings: dict[str, np.ndarray] = {}
        self.item_categories: dict[str, str] = {}
        self.global_mean = np.zeros(embedding_dim, dtype=np.float32)

    def fit(self, interactions: pd.DataFrame) -> "TwoTowerRecommender":
        rng = np.random.default_rng(self.seed)
        users = sorted(interactions["user_id"].astype(str).unique())
        items = sorted(interactions["item_id"].astype(str).unique())
        self.item_categories = dict(zip(interactions["item_id"].astype(str), interactions["category"].astype(str)))
        for item in items:
            self.item_embeddings[item] = rng.normal(0, 0.1, self.embedding_dim).astype(np.float32)
        positives = interactions[interactions["label"] == 1]
        if not positives.empty:
            item_counts = positives["item_id"].astype(str).value_counts()
            for item, count in item_counts.items():
                self.item_embeddings[item] += np.float32(np.log1p(count) * 0.01)
            self.global_mean = np.mean([self.item_embeddings[i] for i in item_counts.index], axis=0).astype(np.float32)
        for user in users:
            user_pos = positives[positives["user_id"].astype(str) == user]["item_id"].astype(str).tolist()
            if user_pos:
                self.user_embeddings[user] = np.mean([self.item_embeddings[i] for i in user_pos], axis=0).astype(np.float32)
            else:
                self.user_embeddings[user] = self.global_mean.copy()
        return self

    def user_vector(self, user_id: str) -> np.ndarray:
        return self.user_embeddings.get(str(user_id), self.global_mean).astype(np.float32)

    def item_vector(self, item_id: str) -> np.ndarray:
        return self.item_embeddings[str(item_id)].astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as fh:
            pickle.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str | Path) -> "TwoTowerRecommender":
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        obj = cls(state["embedding_dim"], state["seed"])
        obj.__dict__.update(state)
        return obj

