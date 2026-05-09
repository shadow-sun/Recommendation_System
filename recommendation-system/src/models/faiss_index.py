from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class VectorIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.item_ids: list[str] = []
        self.matrix = np.empty((0, dim), dtype=np.float32)

    def build(self, item_embeddings: dict[str, np.ndarray]) -> "VectorIndex":
        self.item_ids = list(item_embeddings.keys())
        if self.item_ids:
            matrix = np.vstack([item_embeddings[item] for item in self.item_ids]).astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
            self.matrix = matrix / norms
        else:
            self.matrix = np.empty((0, self.dim), dtype=np.float32)
        return self

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.matrix.size == 0:
            return []
        q = query.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [(self.item_ids[int(i)], float(scores[int(i)])) for i in order]

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as fh:
            pickle.dump({"dim": self.dim, "item_ids": self.item_ids, "matrix": self.matrix}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        obj = cls(payload["dim"])
        obj.item_ids = payload["item_ids"]
        obj.matrix = payload["matrix"]
        return obj

