from __future__ import annotations

from collections import Counter

import pandas as pd


class PopularRecommender:
    def __init__(self) -> None:
        self.item_scores: Counter[str] = Counter()
        self.item_categories: dict[str, str] = {}

    def fit(self, interactions: pd.DataFrame) -> "PopularRecommender":
        positives = interactions[interactions["label"] == 1]
        self.item_scores = Counter(positives["item_id"].astype(str))
        self.item_categories = dict(zip(interactions["item_id"].astype(str), interactions["category"].astype(str)))
        return self

    def recommend(self, k: int = 20, exclude: set[str] | None = None) -> list[dict]:
        exclude = exclude or set()
        rows = []
        for item_id, score in self.item_scores.most_common():
            if item_id in exclude:
                continue
            rows.append({"item_id": item_id, "score": float(score), "category": self.item_categories.get(item_id, "unknown")})
            if len(rows) >= k:
                break
        return rows

