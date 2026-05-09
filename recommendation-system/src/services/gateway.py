from __future__ import annotations

from uuid import uuid4

import pandas as pd

from src.config import get_settings
from src.data import create_sample_dataset
from src.features import FeatureStore
from src.models import DeepFMRanker, PopularRecommender, TwoTowerRecommender, VectorIndex
from src.rerank import mmr_rerank
from src.streaming import process_events


class RecommendationGateway:
    def __init__(
        self,
        store: FeatureStore,
        popular: PopularRecommender,
        two_tower: TwoTowerRecommender,
        ranker: DeepFMRanker,
        index: VectorIndex,
        interactions: pd.DataFrame,
    ) -> None:
        self.store = store
        self.popular = popular
        self.two_tower = two_tower
        self.ranker = ranker
        self.index = index
        self.interactions = interactions
        self.item_meta = interactions.drop_duplicates("item_id").set_index("item_id")[["category", "item_type", "source_dataset"]]

    def recommend(self, user_id: str, k: int = 20) -> dict:
        self.store.request_count += 1
        settings = get_settings()
        k = max(1, min(int(k), 100))
        recent = set(self.store.recent_items.get(str(user_id), []))
        if str(user_id) in self.two_tower.user_embeddings:
            recall = self.index.search(self.two_tower.user_vector(user_id), settings.recall_candidates)
            candidates = [
                {
                    "user_id": str(user_id),
                    "item_id": item_id,
                    "score": score,
                    "recall_score": score,
                    "category": self.two_tower.item_categories.get(item_id, "unknown"),
                    "item_type": "live_room",
                    "source_dataset": "sample",
                }
                for item_id, score in recall
                if item_id not in recent
            ]
        else:
            candidates = [
                {"user_id": str(user_id), **row, "recall_score": row["score"], "item_type": "live_room", "source_dataset": "sample"}
                for row in self.popular.recommend(settings.recall_candidates, exclude=recent)
            ]
        if not candidates:
            candidates = [{"user_id": str(user_id), **row, "recall_score": row["score"], "item_type": "live_room", "source_dataset": "sample"} for row in self.popular.recommend(k)]
        candidate_df = pd.DataFrame(candidates)
        candidate_df["label"] = 0
        scores = self.ranker.predict(candidate_df)
        ranked = []
        for row, score in zip(candidates, scores):
            ranked.append({**row, "score": float(score), "reason": "two_tower+deepfm"})
        ranked.sort(key=lambda x: x["score"], reverse=True)
        penalties = self.store.negative.get(str(user_id), {})
        strategy = self.store.strategy
        final = mmr_rerank(
            ranked[: settings.recall_candidates],
            self.two_tower.item_embeddings,
            k,
            lambda_=float(strategy.get("lambda", settings.mmr_lambda)),
            negative_penalties=penalties,
        )
        return {"user_id": str(user_id), "items": final, "strategy": strategy.get("name", "default"), "request_id": str(uuid4())}

    def feedback(self, user_id: str, item_id: str, behavior_type: str = "click", category: str | None = None) -> dict:
        category = category or self.two_tower.item_categories.get(str(item_id), "unknown")
        label = 0 if behavior_type in {"negative", "dislike", "not_interested"} else 1
        event = {"user_id": str(user_id), "item_id": str(item_id), "behavior_type": behavior_type, "label": label, "category": category}
        self.store.update_event(event)
        return {"ok": True, "event": event}


def build_demo_gateway() -> RecommendationGateway:
    settings = get_settings()
    sample_path = create_sample_dataset(settings.processed_data_dir)
    interactions = pd.read_csv(sample_path)
    store = FeatureStore(max_history=settings.max_history)
    process_events(interactions.head(500).to_dict("records"), store)
    popular = PopularRecommender().fit(interactions)
    two_tower = TwoTowerRecommender(settings.embedding_dim).fit(interactions)
    index = VectorIndex(settings.embedding_dim).build(two_tower.item_embeddings)
    ranker = DeepFMRanker().fit(interactions)
    return RecommendationGateway(store, popular, two_tower, ranker, index, interactions)

