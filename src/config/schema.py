"""Unified data schema for KuaiLive and ml-100k datasets."""
from dataclasses import dataclass
from typing import List, Optional

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

SPARSE_FEATURES = ["user_id", "item_id", "category", "item_type", "behavior_type"]

DENSE_FEATURES = ["popularity", "avg_rating", "num_ratings", "hour", "day_of_week", "recall_score"]

SEQUENCE_FEATURES = ["user_history_items", "user_history_categories"]

CONTEXT_FEATURES = ["hour", "day_of_week"]

ITEM_TYPES = ["movie", "live_room"]

BEHAVIOR_TYPES = ["click", "comment", "like", "gift", "rating", "exposure", "dislike"]


@dataclass
class UnifiedRecord:
    user_id: str
    item_id: str
    item_type: str
    timestamp: float
    behavior_type: str
    label: int
    category: str
    source_dataset: str
    features: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "user_id": self.user_id,
            "item_id": self.item_id,
            "item_type": self.item_type,
            "timestamp": self.timestamp,
            "behavior_type": self.behavior_type,
            "label": self.label,
            "category": self.category,
            "source_dataset": self.source_dataset,
        }
        if self.features:
            d.update(self.features)
        return d


@dataclass
class RecommendationItem:
    item_id: str
    score: float
    rank: int
    category: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "score": round(self.score, 4),
            "rank": self.rank,
            "category": self.category,
            "reason": self.reason,
        }


@dataclass
class RecommendResponse:
    user_id: str
    items: List[RecommendationItem]
    strategy: str
    request_id: str

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "items": [item.to_dict() for item in self.items],
            "strategy": self.strategy,
            "request_id": self.request_id,
        }


@dataclass
class FeedbackRequest:
    user_id: str
    item_id: str
    feedback_type: str
    category: str = ""
    timestamp: float = 0.0
