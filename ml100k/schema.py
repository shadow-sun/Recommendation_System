"""Data schemas for ml-100k recommendation system."""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RecommendationItem:
    item_id: str
    score: float
    rank: int
    category: str
    reason: str

    def to_dict(self):
        return {"item_id": self.item_id, "score": round(self.score, 4), "rank": self.rank, "category": self.category, "reason": self.reason}


@dataclass
class RecommendResponse:
    user_id: str
    items: List[RecommendationItem]
    strategy: str
    request_id: str

    def to_dict(self):
        return {"user_id": self.user_id, "items": [item.to_dict() for item in self.items], "strategy": self.strategy, "request_id": self.request_id}


@dataclass
class FeedbackRequest:
    user_id: str
    item_id: str
    feedback_type: str
    category: str = ""
    timestamp: float = 0.0
