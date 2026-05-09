from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureStore:
    max_history: int = 50
    users: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_items: dict[str, deque[str]] = field(default_factory=dict)
    hot_scores: Counter[str] = field(default_factory=Counter)
    item_categories: dict[str, str] = field(default_factory=dict)
    item_embeddings: dict[str, list[float]] = field(default_factory=dict)
    negative: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    strategy: dict[str, Any] = field(
        default_factory=lambda: {"name": "default", "lambda": 0.7, "explore_ratio": 0.1, "negative_penalty": 0.5}
    )
    request_count: int = 0
    feedback_count: int = 0

    def update_event(self, event: dict[str, Any]) -> None:
        user_id = str(event["user_id"])
        item_id = str(event["item_id"])
        category = str(event.get("category", "unknown"))
        behavior = str(event.get("behavior_type", "click"))
        self.item_categories[item_id] = category
        self.hot_scores[item_id] += 1 if int(event.get("label", 0)) else 0
        self.users.setdefault(user_id, {"events": 0, "positive": 0, "categories": Counter()})
        self.users[user_id]["events"] += 1
        if int(event.get("label", 0)):
            self.users[user_id]["positive"] += 1
            self.users[user_id]["categories"][category] += 1
        self.recent_items.setdefault(user_id, deque(maxlen=self.max_history)).append(item_id)
        if behavior in {"negative", "dislike", "not_interested"}:
            self.add_negative_feedback(user_id, category)

    def add_negative_feedback(self, user_id: str, category: str, penalty: float | None = None) -> None:
        self.feedback_count += 1
        value = float(penalty if penalty is not None else self.strategy.get("negative_penalty", 0.5))
        self.negative[str(user_id)][str(category)] = value

    def category_penalty(self, user_id: str, category: str) -> float:
        return float(self.negative.get(str(user_id), {}).get(str(category), 0.0))

    def set_strategy(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.strategy.update(payload)
        return self.strategy

    def summary(self) -> dict[str, Any]:
        return {
            "users": len(self.users),
            "items": len(self.item_categories),
            "requests": self.request_count,
            "feedback": self.feedback_count,
            "strategy": self.strategy,
        }


MemoryFeatureStore = FeatureStore

