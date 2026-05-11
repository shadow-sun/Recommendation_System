"""Feature store: Redis-backed (with in-memory fallback) feature cache."""
import json
import threading
from typing import Dict, List, Optional


class FeatureStore:
    """Central feature store for online serving.

    Provides access to user features, recent items, item hot scores,
    embeddings, negative feedback, and strategy configuration.
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._store: dict = {}
        self._lock = threading.Lock()

    def _get(self, key: str) -> Optional[str]:
        if self.redis:
            v = self.redis.get(key)
            return v.decode() if isinstance(v, bytes) else v
        with self._lock:
            return self._store.get(key)

    def _set(self, key: str, value: str, ttl: int = 3600) -> None:
        if self.redis:
            self.redis.set(key, value, ex=ttl)
        else:
            with self._lock:
                self._store[key] = value

    def _hgetall(self, key: str) -> dict:
        if self.redis:
            d = self.redis.hgetall(key)
            return {k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v for k, v in d.items()}
        with self._lock:
            raw = self._store.get(key, "{}")
            return json.loads(raw) if isinstance(raw, str) else (raw or {})

    def get_user_recent_items(self, user_id: str, n: int = 50) -> List[str]:
        """Get recent items a user has interacted with."""
        key = f"user:{user_id}:recent"
        if self.redis:
            items = self.redis.lrange(key, 0, n - 1)
            return [i.decode() if isinstance(i, bytes) else i for i in items]
        with self._lock:
            v = self._store.get(key, [])
            return v[:n] if isinstance(v, list) else []

    def get_user_features(self, user_id: str) -> dict:
        """Get stored user features."""
        return self._hgetall(f"user:{user_id}:features")

    def get_item_hot_score(self, item_id: str) -> float:
        """Get item hot score (popularity indicator)."""
        v = self._get(f"item:{item_id}:hot_score")
        return float(v) if v else 0.0

    def get_item_embedding(self, item_id: str) -> Optional[List[float]]:
        """Get cached item embedding if available."""
        v = self._get(f"item:{item_id}:embedding")
        if v:
            return json.loads(v)
        return None

    def get_negative_penalties(self, user_id: str) -> Dict[str, float]:
        """Get negative feedback penalties for a user by category."""
        result = {}
        prefix = f"neg:{user_id}:"
        if self.redis:
            keys = self.redis.keys(f"neg:{user_id}:*")
            for k in keys:
                cat = k.decode().split(":")[-1] if isinstance(k, bytes) else k.split(":")[-1]
                v = self._get(k)
                if v:
                    result[cat] = float(v)
        else:
            with self._lock:
                for k, v in self._store.items():
                    if k.startswith(prefix):
                        cat = k[len(prefix):]
                        result[cat] = float(v)
        return result

    def record_feedback(self, user_id: str, item_id: str, feedback_type: str, category: str) -> None:
        """Record user feedback."""
        if feedback_type in ("not_interested", "dislike", "bad_quality"):
            penalties = self.get_negative_penalties(user_id)
            current = penalties.get(category, 0.0)
            new_penalty = min(current + 0.5, 1.0)
            self._set(f"neg:{user_id}:{category}", str(new_penalty))

    def get_strategy(self, strategy_id: str = "default") -> dict:
        """Get recommendation strategy configuration."""
        v = self._get(f"strategy:{strategy_id}")
        if v:
            return json.loads(v)
        return {"mmr_lambda": 0.7, "explore_ratio": 0.1, "negative_penalty": 0.5}

    def set_strategy(self, strategy_id: str, config: dict) -> None:
        """Set recommendation strategy configuration."""
        self._set(f"strategy:{strategy_id}", json.dumps(config))
