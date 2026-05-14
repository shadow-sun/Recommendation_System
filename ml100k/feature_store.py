"""Feature store: Redis-backed (with in-memory fallback) feature cache."""
import json
import threading
from typing import Dict, List, Optional


class FeatureStore:
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._store = {}
        self._lock = threading.Lock()

    def _get(self, key):
        if self.redis:
            v = self.redis.get(key)
            return v.decode() if isinstance(v, bytes) else v
        with self._lock:
            return self._store.get(key)

    def _set(self, key, value, ttl=3600):
        if self.redis:
            self.redis.set(key, value, ex=ttl)
        else:
            with self._lock:
                self._store[key] = value

    def get_user_recent_items(self, user_id, n=50):
        key = f"user:{user_id}:recent"
        if self.redis:
            items = self.redis.lrange(key, 0, n - 1)
            return [i.decode() if isinstance(i, bytes) else i for i in items]
        with self._lock:
            v = self._store.get(key, [])
            return v[:n] if isinstance(v, list) else []

    def get_negative_penalties(self, user_id):
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
                        result[k[len(prefix):]] = float(v)
        return result

    def record_feedback(self, user_id, item_id, feedback_type, category):
        if feedback_type in ("not_interested", "dislike", "bad_quality"):
            penalties = self.get_negative_penalties(user_id)
            current = penalties.get(category, 0.0)
            self._set(f"neg:{user_id}:{category}", str(min(current + 0.5, 1.0)))

    def get_strategy(self, strategy_id="default"):
        v = self._get(f"strategy:{strategy_id}")
        if v:
            return json.loads(v)
        return {"mmr_lambda": 0.7, "explore_ratio": 0.1, "negative_penalty": 0.5}

    def set_strategy(self, strategy_id, cfg):
        self._set(f"strategy:{strategy_id}", json.dumps(cfg))
