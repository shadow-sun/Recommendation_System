"""Lightweight event consumer: reads from queue and updates Redis feature store.

Works with both in-memory EventQueue and Kafka-backed queues.
"""
import json
import logging
import threading
import time
from typing import Optional

from src.streaming.producer import get_event_queue

logger = logging.getLogger(__name__)


class RedisFeatureUpdater:
    """Updates Redis with real-time user/item features from behavior events.

    Uses a Python dict as fallback when Redis is unavailable.
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._store: dict = {}  # In-memory fallback store

    def _rget(self, key: str) -> Optional[str]:
        if self.redis:
            return self.redis.get(key)
        return self._store.get(key)

    def _rset(self, key: str, value: str, ttl: int = 3600) -> None:
        if self.redis:
            self.redis.set(key, value, ex=ttl)
        else:
            self._store[key] = value

    def _rhset(self, key: str, mapping: dict) -> None:
        if self.redis:
            self.redis.hset(key, mapping=mapping)
        else:
            existing = self._store.get(key, "{}")
            try:
                d = json.loads(existing) if isinstance(existing, str) else (existing or {})
            except json.JSONDecodeError:
                d = {}
            d.update(mapping)
            self._store[key] = json.dumps(d)

    def _rpush(self, key: str, value: str, maxlen: int = 50) -> None:
        if self.redis:
            self.redis.lpush(key, value)
            self.redis.ltrim(key, 0, maxlen - 1)
        else:
            lst = self._store.get(key, [])
            if not isinstance(lst, list):
                lst = []
            lst.insert(0, value)
            lst = lst[:maxlen]
            self._store[key] = lst

    def update_hot_score(self, item_id: str, increment: float = 1.0) -> None:
        key = f"item:{item_id}:hot_score"
        current = self._rget(key)
        score = float(current) + increment if current else increment
        self._rset(key, str(score))

    def update_user_recent(self, user_id: str, item_id: str) -> None:
        key = f"user:{user_id}:recent"
        self._rpush(key, item_id)

    def update_user_features(self, user_id: str, features: dict) -> None:
        key = f"user:{user_id}:features"
        self._rhset(key, features)

    def update_negative_feedback(self, user_id: str, category: str, penalty: float) -> None:
        key = f"neg:{user_id}:{category}"
        self._rset(key, str(penalty))

    def get_negative_feedback(self, user_id: str) -> dict:
        """Get all negative feedback penalties for a user."""
        result = {}
        if self.redis:
            keys = self.redis.keys(f"neg:{user_id}:*")
            for k in keys:
                cat = k.decode().split(":")[-1] if isinstance(k, bytes) else k.split(":")[-1]
                v = self._rget(k)
                if v:
                    result[cat] = float(v)
        else:
            prefix = f"neg:{user_id}:"
            for k, v in self._store.items():
                if k.startswith(prefix):
                    cat = k[len(prefix):]
                    result[cat] = float(v)
        return result

    def get_user_recent(self, user_id: str, n: int = 50) -> list:
        key = f"user:{user_id}:recent"
        if self.redis:
            items = self.redis.lrange(key, 0, n - 1)
            return [i.decode() if isinstance(i, bytes) else i for i in items]
        lst = self._store.get(key, [])
        return lst[:n] if isinstance(lst, list) else []


class EventConsumer:
    """Consumes behavior events from the queue (in-memory or Kafka)."""

    def __init__(
        self,
        queue=None,
        updater: Optional[RedisFeatureUpdater] = None,
    ):
        self.queue = queue if queue is not None else get_event_queue()
        self.updater = updater or RedisFeatureUpdater()
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.stats = {"processed": 0, "errors": 0}

    def start(self) -> None:
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while self.running:
            event = self.queue.get(timeout=0.5)
            if event is None:
                continue

            try:
                self._process(event)
                self.stats["processed"] += 1
            except Exception:
                self.stats["errors"] += 1

    def _process(self, event: dict) -> None:
        user_id = event["user_id"]
        item_id = event["item_id"]
        category = event.get("category", "")

        self.updater.update_user_recent(user_id, item_id)
        self.updater.update_hot_score(item_id)

        if event.get("behavior_type") == "dislike":
            current = self.updater.get_negative_feedback(user_id)
            penalty = current.get(category, 0.0) + 0.5
            self.updater.update_negative_feedback(user_id, category, min(penalty, 1.0))
