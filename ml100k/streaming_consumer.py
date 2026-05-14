"""Lightweight event consumer: reads from queue and updates feature store."""
import json
import threading
from typing import Optional

from .streaming_producer import EventQueue, get_event_queue


class RedisFeatureUpdater:
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._store = {}

    def _rget(self, key):
        if self.redis:
            return self.redis.get(key)
        return self._store.get(key)

    def _rset(self, key, value, ttl=3600):
        if self.redis:
            self.redis.set(key, value, ex=ttl)
        else:
            self._store[key] = value

    def _rpush(self, key, value, maxlen=50):
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

    def update_hot_score(self, item_id, increment=1.0):
        key = f"item:{item_id}:hot_score"
        current = self._rget(key)
        score = float(current) + increment if current else increment
        self._rset(key, str(score))

    def update_user_recent(self, user_id, item_id):
        self._rpush(f"user:{user_id}:recent", item_id)

    def update_negative_feedback(self, user_id, category, penalty):
        self._rset(f"neg:{user_id}:{category}", str(penalty))

    def get_negative_feedback(self, user_id):
        result = {}
        prefix = f"neg:{user_id}:"
        if self.redis:
            keys = self.redis.keys(f"neg:{user_id}:*")
            for k in keys:
                cat = k.decode().split(":")[-1] if isinstance(k, bytes) else k.split(":")[-1]
                v = self._rget(k)
                if v:
                    result[cat] = float(v)
        else:
            for k, v in self._store.items():
                if k.startswith(prefix):
                    result[k[len(prefix):]] = float(v)
        return result


class EventConsumer:
    def __init__(self, queue=None, updater=None):
        self.queue = queue or get_event_queue()
        self.updater = updater or RedisFeatureUpdater()
        self.running = False
        self._thread = None
        self.stats = {"processed": 0, "errors": 0}

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        while self.running:
            event = self.queue.get(timeout=0.5)
            if event is None:
                continue
            try:
                self._process(event)
                self.stats["processed"] += 1
            except Exception:
                self.stats["errors"] += 1

    def _process(self, event):
        user_id = event["user_id"]
        item_id = event["item_id"]
        category = event.get("category", "")
        self.updater.update_user_recent(user_id, item_id)
        self.updater.update_hot_score(item_id)
        if event.get("behavior_type") == "dislike":
            current = self.updater.get_negative_feedback(user_id)
            penalty = current.get(category, 0.0) + 0.5
            self.updater.update_negative_feedback(user_id, category, min(penalty, 1.0))
