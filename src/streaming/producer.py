"""Lightweight event producer for simulating real-time behavior stream.

Supports both in-memory queue and Kafka (via kafka_bridge).
"""
import json
import logging
import threading
import time
from collections import deque
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class EventQueue:
    """Thread-safe in-memory event queue (kept for backward compatibility)."""

    def __init__(self, maxsize: int = 10000):
        self._queue: deque = deque(maxlen=maxsize)
        self._lock = threading.Lock()

    def put(self, event: dict) -> None:
        with self._lock:
            self._queue.append(event)

    def get(self, timeout: float = 1.0) -> Optional[dict]:
        with self._lock:
            if self._queue:
                return self._queue.popleft()
        return None

    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def close(self) -> None:
        pass


_event_queue: Optional[EventQueue] = None
_lock = threading.Lock()


def get_event_queue(cfg=None):
    global _event_queue
    if _event_queue is not None:
        return _event_queue

    with _lock:
        if _event_queue is not None:
            return _event_queue

        try:
            from KuaiLive.kafka_bridge import create_event_queue

            _event_queue = create_event_queue(cfg)
        except ImportError:
            _event_queue = EventQueue()
            logger.info("kafka_bridge unavailable, using in-memory queue")
        except Exception:
            _event_queue = EventQueue()
            logger.exception("Kafka init failed, falling back to in-memory queue")

    return _event_queue


class BehaviorProducer:
    """Reads interaction logs and publishes events to the queue in time order."""

    def __init__(self, df: pd.DataFrame, queue=None, speedup: float = 1000.0):
        self.df = df.sort_values("timestamp")
        self.queue = queue if queue is not None else get_event_queue()
        self.speedup = speedup
        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        prev_ts = None
        for _, row in self.df.iterrows():
            if not self.running:
                break
            current_ts = row["timestamp"]
            if prev_ts is not None:
                wait = (current_ts - prev_ts) / self.speedup
                if wait > 0:
                    time.sleep(min(wait, 0.1))
            event = {
                "user_id": str(row["user_id"]),
                "item_id": str(row["item_id"]),
                "behavior_type": str(row["behavior_type"]),
                "category": str(row.get("category", "")),
                "timestamp": float(current_ts),
                "source_dataset": str(row.get("source_dataset", "")),
            }
            self.queue.put(event)
            prev_ts = current_ts

    def get_queue(self):
        return self.queue
