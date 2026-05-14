"""Lightweight event producer for simulating real-time behavior stream."""
import time
import threading
from collections import deque
from typing import Optional

import pandas as pd


class EventQueue:
    def __init__(self, maxsize=10000):
        self._queue = deque(maxlen=maxsize)
        self._lock = threading.Lock()

    def put(self, event):
        with self._lock:
            self._queue.append(event)

    def get(self, timeout=1.0):
        with self._lock:
            if self._queue:
                return self._queue.popleft()
        return None

    def size(self):
        with self._lock:
            return len(self._queue)


_event_queue = EventQueue()


class BehaviorProducer:
    def __init__(self, df, queue=None, speedup=1000.0):
        self.df = df.sort_values("timestamp")
        self.queue = queue or _event_queue
        self.speedup = speedup
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
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
            }
            self.queue.put(event)
            prev_ts = current_ts


def get_event_queue():
    return _event_queue
