"""Kafka bridge — drop-in replacement for the in-memory EventQueue.

Provides ``KafkaEventQueue`` with the same ``put()`` / ``get()`` / ``size()``
interface so the existing streaming producer and consumer work unchanged.

Usage::

    from ml1m.config import config
    from ml1m.kafka_bridge import create_event_queue

    queue = create_event_queue(config)  # auto-picks Kafka or in-memory
    queue.put({"user_id": "1", "item_id": "42"})
    event = queue.get()
"""
import json
import logging
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------

class _InMemoryQueue:
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


# ---------------------------------------------------------------------------
# Kafka-backed queue
# ---------------------------------------------------------------------------

class KafkaEventQueue:
    """Event queue backed by a Kafka topic.

    * ``put()``  → produces a message to the topic (sync, ack=1).
    * ``get()``  → consumes from an internal buffer filled by a background
                   consumer thread.
    * ``size()`` → returns the buffered message count.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "ml1m_events",
        consumer_group: str = "ml1m_consumer",
        buffer_size: int = 10000,
    ):
        self.topic = topic
        self._buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._running = False

        try:
            from kafka import KafkaConsumer, KafkaProducer
        except ImportError:
            raise ImportError(
                "kafka-python is required for KafkaEventQueue. "
                "Install it with: pip install kafka-python"
            )

        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            acks=1,
            retries=3,
            max_block_ms=5000,
        )

        self._ensure_topic(bootstrap_servers, topic)

        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=consumer_group,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")) if b else None,
            consumer_timeout_ms=1000,
            session_timeout_ms=30000,
            max_poll_records=500,
        )

        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()

    def _ensure_topic(self, bootstrap_servers: str, topic: str) -> None:
        try:
            from kafka.admin import KafkaAdminClient, NewTopic
        except ImportError:
            return

        try:
            admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
            existing = admin.list_topics()
            if topic not in existing:
                admin.create_topics([
                    NewTopic(name=topic, num_partitions=1, replication_factor=1)
                ])
                logger.info("Created Kafka topic '%s'", topic)
            admin.close()
        except Exception:
            pass

    def put(self, event: dict) -> None:
        try:
            self._producer.send(self.topic, event)
        except Exception:
            logger.exception("Failed to publish event to Kafka topic '%s'", self.topic)

    def _consume_loop(self) -> None:
        while self._running:
            try:
                records = self._consumer.poll(timeout_ms=500)
                for partitions, messages in records.items():
                    for msg in messages:
                        if msg.value is not None:
                            with self._lock:
                                self._buffer.append(msg.value)
            except Exception:
                logger.exception("Kafka consumer polling error")
                time.sleep(1.0)

    def get(self, timeout: float = 1.0) -> Optional[dict]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._buffer:
                    return self._buffer.popleft()
            time.sleep(0.05)
        return None

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def close(self) -> None:
        self._running = False
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        try:
            self._consumer.close(autocommit=False)
        except Exception:
            pass
        try:
            self._producer.close(timeout=0)
        except Exception:
            pass

    def flush(self, timeout: float = 5.0) -> None:
        try:
            self._producer.flush(timeout=timeout)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _check_kafka_connection(bootstrap_servers: str, timeout: float = 5.0) -> bool:
    try:
        from kafka import KafkaAdminClient
        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            request_timeout_ms=int(timeout * 1000),
        )
        admin.list_topics()
        admin.close()
        return True
    except Exception:
        return False


def create_event_queue(cfg=None):
    """Return an event queue — Kafka if configured, otherwise in-memory.

    When *cfg.kafka.enabled* is True, attempts to connect to Kafka and
    verifies the connection is healthy.  Falls back to in-memory on failure.
    """
    if cfg is None:
        from .config import config as _cfg
        cfg = _cfg

    if getattr(cfg, "kafka", None) and cfg.kafka.enabled:
        logger.info(
            "Kafka mode: brokers=%s topic=%s",
            cfg.kafka.bootstrap_servers,
            cfg.kafka.topic,
        )
        try:
            if not _check_kafka_connection(cfg.kafka.bootstrap_servers):
                raise ConnectionError("Kafka broker unreachable")

            queue = KafkaEventQueue(
                bootstrap_servers=cfg.kafka.bootstrap_servers,
                topic=cfg.kafka.topic,
                consumer_group=cfg.kafka.consumer_group,
            )
            logger.info("Kafka connection OK")
            return queue
        except ImportError:
            logger.warning("kafka-python not installed, falling back to in-memory queue")
        except Exception:
            logger.exception("Kafka init failed, falling back to in-memory queue")

    logger.info("Using in-memory event queue (Kafka disabled or unavailable)")
    return _InMemoryQueue()
