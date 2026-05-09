from __future__ import annotations

from collections.abc import Iterable

from src.features import FeatureStore


def process_events(events: Iterable[dict], store: FeatureStore) -> int:
    count = 0
    for event in events:
        store.update_event(event)
        count += 1
    return count

