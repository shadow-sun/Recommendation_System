from __future__ import annotations

import json
import time
from pathlib import Path

from src.evaluation import ils
from src.services import build_demo_gateway


def run_closed_loop(user_id: str = "u1", k: int = 20, output: str | Path | None = None) -> dict:
    gateway = build_demo_gateway()
    start = time.perf_counter()
    first = gateway.recommend(user_id, k)
    latency_ms = (time.perf_counter() - start) * 1000
    first_items = first["items"]
    feedback_item = first_items[0]
    feedback_category = feedback_item["category"]
    before_category_count = sum(1 for item in first_items if item["category"] == feedback_category)
    gateway.feedback(user_id, feedback_item["item_id"], "not_interested", feedback_category)
    second = gateway.recommend(user_id, k)
    after_category_count = sum(1 for item in second["items"] if item["category"] == feedback_category)
    report = {
        "user_id": user_id,
        "latency_ms": latency_ms,
        "first_count": len(first_items),
        "second_count": len(second["items"]),
        "negative_category": feedback_category,
        "negative_category_before": before_category_count,
        "negative_category_after": after_category_count,
        "ils": ils([item["item_id"] for item in second["items"]], gateway.two_tower.item_embeddings),
        "metrics": gateway.store.summary(),
    }
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

