from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.services.gateway import build_demo_gateway


class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    behavior_type: str = "click"
    category: str | None = None


@lru_cache(maxsize=1)
def gateway():
    return build_demo_gateway()


app = FastAPI(title="KuaiLive ml-100k Recommendation System")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/recommend")
def recommend(user_id: str, k: int = 20) -> dict[str, Any]:
    return gateway().recommend(user_id, k)


@app.post("/feedback")
def feedback(payload: FeedbackRequest) -> dict[str, Any]:
    return gateway().feedback(payload.user_id, payload.item_id, payload.behavior_type, payload.category)


@app.get("/metrics/summary")
def metrics_summary() -> dict[str, Any]:
    return gateway().store.summary()


@app.get("/admin/strategy")
def get_strategy() -> dict[str, Any]:
    return gateway().store.strategy


@app.post("/admin/strategy")
def set_strategy(payload: dict[str, Any]) -> dict[str, Any]:
    return gateway().store.set_strategy(payload)

