"""Tests for recommendation API endpoints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.services.gateway import app, set_services
from src.services.feature_store import FeatureStore
from src.services.recall import RecallService
from src.services.ranker import RankService


def create_test_services():
    """Create test service instances with some popular items."""
    store = FeatureStore()
    popular = [f"item_{i}" for i in range(100)]
    recall = RecallService(
        feature_store=store,
        popular_items=popular,
    )
    rank = RankService(feature_store=store)
    return recall, rank, store


recall_svc, rank_svc, feat_store = create_test_services()
set_services(recall_svc, rank_svc, feat_store)

client = TestClient(app)


def test_health():
    """Test health endpoint."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_recommend_new_user():
    """Test recommendation for a new (cold-start) user."""
    resp = client.get("/recommend?user_id=new_user_999&k=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "new_user_999"
    assert len(data["items"]) == 10
    assert "request_id" in data
    assert "latency_ms" in data


def test_recommend_with_k():
    """Test different k values."""
    for k in [5, 10, 20]:
        resp = client.get(f"/recommend?user_id=test_user&k={k}")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == k


def test_feedback():
    """Test feedback endpoint."""
    resp = client.post("/feedback", json={
        "user_id": "test_user",
        "item_id": "item_1",
        "feedback_type": "not_interested",
        "category": "Action",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_feedback_reduces_category():
    """Test that negative feedback reduces category exposure."""
    # Record negative feedback
    client.post("/feedback", json={
        "user_id": "feedback_user",
        "item_id": "item_50",
        "feedback_type": "not_interested",
        "category": "Action",
    })

    # The feature store should record it
    penalties = feat_store.get_negative_penalties("feedback_user")
    assert len(penalties) > 0
    assert "Action" in penalties


def test_metrics():
    """Test metrics endpoint."""
    resp = client.get("/metrics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "latency_p50_ms" in data


def test_strategy():
    """Test strategy management endpoints."""
    # Set strategy
    resp = client.post("/admin/strategy?strategy_id=test_strat", json={
        "mmr_lambda": 0.8,
        "explore_ratio": 0.2,
    })
    assert resp.status_code == 200

    # Get strategy
    resp = client.get("/admin/strategy?strategy_id=test_strat")
    assert resp.status_code == 200
    data = resp.json()
    assert data["mmr_lambda"] == 0.8


if __name__ == "__main__":
    test_health()
    test_recommend_new_user()
    test_recommend_with_k()
    test_feedback()
    test_feedback_reduces_category()
    test_metrics()
    test_strategy()
    print("All API tests passed!")
