"""FastAPI recommendation gateway with full recall → rank → rerank pipeline."""
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

from src.config.settings import config
from src.config.schema import FeedbackRequest, RecommendResponse, RecommendationItem
from src.services.feature_store import FeatureStore
from src.services.recall import RecallService
from src.services.ranker import RankService
from src.rerank.mmr import MMRReranker

app = FastAPI(
    title="Recommendation System",
    description="Two-Tower Recall + DeepFM Rank + MMR Rerank",
    version="1.0.0",
)

feature_store: FeatureStore = FeatureStore()
recall_service: Optional[RecallService] = None
rank_service: Optional[RankService] = None
reranker: MMRReranker = MMRReranker()

request_stats: Dict[str, int] = {"total": 0, "cold_start": 0, "errors": 0}
latency_samples: List[float] = []


def _load_vocab(name: str) -> dict:
    p = config.model_dir / "vocabs" / f"{name}_vocab.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_popular_items() -> dict:
    p = config.data.processed_dir / "unified_train.parquet"
    if p.exists():
        import pandas as pd
        df = pd.read_parquet(p)
        result = {}
        for source in ["ml-100k", "kualive"]:
            src = df[df["source_dataset"] == source]
            result[source] = src.groupby("item_id").size().sort_values(ascending=False).head(500).index.tolist()
        return result
    return {}


@app.on_event("startup")
def startup() -> None:
    global recall_service, rank_service

    user_vocab = _load_vocab("user_id")
    item_vocab = _load_vocab("item_id")
    category_vocab = _load_vocab("category")
    popular_items = _load_popular_items()

    # Load Two-Tower model
    two_tower = None
    tt_path = config.model_dir / "two_tower" / "two_tower_best.pt"
    if tt_path.exists():
        import torch
        from src.models.two_tower import TwoTowerModel
        two_tower = TwoTowerModel(
            user_vocab_size=len(user_vocab),
            item_vocab_size=len(item_vocab),
            category_vocab_size=len(category_vocab),
            embedding_dim=config.model.embedding_dim,
            category_embedding_dim=config.model.category_embedding_dim,
            hidden_units=tuple(config.model.hidden_units),
            temperature=config.model.temperature,
        )
        ckpt = torch.load(tt_path, map_location="cpu", weights_only=False)
        two_tower.load_state_dict(ckpt["model_state_dict"])
        two_tower.eval()

    # Load per-source Faiss indices
    indexers = {}
    for source in ["ml-100k", "kualive"]:
        idx_dir = config.model_dir / "faiss" / source
        if (idx_dir / "faiss.index").exists():
            from src.models.indexer import FaissIndexer
            idx = FaissIndexer()
            idx.load(idx_dir)
            indexers[source] = idx

    recall_service = RecallService(
        indexers=indexers,
        feature_store=feature_store,
        popular_items=popular_items,
        model=two_tower,
        user_vocab=user_vocab,
        item_vocab=item_vocab,
        category_vocab=category_vocab,
    )
    rank_service = RankService(
        model=None,
        feature_store=feature_store,
        user_vocab=user_vocab,
        item_vocab=item_vocab,
        category_vocab=category_vocab,
    )

    loaded = []
    if two_tower: loaded.append("TwoTower")
    for s in indexers: loaded.append(f"Faiss[{s}]")
    print(f"Models loaded: {', '.join(loaded) if loaded else 'none (run train_pipeline.py first)'}")


def set_services(
    recall: RecallService,
    rank: RankService,
    store: Optional[FeatureStore] = None,
) -> None:
    """Set service instances (called after model loading)."""
    global recall_service, rank_service, feature_store
    recall_service = recall
    rank_service = rank
    if store:
        feature_store = store


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "recall_available": recall_service is not None,
        "rank_available": rank_service is not None and rank_service.model is not None,
    }


@app.get("/recommend")
def recommend(
    user_id: str = Query(..., description="User ID"),
    k: int = Query(20, ge=1, le=100, description="Number of recommendations"),
    strategy: str = Query("default", description="Strategy name"),
) -> dict:
    """Get personalized recommendations."""
    start = time.time()
    request_id = str(uuid.uuid4())[:8]
    request_stats["total"] += 1

    try:
        # 1. Get strategy config
        strategy_config = feature_store.get_strategy(strategy)

        # 2. Recall
        user_recent = feature_store.get_user_recent_items(user_id)
        candidates = recall_service.recall(user_id, k=config.recall.top_k, exclude_items=user_recent)

        if not user_recent:
            request_stats["cold_start"] += 1

        # 3. Rank
        ranked = rank_service.rank(user_id, candidates)

        # 4. Get negative feedback penalties
        neg_penalties = feature_store.get_negative_penalties(user_id)

        # 5. Rerank with MMR
        reranker.mmr_lambda = strategy_config.get("mmr_lambda", config.rank.mmr_lambda)
        reranker.explore_ratio = strategy_config.get("explore_ratio", config.rank.explore_ratio)
        reranker.negative_penalty = strategy_config.get("negative_penalty", config.rank.negative_penalty)

        final_items = reranker.rerank(
            candidates=ranked,
            k=k,
            negative_categories=neg_penalties,
        )

        # Build response
        items = [
            RecommendationItem(
                item_id=item["item_id"],
                score=item.get("score", 0.0),
                rank=item.get("rank", i + 1),
                category=item.get("category", ""),
                reason=item.get("reason", "two_tower+deepfm+mmr"),
            )
            for i, item in enumerate(final_items)
        ]

        resp = RecommendResponse(
            user_id=user_id,
            items=items,
            strategy=strategy,
            request_id=request_id,
        )

        elapsed = (time.time() - start) * 1000
        latency_samples.append(elapsed)
        if len(latency_samples) > 1000:
            latency_samples.pop(0)

        result = resp.to_dict()
        result["latency_ms"] = round(elapsed, 2)
        return result

    except Exception as e:
        request_stats["errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    """Record user feedback."""
    feature_store.record_feedback(
        req.user_id, req.item_id, req.feedback_type, req.category
    )
    return {"status": "ok", "message": f"Feedback recorded: {req.feedback_type}"}


@app.get("/metrics/summary")
def metrics_summary() -> dict:
    """Get current metrics summary."""
    latencies = latency_samples
    if latencies:
        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
    else:
        p50 = p95 = p99 = 0

    return {
        "total_requests": request_stats["total"],
        "cold_start_ratio": (
            request_stats["cold_start"] / max(request_stats["total"], 1)
        ),
        "errors": request_stats["errors"],
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
    }


@app.post("/admin/strategy")
def set_strategy(strategy_id: str = "default", config_dict: dict = None) -> dict:
    """Set strategy configuration."""
    if config_dict is None:
        raise HTTPException(status_code=400, detail="config_dict required")
    feature_store.set_strategy(strategy_id, config_dict)
    return {"status": "ok", "strategy_id": strategy_id}


@app.get("/admin/strategy")
def get_strategy(strategy_id: str = "default") -> dict:
    """Get strategy configuration."""
    return feature_store.get_strategy(strategy_id)


def run_gateway(host: str = "0.0.0.0", port: int = 3000) -> None:
    """Run the FastAPI gateway."""
    uvicorn.run(app, host=host, port=port)
