"""FastAPI recommendation gateway for KuaiLive subsystem."""
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
import uvicorn

from .config import config
from .schema import FeedbackRequest, RecommendResponse, RecommendationItem
from .feature_store import FeatureStore
from .mmr import MMRReranker

app = FastAPI(
    title="KuaiLive Recommendation System",
    description="Two-Tower Recall + DeepFM Rank + MMR Rerank for KuaiLive",
    version="1.0.0",
)

feature_store: FeatureStore = FeatureStore()
two_tower_model: Optional[object] = None
faiss_indexer: Optional[object] = None
user_vocab: dict = {}
item_vocab: dict = {}
category_vocab: dict = {}
popular_items: List[str] = []
reranker: MMRReranker = MMRReranker()

request_stats: Dict[str, int] = {"total": 0, "cold_start": 0, "errors": 0}
latency_samples: List[float] = []


def _load_vocab(name: str) -> dict:
    p = config.model_dir / "vocabs" / f"{name}_vocab.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_popular_items() -> List[str]:
    p = config.data.processed_dir / "train.parquet"
    if p.exists():
        import pandas as pd
        df = pd.read_parquet(p)
        return df.groupby("item_id").size().sort_values(ascending=False).head(500).index.tolist()
    return []


@app.on_event("startup")
def startup() -> None:
    global two_tower_model, faiss_indexer, user_vocab, item_vocab, category_vocab, popular_items
    import torch

    user_vocab = _load_vocab("user_id")
    item_vocab = _load_vocab("item_id")
    category_vocab = _load_vocab("category")
    popular_items = _load_popular_items()

    # Load Two-Tower model
    tt_path = config.model_dir / "two_tower" / "two_tower_best.pt"
    if tt_path.exists():
        from .two_tower import TwoTowerModel
        two_tower_model = TwoTowerModel(
            user_vocab_size=len(user_vocab),
            item_vocab_size=len(item_vocab),
            category_vocab_size=len(category_vocab),
            embedding_dim=config.model.embedding_dim,
            category_embedding_dim=config.model.category_embedding_dim,
            hidden_units=tuple(config.model.hidden_units),
            temperature=config.model.temperature,
        )
        ckpt = torch.load(tt_path, map_location="cpu", weights_only=False)
        two_tower_model.load_state_dict(ckpt["model_state_dict"])
        two_tower_model.eval()

    # Load Faiss index
    idx_dir = config.model_dir / "faiss"
    if (idx_dir / "faiss.index").exists():
        from .indexer import FaissIndexer
        faiss_indexer = FaissIndexer()
        faiss_indexer.load(idx_dir)

    loaded = []
    if two_tower_model:
        loaded.append("TwoTower")
    if faiss_indexer:
        loaded.append("Faiss")
    print(f"Models loaded: {', '.join(loaded) if loaded else 'none (run train.py first)'}")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_available": two_tower_model is not None,
        "index_available": faiss_indexer is not None,
    }


@app.get("/recommend")
def recommend(
    user_id: str = Query(..., description="User ID"),
    k: int = Query(20, ge=1, le=100, description="Number of recommendations"),
    strategy: str = Query("default", description="Strategy name"),
) -> dict:
    start = time.time()
    request_id = str(uuid.uuid4())[:8]
    request_stats["total"] += 1

    try:
        strategy_config = feature_store.get_strategy(strategy)
        user_recent = feature_store.get_user_recent_items(user_id)
        exclude = set(user_recent)

        # Recall
        candidates = []
        if two_tower_model and faiss_indexer and user_recent:
            import torch
            import numpy as np
            uid_idx = user_vocab.get(user_id, 1)
            history_indices = [item_vocab.get(iid, 0) for iid in user_recent[-config.feature.max_seq_len:]]
            pad_len = config.feature.max_seq_len - len(history_indices)
            history = history_indices + [0] * pad_len
            mask = [1.0] * len(history_indices) + [0.0] * pad_len

            uid_t = torch.tensor([uid_idx], dtype=torch.long)
            hist_t = torch.tensor([history], dtype=torch.long)
            mask_t = torch.tensor([mask], dtype=torch.float32)

            with torch.no_grad():
                user_emb = two_tower_model.get_user_embeddings(uid_t, hist_t, mask_t).cpu().numpy()
            results = faiss_indexer.search(user_emb, k=config.recall.top_k, exclude_ids=list(exclude))
            candidates = [{"item_id": r[0], "score": r[1], "category": ""} for r in results[0]] if results else []
        else:
            request_stats["cold_start"] += 1

        if not candidates:
            for item in popular_items:
                if item not in exclude:
                    candidates.append({"item_id": item, "score": 1.0, "category": ""})
                if len(candidates) >= k:
                    break

        # Rank (scores already from Faiss)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Rerank with MMR
        neg_penalties = feature_store.get_negative_penalties(user_id)
        reranker.mmr_lambda = strategy_config.get("mmr_lambda", config.rank.mmr_lambda)
        reranker.explore_ratio = strategy_config.get("explore_ratio", config.rank.explore_ratio)
        reranker.negative_penalty = strategy_config.get("negative_penalty", config.rank.negative_penalty)

        final_items = reranker.rerank(candidates=candidates, k=k, negative_categories=neg_penalties)

        items = [
            RecommendationItem(
                item_id=item["item_id"],
                score=item.get("score", 0.0),
                rank=item.get("rank", i + 1),
                category=item.get("category", ""),
                reason=item.get("reason", "two_tower+mmr"),
            )
            for i, item in enumerate(final_items)
        ]

        resp = RecommendResponse(user_id=user_id, items=items, strategy=strategy, request_id=request_id)

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
    feature_store.record_feedback(req.user_id, req.item_id, req.feedback_type, req.category)
    return {"status": "ok", "message": f"Feedback recorded: {req.feedback_type}"}


@app.get("/metrics/summary")
def metrics_summary() -> dict:
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
        "cold_start_ratio": request_stats["cold_start"] / max(request_stats["total"], 1),
        "errors": request_stats["errors"],
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
    }


@app.post("/admin/strategy")
def set_strategy(strategy_id: str = "default", config_dict: dict = None) -> dict:
    if config_dict is None:
        raise HTTPException(status_code=400, detail="config_dict required")
    feature_store.set_strategy(strategy_id, config_dict)
    return {"status": "ok", "strategy_id": strategy_id}


@app.get("/admin/strategy")
def get_strategy(strategy_id: str = "default") -> dict:
    return feature_store.get_strategy(strategy_id)


def run_gateway(host: str = "0.0.0.0", port: int = 3001) -> None:
    uvicorn.run(app, host=host, port=port)
