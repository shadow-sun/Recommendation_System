"""FastAPI recommendation gateway for the ml-1m subsystem."""
import json
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from ml100k.feature_store import FeatureStore
from ml100k.mmr import MMRReranker
from ml100k.schema import FeedbackRequest, RecommendResponse, RecommendationItem
from ml100k.two_tower import TwoTowerModel

from .config import config
from .data_loader import load_items, load_ratings, load_users

app = FastAPI(
    title="MovieLens 1M Recommendation System",
    description="Two-Tower Recall + MMR Rerank for MovieLens 1M",
    version="1.0.0",
)

feature_store: FeatureStore = FeatureStore()
two_tower_model = None
faiss_indexer = None
user_vocab = {}
item_vocab = {}
category_vocab = {}
popular_items: List[str] = []
valid_user_ids = set()
dataset_user_histories: Dict[str, List[str]] = {}
item_metadata: Dict[str, Dict[str, str]] = {}
item_categories: Dict[str, str] = {}
reranker: MMRReranker = MMRReranker()

request_stats: Dict[str, int] = {
    "total": 0,
    "success": 0,
    "cold_start": 0,
    "fallback": 0,
    "popular": 0,
    "client_errors": 0,
    "server_errors": 0,
    "feedback": 0,
}
latency_samples: List[float] = []

PRESET_STRATEGIES = {
    "default": {
        "source": "model",
        "mmr_lambda": 0.7,
        "explore_ratio": 0.1,
        "negative_penalty": 0.5,
        "category_diversity_weight": 0.15,
        "reason": "two_tower+mmr",
    },
    "diverse": {
        "source": "model",
        "mmr_lambda": 0.35,
        "explore_ratio": 0.0,
        "negative_penalty": 0.5,
        "category_diversity_weight": 0.55,
        "reason": "two_tower+diverse",
    },
    "popular": {
        "source": "popular",
        "mmr_lambda": 1.0,
        "explore_ratio": 0.0,
        "negative_penalty": 0.5,
        "category_diversity_weight": 0.0,
        "reason": "popular",
    },
}

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _load_vocab(name: str) -> dict:
    suffix = os.getenv("ML1M_MODEL_SUFFIX", "")
    p = config.model_dir / f"vocabs{suffix}" / f"{name}_vocab.json"
    if not p.exists():
        p = config.model_dir / "vocabs" / f"{name}_vocab.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_popular_items() -> List[str]:
    try:
        df = load_ratings()
    except FileNotFoundError:
        return []
    df["item_id"] = df["item_id"].astype(str)
    return df.groupby("item_id").size().sort_values(ascending=False).head(1000).index.tolist()


def _load_user_histories() -> Dict[str, List[str]]:
    try:
        df = load_ratings()
    except FileNotFoundError:
        return {}
    df = df[df["rating"] >= config.data.pos_threshold].sort_values(["user_id", "timestamp"])
    histories: Dict[str, List[str]] = {}
    for uid, g in df.groupby("user_id", sort=False):
        histories[str(uid)] = g["item_id"].astype(str).tail(config.feature.max_seq_len).tolist()
    return histories


def _load_valid_user_ids() -> set:
    try:
        users = load_users()
    except FileNotFoundError:
        try:
            ratings = load_ratings()
        except FileNotFoundError:
            return set()
        return set(ratings["user_id"].astype(str).unique().tolist())
    return set(users["user_id"].astype(str).unique().tolist())


def _load_item_metadata() -> Dict[str, Dict[str, str]]:
    try:
        items = load_items()
    except FileNotFoundError:
        return {}
    return {
        str(row.item_id): {"title": str(row.title), "category": str(row.category)}
        for row in items.itertuples(index=False)
    }


def _load_item_categories() -> Dict[str, str]:
    metadata = item_metadata or _load_item_metadata()
    return {item_id: meta.get("category", "") for item_id, meta in metadata.items()}


def _adjusted_display_score(item: dict, negative_categories: Dict[str, float]) -> float:
    if "_adjusted_score" in item:
        return item["_adjusted_score"]
    category = item.get("category", "")
    penalty = negative_categories.get(category, 0.0) * reranker.negative_penalty
    return item.get("score", 0.0) * (1.0 - penalty)


def _strategy_config(strategy: str) -> dict:
    strategy_id = (strategy or "default").strip().lower()
    if strategy_id not in PRESET_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy={strategy}. Use one of: default, diverse, popular.",
        )
    cfg = dict(PRESET_STRATEGIES[strategy_id])
    stored = feature_store.get_strategy(strategy_id)
    if stored:
        cfg.update(stored)
    cfg["id"] = strategy_id
    return cfg


def _popular_candidates(exclude: set, limit: int) -> List[dict]:
    pool = [item for item in popular_items if item not in exclude]
    total = max(len(pool), 1)
    candidates = []
    for rank, item in enumerate(pool[:limit]):
        meta = item_metadata.get(item, {})
        score = 0.35 + 0.6 * (1.0 - rank / total)
        candidates.append({
            "item_id": item,
            "title": meta.get("title", ""),
            "score": round(score, 6),
            "category": meta.get("category", ""),
            "reason": "popular",
        })
    return candidates


def _diverse_select(candidates: List[dict], k: int, negative_categories: Dict[str, float]) -> List[dict]:
    for item in candidates:
        item["_adjusted_score"] = _adjusted_display_score(item, negative_categories)
    ranked = sorted(candidates, key=lambda x: x["_adjusted_score"], reverse=True)[: max(200, k * 8)]
    buckets: "OrderedDict[str, List[dict]]" = OrderedDict()
    for item in ranked:
        category = item.get("category") or "Unknown"
        buckets.setdefault(category, []).append(item)

    selected = []
    while len(selected) < k and buckets:
        for category in list(buckets.keys()):
            bucket = buckets.get(category, [])
            if not bucket:
                buckets.pop(category, None)
                continue
            item = bucket.pop(0)
            item["reason"] = "two_tower+diverse"
            selected.append(item)
            if len(selected) >= k:
                break
            if not bucket:
                buckets.pop(category, None)

    for idx, item in enumerate(selected):
        item["rank"] = idx + 1
    return selected


def _ensure_serving_data() -> None:
    global popular_items, valid_user_ids, dataset_user_histories, item_metadata, item_categories
    if not valid_user_ids:
        valid_user_ids = _load_valid_user_ids()
    if not dataset_user_histories:
        dataset_user_histories = _load_user_histories()
    if not item_metadata:
        item_metadata = _load_item_metadata()
    if not item_categories:
        item_categories = _load_item_categories()
    if not popular_items:
        popular_items = _load_popular_items()


def _ensure_model_artifacts() -> None:
    global two_tower_model, faiss_indexer, user_vocab, item_vocab, category_vocab
    if not user_vocab:
        user_vocab = _load_vocab("user_id")
    if not item_vocab:
        item_vocab = _load_vocab("item_id")
    if not category_vocab:
        category_vocab = _load_vocab("category")

    suffix = os.getenv("ML1M_MODEL_SUFFIX", "")
    if two_tower_model is None:
        tt_path = config.model_dir / f"two_tower{suffix}" / "two_tower_best.pt"
        if not tt_path.exists():
            tt_path = config.model_dir / "two_tower" / "two_tower_best.pt"
        if tt_path.exists() and user_vocab and item_vocab and category_vocab:
            import torch

            two_tower = TwoTowerModel(
                user_vocab_size=len(user_vocab),
                item_vocab_size=len(item_vocab),
                category_vocab_size=len(category_vocab),
                embedding_dim=config.model.embedding_dim,
                category_embedding_dim=config.model.category_embedding_dim,
                hidden_units=tuple(config.model.hidden_units),
                temperature=config.model.temperature,
                max_seq_len=config.feature.max_seq_len,
                dropout=config.model.dropout,
            )
            ckpt = torch.load(tt_path, map_location="cpu", weights_only=False)
            two_tower.load_state_dict(ckpt["model_state_dict"])
            two_tower.eval()
            two_tower_model = two_tower

    if faiss_indexer is None:
        idx_dir = config.model_dir / f"faiss{suffix}"
        if not (idx_dir / "faiss.index").exists():
            idx_dir = config.model_dir / "faiss"
        if (idx_dir / "faiss.index").exists():
            from ml100k.indexer import FaissIndexer

            idx = FaissIndexer()
            idx.load(idx_dir)
            faiss_indexer = idx


@app.on_event("startup")
def startup():
    global two_tower_model, faiss_indexer, user_vocab, item_vocab, category_vocab
    global popular_items, valid_user_ids, dataset_user_histories, item_metadata

    _ensure_serving_data()
    _ensure_model_artifacts()

    loaded = []
    if two_tower_model:
        loaded.append("TwoTower")
    if faiss_indexer:
        loaded.append("Faiss")
    print(f"ml-1m models loaded: {', '.join(loaded) if loaded else 'none (run ml1m/scripts/train.py first)'}")


@app.get("/")
def index():
    dashboard = STATIC_DIR / "dashboard.html"
    if dashboard.exists():
        return FileResponse(dashboard)
    return {"message": "MovieLens 1M dashboard not found"}


@app.get("/health")
def health():
    _ensure_serving_data()
    _ensure_model_artifacts()
    return {
        "status": "ok",
        "dataset": "ml-1m",
        "model_available": two_tower_model is not None,
        "index_available": faiss_indexer is not None,
        "known_users": len(dataset_user_histories),
        "valid_users": len(valid_user_ids),
        "known_items": len(item_metadata),
        "user_id_min": min((int(uid) for uid in valid_user_ids if uid.isdigit()), default=None),
        "user_id_max": max((int(uid) for uid in valid_user_ids if uid.isdigit()), default=None),
    }


@app.get("/recommend")
def recommend(
    user_id: str = Query(...),
    k: int = Query(20, ge=1, le=100),
    strategy: str = Query("default"),
):
    start = time.time()
    request_id = str(uuid.uuid4())[:8]
    request_stats["total"] += 1

    try:
        _ensure_serving_data()
        _ensure_model_artifacts()
        if valid_user_ids and user_id not in valid_user_ids:
            numeric_ids = [int(uid) for uid in valid_user_ids if uid.isdigit()]
            range_hint = ""
            if numeric_ids:
                range_hint = f" in range {min(numeric_ids)}-{max(numeric_ids)}"
            raise HTTPException(
                status_code=400,
                detail=f"Unknown ml-1m user_id={user_id}. Please use a MovieLens 1M user ID{range_hint}.",
            )

        strategy_config = _strategy_config(strategy)
        user_recent = feature_store.get_user_recent_items(user_id)
        if not user_recent:
            user_recent = dataset_user_histories.get(user_id, [])
        if not user_recent:
            raise HTTPException(
                status_code=400,
                detail=f"user_id={user_id} has no positive history available for recommendation.",
            )
        exclude = set(user_recent)

        candidates = []
        if strategy_config["source"] == "popular":
            request_stats["popular"] += 1
            candidates = _popular_candidates(exclude, limit=max(k, config.recall.top_k))
        elif two_tower_model and faiss_indexer and user_recent:
            import torch

            uid_idx = user_vocab.get(user_id, 1)
            history_indices = [item_vocab.get(iid, 0) for iid in user_recent[-config.feature.max_seq_len:]]
            history_indices = [iid for iid in history_indices if iid != 0]
            pad_len = config.feature.max_seq_len - len(history_indices)
            history = history_indices + [0] * pad_len
            mask = [1.0] * len(history_indices) + [0.0] * pad_len

            uid_t = torch.tensor([uid_idx], dtype=torch.long)
            hist_t = torch.tensor([history], dtype=torch.long)
            mask_t = torch.tensor([mask], dtype=torch.float32)

            with torch.no_grad():
                user_emb = two_tower_model.get_user_embeddings(uid_t, hist_t, mask_t).cpu().numpy()
            results = faiss_indexer.search(user_emb, k=config.recall.top_k, exclude_ids=list(exclude))
            if results:
                for iid, score in results[0]:
                    meta = item_metadata.get(iid, {})
                    candidates.append({
                        "item_id": iid,
                        "title": meta.get("title", ""),
                        "score": score,
                        "category": meta.get("category", ""),
                        "reason": strategy_config["reason"],
                    })
        else:
            request_stats["cold_start"] += 1

        if not candidates:
            request_stats["fallback"] += 1
            candidates = _popular_candidates(exclude, limit=k)
            for item in candidates:
                item["reason"] = "popular_fallback"

        candidates.sort(key=lambda x: x["score"], reverse=True)
        neg_penalties = feature_store.get_negative_penalties(user_id)
        reranker.mmr_lambda = strategy_config.get("mmr_lambda", config.rank.mmr_lambda)
        reranker.explore_ratio = strategy_config.get("explore_ratio", config.rank.explore_ratio)
        reranker.negative_penalty = strategy_config.get("negative_penalty", config.rank.negative_penalty)
        reranker.category_diversity_weight = strategy_config.get(
            "category_diversity_weight",
            config.rank.category_diversity_weight,
        )
        if strategy_config["id"] == "diverse":
            final_items = _diverse_select(candidates=candidates, k=k, negative_categories=neg_penalties)
        else:
            final_items = reranker.rerank(candidates=candidates, k=k, negative_categories=neg_penalties)

        items = [
            RecommendationItem(
                item_id=item["item_id"],
                title=item.get("title", ""),
                score=_adjusted_display_score(item, neg_penalties),
                raw_score=item.get("score", 0.0),
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
        request_stats["success"] += 1
        return result
    except HTTPException:
        request_stats["client_errors"] += 1
        raise
    except Exception as e:
        request_stats["server_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if valid_user_ids and req.user_id not in valid_user_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ml-1m user_id={req.user_id}. Please use a valid MovieLens 1M user ID.",
        )
    feature_store.record_feedback(req.user_id, req.item_id, req.feedback_type, req.category)
    event = {
        "user_id": req.user_id,
        "item_id": req.item_id,
        "feedback_type": req.feedback_type,
        "category": req.category,
        "timestamp": time.time(),
    }
    feature_store.save_feedback_event(event)
    request_stats["feedback"] += 1
    return {"status": "ok", "message": f"Feedback recorded: {req.feedback_type}"}


@app.get("/feedback/summary")
def feedback_summary(limit: int = Query(20, ge=1, le=100)):
    penalties = {}
    if hasattr(feature_store, "_store"):
        for key, value in feature_store._store.items():
            if key.startswith("neg:"):
                _, user_id, category = key.split(":", 2)
                penalties.setdefault(user_id, {})[category] = float(value)
    return {
        "total_feedback": request_stats["feedback"],
        "recent": feature_store.list_feedback(limit=limit),
        "negative_penalties": penalties,
    }


@app.get("/feedback/user/{user_id}")
def user_feedback(user_id: str, limit: int = Query(20, ge=1, le=100)):
    if valid_user_ids and user_id not in valid_user_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ml-1m user_id={user_id}. Please use a valid MovieLens 1M user ID.",
        )
    return {
        "user_id": user_id,
        "recent": feature_store.list_user_feedback(user_id, limit=limit),
        "negative_penalties": feature_store.get_negative_penalties(user_id),
        "recent_items": feature_store.get_user_recent_items(user_id),
    }


@app.get("/metrics/summary")
def metrics_summary():
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
        "successful_requests": request_stats["success"],
        "cold_start_ratio": request_stats["cold_start"] / max(request_stats["total"], 1),
        "fallback_requests": request_stats["fallback"],
        "fallback_ratio": request_stats["fallback"] / max(request_stats["total"], 1),
        "popular_requests": request_stats["popular"],
        "popular_ratio": request_stats["popular"] / max(request_stats["total"], 1),
        "client_errors": request_stats["client_errors"],
        "server_errors": request_stats["server_errors"],
        "errors": request_stats["client_errors"] + request_stats["server_errors"],
        "feedback_count": request_stats["feedback"],
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
    }


@app.post("/admin/strategy")
def set_strategy(strategy_id: str = "default", config_dict: dict = Body(...)):
    feature_store.set_strategy(strategy_id, config_dict)
    return {"status": "ok", "strategy_id": strategy_id}


@app.get("/admin/strategy")
def get_strategy(strategy_id: str = "default"):
    return feature_store.get_strategy(strategy_id)


def run_gateway(host="0.0.0.0", port=3006):
    uvicorn.run(app, host=host, port=port)
