"""FastAPI recommendation gateway for KuaiLive subsystem."""
import json
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from .auth import get_auth_store
from .config import config
from .schema import FeedbackRequest, RecommendResponse, RecommendationItem
from .feature_store import FeatureStore
from .mmr import MMRReranker


feature_store: FeatureStore = FeatureStore()
sessions: Dict[str, str] = {}  # token → username


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class ApproveRequest(BaseModel):
    username: str
    role: str


def _require_auth(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    username = sessions.get(token)
    if not username:
        raise HTTPException(status_code=401, detail="请先登录")
    return username


def _require_role(request: Request, roles: list):
    username = _require_auth(request)
    user = get_auth_store().authenticate(username, "")
    if user is None:
        # Re-fetch from store
        store = get_auth_store()
        for u in store.list_users():
            if u["username"] == username:
                user_role = u["role"]
                break
        else:
            raise HTTPException(status_code=403, detail="无权访问")
    else:
        user_role = user.role
    # Actually need to get current role from store
    store = get_auth_store()
    u2 = store._users.get(username)
    user_role = u2.role if u2 else "user"
    if user_role not in roles:
        raise HTTPException(status_code=403, detail="权限不足，需要: " + "/".join(roles))
    return username
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models and vocabularies."""
    global two_tower_model, faiss_indexer, user_vocab, item_vocab, category_vocab, popular_items
    import torch

    user_vocab = _load_vocab("user_id")
    item_vocab = _load_vocab("item_id")
    category_vocab = _load_vocab("category")
    popular_items = _load_popular_items()

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

    idx_dir = config.model_dir / "faiss"
    if (idx_dir / "lightgcn" / "faiss.index").exists():
        from .indexer import FaissIndexer
        faiss_indexer = FaissIndexer()
        faiss_indexer.load(idx_dir / "lightgcn")

    loaded = []
    if two_tower_model:
        loaded.append("TwoTower")
    if faiss_indexer:
        loaded.append("Faiss")
    print(f"Models loaded: {', '.join(loaded) if loaded else 'none (run train.py first)'}")

    yield

    # Shutdown: nothing to clean up currently
    print("Gateway shutting down")


app = FastAPI(
    title="KuaiLive Recommendation System",
    description="LightGCN/Two-Tower Recall + MMR Rerank for KuaiLive",
    version="1.0.0",
    lifespan=lifespan,
)

_ROOT = Path(__file__).resolve().parent
static_dir = _ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def index():
    p = _ROOT / "static" / "dashboard.html"
    return FileResponse(p) if p.exists() else {"message": "Dashboard not found"}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_available": two_tower_model is not None,
        "index_available": faiss_indexer is not None,
    }


# ── Auth endpoints ────────────────────────────────────────────────────────

@app.post("/auth/register")
def auth_register(req: RegisterRequest):
    store = get_auth_store()
    user = store.register(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=409, detail="用户名已存在")
    return {"status": "ok", "username": user.username, "role": user.role}


@app.post("/auth/login")
def auth_login(req: LoginRequest):
    store = get_auth_store()
    user = store.authenticate(req.username, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = secrets.token_hex(32)
    sessions[token] = user.username
    return {"status": "ok", "username": user.username, "role": user.role, "token": token}


@app.get("/auth/me")
def auth_me(request: Request):
    username = _require_auth(request)
    store = get_auth_store()
    u = store._users.get(username)
    if u is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"username": u.username, "role": u.role, "approved": u.approved}


@app.post("/auth/logout")
def auth_logout(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    sessions.pop(token, None)
    return {"status": "ok"}


@app.get("/auth/users")
def auth_list_users(request: Request):
    _require_role(request, ["admin"])
    store = get_auth_store()
    return {"users": store.list_users()}


@app.post("/auth/approve")
def auth_approve(req: ApproveRequest, request: Request):
    _require_role(request, ["admin"])
    store = get_auth_store()
    if not store.promote_user(req.username, req.role):
        raise HTTPException(status_code=400, detail="用户不存在或角色无效")
    return {"status": "ok", "username": req.username, "role": req.role}


@app.delete("/auth/users/{username}")
def auth_delete_user(username: str, request: Request):
    _require_role(request, ["admin"])
    store = get_auth_store()
    if not store.delete_user(username):
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"status": "ok"}


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
def metrics_summary(request: Request) -> dict:
    _require_role(request, ["admin", "operator"])
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
def set_strategy(strategy_id: str = "default", config_dict: dict = None, request: Request = None) -> dict:
    _require_role(request, ["admin", "operator"])
    if config_dict is None:
        raise HTTPException(status_code=400, detail="config_dict required")
    feature_store.set_strategy(strategy_id, config_dict)
    return {"status": "ok", "strategy_id": strategy_id}


@app.get("/admin/strategy")
def get_strategy(strategy_id: str = "default") -> dict:
    return feature_store.get_strategy(strategy_id)


def run_gateway(host: str = "0.0.0.0", port: int = 3001) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_gateway()
