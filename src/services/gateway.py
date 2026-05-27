"""FastAPI recommendation gateway — ml-100k recall → rank → rerank pipeline."""
import json
import secrets
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from src.auth import get_auth_store
from src.config.settings import config, ROOT
from src.config.schema import FeedbackRequest, RecommendResponse, RecommendationItem
from src.services.feature_store import FeatureStore
from src.services.recall import RecallService
from src.services.ranker import RankService
from src.rerank.mmr import MMRReranker


# ── Auth models ────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class ApproveRequest(BaseModel):
    username: str
    role: str


# ── Session & auth helpers ─────────────────────────────────────────────────

sessions: Dict[str, str] = {}  # token → username


def _require_auth(request: Request) -> str:
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    username = sessions.get(token)
    if not username:
        raise HTTPException(status_code=401, detail="请先登录")
    return username


def _require_role(request: Request, roles: list) -> str:
    username = _require_auth(request)
    store = get_auth_store()
    user = store.get_user(username)
    user_role = user.role if user else "user"
    if user_role not in roles:
        raise HTTPException(status_code=403, detail="权限不足，需要: " + "/".join(roles))
    return username


# ── App & services ─────────────────────────────────────────────────────────

app = FastAPI(
    title="MovieLens Recommendation System",
    description="Two-Tower Recall + MMR Rerank for MovieLens 100K",
    version="1.0.0",
)

feature_store: FeatureStore = FeatureStore()
recall_service: Optional[RecallService] = None
rank_service: Optional[RankService] = None
reranker: MMRReranker = MMRReranker()

request_stats: Dict[str, int] = {"total": 0, "cold_start": 0, "errors": 0}
latency_samples: List[float] = []


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_vocab(name: str) -> dict:
    p = config.model_dir / "vocabs" / f"{name}_vocab.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_vocab_from(vocab_dir: Path, name: str) -> dict:
    p = vocab_dir / f"{name}_vocab.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_popular_items() -> dict:
    import pandas as pd

    # Try ml100k processed data path first
    from ml100k.config import config as ml100k_config
    p = ml100k_config.data.processed_dir / "train.parquet"
    if not p.exists():
        p = config.data.processed_dir / "train.parquet"
    if not p.exists():
        p = config.data.processed_dir / "unified_train.parquet"

    result = {}
    if p.exists():
        df = pd.read_parquet(p)
        if "source_dataset" in df.columns:
            df = df[df["source_dataset"] == "ml-100k"]
        result["ml-100k"] = df.groupby("item_id").size().sort_values(ascending=False).head(500).index.tolist()
        return result

    # Fallback: compute popular items from raw MovieLens ratings
    raw_dir = ml100k_config.data.raw_dir / "ml-100k"
    if not raw_dir.is_dir():
        raw_dir = ml100k_config.data.raw_dir
    ratings_path = raw_dir / "u.data"
    if ratings_path.exists():
        cols = ["user_id", "item_id", "rating", "timestamp"]
        df = pd.read_csv(ratings_path, sep="\t", names=cols)
        df["item_id"] = df["item_id"].astype(str)
        result["ml-100k"] = df.groupby("item_id").size().sort_values(ascending=False).head(500).index.tolist()
        return result

    return {}


# ── Startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup() -> None:
    global recall_service, rank_service

    ml100k_model_dir = ROOT / "ml100k" / "models"
    user_vocab = _load_vocab_from(ml100k_model_dir / "vocabs", "user_id")
    item_vocab = _load_vocab_from(ml100k_model_dir / "vocabs", "item_id")
    category_vocab = _load_vocab_from(ml100k_model_dir / "vocabs", "category")
    if not user_vocab:
        user_vocab = _load_vocab("user_id")
        item_vocab = _load_vocab("item_id")
        category_vocab = _load_vocab("category")
    popular_items = _load_popular_items()

    two_tower = None
    tt_path = ml100k_model_dir / "two_tower" / "two_tower_best.pt"
    if not tt_path.exists():
        tt_path = config.model_dir / "two_tower" / "two_tower_best.pt"
    if tt_path.exists():
        import torch
        from ml100k.two_tower import TwoTowerModel
        two_tower = TwoTowerModel(
            user_vocab_size=len(user_vocab),
            item_vocab_size=len(item_vocab),
            category_vocab_size=len(category_vocab),
            embedding_dim=64,
            category_embedding_dim=16,
            hidden_units=(128, 64),
            temperature=0.07,
            max_seq_len=50,
            dropout=0.15,
        )
        ckpt = torch.load(tt_path, map_location="cpu", weights_only=False)
        two_tower.load_state_dict(ckpt["model_state_dict"])
        two_tower.eval()

    indexers = {}
    idx_dir = ml100k_model_dir / "faiss"
    if not (idx_dir / "faiss.index").exists():
        idx_dir = config.model_dir / "faiss" / "ml-100k"
    if (idx_dir / "faiss.index").exists():
        from src.models.indexer import FaissIndexer
        idx = FaissIndexer()
        idx.load(idx_dir)
        indexers["ml-100k"] = idx

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
    if two_tower: loaded.append("TwoTower[ml-100k]")
    for s in indexers: loaded.append(f"Faiss[{s}]")
    print(f"Models loaded: {', '.join(loaded) if loaded else 'none (run train_pipeline.py first)'}")


def set_services(
    recall: RecallService,
    rank: RankService,
    store: Optional[FeatureStore] = None,
) -> None:
    global recall_service, rank_service, feature_store
    recall_service = recall
    rank_service = rank
    if store:
        feature_store = store


# ── Static files ───────────────────────────────────────────────────────────

_STATIC_DIR = ROOT / "src" / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def index():
    p = _STATIC_DIR / "dashboard.html"
    return FileResponse(p) if p.exists() else {"message": "Dashboard not found"}


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "recall_available": recall_service is not None,
        "rank_available": rank_service is not None and rank_service.model is not None,
    }


# ── Auth endpoints ─────────────────────────────────────────────────────────

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
    u = store.get_user(username)
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


# ── Recommend ──────────────────────────────────────────────────────────────

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
        candidates = recall_service.recall(user_id, k=config.recall.top_k, exclude_items=user_recent)

        if not user_recent:
            request_stats["cold_start"] += 1

        ranked = rank_service.rank(user_id, candidates)
        neg_penalties = feature_store.get_negative_penalties(user_id)

        reranker.mmr_lambda = strategy_config.get("mmr_lambda", config.rank.mmr_lambda)
        reranker.explore_ratio = strategy_config.get("explore_ratio", config.rank.explore_ratio)
        reranker.negative_penalty = strategy_config.get("negative_penalty", config.rank.negative_penalty)

        final_items = reranker.rerank(
            candidates=ranked,
            k=k,
            negative_categories=neg_penalties,
        )

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


# ── Feedback ───────────────────────────────────────────────────────────────

@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    feature_store.record_feedback(
        req.user_id, req.item_id, req.feedback_type, req.category
    )
    return {"status": "ok", "message": f"Feedback recorded: {req.feedback_type}"}


# ── Metrics (protected) ────────────────────────────────────────────────────

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
        "cold_start_ratio": (
            request_stats["cold_start"] / max(request_stats["total"], 1)
        ),
        "errors": request_stats["errors"],
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
    }


# ── Strategy (protected) ───────────────────────────────────────────────────

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


# ── Run ────────────────────────────────────────────────────────────────────

def run_gateway(host: str = "0.0.0.0", port: int = 3000) -> None:
    uvicorn.run(app, host=host, port=port)
