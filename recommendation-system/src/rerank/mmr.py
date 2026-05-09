from __future__ import annotations

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def mmr_rerank(
    candidates: list[dict],
    embeddings: dict[str, np.ndarray],
    k: int,
    lambda_: float = 0.7,
    negative_penalties: dict[str, float] | None = None,
) -> list[dict]:
    negative_penalties = negative_penalties or {}
    remaining = [dict(c) for c in candidates]
    selected: list[dict] = []
    while remaining and len(selected) < k:
        best_idx = 0
        best_score = -float("inf")
        for idx, candidate in enumerate(remaining):
            item_id = str(candidate["item_id"])
            relevance = float(candidate.get("score", 0.0))
            category = str(candidate.get("category", "unknown"))
            relevance *= 1.0 - float(negative_penalties.get(category, 0.0))
            if selected and item_id in embeddings:
                sim = max(
                    _cosine(embeddings[item_id], embeddings[str(s["item_id"])])
                    for s in selected
                    if str(s["item_id"]) in embeddings
                )
            else:
                sim = 0.0
            mmr_score = lambda_ * relevance - (1.0 - lambda_) * sim
            if mmr_score > best_score:
                best_idx = idx
                best_score = mmr_score
        chosen = remaining.pop(best_idx)
        chosen["score"] = float(best_score)
        selected.append(chosen)
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
        reason = item.get("reason", "two_tower+deepfm")
        item["reason"] = reason if reason.endswith("+mmr") else f"{reason}+mmr"
    return selected

