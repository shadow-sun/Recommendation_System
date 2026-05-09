from __future__ import annotations

import math

import numpy as np


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def auc_logloss(labels: list[int], scores: list[float]) -> dict[str, float]:
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.clip(np.asarray(scores, dtype=float), 1e-15, 1.0 - 1e-15)
    loss = -float(np.mean(labels_arr * np.log(scores_arr) + (1 - labels_arr) * np.log(1 - scores_arr)))
    if len(set(labels)) < 2:
        return {"auc": 0.5, "logloss": loss}
    order = np.argsort(scores_arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores_arr) + 1)
    pos = labels_arr == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    auc = float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return {"auc": auc, "logloss": loss}


def ils(item_ids: list[str], embeddings: dict[str, np.ndarray]) -> float:
    pairs = []
    for i, left in enumerate(item_ids):
        for right in item_ids[i + 1 :]:
            if left in embeddings and right in embeddings:
                a, b = embeddings[left], embeddings[right]
                pairs.append(float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)))
    return float(np.mean(pairs)) if pairs else 0.0
