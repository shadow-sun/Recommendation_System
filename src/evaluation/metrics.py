"""Evaluation metrics for recommendation system."""
from typing import Dict, List, Optional, Set

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def recall_at_k(
    recommended: List[str],
    relevant: List[str],
    k: int,
) -> float:
    """Recall@K: fraction of relevant items found in top-K recommendations."""
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    hits = top_k & set(relevant)
    return len(hits) / min(len(relevant), k)


def precision_at_k(
    recommended: List[str],
    relevant: List[str],
    k: int,
) -> float:
    """Precision@K: fraction of top-K recommendations that are relevant."""
    if k <= 0:
        return 0.0
    top_k = set(recommended[:k])
    hits = top_k & set(relevant)
    return len(hits) / k


def hit_rate_at_k(
    recommended: List[str],
    relevant: List[str],
    k: int,
) -> float:
    """HitRate@K: whether top-K recommendations contain any relevant item."""
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    return 1.0 if top_k & set(relevant) else 0.0


def diversity_at_k(
    recommended: List[str],
    item_categories: Dict[str, str],
    k: int,
) -> float:
    """Category diversity@K: unique categories divided by returned top-K items."""
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    categories = [item_categories.get(item, "") for item in top_k]
    categories = [category for category in categories if category]
    if not categories:
        return 0.0
    return len(set(categories)) / len(categories)


def ndcg_at_k(
    recommended: List[str],
    relevant: List[str],
    k: int,
    relevance_scores: Optional[Dict[str, float]] = None,
) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain."""
    if not relevant:
        return 0.0

    rel_set = set(relevant)
    scores = relevance_scores or {r: 1.0 for r in relevant}

    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in rel_set:
            rel = scores.get(item, 1.0)
            dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed

    ideal_scores = sorted(
        [scores.get(r, 1.0) for r in relevant], reverse=True
    )[:k]
    idcg = sum(s / np.log2(i + 2) for i, s in enumerate(ideal_scores))

    return dcg / idcg if idcg > 0 else 0.0


def ils_diversity(
    recommended_lists: List[List[str]],
    item_embeddings: Dict[str, np.ndarray],
) -> float:
    """Intra-List Similarity: average pairwise cosine similarity.

    Lower ILS = more diverse recommendations.
    """
    total_sim = 0.0
    count = 0

    for recs in recommended_lists:
        embs = []
        for item in recs:
            emb = item_embeddings.get(item)
            if emb is not None:
                embs.append(emb)

        if len(embs) < 2:
            continue

        embs = np.array(embs)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / norms
        sim_matrix = np.dot(embs, embs.T)

        n = len(embs)
        pairwise_sim = (sim_matrix.sum() - n) / (n * (n - 1))
        total_sim += pairwise_sim
        count += 1

    return total_sim / max(count, 1)


def category_coverage(
    recommended_lists: List[List[str]],
    item_categories: Dict[str, str],
    total_categories: int,
) -> float:
    """Fraction of total categories covered by recommendations."""
    covered: Set[str] = set()
    for recs in recommended_lists:
        for item in recs:
            cat = item_categories.get(item, "")
            if cat:
                covered.add(cat)
    return len(covered) / max(total_categories, 1)


def evaluate_recommendations(
    user_recommendations: Dict[str, List[str]],
    user_relevant: Dict[str, List[str]],
    k_values: List[int] = [5, 10, 20],
    item_categories: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Compute content-inclusion metrics for all users.

    NDCG is intentionally omitted from default reports because the current
    experiment objective values inclusion over rank position.
    """
    results = {}
    for k in k_values:
        recalls = []
        precisions = []
        hit_rates = []
        diversities = []
        for uid, recs in user_recommendations.items():
            rel = user_relevant.get(uid, [])
            recalls.append(recall_at_k(recs, rel, k))
            precisions.append(precision_at_k(recs, rel, k))
            hit_rates.append(hit_rate_at_k(recs, rel, k))
            if item_categories is not None:
                diversities.append(diversity_at_k(recs, item_categories, k))
        results[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        results[f"precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
        results[f"hit_rate@{k}"] = float(np.mean(hit_rates)) if hit_rates else 0.0
        if item_categories is not None:
            results[f"diversity@{k}"] = float(np.mean(diversities)) if diversities else 0.0
    return results


def compute_auc_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute AUC and LogLoss."""
    return {
        "auc": float(roc_auc_score(y_true, y_pred)) if len(set(y_true)) > 1 else 0.5,
        "logloss": float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7))),
    }
