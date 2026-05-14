"""Evaluation metrics for recommendation system."""
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    hits = top_k & set(relevant)
    return len(hits) / len(set(relevant))


def precision_at_k(recommended, relevant, k):
    if not recommended or k <= 0:
        return 0.0
    top_k = recommended[:k]
    hits = set(top_k) & set(relevant)
    return len(hits) / min(len(top_k), k)


def hit_rate_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return 1.0 if set(recommended[:k]) & set(relevant) else 0.0


def diversity_at_k(recommended, item_categories, k):
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    categories = [item_categories.get(item, "") for item in top_k]
    categories = [category for category in categories if category]
    if not categories:
        return 0.0
    return len(set(categories)) / len(categories)


def ndcg_at_k(recommended, relevant, k, relevance_scores=None):
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    scores = relevance_scores or {r: 1.0 for r in relevant}
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in rel_set:
            dcg += scores.get(item, 1.0) / np.log2(i + 2)
    ideal_scores = sorted([scores.get(r, 1.0) for r in relevant], reverse=True)[:k]
    idcg = sum(s / np.log2(i + 2) for i, s in enumerate(ideal_scores))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(user_recommendations, user_relevant, k_values=[5, 10, 20]):
    results = {}
    for k in k_values:
        recalls = []
        precisions = []
        hit_rates = []
        for uid, recs in user_recommendations.items():
            rel = user_relevant.get(uid, [])
            recalls.append(recall_at_k(recs, rel, k))
            precisions.append(precision_at_k(recs, rel, k))
            hit_rates.append(hit_rate_at_k(recs, rel, k))
        results[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        results[f"precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
        results[f"hit_rate@{k}"] = float(np.mean(hit_rates)) if hit_rates else 0.0
    return results


def compute_auc_logloss(y_true, y_pred):
    return {
        "auc": float(roc_auc_score(y_true, y_pred)) if len(set(y_true)) > 1 else 0.5,
        "logloss": float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7))),
    }
