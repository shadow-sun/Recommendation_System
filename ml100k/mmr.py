"""MMR (Maximal Marginal Relevance) diversity re-ranking."""
import random
from typing import Dict, List, Optional

import numpy as np


class MMRReranker:
    def __init__(self, mmr_lambda=0.7, explore_ratio=0.1, negative_penalty=0.5, category_diversity_weight=0.15):
        self.mmr_lambda = mmr_lambda
        self.explore_ratio = explore_ratio
        self.negative_penalty = negative_penalty
        self.category_diversity_weight = category_diversity_weight

    def rerank(self, candidates, k=20, negative_categories=None, item_embeddings=None, random_pool=None):
        if len(candidates) <= k:
            return candidates
        negative_categories = negative_categories or {}
        item_embeddings = item_embeddings or {}
        explore_n = max(1, int(k * self.explore_ratio))
        for c in candidates:
            cat = c.get("category", "")
            penalty = negative_categories.get(cat, 0.0) * self.negative_penalty
            c["_adjusted_score"] = c["score"] * (1.0 - penalty)
        candidates = sorted(candidates, key=lambda x: x["_adjusted_score"], reverse=True)
        selected = []
        remaining = list(candidates)
        for _ in range(k - explore_n):
            if not remaining:
                break
            best_item, best_score = None, -float("inf")
            for item in remaining[:max(50, k * 2)]:
                relevance = item["_adjusted_score"]
                sim_penalty = 0.0
                if item_embeddings and selected:
                    item_emb = item_embeddings.get(item["item_id"])
                    if item_emb is not None:
                        max_sim = 0.0
                        for sel in selected:
                            sel_emb = item_embeddings.get(sel["item_id"])
                            if sel_emb is not None:
                                sim = np.dot(item_emb, sel_emb) / (np.linalg.norm(item_emb) * np.linalg.norm(sel_emb) + 1e-8)
                                max_sim = max(max_sim, sim)
                        sim_penalty = max_sim
                cat_penalty = 0.0
                if self.category_diversity_weight > 0 and selected:
                    selected_cats = {s.get("category", "") for s in selected}
                    if item.get("category", "") in selected_cats:
                        cat_penalty = self.category_diversity_weight
                mmr_score = self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * sim_penalty - cat_penalty
                if mmr_score > best_score:
                    best_score, best_item = mmr_score, item
            if best_item:
                selected.append(best_item)
                remaining.remove(best_item)
            else:
                selected.append(remaining.pop(0))
        if len(selected) < k and remaining:
            for item in remaining:
                if item not in selected:
                    selected.append(item)
                    if len(selected) >= k:
                        break
        if random_pool and explore_n > 0:
            pool_filtered = [r for r in random_pool if r not in selected]
            if pool_filtered:
                explore_items = random.sample(pool_filtered, min(explore_n, len(pool_filtered)))
                positions = sorted(random.sample(range(len(selected) + 1), len(explore_items)))
                for pos, item in zip(positions, explore_items):
                    item["reason"] = "explore"
                    selected.insert(min(pos, len(selected)), item)
        for i, item in enumerate(selected[:k]):
            item["rank"] = i + 1
            if "reason" not in item:
                item["reason"] = "two_tower+mmr"
        return selected[:k]
