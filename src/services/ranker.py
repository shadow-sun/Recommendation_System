"""Ranking service: DeepFM model scoring for candidate items."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.config.settings import config
from src.services.feature_store import FeatureStore


class RankService:
    """Ranks recall candidates using DeepFM model."""

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        feature_store: Optional[FeatureStore] = None,
        user_vocab: Optional[Dict[str, int]] = None,
        item_vocab: Optional[Dict[str, int]] = None,
        category_vocab: Optional[Dict[str, int]] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.feature_store = feature_store or FeatureStore()
        self.user_vocab = user_vocab or {}
        self.item_vocab = item_vocab or {}
        self.category_vocab = category_vocab or {}
        self.device = device

    def rank(
        self,
        user_id: str,
        candidates: List[Tuple[str, float]],
    ) -> List[dict]:
        """Score and rank candidates.

        Args:
            user_id: User requesting recommendations.
            candidates: List of (item_id, recall_score) tuples.

        Returns:
            List of dicts with item_id, score, category, etc. sorted by score desc.
        """
        if not candidates:
            return []

        if self.model is None:
            # No DeepFM: use recall scores directly
            results = []
            for i, (item_id, recall_score) in enumerate(candidates):
                results.append({
                    "item_id": item_id,
                    "score": recall_score,
                    "category": "",
                    "recall_score": recall_score,
                })
            return sorted(results, key=lambda x: x["score"], reverse=True)

        # With DeepFM: prepare features and predict
        try:
            return self._deepfm_rank(user_id, candidates)
        except Exception:
            # Fallback to recall score
            results = []
            for i, (item_id, recall_score) in enumerate(candidates):
                results.append({
                    "item_id": item_id,
                    "score": recall_score,
                    "category": "",
                })
            return sorted(results, key=lambda x: x["score"], reverse=True)

    def _deepfm_rank(self, user_id: str, candidates: List[Tuple[str, float]]) -> List[dict]:
        """Use DeepFM model for scoring."""
        item_ids = [c[0] for c in candidates]
        recall_scores = [c[1] for c in candidates]

        uid_idx = self.user_vocab.get(user_id, 1)
        iid_indices = [self.item_vocab.get(iid, 1) for iid in item_ids]
        cat_indices = [self.category_vocab.get("", 1) for _ in item_ids]

        n = len(candidates)
        sparse_inputs = torch.tensor(
            [[uid_idx, iid_indices[i], cat_indices[i]] for i in range(n)],
            dtype=torch.long,
        ).to(self.device)
        dense_inputs = torch.tensor(
            [[recall_scores[i], 0.0, 0.0] for i in range(n)],
            dtype=torch.float32,
        ).to(self.device)

        with torch.no_grad():
            preds = self.model.predict(sparse_inputs, dense_inputs).squeeze(-1).cpu().numpy()

        results = []
        for i, (item_id, recall_score) in enumerate(candidates):
            results.append({
                "item_id": item_id,
                "score": float(preds[i]) if i < len(preds) else recall_score,
                "category": "",
                "recall_score": recall_score,
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)
