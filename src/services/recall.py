"""Recall service: Faiss ANN search + popular fallback for cold start."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.config.settings import config
from src.models.indexer import FaissIndexer
from src.services.feature_store import FeatureStore


class RecallService:
    """Recall layer: returns candidate items using Faiss + popular fallback."""

    def __init__(
        self,
        indexers: Optional[Dict[str, FaissIndexer]] = None,
        feature_store: Optional[FeatureStore] = None,
        popular_items: Optional[Dict[str, List[str]]] = None,
        model: Optional[torch.nn.Module] = None,
        user_vocab: Optional[Dict[str, int]] = None,
        item_vocab: Optional[Dict[str, int]] = None,
        category_vocab: Optional[Dict[str, int]] = None,
        device: str = "cpu",
    ):
        self.indexers = indexers or {}
        self.feature_store = feature_store or FeatureStore()
        self.popular_items = popular_items or {}
        self.model = model
        self.user_vocab = user_vocab or {}
        self.item_vocab = item_vocab or {}
        self.category_vocab = category_vocab or {}
        self.device = device
        self.top_k = config.recall.top_k

    def _user_source(self, user_id: str) -> str:
        return "kualive" if user_id.startswith("kuai_") else "ml-100k"

    def recall(
        self,
        user_id: str,
        k: int = 500,
        exclude_items: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Get recall candidates for a user."""
        recent = self.feature_store.get_user_recent_items(user_id)
        exclude = set(exclude_items or []) | set(recent)
        source = self._user_source(user_id)
        indexer = self.indexers.get(source)
        pop = self.popular_items.get(source, [])

        if not self.model or not indexer or not recent:
            results = []
            for item in pop:
                if item not in exclude:
                    results.append((item, 1.0))
                if len(results) >= k:
                    break
            return results

        try:
            user_emb = self._compute_user_embedding(user_id, recent)
            if user_emb is None:
                return self._popular_fallback(k, exclude, pop)

            query = user_emb.reshape(1, -1)
            search_results = indexer.search(query, k, list(exclude))
            return search_results[0] if search_results else self._popular_fallback(k, exclude, pop)
        except Exception:
            return self._popular_fallback(k, exclude, pop)

    def _compute_user_embedding(self, user_id: str, recent_items: List[str]) -> Optional[np.ndarray]:
        """Compute user embedding from the Two-Tower model."""
        if self.model is None:
            return None

        uid_idx = self.user_vocab.get(user_id, 1)
        history_indices = [self.item_vocab.get(iid, 0) for iid in recent_items[-50:]]
        pad_len = 50 - len(history_indices)
        history = history_indices + [0] * pad_len
        mask = [1.0] * len(history_indices) + [0.0] * pad_len

        uid_t = torch.tensor([uid_idx], dtype=torch.long).to(self.device)
        hist_t = torch.tensor([history], dtype=torch.long).to(self.device)
        mask_t = torch.tensor([mask], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            emb = self.model.get_user_embeddings(uid_t, hist_t, mask_t)
        return emb.cpu().numpy()

    def _popular_fallback(self, k: int, exclude: set, pop: list = None) -> List[Tuple[str, float]]:
        results = []
        source = pop if pop is not None else []
        for item in source:
            if item not in exclude:
                results.append((item, 1.0))
            if len(results) >= k:
                break
        return results
