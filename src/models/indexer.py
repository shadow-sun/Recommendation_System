"""Faiss index builder and searcher for item embeddings."""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
import torch

from src.config.settings import config


class FaissIndexer:
    """Build, save, load, and search Faiss index for item embeddings."""

    def __init__(
        self,
        dim: int = 64,
        index_type: Optional[str] = None,
        nlist: int = 256,
    ):
        self.dim = dim
        self.index_type = index_type or config.recall.faiss_index_type
        self.nlist = nlist or config.recall.faiss_nlist
        self.index: Optional[faiss.Index] = None
        self.item_ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}

    def build(
        self,
        item_ids: List[str],
        embeddings: np.ndarray,
        categories: Optional[List[str]] = None,
    ) -> "FaissIndexer":
        """Build a Faiss index from item embeddings.

        Args:
            item_ids: List of item ID strings.
            embeddings: [N, dim] numpy array of normalized embeddings.
            categories: Optional list of category strings per item.
        """
        self.item_ids = item_ids
        self.id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        self.embeddings = embeddings.astype(np.float32)

        if "Flat" in self.index_type:
            self.index = faiss.IndexFlatIP(self.dim)  # Inner product for normalized vectors
        elif "IVF" in self.index_type:
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            n_train = min(len(embeddings), max(self.nlist * 40, 1000))
            self.index.train(embeddings[:n_train])
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        self.index.add(embeddings)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = config.recall.faiss_nprobe
        return self

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 500,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, float]]]:
        """Search for top-k most similar items.

        Args:
            query_embeddings: [B, dim] normalized query vectors.
            k: Number of results per query.
            exclude_ids: Optional list of item IDs to exclude.

        Returns:
            List of lists of (item_id, score) tuples.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        query = query_embeddings.astype(np.float32)
        search_k = min(k + len(exclude_ids or []), self.index.ntotal)
        distances, indices = self.index.search(query, search_k)

        results = []
        exclude_set = set(exclude_ids or [])
        for dist_row, idx_row in zip(distances, indices):
            items = []
            for d, idx in zip(dist_row, idx_row):
                if idx < 0 or idx >= len(self.item_ids):
                    continue
                iid = self.item_ids[idx]
                if iid in exclude_set:
                    continue
                items.append((iid, float(d)))
                if len(items) >= k:
                    break
            results.append(items)
        return results

    def get_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific item."""
        idx = self.id_to_idx.get(item_id)
        if idx is None or idx >= len(self.embeddings):
            return None
        return self.embeddings[idx]

    def save(self, path: Path) -> None:
        """Save index and metadata to disk."""
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "item_ids.json", "w") as f:
            json.dump(self.item_ids, f)
        with open(path / "id_to_idx.json", "w") as f:
            json.dump(self.id_to_idx, f)

    def load(self, path: Path) -> "FaissIndexer":
        """Load index and metadata from disk."""
        self.index = faiss.read_index(str(path / "faiss.index"))
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = config.recall.faiss_nprobe
        self.embeddings = np.load(path / "embeddings.npy")
        with open(path / "item_ids.json", "r") as f:
            self.item_ids = json.load(f)
        with open(path / "id_to_idx.json", "r") as f:
            self.id_to_idx = json.load(f)
        self.dim = self.embeddings.shape[1]
        return self


def build_faiss_index_from_model(
    model: torch.nn.Module,
    item_ids: List[str],
    categories: List[str],
    item_id_map: Dict[str, int],
    category_map: Dict[str, int],
    device: str = "cpu",
    batch_size: int = 256,
) -> FaissIndexer:
    """Compute item embeddings from a trained Two-Tower model and build Faiss index.

    Args:
        model: Trained TwoTowerModel.
        item_ids: List of all item IDs (strings).
        categories: List of category strings per item.
        item_id_map: Mapping from item_id string to index.
        category_map: Mapping from category string to index.
        device: Computation device.
        batch_size: Batch size for inference.
    """
    model.eval()
    model.to(device)
    dim = model.item_tower.output_dim

    item_idx = torch.tensor(
        [item_id_map.get(iid, 0) for iid in item_ids], dtype=torch.long
    )
    cat_idx = torch.tensor(
        [category_map.get(cat, 0) for cat in categories], dtype=torch.long
    )

    all_embs = []
    for i in range(0, len(item_ids), batch_size):
        i_batch = item_idx[i:i+batch_size].to(device)
        c_batch = cat_idx[i:i+batch_size].to(device)
        with torch.no_grad():
            emb = model.get_item_embeddings(i_batch, c_batch)
        all_embs.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)

    indexer = FaissIndexer(dim=dim)
    indexer.build(item_ids, embeddings, categories)
    return indexer
