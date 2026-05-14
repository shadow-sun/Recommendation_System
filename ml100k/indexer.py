"""Faiss index builder and searcher for item embeddings."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
import torch

from .config import config


class FaissIndexer:
    def __init__(self, dim=64, index_type=None, nlist=256):
        self.dim = dim
        self.index_type = index_type or config.recall.faiss_index_type
        self.nlist = nlist or config.recall.faiss_nlist
        self.index = None
        self.item_ids = []
        self.id_to_idx = {}
        self.embeddings = None

    def build(self, item_ids, embeddings):
        self.item_ids = item_ids
        self.id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
        self.embeddings = embeddings.astype(np.float32)
        if "Flat" in self.index_type:
            self.index = faiss.IndexFlatIP(self.dim)
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

    def search(self, query_embeddings, k=500, exclude_ids=None):
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

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "item_ids.json", "w") as f:
            json.dump(self.item_ids, f)
        with open(path / "id_to_idx.json", "w") as f:
            json.dump(self.id_to_idx, f)

    def load(self, path):
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


def build_faiss_index_from_model(model, item_ids, categories, item_id_map, category_map, device="cpu", batch_size=256):
    model.eval()
    model.to(device)
    dim = model.item_tower.output_dim
    item_idx = torch.tensor([item_id_map.get(iid, 0) for iid in item_ids], dtype=torch.long)
    cat_idx = torch.tensor([category_map.get(cat, 0) for cat in categories], dtype=torch.long)
    all_embs = []
    for i in range(0, len(item_ids), batch_size):
        i_batch = item_idx[i:i+batch_size].to(device)
        c_batch = cat_idx[i:i+batch_size].to(device)
        with torch.no_grad():
            emb = model.get_item_embeddings(i_batch, c_batch)
        all_embs.append(emb.cpu().numpy())
    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
    indexer = FaissIndexer(dim=dim)
    indexer.build(item_ids, embeddings)
    return indexer
