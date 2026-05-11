"""Tests for Two-Tower model training, embedding export, and Faiss."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader


def test_two_tower_forward():
    """Test Two-Tower model forward pass."""
    from src.models.two_tower import TwoTowerModel

    model = TwoTowerModel(
        user_vocab_size=1000,
        item_vocab_size=500,
        category_vocab_size=20,
        embedding_dim=64,
        category_embedding_dim=16,
        hidden_units=(128, 64),
    )

    batch_size = 32
    user_ids = torch.randint(0, 1000, (batch_size,))
    item_ids = torch.randint(0, 500, (batch_size,))
    categories = torch.randint(0, 20, (batch_size,))
    user_history = torch.randint(0, 500, (batch_size, 50))
    history_mask = torch.ones(batch_size, 50)

    user_emb, item_emb = model(user_ids, item_ids, categories, user_history, history_mask)
    assert user_emb.shape == (batch_size, 64)
    assert item_emb.shape == (batch_size, 64)

    # Check L2 normalization
    user_norms = user_emb.norm(dim=1)
    item_norms = item_emb.norm(dim=1)
    assert torch.allclose(user_norms, torch.ones_like(user_norms), atol=1e-5)
    assert torch.allclose(item_norms, torch.ones_like(item_norms), atol=1e-5)


def test_sampled_softmax_loss():
    """Test sampled softmax loss computation."""
    from src.models.two_tower import sampled_softmax_loss

    batch_size = 16
    emb_dim = 64
    user_emb = torch.randn(batch_size, emb_dim)
    item_emb = torch.randn(batch_size, emb_dim)
    user_emb = torch.nn.functional.normalize(user_emb, dim=1)
    item_emb = torch.nn.functional.normalize(item_emb, dim=1)

    loss = sampled_softmax_loss(user_emb, item_emb)
    assert loss.item() > 0


def test_faiss_indexer():
    """Test Faiss index build, save, load, and search."""
    import tempfile
    from src.models.indexer import FaissIndexer

    np.random.seed(42)
    n_items = 100
    dim = 64
    item_ids = [f"item_{i}" for i in range(n_items)]
    embeddings = np.random.randn(n_items, dim).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    indexer = FaissIndexer(dim=dim, index_type="Flat")
    indexer.build(item_ids, embeddings)

    # Test search
    query = np.random.randn(1, dim).astype(np.float32)
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    results = indexer.search(query, k=10)
    assert len(results) == 1
    assert len(results[0]) == 10
    assert all(isinstance(r[0], str) for r in results[0])
    assert all(isinstance(r[1], float) for r in results[0])

    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "faiss_test"
        indexer.save(path)
        loaded = FaissIndexer().load(path)
        assert loaded.index.ntotal == n_items
        results2 = loaded.search(query, k=10)
        assert results[0] == results2[0]


if __name__ == "__main__":
    test_two_tower_forward()
    test_sampled_softmax_loss()
    test_faiss_indexer()
    print("All recall tests passed!")
