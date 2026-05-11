"""End-to-end training pipeline.

Data loading → Feature building → Two-Tower training → Embedding export
→ Faiss index build → DeepFM training → Evaluation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import config
from src.data.ml100k_loader import load_ml100k
from src.data.unified_converter import convert_ml100k, convert_kualive, combine_datasets
from src.data.data_splitter import leave_last_out_split, save_splits
from src.data.base_dataset import (
    TwoTowerDataset, build_user_history, collate_fn,
)
from src.features.vocab_builder import build_vocabs_from_df, save_all_vocabs
from src.models.two_tower import TwoTowerModel
from src.models.trainer import Trainer, two_tower_loss_fn
from src.models.indexer import build_faiss_index_from_model, FaissIndexer
from src.evaluation.metrics import evaluate_recommendations, recall_at_k
from src.evaluation.reporter import EvaluationReport


def main():
    print("=" * 60)
    print("  Recommendation System — Training Pipeline")
    print("=" * 60)

    # 1. Load and convert data
    print("\n[1/8] Loading data...")
    ml_ratings, ml_items, _ = load_ml100k()
    ml_unified = convert_ml100k(ml_ratings, ml_items)
    kuai_unified = convert_kualive()
    combined = combine_datasets(ml_unified, kuai_unified)
    print(f"  Combined dataset: {len(combined)} interactions")
    print(f"  Users: {combined['user_id'].nunique()}, Items: {combined['item_id'].nunique()}")

    # 2. Per-user leave-last-out split (avoids zero item overlap on high-churn data)
    print("\n[2/8] Splitting data (per-user leave-last-out)...")
    train, val, test = leave_last_out_split(combined)
    processed_dir = config.data.processed_dir
    save_splits(train, val, test, str(processed_dir))
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Diagnose item overlap between train and test
    for source in ["ml-100k", "kualive"]:
        tr_items = set(train[train["source_dataset"] == source]["item_id"].unique())
        te_items = set(test[test["source_dataset"] == source]["item_id"].unique())
        overlap = len(tr_items & te_items)
        pct = overlap / max(len(te_items), 1) * 100
        print(f"  [{source}] train items={len(tr_items)}, test items={len(te_items)}, overlap={overlap} ({pct:.1f}% of test)")

    # 3. Build vocabularies
    print("\n[3/8] Building vocabularies...")
    vocabs = build_vocabs_from_df(train)
    for name, v in vocabs.items():
        print(f"  {name}: {len(v)} unique values")

    user_vocab = vocabs["user_id"]
    item_vocab = vocabs["item_id"]
    category_vocab = vocabs["category"]

    # 4. Build datasets
    print("\n[4/8] Building datasets...")
    user_history = build_user_history(train, item_vocab)

    train_ds = TwoTowerDataset(
        train, user_vocab, item_vocab, category_vocab,
        user_history, config.feature.max_seq_len,
    )
    val_ds = TwoTowerDataset(
        val, user_vocab, item_vocab, category_vocab,
        user_history, config.feature.max_seq_len,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.model.batch_size, shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.model.batch_size, shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 5. Train Two-Tower model
    print("\n[5/8] Training Two-Tower model...")
    two_tower = TwoTowerModel(
        user_vocab_size=len(user_vocab),
        item_vocab_size=len(item_vocab),
        category_vocab_size=len(category_vocab),
        embedding_dim=config.model.embedding_dim,
        category_embedding_dim=config.model.category_embedding_dim,
        hidden_units=tuple(config.model.hidden_units),
        temperature=config.model.temperature,
    )

    trainer = Trainer(
        two_tower,
        lr=config.model.learning_rate,
        model_dir=config.model_dir / "two_tower",
    )

    # Delete stale checkpoints from previous runs (different vocab sizes)
    import os as _os
    _old_ckpt = config.model_dir / "two_tower" / "two_tower_best.pt"
    if _old_ckpt.exists():
        _old_ckpt.unlink()

    history = trainer.fit(
        train_loader, val_loader, two_tower_loss_fn,
        epochs=config.model.epochs,
        patience=config.model.early_stopping_patience,
        checkpoint_name="two_tower_best.pt",
    )
    trainer.save("two_tower_best.pt")

    # 6. Build separate Faiss indices per source dataset
    print("\n[6/8] Building Faiss indexes...")
    two_tower.eval()

    item_id_to_category = dict(zip(train["item_id"], train["category"]))
    item_id_to_source = dict(zip(train["item_id"], train["source_dataset"]))

    for source in ["ml-100k", "kualive"]:
        src_items = sorted(train[train["source_dataset"] == source]["item_id"].unique())
        src_categories = [item_id_to_category.get(iid, "") for iid in src_items]
        if not src_items:
            continue

        faiss_indexer = build_faiss_index_from_model(
            two_tower,
            src_items,
            src_categories,
            item_vocab,
            category_vocab,
            device=config.device,
        )
        faiss_dir = config.model_dir / "faiss" / source
        faiss_indexer.save(faiss_dir)
        print(f"  Faiss [{source}] saved to {faiss_dir}, ntotal={faiss_indexer.index.ntotal}")

    # 7. Compute popular items per source dataset
    print("\n[7/8] Computing baselines...")
    popular_items = {}
    for source in ["ml-100k", "kualive"]:
        src_train = train[train["source_dataset"] == source]
        popular_items[source] = (
            src_train.groupby("item_id").size().sort_values(ascending=False).head(500).index.tolist()
        )

    # Load per-source Faiss indices
    faiss_indexers = {}
    for source in ["ml-100k", "kualive"]:
        idx_dir = config.model_dir / "faiss" / source
        if (idx_dir / "faiss.index").exists():
            idx = FaissIndexer()
            idx.load(idx_dir)
            faiss_indexers[source] = idx

    # 8. Evaluate recall on test set
    print("\n[8/8] Evaluating...")
    test_users = test["user_id"].unique()
    warm_users = [u for u in test_users if user_history.get(u)]
    cold_users = [u for u in test_users if not user_history.get(u)]
    print(f"  Test users: {len(test_users)} ({len(warm_users)} warm, {len(cold_users)} cold)")

    eval_users = (warm_users[:100] if warm_users else test_users[:100])
    device = config.device

    user_recs = {}
    user_relevant = {}
    for uid in eval_users:
        recent = user_history.get(uid, [])
        relevant_items = test[test["user_id"] == uid]["item_id"].tolist()
        user_relevant[uid] = relevant_items

        user_source = "kualive" if uid.startswith("kuai_") else "ml-100k"
        fl = faiss_indexers.get(user_source)
        pop = popular_items.get(user_source, [])

        if recent and fl:
            uid_idx = user_vocab.get(uid, 1)
            hist = recent[-config.feature.max_seq_len:]
            pad_len = config.feature.max_seq_len - len(hist)
            hist_padded = hist + [0] * pad_len
            mask = [1.0] * len(hist) + [0.0] * pad_len

            with torch.no_grad():
                user_emb = two_tower.get_user_embeddings(
                    torch.tensor([uid_idx], device=device),
                    torch.tensor([hist_padded], device=device),
                    torch.tensor([mask], dtype=torch.float32, device=device),
                ).cpu().numpy()
            results = fl.search(user_emb, k=20)
            user_recs[uid] = [r[0] for r in results[0]] if results else pop[:20]
        else:
            user_recs[uid] = pop[:20]

    # Check overlap
    all_pop = popular_items["ml-100k"][:20] + popular_items["kualive"][:20]
    pop_in_test = len(set(all_pop) & set(test["item_id"].unique()))
    print(f"  Popular-20 overlap with test items: {pop_in_test}")

    # Diagnostic: print first 3 warm users
    for uid in eval_users[:3]:
        print(f"  DEBUG user={uid}")
        print(f"    recent_history={user_history.get(uid, [])}")
        print(f"    test_relevant={user_relevant.get(uid, [])[:5]}")
        print(f"    recommended={user_recs.get(uid, [])[:5]}")
        print(f"    overlap={set(user_recs.get(uid, [])) & set(user_relevant.get(uid, []))}")

    report = EvaluationReport("training_evaluation")
    metrics = evaluate_recommendations(user_recs, user_relevant, k_values=[5, 10, 20])
    report.add_section("Offline Metrics", metrics)
    report.print()

    # Save vocabs for serving
    save_all_vocabs(vocabs, config.model_dir / "vocabs")

    print("\nTraining pipeline complete!")
    print(f"Models saved to: {config.model_dir}")


if __name__ == "__main__":
    main()
