"""Train ml100k Two-Tower pointwise retrieval model."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml100k.config import config
from ml100k.data_loader import load_ml100k_for_training
from ml100k.dataset import TwoTowerDataset, build_user_history, collate_fn
from ml100k.indexer import build_faiss_index_from_model
from ml100k.metrics import diversity_at_k, hit_rate_at_k, precision_at_k, recall_at_k
from ml100k.trainer import Trainer, two_tower_loss_fn
from ml100k.two_tower import TwoTowerModel
from ml100k.vocab_builder import build_vocabs_from_df, save_all_vocabs


def parse_args():
    parser = argparse.ArgumentParser(description="Train ml100k Two-Tower pointwise model")
    parser.add_argument("--epochs", type=int, default=config.model.epochs)
    parser.add_argument("--batch-size", type=int, default=config.model.batch_size)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-faiss", action="store_true")
    parser.add_argument("--model-suffix", type=str, default="")
    return parser.parse_args()


def chronological_split(df):
    train_parts, val_parts, test_parts = [], [], []
    for _, user_df in df.sort_values("timestamp").groupby("user_id", sort=False):
        if len(user_df) < 3:
            train_parts.append(user_df)
            continue
        train_parts.append(user_df.iloc[:-2])
        val_parts.append(user_df.iloc[-2:-1])
        test_parts.append(user_df.iloc[-1:])
    train = df.iloc[0:0].copy() if not train_parts else pd.concat(train_parts, ignore_index=True)
    val = df.iloc[0:0].copy() if not val_parts else pd.concat(val_parts, ignore_index=True)
    test = df.iloc[0:0].copy() if not test_parts else pd.concat(test_parts, ignore_index=True)
    return train, val, test


def limited_loader(loader, max_steps):
    if max_steps is None:
        return loader
    batches = []
    for step, batch in enumerate(loader, start=1):
        batches.append(batch)
        if step >= max_steps:
            break
    return batches


def evaluate_full_sort(model, test, user_history, user_vocab, item_vocab, category_vocab, item_categories, k_values=(5, 10, 20)):
    model.eval()
    idx_to_item = {idx: item for item, idx in item_vocab.items() if idx > 1}
    item_ids = [idx_to_item[idx] for idx in sorted(idx_to_item)]
    item_tensor = torch.tensor([item_vocab[iid] for iid in item_ids], dtype=torch.long, device=config.device)
    cat_tensor = torch.tensor(
        [category_vocab.get(item_categories.get(iid, ""), 0) for iid in item_ids],
        dtype=torch.long,
        device=config.device,
    )
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    metrics.update({f"hit_rate@{k}": [] for k in k_values})
    metrics.update({f"diversity@{k}": [] for k in k_values})

    with torch.no_grad():
        item_emb = model.get_item_embeddings(item_tensor, cat_tensor)
        for uid, user_df in test.groupby("user_id"):
            relevant = user_df[user_df["label"] == 1]["item_id"].astype(str).tolist()
            if not relevant:
                continue
            hist = user_history.get(str(uid), [])[-config.feature.max_seq_len:]
            hist_padded = hist + [0] * (config.feature.max_seq_len - len(hist))
            mask = [1.0] * len(hist) + [0.0] * (config.feature.max_seq_len - len(hist))
            user_emb = model.get_user_embeddings(
                torch.tensor([user_vocab.get(str(uid), 1)], dtype=torch.long, device=config.device),
                torch.tensor([hist_padded], dtype=torch.long, device=config.device),
                torch.tensor([mask], dtype=torch.float32, device=config.device),
            )
            scores = torch.matmul(user_emb, item_emb.t()).squeeze(0).cpu().numpy()
            seen = {idx_to_item[i] for i in hist if i in idx_to_item}
            ranked = [
                item_ids[i]
                for i in np.argsort(-scores)
                if item_ids[i] not in seen
            ][: max(k_values)]
            for k in k_values:
                metrics[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
                metrics[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
                metrics[f"hit_rate@{k}"].append(hit_rate_at_k(ranked, relevant, k))
                metrics[f"diversity@{k}"].append(diversity_at_k(ranked, item_categories, k))
    return {key: float(np.mean(vals)) if vals else 0.0 for key, vals in metrics.items()}


def main():
    args = parse_args()
    torch.manual_seed(config.model.seed)
    np.random.seed(config.model.seed)

    print("Training ml100k: Two-Tower + Pointwise")
    df = load_ml100k_for_training()
    train, val, test = chronological_split(df)
    print(f"  Split: train={len(train)}, val={len(val)}, test={len(test)}")

    vocabs = build_vocabs_from_df(train)
    user_vocab = vocabs["user_id"]
    item_vocab = vocabs["item_id"]
    category_vocab = vocabs["category"]
    user_history = build_user_history(train, item_vocab, positive_only=True)

    train_ds = TwoTowerDataset(
        train,
        user_vocab,
        item_vocab,
        category_vocab,
        user_history,
        config.feature.max_seq_len,
        use_causal_history=True,
    )
    val_ds = TwoTowerDataset(val, user_vocab, item_vocab, category_vocab, user_history, config.feature.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TwoTowerModel(
        user_vocab_size=len(user_vocab),
        item_vocab_size=len(item_vocab),
        category_vocab_size=len(category_vocab),
        embedding_dim=config.model.embedding_dim,
        category_embedding_dim=config.model.category_embedding_dim,
        hidden_units=tuple(config.model.hidden_units),
        temperature=config.model.temperature,
        max_seq_len=config.feature.max_seq_len,
        dropout=config.model.dropout,
    )
    model_dir = config.model_dir / "two_tower"
    trainer = Trainer(model, lr=config.model.learning_rate, model_dir=model_dir)
    checkpoint = f"two_tower_best{args.model_suffix}.pt"
    trainer.fit(
        limited_loader(train_loader, args.max_steps),
        val_loader,
        two_tower_loss_fn,
        epochs=args.epochs,
        patience=config.model.early_stopping_patience,
        checkpoint_name=checkpoint,
    )
    trainer.save(checkpoint)

    item_categories = dict(zip(train["item_id"].astype(str), train["category"].astype(str)))
    metrics = evaluate_full_sort(model, test, user_history, user_vocab, item_vocab, category_vocab, item_categories)
    print("  Offline metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.6f}")

    save_all_vocabs(vocabs, config.model_dir / "vocabs")
    if not args.skip_faiss:
        item_ids = sorted(train["item_id"].astype(str).unique().tolist())
        categories = [item_categories.get(iid, "") for iid in item_ids]
        indexer = build_faiss_index_from_model(model, item_ids, categories, item_vocab, category_vocab, config.device)
        indexer.save(config.model_dir / "faiss")
        print(f"  Faiss index saved to {config.model_dir / 'faiss'}")

    print(f"ml100k training complete. Model saved to {model_dir / checkpoint}")


if __name__ == "__main__":
    main()
