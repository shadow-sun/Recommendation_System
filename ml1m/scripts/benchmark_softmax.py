"""Benchmark full-softmax vs sampled-softmax training cost on MovieLens 1M.

The benchmark keeps the same ml1m data pipeline, next-item samples, Two-Tower
model shape, optimizer, batch size, and number of measured steps. It changes
only the softmax candidate set:

* full-softmax: every catalog item is a class.
* sampled-softmax: the positive target plus sampled global negatives.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml1m.config import config
from ml1m.data_loader import load_ml1m_for_training
from ml100k import retrieval_framework as rf
from ml100k.two_tower import TwoTowerModel
from ml100k.vocab_builder import build_vocabs_from_df

rf.config = config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(batch_size: int, steps: int):
    df = load_ml1m_for_training()
    sequences = rf.build_positive_sequences(df)
    train_samples, _, _ = rf.split_sequences(
        sequences,
        eval_target_count=config.model.eval_target_count,
    )
    positive_df = df[df["label"] == 1].copy()
    vocabs = build_vocabs_from_df(positive_df)
    item_categories = positive_df.drop_duplicates("item_id").set_index("item_id")["category"].to_dict()
    all_items = sorted(vocabs["item_id"].keys(), key=lambda x: vocabs["item_id"][x])
    all_items = [iid for iid in all_items if iid not in {"<PAD>", "<UNK>"}]

    full_train_sample_count = len(train_samples)
    # Keep enough samples for warmup + measured steps without timing a full epoch.
    needed = max((steps + 4) * batch_size, batch_size)
    train_samples = train_samples[: min(len(train_samples), needed)]
    ds = rf.NextItemDataset(
        train_samples,
        vocabs["user_id"],
        vocabs["item_id"],
        config.feature.max_seq_len,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=rf.next_item_collate)
    return loader, vocabs, item_categories, all_items, len(df), len(train_samples), full_train_sample_count


def make_model(vocabs):
    return TwoTowerModel(
        user_vocab_size=len(vocabs["user_id"]),
        item_vocab_size=len(vocabs["item_id"]),
        category_vocab_size=len(vocabs["category"]),
        embedding_dim=config.model.embedding_dim,
        category_embedding_dim=config.model.category_embedding_dim,
        hidden_units=tuple(config.model.hidden_units),
        temperature=config.model.temperature,
        max_seq_len=config.feature.max_seq_len,
        dropout=config.model.dropout,
    )


def sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def full_softmax_step(model, batch, item_tensor, category_tensor, vocab_col_tensor, optimizer, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    user_emb = model.get_user_embeddings(batch["user_id"], batch["history"], batch["history_mask"])
    item_emb = model.get_item_embeddings(item_tensor, category_tensor)
    logits = torch.matmul(user_emb, item_emb.t()) / config.model.temperature

    valid = (batch["history"] > 0) & (batch["history"] < vocab_col_tensor.numel())
    if valid.any():
        history_cols = torch.full_like(batch["history"], -1)
        history_cols[valid] = vocab_col_tensor[batch["history"][valid]]
        mask = torch.zeros_like(logits, dtype=torch.bool)
        rows = torch.arange(batch["history"].size(0), device=device).unsqueeze(1).expand_as(batch["history"])
        valid_cols = history_cols >= 0
        mask[rows[valid_cols], history_cols[valid_cols]] = True
        logits = logits.masked_fill(mask, -1e9)

    targets = vocab_col_tensor[batch["target"]]
    loss = F.cross_entropy(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def sampled_softmax_step(
    model,
    batch,
    item_tensor,
    category_tensor,
    vocab_col_tensor,
    sampler_probs,
    negatives: int,
    optimizer,
    device,
):
    batch = {k: v.to(device) for k, v in batch.items()}
    batch_size = batch["target"].size(0)
    target_cols = vocab_col_tensor[batch["target"]]
    sampled_cols = torch.multinomial(sampler_probs, num_samples=negatives, replacement=False)
    candidate_cols = torch.cat([target_cols, sampled_cols], dim=0)
    candidate_cols = torch.unique(candidate_cols, sorted=True)
    col_to_pos = torch.full((item_tensor.numel(),), -1, dtype=torch.long, device=device)
    col_to_pos[candidate_cols] = torch.arange(candidate_cols.numel(), device=device)
    targets = col_to_pos[target_cols]
    candidate_items = item_tensor[candidate_cols]
    candidate_categories = category_tensor[candidate_cols]

    user_emb = model.get_user_embeddings(batch["user_id"], batch["history"], batch["history_mask"])
    item_emb = model.get_item_embeddings(candidate_items, candidate_categories)
    logits = torch.matmul(user_emb, item_emb.t()) / config.model.temperature
    loss = F.cross_entropy(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item()), int(candidate_cols.numel()), batch_size


def time_loop(name, step_fn, loader, steps: int, warmup: int, device: str):
    losses = []
    timings = []
    batches = iter(loader)
    for i in range(warmup + steps):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        sync_if_needed(device)
        start = time.perf_counter()
        result = step_fn(batch)
        sync_if_needed(device)
        elapsed = time.perf_counter() - start
        loss = result[0] if isinstance(result, tuple) else result
        if i >= warmup:
            timings.append(elapsed)
            losses.append(loss)
    total = sum(timings)
    return {
        "name": name,
        "steps": steps,
        "total_seconds": total,
        "seconds_per_step": total / max(steps, 1),
        "samples_per_second": (steps * loader.batch_size) / max(total, 1e-9),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sampled-negatives", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("ml1m/reports/softmax_benchmark.json"))
    args = parser.parse_args()

    set_seed(config.model.seed)
    device = config.device
    loader, vocabs, item_categories, all_items, interactions, samples, full_train_samples = build_loader(
        args.batch_size,
        args.steps,
    )

    item_vocab = vocabs["item_id"]
    category_vocab = vocabs["category"]
    item_tensor = torch.tensor([item_vocab[iid] for iid in all_items], dtype=torch.long, device=device)
    category_tensor = torch.tensor(
        [category_vocab.get(item_categories.get(iid, ""), 0) for iid in all_items],
        dtype=torch.long,
        device=device,
    )
    vocab_idx_to_col = {item_vocab[iid]: col for col, iid in enumerate(all_items)}
    vocab_col_tensor = torch.full((max(item_vocab.values()) + 1,), -1, dtype=torch.long, device=device)
    for vocab_idx, col in vocab_idx_to_col.items():
        vocab_col_tensor[vocab_idx] = col

    sampler_probs = torch.ones(len(all_items), dtype=torch.float32, device=device)
    sampler_probs = sampler_probs / sampler_probs.sum()

    full_model = make_model(vocabs).to(device)
    sampled_model = make_model(vocabs).to(device)
    sampled_model.load_state_dict(full_model.state_dict())
    full_opt = torch.optim.AdamW(full_model.parameters(), lr=config.model.learning_rate)
    sampled_opt = torch.optim.AdamW(sampled_model.parameters(), lr=config.model.learning_rate)

    full = time_loop(
        "full_softmax",
        lambda batch: full_softmax_step(
            full_model,
            batch,
            item_tensor,
            category_tensor,
            vocab_col_tensor,
            full_opt,
            device,
        ),
        loader,
        args.steps,
        args.warmup,
        device,
    )
    sampled = time_loop(
        f"sampled_softmax_{args.sampled_negatives}",
        lambda batch: sampled_softmax_step(
            sampled_model,
            batch,
            item_tensor,
            category_tensor,
            vocab_col_tensor,
            sampler_probs,
            args.sampled_negatives,
            sampled_opt,
            device,
        ),
        loader,
        args.steps,
        args.warmup,
        device,
    )

    full_epoch_steps = int(np.ceil(full_train_samples / args.batch_size))
    result = {
        "dataset": "ml-1m",
        "device": device,
        "interactions_after_filter": interactions,
        "train_samples_used": samples,
        "full_train_samples": full_train_samples,
        "full_epoch_steps": full_epoch_steps,
        "catalog_items": len(all_items),
        "batch_size": args.batch_size,
        "measured_steps": args.steps,
        "warmup_steps": args.warmup,
        "sampled_negatives": args.sampled_negatives,
        "full_softmax": full,
        "sampled_softmax": sampled,
        "sampled_speedup": full["seconds_per_step"] / sampled["seconds_per_step"],
        "estimated_full_epoch_seconds": full["seconds_per_step"] * full_epoch_steps,
        "estimated_sampled_epoch_seconds": sampled["seconds_per_step"] * full_epoch_steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
