"""MovieLens 1M retrieval training pipeline.

This follows the ml100k training approach:
chronological positive sequences -> next-item full-softmax training ->
full-sort retrieval evaluation -> Faiss index export.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import random
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml1m.config import config
from ml1m.data_loader import load_ml1m_for_training
from ml100k import indexer as shared_indexer
from ml100k import retrieval_framework as rf
from ml100k.reporter import EvaluationReport
from ml100k.two_tower import TwoTowerModel
from ml100k.vocab_builder import build_vocabs_from_df, save_all_vocabs

rf.config = config
shared_indexer.config = config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.model.epochs)
    parser.add_argument("--max-steps", type=int, default=None, help="Limit batches per epoch for smoke testing.")
    parser.add_argument("--batch-size", type=int, default=config.model.batch_size)
    parser.add_argument("--skip-faiss", action="store_true", help="Skip Faiss index export after training/evaluation.")
    parser.add_argument("--model-suffix", default="", help="Optional suffix for output dirs, e.g. _smoke.")
    args = parser.parse_args()

    set_seed(config.model.seed)
    two_tower_dir = config.model_dir / f"two_tower{args.model_suffix}"
    faiss_dir = config.model_dir / f"faiss{args.model_suffix}"
    vocab_dir = config.model_dir / f"vocabs{args.model_suffix}"

    print("=" * 60)
    print("  MovieLens 1M Retrieval Training Pipeline")
    print("=" * 60)

    print("\n[1/7] Loading ml-1m data...")
    df = load_ml1m_for_training()
    print(f"  Interactions: {len(df)}")
    print(f"  Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")
    print(f"  Positive interactions: {int(df['label'].sum())}")

    print("\n[2/7] Building chronological positive sequences...")
    sequences = rf.build_positive_sequences(df)
    train_samples, val_samples, test_samples = rf.split_sequences(
        sequences,
        eval_target_count=config.model.eval_target_count,
    )
    if not train_samples or not val_samples or not test_samples:
        raise RuntimeError("Not enough positive sequences to train/evaluate retrieval model.")

    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val users: {len(val_samples)}")
    print(f"  Test users: {len(test_samples)}")
    print(f"  Eval targets/user: {config.model.eval_target_count}")

    print("\n[3/7] Building vocabularies and processed artifacts...")
    positive_df = df[df["label"] == 1].copy()
    vocabs = build_vocabs_from_df(positive_df)
    user_vocab = vocabs["user_id"]
    item_vocab = vocabs["item_id"]
    category_vocab = vocabs["category"]

    processed_dir = config.data.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    rf.samples_to_frame(train_samples).to_parquet(processed_dir / "train.parquet", index=False)
    rf.samples_to_frame(val_samples).to_parquet(processed_dir / "val.parquet", index=False)
    rf.samples_to_frame(test_samples).to_parquet(processed_dir / "test.parquet", index=False)
    save_all_vocabs(vocabs, vocab_dir)

    for name, vocab in vocabs.items():
        print(f"  {name}: {len(vocab)} values")

    print("\n[4/7] Creating next-item datasets...")
    train_ds = rf.NextItemDataset(train_samples, user_vocab, item_vocab, config.feature.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=rf.next_item_collate,
        generator=torch.Generator().manual_seed(config.model.seed),
    )
    print(f"  Train batches: {len(train_loader)}")

    print("\n[5/7] Training two-tower retrieval model...")
    two_tower = TwoTowerModel(
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

    item_categories = positive_df.drop_duplicates("item_id").set_index("item_id")["category"].to_dict()
    all_items = sorted(item_vocab.keys(), key=lambda x: item_vocab[x])
    all_items = [iid for iid in all_items if iid not in {"<PAD>", "<UNK>"}]

    trainer = rf.FullSoftmaxRetrievalTrainer(
        model=two_tower,
        all_item_ids=all_items,
        item_vocab=item_vocab,
        item_categories=item_categories,
        category_vocab=category_vocab,
        device=config.device,
        model_dir=two_tower_dir,
        lr=config.model.learning_rate,
        temperature=config.model.temperature,
    )

    trainer.fit(
        train_loader=train_loader,
        val_samples=val_samples,
        user_vocab=user_vocab,
        item_vocab=item_vocab,
        epochs=args.epochs,
        patience=config.model.early_stopping_patience,
        checkpoint_name="two_tower_best.pt",
        max_steps_per_epoch=args.max_steps,
    )

    print("\n[6/7] Evaluating full-sort test ranking...")
    test_metrics = trainer.evaluate(test_samples, user_vocab, item_vocab, k_values=(5, 10, 20))
    report = EvaluationReport("ml-1m Retrieval Evaluation")
    report.add_section("Full-Sort Test Metrics", test_metrics)
    report.print()

    if args.skip_faiss:
        print("\n[7/7] Skipping Faiss index export.")
    else:
        print("\n[7/7] Building Faiss index from best checkpoint...")
        two_tower.eval()
        faiss_indexer = shared_indexer.build_faiss_index_from_model(
            two_tower,
            all_items,
            [item_categories.get(iid, "") for iid in all_items],
            item_vocab,
            category_vocab,
            device=config.device,
        )
        faiss_indexer.save(faiss_dir)
        print(f"  Faiss index saved to {faiss_dir}, ntotal={faiss_indexer.index.ntotal}")

    checkpoint = two_tower_dir / "two_tower_best.pt"
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    print(f"  Saved checkpoint objective: {ckpt.get('training_objective', 'unknown')}")
    print("\nml-1m retrieval training complete.")
    if args.max_steps is not None:
        print(f"Smoke mode used max_steps={args.max_steps}; remove it for full training.")
    print(f"Models saved to: {config.model_dir}")


if __name__ == "__main__":
    main()
