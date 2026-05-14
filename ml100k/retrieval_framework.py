"""Full-sort retrieval training framework for MovieLens 100K.

Trains a two-tower model as next-item retrieval: given a user's chronological
positive history, predict the next positive item from the complete item catalog.
Uses full-softmax cross-entropy with label smoothing.
"""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import config
from .metrics import diversity_at_k, hit_rate_at_k, precision_at_k, recall_at_k


@dataclass
class RetrievalSample:
    user_id: str
    history: List[str]
    target: str


@dataclass
class RetrievalEvalSample:
    user_id: str
    history: List[str]
    targets: List[str]


def relevant_items(sample: Union[RetrievalSample, RetrievalEvalSample]) -> List[str]:
    if hasattr(sample, "targets"):
        return list(sample.targets)
    return [sample.target]


def build_positive_sequences(df: pd.DataFrame) -> Dict[str, List[str]]:
    positives = df[df["label"] == 1].sort_values(["user_id", "timestamp"])
    return positives.groupby("user_id")["item_id"].apply(list).to_dict()


def split_sequences(
    sequences: Dict[str, List[str]],
    min_train_positives: int = 1,
    eval_target_count: int = 1,
) -> Tuple[List[RetrievalSample], List[RetrievalEvalSample], List[RetrievalEvalSample]]:
    train_samples: List[RetrievalSample] = []
    val_samples: List[RetrievalEvalSample] = []
    test_samples: List[RetrievalEvalSample] = []
    eval_target_count = max(1, eval_target_count)

    for uid, items in sequences.items():
        deduped = list(dict.fromkeys(items))
        holdout_count = eval_target_count * 2
        if len(deduped) < min_train_positives + holdout_count:
            continue

        train_items = deduped[:-holdout_count]
        val_targets = deduped[-holdout_count:-eval_target_count]
        test_targets = deduped[-eval_target_count:]

        for i in range(1, len(train_items)):
            train_samples.append(RetrievalSample(uid, train_items[:i], train_items[i]))

        if train_items:
            val_samples.append(RetrievalEvalSample(uid, train_items, val_targets))
            test_samples.append(RetrievalEvalSample(uid, train_items + val_targets, test_targets))

    return train_samples, val_samples, test_samples


class NextItemDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[RetrievalSample],
        user_vocab: Dict[str, int],
        item_vocab: Dict[str, int],
        max_seq_len: int,
    ):
        self.samples = list(samples)
        self.user_vocab = user_vocab
        self.item_vocab = item_vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        full_history = [self.item_vocab.get(iid, 0) for iid in sample.history]
        full_history = [iid for iid in full_history if iid != 0]
        if len(full_history) > 2 and random.random() < config.model.history_crop_prob:
            min_len = max(1, len(full_history) // 4)
            keep = random.randint(min_len, len(full_history))
            full_history = full_history[-keep:]
        if len(full_history) > 2 and config.model.history_dropout_prob > 0:
            kept = [iid for iid in full_history if random.random() >= config.model.history_dropout_prob]
            full_history = kept or [full_history[-1]]
        history = full_history[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(history)
        return {
            "user_id": torch.tensor(self.user_vocab.get(sample.user_id, 1), dtype=torch.long),
            "history": torch.tensor(history + [0] * pad_len, dtype=torch.long),
            "history_mask": torch.tensor([1.0] * len(history) + [0.0] * pad_len, dtype=torch.float32),
            "target": torch.tensor(self.item_vocab.get(sample.target, 0), dtype=torch.long),
        }


def next_item_collate(batch: List[dict]) -> dict:
    return {
        "user_id": torch.stack([b["user_id"] for b in batch]),
        "history": torch.stack([b["history"] for b in batch]),
        "history_mask": torch.stack([b["history_mask"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }


def samples_to_frame(samples: Iterable[RetrievalSample]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        for item_id in sample.history:
            rows.append({"user_id": sample.user_id, "item_id": item_id})
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


class FullSoftmaxRetrievalTrainer:
    def __init__(
        self,
        model,
        all_item_ids: Sequence[str],
        item_vocab: Dict[str, int],
        item_categories: Dict[str, str],
        category_vocab: Dict[str, int],
        device: str,
        model_dir: Path,
        lr: float,
        temperature: float,
    ):
        self.model = model.to(device)
        self.all_item_ids = list(all_item_ids)
        self.item_vocab = item_vocab
        self.item_categories = item_categories
        self.category_vocab = category_vocab
        self.device = device
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.model.weight_decay)
        self.temperature = temperature

        self.item_tensor = torch.tensor([item_vocab[iid] for iid in self.all_item_ids], dtype=torch.long, device=device)
        self.category_tensor = torch.tensor(
            [category_vocab.get(item_categories.get(iid, ""), 0) for iid in self.all_item_ids],
            dtype=torch.long,
            device=device,
        )
        self.vocab_idx_to_col = {item_vocab[iid]: col for col, iid in enumerate(self.all_item_ids)}
        max_vocab_idx = max(item_vocab.values())
        self.vocab_col_tensor = torch.full((max_vocab_idx + 1,), -1, dtype=torch.long, device=device)
        for vocab_idx, col in self.vocab_idx_to_col.items():
            self.vocab_col_tensor[vocab_idx] = col

    def _target_columns(self, targets: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.vocab_idx_to_col[int(t)] for t in targets.cpu().tolist()], dtype=torch.long, device=self.device)

    def _logits(self, batch: dict) -> torch.Tensor:
        user_emb = self.model.get_user_embeddings(batch["user_id"], batch["history"], batch["history_mask"])
        item_emb = self.model.get_item_embeddings(self.item_tensor, self.category_tensor)
        return torch.matmul(user_emb, item_emb.t()) / self.temperature

    def _mask_history_logits(self, logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Avoid treating known positive history items as full-softmax negatives."""
        valid = (history > 0) & (history < self.vocab_col_tensor.numel())
        if not valid.any():
            return logits
        history_cols = torch.full_like(history, -1)
        history_cols[valid] = self.vocab_col_tensor[history[valid]]
        mask = torch.zeros_like(logits, dtype=torch.bool)
        row_idx = torch.arange(history.size(0), device=history.device).unsqueeze(1).expand_as(history)
        valid_cols = history_cols >= 0
        mask[row_idx[valid_cols], history_cols[valid_cols]] = True
        return logits.masked_fill(mask, -1e9)

    def _hard_negative_bpr_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        k = min(config.model.hard_negative_count, logits.size(1) - 1)
        if k <= 0:
            return logits.new_tensor(0.0)
        pos_scores = logits.gather(1, targets.unsqueeze(1))
        negative_logits = logits.clone()
        negative_logits.scatter_(1, targets.unsqueeze(1), -1e9)
        hard_negatives = negative_logits.topk(k=k, dim=1).values
        return -F.logsigmoid(pos_scores - hard_negatives).mean()

    def train_epoch(self, loader: DataLoader, max_steps: int = None) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            targets = self._target_columns(batch["target"])
            logits = self._mask_history_logits(self._logits(batch), batch["history"])
            ce_loss = F.cross_entropy(logits, targets, label_smoothing=config.model.label_smoothing)
            bpr_loss = self._hard_negative_bpr_loss(logits, targets)
            loss = ce_loss + config.model.bpr_loss_weight * bpr_loss
            self.optimizer.zero_grad()
            loss.backward()
            if config.model.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.model.grad_clip_norm)
            self.optimizer.step()
            total_loss += loss.item()
            steps += 1
            if max_steps is not None and step >= max_steps:
                break
        return total_loss / max(steps, 1)

    @torch.no_grad()
    def evaluate(self, samples: Sequence[Union[RetrievalSample, RetrievalEvalSample]], user_vocab: Dict[str, int], item_vocab: Dict[str, int], k_values=(5, 10, 20)) -> Dict[str, float]:
        self.model.eval()
        metrics = {f"recall@{k}": [] for k in k_values}
        metrics.update({f"precision@{k}": [] for k in k_values})
        metrics.update({f"hit_rate@{k}": [] for k in k_values})
        metrics.update({f"diversity@{k}": [] for k in k_values})

        item_emb = self.model.get_item_embeddings(self.item_tensor, self.category_tensor)
        for sample in samples:
            history_ids = [item_vocab.get(iid, 0) for iid in sample.history[-config.feature.max_seq_len:]]
            history_ids = [iid for iid in history_ids if iid != 0]
            relevant = [iid for iid in relevant_items(sample) if iid in item_vocab]
            if not history_ids or not relevant:
                continue

            pad_len = config.feature.max_seq_len - len(history_ids)
            user_id = torch.tensor([user_vocab.get(sample.user_id, 1)], dtype=torch.long, device=self.device)
            history = torch.tensor([history_ids + [0] * pad_len], dtype=torch.long, device=self.device)
            mask = torch.tensor([[1.0] * len(history_ids) + [0.0] * pad_len], dtype=torch.float32, device=self.device)
            user_emb = self.model.get_user_embeddings(user_id, history, mask)
            scores = torch.matmul(user_emb, item_emb.t()).squeeze(0).cpu().numpy()

            seen = set(sample.history)
            for col, item_id in enumerate(self.all_item_ids):
                if item_id in seen:
                    scores[col] = -np.inf

            ranked_cols = np.argsort(-scores)[: max(k_values)]
            ranked = [self.all_item_ids[col] for col in ranked_cols]
            for k in k_values:
                metrics[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
                metrics[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
                metrics[f"hit_rate@{k}"].append(hit_rate_at_k(ranked, relevant, k))
                metrics[f"diversity@{k}"].append(diversity_at_k(ranked, self.item_categories, k))

        return {key: float(np.mean(vals)) if vals else 0.0 for key, vals in metrics.items()}

    def fit(
        self,
        train_loader: DataLoader,
        val_samples: Sequence[Union[RetrievalSample, RetrievalEvalSample]],
        user_vocab: Dict[str, int],
        item_vocab: Dict[str, int],
        epochs: int,
        patience: int,
        checkpoint_name: str = "two_tower_best.pt",
        max_steps_per_epoch: int = None,
    ) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_recall@10": [], "val_precision@10": [], "val_hit_rate@10": [], "val_diversity@10": []}
        best_metric = -1.0
        bad_epochs = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, max_steps=max_steps_per_epoch)
            scheduler.step()
            val_metrics = self.evaluate(val_samples, user_vocab, item_vocab, k_values=(5, 10, 20))
            valid_metric = val_metrics["precision@10"]
            history["train_loss"].append(train_loss)
            history["val_recall@10"].append(val_metrics["recall@10"])
            history["val_precision@10"].append(valid_metric)
            history["val_hit_rate@10"].append(val_metrics["hit_rate@10"])
            history["val_diversity@10"].append(val_metrics["diversity@10"])

            marker = ""
            if valid_metric > best_metric:
                best_metric = valid_metric
                bad_epochs = 0
                self.save(checkpoint_name, history)
                marker = " [BEST]"
            else:
                bad_epochs += 1

            print(
                f"Epoch {epoch:3d}/{epochs} | train_loss: {train_loss:.4f} | "
                f"val_precision@10: {valid_metric:.4f} | "
                f"val_recall@10: {val_metrics['recall@10']:.4f} | "
                f"val_hit_rate@10: {val_metrics['hit_rate@10']:.4f} | "
                f"val_diversity@10: {val_metrics['diversity@10']:.4f} | "
                f"best_precision@10: {best_metric:.4f}{marker}"
            )

            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.load(checkpoint_name)
        return history

    def save(self, name: str, history: Dict[str, List[float]]) -> Path:
        path = self.model_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics_history": history,
                "training_objective": "full_softmax_next_item_retrieval",
            },
            path,
        )
        return path

    def load(self, name: str) -> None:
        path = self.model_dir / name
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        print(f"Loaded checkpoint from {path}")
