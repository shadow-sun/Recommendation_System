"""Sampled-softmax retrieval training framework for KuaiLive.

Trains a two-tower model as next-item retrieval: given a user's chronological
positive history, predict the next positive item using in-batch negatives +
randomly sampled negatives. Uses sampled softmax cross-entropy with label
smoothing and hard-negative BPR loss.
"""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

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


@dataclass
class UserTrainSequence:
    user_id: str
    train_items: List[str]


def relevant_items(sample: Union[RetrievalSample, RetrievalEvalSample]) -> List[str]:
    if hasattr(sample, "targets"):
        return list(sample.targets)
    return [sample.target]


def build_positive_sequences(df: pd.DataFrame) -> Dict[str, List[str]]:
    positives = df[df["label"] == 1].sort_values(["user_id", "timestamp"])
    return positives.groupby("user_id")["item_id"].apply(list).to_dict()


def build_user_negative_items(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build per-user explicit negative item lists from KuaiLive negative feedback."""
    negatives = df[df["label"] == 0].sort_values(["user_id", "timestamp"])
    if negatives.empty:
        return {}
    return negatives.groupby("user_id")["item_id"].apply(lambda s: list(dict.fromkeys(s))).to_dict()


def build_item_sampling_weights(
    df: pd.DataFrame,
    all_item_ids: Sequence[str],
    alpha: float = 0.75,
) -> np.ndarray:
    """Popularity-smoothed item sampling weights for large-catalog negatives."""
    counts = df[df["label"] == 1]["item_id"].value_counts()
    weights = np.array([float(counts.get(iid, 1.0)) for iid in all_item_ids], dtype=np.float64)
    weights = np.power(np.maximum(weights, 1.0), alpha)
    weights = weights / weights.sum()
    return weights.astype(np.float32)


def split_sequences(
    sequences: Dict[str, List[str]],
    min_train_positives: int = 1,
    eval_target_count: int = 1,
    max_train_samples_per_user: Optional[int] = None,
    seed: int = 2026,
) -> Tuple[List[RetrievalSample], List[RetrievalEvalSample], List[RetrievalEvalSample]]:
    train_samples: List[RetrievalSample] = []
    val_samples: List[RetrievalEvalSample] = []
    test_samples: List[RetrievalEvalSample] = []
    eval_target_count = max(1, eval_target_count)
    rng = random.Random(seed)

    for uid, items in sequences.items():
        deduped = list(dict.fromkeys(items))
        holdout_count = eval_target_count * 2
        if len(deduped) < min_train_positives + holdout_count:
            continue

        train_items = deduped[:-holdout_count]
        val_targets = deduped[-holdout_count:-eval_target_count]
        test_targets = deduped[-eval_target_count:]

        train_positions = list(range(1, len(train_items)))
        if max_train_samples_per_user and len(train_positions) > max_train_samples_per_user:
            recent_count = max(1, max_train_samples_per_user // 2)
            recent = train_positions[-recent_count:]
            earlier = train_positions[:-recent_count]
            sampled = rng.sample(earlier, max_train_samples_per_user - recent_count)
            train_positions = sorted(sampled + recent)

        for i in train_positions:
            train_samples.append(RetrievalSample(uid, train_items[:i], train_items[i]))

        if train_items:
            val_samples.append(RetrievalEvalSample(uid, train_items, val_targets))
            test_samples.append(RetrievalEvalSample(uid, train_items + val_targets, test_targets))

    return train_samples, val_samples, test_samples


def split_user_sequences(
    sequences: Dict[str, List[str]],
    min_train_positives: int = 1,
    eval_target_count: int = 1,
) -> Tuple[List[UserTrainSequence], List[RetrievalEvalSample], List[RetrievalEvalSample]]:
    """Split positive sequences while keeping train data as compact per-user sequences."""
    train_sequences: List[UserTrainSequence] = []
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
        if len(train_items) < 2:
            continue

        train_sequences.append(UserTrainSequence(uid, train_items))
        val_samples.append(RetrievalEvalSample(uid, train_items, val_targets))
        test_samples.append(RetrievalEvalSample(uid, train_items + val_targets, test_targets))

    return train_sequences, val_samples, test_samples


def partition_train_sequences(
    train_sequences: Sequence[UserTrainSequence],
    stage_count: int,
) -> List[List[UserTrainSequence]]:
    """Partition users into deterministic stages by recency of their last train item."""
    stage_count = max(1, int(stage_count))
    ordered = sorted(train_sequences, key=lambda s: s.train_items[-1])
    stages = [[] for _ in range(stage_count)]
    for idx, sequence in enumerate(ordered):
        stage_idx = min(stage_count - 1, idx * stage_count // max(len(ordered), 1))
        stages[stage_idx].append(sequence)
    return [stage for stage in stages if stage]


def build_replay_samples(
    seen_sequences: Sequence[UserTrainSequence],
    replay_ratio: float,
    max_samples: int,
    max_train_samples_per_user: Optional[int],
    seed: int,
) -> List[RetrievalSample]:
    """Reservoir-style replay from previously seen user histories."""
    if replay_ratio <= 0 or not seen_sequences:
        return []
    rng = random.Random(seed)
    per_user_limit = max_train_samples_per_user
    target_total = min(
        max_samples,
        max(1, int(sum(max(0, len(s.train_items) - 1) for s in seen_sequences) * replay_ratio)),
    )
    samples: List[RetrievalSample] = []

    shuffled = list(seen_sequences)
    rng.shuffle(shuffled)
    for sequence in shuffled:
        positions = list(range(1, len(sequence.train_items)))
        if per_user_limit and len(positions) > per_user_limit:
            positions = sorted(rng.sample(positions, per_user_limit))
        if positions:
            pos = rng.choice(positions)
            samples.append(RetrievalSample(sequence.user_id, sequence.train_items[:pos], sequence.train_items[pos]))
        if len(samples) >= target_total:
            break

    return samples


def samples_to_frame(samples: Iterable[RetrievalSample]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        for item_id in sample.history:
            rows.append({"user_id": sample.user_id, "item_id": item_id})
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


class NextItemDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[RetrievalSample],
        user_vocab: Dict[str, int],
        item_vocab: Dict[str, int],
        category_vocab: Dict[str, int],
        item_categories: Dict[str, str],
        max_seq_len: int,
        user_negative_items: Optional[Dict[str, List[str]]] = None,
        explicit_negative_count: int = 0,
    ):
        self.samples = list(samples)
        self.user_vocab = user_vocab
        self.item_vocab = item_vocab
        self.category_vocab = category_vocab
        self.item_categories = item_categories
        self.max_seq_len = max_seq_len
        self.user_negative_items = user_negative_items or {}
        self.explicit_negative_count = max(0, explicit_negative_count)

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

        target_idx = self.item_vocab.get(sample.target, 0)
        target_cat = self.category_vocab.get(
            self.item_categories.get(sample.target, ""), 0
        )

        explicit_negatives = self._sample_explicit_negatives(sample, full_history)

        return {
            "user_id": torch.tensor(self.user_vocab.get(sample.user_id, 1), dtype=torch.long),
            "history": torch.tensor(history + [0] * pad_len, dtype=torch.long),
            "history_mask": torch.tensor([1.0] * len(history) + [0.0] * pad_len, dtype=torch.float32),
            "target": torch.tensor(target_idx, dtype=torch.long),
            "target_category": torch.tensor(target_cat, dtype=torch.long),
            "explicit_negative": torch.tensor([x[0] for x in explicit_negatives], dtype=torch.long),
            "explicit_negative_category": torch.tensor([x[1] for x in explicit_negatives], dtype=torch.long),
            "explicit_negative_mask": torch.tensor([x[2] for x in explicit_negatives], dtype=torch.float32),
        }

    def _sample_explicit_negatives(self, sample: RetrievalSample, full_history: List[int]) -> List[Tuple[int, int, float]]:
        if self.explicit_negative_count <= 0:
            return []

        blocked = set(full_history)
        blocked.add(self.item_vocab.get(sample.target, 0))
        candidates = []
        for item_id in self.user_negative_items.get(sample.user_id, []):
            item_idx = self.item_vocab.get(item_id, 0)
            if item_idx == 0 or item_idx in blocked:
                continue
            cat_idx = self.category_vocab.get(self.item_categories.get(item_id, ""), 0)
            candidates.append((item_idx, cat_idx, 1.0))

        if len(candidates) > self.explicit_negative_count:
            candidates = random.sample(candidates, self.explicit_negative_count)

        pad_count = self.explicit_negative_count - len(candidates)
        if pad_count > 0:
            candidates.extend([(0, 0, 0.0)] * pad_count)
        return candidates


class StreamingNextItemDataset(IterableDataset):
    """Stream next-item samples from compact per-user sequences plus replay samples."""

    def __init__(
        self,
        train_sequences: Sequence[UserTrainSequence],
        user_vocab: Dict[str, int],
        item_vocab: Dict[str, int],
        category_vocab: Dict[str, int],
        item_categories: Dict[str, str],
        max_seq_len: int,
        user_negative_items: Optional[Dict[str, List[str]]] = None,
        explicit_negative_count: int = 0,
        max_train_samples_per_user: Optional[int] = None,
        replay_samples: Optional[Sequence[RetrievalSample]] = None,
        seed: int = 2026,
        shuffle: bool = True,
    ):
        self.train_sequences = list(train_sequences)
        self.replay_samples = list(replay_samples or [])
        self.user_vocab = user_vocab
        self.item_vocab = item_vocab
        self.category_vocab = category_vocab
        self.item_categories = item_categories
        self.max_seq_len = max_seq_len
        self.user_negative_items = user_negative_items or {}
        self.explicit_negative_count = max(0, explicit_negative_count)
        self.max_train_samples_per_user = max_train_samples_per_user
        self.seed = seed
        self.shuffle = shuffle

    def __iter__(self):
        rng = random.Random(self.seed)
        samples = list(self._iter_sequence_samples(rng)) + list(self.replay_samples)
        if self.shuffle:
            rng.shuffle(samples)
        encoder = NextItemDataset(
            samples,
            self.user_vocab,
            self.item_vocab,
            self.category_vocab,
            self.item_categories,
            self.max_seq_len,
            user_negative_items=self.user_negative_items,
            explicit_negative_count=self.explicit_negative_count,
        )
        for idx in range(len(encoder)):
            yield encoder[idx]

    def __len__(self) -> int:
        total = len(self.replay_samples)
        for sequence in self.train_sequences:
            count = max(0, len(sequence.train_items) - 1)
            if self.max_train_samples_per_user:
                count = min(count, self.max_train_samples_per_user)
            total += count
        return total

    def _iter_sequence_samples(self, rng: random.Random) -> Iterable[RetrievalSample]:
        sequences = list(self.train_sequences)
        if self.shuffle:
            rng.shuffle(sequences)

        for sequence in sequences:
            positions = list(range(1, len(sequence.train_items)))
            if self.max_train_samples_per_user and len(positions) > self.max_train_samples_per_user:
                recent_count = max(1, self.max_train_samples_per_user // 2)
                recent = positions[-recent_count:]
                earlier = positions[:-recent_count]
                sampled = rng.sample(earlier, self.max_train_samples_per_user - recent_count)
                positions = sorted(sampled + recent)
            if self.shuffle:
                rng.shuffle(positions)

            for pos in positions:
                yield RetrievalSample(sequence.user_id, sequence.train_items[:pos], sequence.train_items[pos])


def next_item_collate(batch: List[dict]) -> dict:
    return {
        "user_id": torch.stack([b["user_id"] for b in batch]),
        "history": torch.stack([b["history"] for b in batch]),
        "history_mask": torch.stack([b["history_mask"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "target_category": torch.stack([b["target_category"] for b in batch]),
        "explicit_negative": torch.stack([b["explicit_negative"] for b in batch]),
        "explicit_negative_category": torch.stack([b["explicit_negative_category"] for b in batch]),
        "explicit_negative_mask": torch.stack([b["explicit_negative_mask"] for b in batch]),
    }


class SampledSoftmaxTrainer:
    """Two-tower trainer using in-batch, popularity-sampled, and explicit negatives."""

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
        item_sampling_weights: Optional[Sequence[float]] = None,
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

        # Pre-computed tensors for random negative sampling
        self.item_tensor = torch.tensor(
            [item_vocab[iid] for iid in self.all_item_ids], dtype=torch.long, device=device
        )
        self.category_tensor = torch.tensor(
            [category_vocab.get(item_categories.get(iid, ""), 0) for iid in self.all_item_ids],
            dtype=torch.long, device=device,
        )
        self.num_items = len(self.all_item_ids)
        self.num_negatives = config.model.num_sampled_negatives
        self.all_item_ids_np = np.array(self.all_item_ids)
        if item_sampling_weights is None:
            item_sampling_weights = np.full(self.num_items, 1.0 / self.num_items, dtype=np.float32)
        weights = torch.tensor(item_sampling_weights, dtype=torch.float32, device=device)
        self.item_sampling_probs = weights / weights.sum()
        self.item_log_probs = torch.log(self.item_sampling_probs.clamp_min(1e-12))

    def _mask_blocked_candidates(
        self,
        logits: torch.Tensor,
        history: torch.Tensor,
        targets: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> torch.Tensor:
        """Mask known positives and duplicate target candidates for each user."""
        blocked = torch.cat([history, targets.unsqueeze(1)], dim=1)
        valid_blocked = blocked > 0
        mask = (candidate_items.view(1, -1, 1) == blocked.unsqueeze(1)) & valid_blocked.unsqueeze(1)
        mask = mask.any(dim=-1)
        row_idx = torch.arange(targets.size(0), device=logits.device)
        mask[row_idx, row_idx] = False
        return logits.masked_fill(mask, -1e9)

    def _sample_random_negatives(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled_cols = torch.multinomial(
            self.item_sampling_probs,
            num_samples=self.num_negatives,
            replacement=True,
        )
        neg_item_ids = self.item_tensor[sampled_cols]
        neg_cat_ids = self.category_tensor[sampled_cols]
        neg_log_probs = self.item_log_probs[sampled_cols]
        return neg_item_ids, neg_cat_ids, neg_log_probs

    def train_epoch(self, loader: DataLoader, max_steps: int = None) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0

        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            B = batch["user_id"].size(0)

            # User embeddings
            user_emb = self.model.get_user_embeddings(
                batch["user_id"], batch["history"], batch["history_mask"]
            )

            # Positive item embeddings
            pos_item_emb = self.model.get_item_embeddings(
                batch["target"], batch["target_category"]
            )

            # Popularity-smoothed random negatives.
            neg_item_ids, neg_cat_ids, neg_log_probs = self._sample_random_negatives()
            neg_item_emb = self.model.get_item_embeddings(neg_item_ids, neg_cat_ids)

            # Concatenate candidates: [pos_1, ..., pos_B, neg_1, ..., neg_N]
            all_item_emb = torch.cat([pos_item_emb, neg_item_emb], dim=0)
            all_candidate_items = torch.cat([batch["target"], neg_item_ids], dim=0)

            # Score matrix [B, B+N]
            logits = torch.matmul(user_emb, all_item_emb.t()) / self.temperature
            if config.model.sampled_softmax_correction and self.num_negatives > 0:
                random_correction = neg_log_probs + np.log(float(self.num_negatives))
                logits[:, B:] = logits[:, B:] - random_correction.view(1, -1)

            # Mask known positives from history and duplicate positive candidates.
            logits = self._mask_blocked_candidates(logits, batch["history"], batch["target"], all_candidate_items)

            # Labels: positive at position i for user i
            targets = torch.arange(B, device=self.device, dtype=torch.long)

            retrieval_loss = self._retrieval_loss(logits, targets)
            explicit_loss = self._explicit_negative_bpr_loss(batch, user_emb, pos_item_emb)

            loss = retrieval_loss + config.model.explicit_negative_weight * explicit_loss

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

    def _retrieval_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_name = config.model.retrieval_loss.lower()
        if loss_name == "sampled_softmax":
            ce_loss = self._masked_cross_entropy(
                logits,
                targets,
                label_smoothing=config.model.label_smoothing,
            )
            bpr_loss = self._sampled_bpr_loss(logits, targets)
            return ce_loss + config.model.bpr_loss_weight * bpr_loss
        if loss_name == "bpr":
            return self._sampled_bpr_loss(logits, targets)
        if loss_name == "pairwise_hinge":
            return self._sampled_hinge_loss(logits, targets)
        raise ValueError(f"Unsupported retrieval_loss: {config.model.retrieval_loss}")

    def _sampled_bpr_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """BPR loss using the hardest negatives from the sampled candidate pool."""
        k = min(config.model.hard_negative_count, logits.size(1) - 1)
        if k <= 0:
            return logits.new_tensor(0.0)
        pos_scores = logits.gather(1, targets.unsqueeze(1))
        negative_logits = logits.clone()
        negative_logits.scatter_(1, targets.unsqueeze(1), -1e9)
        hard_negatives = negative_logits.topk(k=k, dim=1).values
        return -F.logsigmoid(pos_scores - hard_negatives).mean()

    def _sampled_hinge_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """ReLU margin ranking loss: relu(margin - positive + hard_negative)."""
        k = min(config.model.hard_negative_count, logits.size(1) - 1)
        if k <= 0:
            return logits.new_tensor(0.0)
        pos_scores = logits.gather(1, targets.unsqueeze(1))
        negative_logits = logits.clone()
        negative_logits.scatter_(1, targets.unsqueeze(1), -1e9)
        hard_negatives = negative_logits.topk(k=k, dim=1).values
        margin = float(config.model.hinge_margin)
        return F.relu(margin - pos_scores + hard_negatives).mean()

    def _masked_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        label_smoothing: float,
    ) -> torch.Tensor:
        """Cross-entropy where label smoothing is distributed over valid candidates only."""
        valid = logits > -1e8
        log_probs = F.log_softmax(logits, dim=1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        eps = max(0.0, float(label_smoothing))
        if eps <= 0:
            return nll.mean()

        valid_count = valid.sum(dim=1).clamp_min(1)
        smooth_loss = -(log_probs.masked_fill(~valid, 0.0).sum(dim=1) / valid_count)
        return ((1.0 - eps) * nll + eps * smooth_loss).mean()

    def _explicit_negative_bpr_loss(
        self,
        batch: dict,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Pair each positive target with the user's explicit negative feedback items."""
        neg_mask = batch["explicit_negative_mask"]
        if neg_mask.numel() == 0 or not (neg_mask > 0).any():
            return user_emb.new_tensor(0.0)

        neg_emb = self.model.get_item_embeddings(
            batch["explicit_negative"].reshape(-1),
            batch["explicit_negative_category"].reshape(-1),
        )
        neg_emb = neg_emb.view(user_emb.size(0), -1, user_emb.size(1))
        pos_scores = (user_emb * pos_item_emb).sum(dim=-1, keepdim=True) / self.temperature
        neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1) / self.temperature
        losses = -F.logsigmoid(pos_scores - neg_scores) * neg_mask
        return losses.sum() / neg_mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def evaluate(
        self,
        samples: Sequence[Union[RetrievalSample, RetrievalEvalSample]],
        user_vocab: Dict[str, int],
        item_vocab: Dict[str, int],
        k_values=(5, 10, 20),
    ) -> Dict[str, float]:
        """Full-sort evaluation over all items."""
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
            uid_t = torch.tensor([user_vocab.get(sample.user_id, 1)], dtype=torch.long, device=self.device)
            hist_t = torch.tensor([history_ids + [0] * pad_len], dtype=torch.long, device=self.device)
            mask_t = torch.tensor([[1.0] * len(history_ids) + [0.0] * pad_len], dtype=torch.float32, device=self.device)

            user_emb = self.model.get_user_embeddings(uid_t, hist_t, mask_t)
            scores = torch.matmul(user_emb, item_emb.t()).squeeze(0).cpu().numpy()

            seen = set(sample.history)
            seen_mask = np.isin(self.all_item_ids_np, list(seen))
            scores[seen_mask] = -np.inf

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
        history = {"train_loss": [], "val_recall@10": [], "val_precision@10": [],
                    "val_hit_rate@10": [], "val_diversity@10": []}
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

    def fit_staged(
        self,
        train_stages: Sequence[Sequence[UserTrainSequence]],
        val_samples: Sequence[Union[RetrievalSample, RetrievalEvalSample]],
        user_vocab: Dict[str, int],
        item_vocab: Dict[str, int],
        category_vocab: Dict[str, int],
        item_categories: Dict[str, str],
        user_negative_items: Dict[str, List[str]],
        epochs: int,
        patience: int,
        batch_size: int,
        checkpoint_name: str = "two_tower_best.pt",
        max_steps_per_epoch: int = None,
    ) -> Dict[str, List[float]]:
        history = {
            "train_loss": [],
            "val_recall@10": [],
            "val_precision@10": [],
            "val_hit_rate@10": [],
            "val_diversity@10": [],
            "stage": [],
            "replay_samples": [],
        }
        best_metric = -1.0
        bad_epochs = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        seen_sequences: List[UserTrainSequence] = []
        seen_stage_ids = set()
        stages = [list(stage) for stage in train_stages if stage]
        if not stages:
            raise ValueError("fit_staged requires at least one non-empty train stage.")

        for epoch in range(1, epochs + 1):
            stage_idx = (epoch - 1) % len(stages)
            current_stage = stages[stage_idx]
            if stage_idx not in seen_stage_ids:
                seen_sequences.extend(current_stage)
                seen_stage_ids.add(stage_idx)

            replay_samples = build_replay_samples(
                seen_sequences,
                replay_ratio=config.model.replay_ratio,
                max_samples=config.model.replay_max_samples,
                max_train_samples_per_user=config.model.max_train_samples_per_user,
                seed=config.model.seed + epoch,
            )
            train_ds = StreamingNextItemDataset(
                current_stage,
                user_vocab,
                item_vocab,
                category_vocab,
                item_categories,
                config.feature.max_seq_len,
                user_negative_items=user_negative_items,
                explicit_negative_count=config.model.per_user_negative_count,
                max_train_samples_per_user=config.model.max_train_samples_per_user,
                replay_samples=replay_samples,
                seed=config.model.seed + epoch,
                shuffle=True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                collate_fn=next_item_collate,
                drop_last=True,
            )

            train_loss = self.train_epoch(train_loader, max_steps=max_steps_per_epoch)
            scheduler.step()
            val_metrics = self.evaluate(val_samples, user_vocab, item_vocab, k_values=(5, 10, 20))
            valid_metric = val_metrics["precision@10"]
            history["train_loss"].append(train_loss)
            history["val_recall@10"].append(val_metrics["recall@10"])
            history["val_precision@10"].append(valid_metric)
            history["val_hit_rate@10"].append(val_metrics["hit_rate@10"])
            history["val_diversity@10"].append(val_metrics["diversity@10"])
            history["stage"].append(stage_idx + 1)
            history["replay_samples"].append(len(replay_samples))

            marker = ""
            if valid_metric > best_metric:
                best_metric = valid_metric
                bad_epochs = 0
                self.save(checkpoint_name, history, training_objective=f"staged_{config.model.retrieval_loss}_retrieval")
                marker = " [BEST]"
            else:
                bad_epochs += 1

            print(
                f"Epoch {epoch:3d}/{epochs} | stage: {stage_idx + 1}/{len(stages)} | "
                f"replay: {len(replay_samples)} | train_loss: {train_loss:.4f} | "
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

    def save(
        self,
        name: str,
        history: Dict[str, List[float]],
        training_objective: str = "sampled_softmax_next_item_retrieval",
    ) -> Path:
        path = self.model_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics_history": history,
                "training_objective": training_objective,
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
