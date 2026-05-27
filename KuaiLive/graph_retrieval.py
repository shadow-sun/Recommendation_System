"""LightGCN graph retrieval framework for KuaiLive."""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import config
from .metrics import diversity_at_k, hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k
from .retrieval_framework import RetrievalEvalSample, relevant_items


def build_positive_user_items(df: pd.DataFrame) -> Dict[str, List[str]]:
    positives = df[df["label"] == 1].sort_values(["user_id", "timestamp"])
    return positives.groupby("user_id")["item_id"].apply(lambda s: list(dict.fromkeys(s))).to_dict()


def build_lightgcn_edges(
    user_items: Dict[str, List[str]],
    user_vocab: Dict[str, int],
    item_vocab: Dict[str, int],
) -> np.ndarray:
    edges = []
    for user_id, items in user_items.items():
        user_idx = user_vocab.get(user_id, 0)
        if user_idx == 0:
            continue
        for item_id in items:
            item_idx = item_vocab.get(item_id, 0)
            if item_idx != 0:
                edges.append((user_idx, item_idx))
    return np.array(edges, dtype=np.int64)


def build_item_cooccurrence(
    user_items: Dict[str, List[str]],
    topn: int = 200,
    window: int = 20,
    recent_decay: float = 0.85,
) -> Dict[str, List[Tuple[str, float]]]:
    """Build item-to-item sequential co-occurrence lists from train histories."""
    scores: Dict[str, Dict[str, float]] = {}
    item_counts: Dict[str, int] = {}
    window = max(1, int(window))

    for items in user_items.values():
        deduped = list(dict.fromkeys(items))
        for item in deduped:
            item_counts[item] = item_counts.get(item, 0) + 1
        for right_pos, target in enumerate(deduped):
            left = max(0, right_pos - window)
            for hist_pos in range(left, right_pos):
                source = deduped[hist_pos]
                if source == target:
                    continue
                distance = right_pos - hist_pos
                weight = recent_decay ** max(distance - 1, 0)
                scores.setdefault(source, {})
                scores[source][target] = scores[source].get(target, 0.0) + weight

    result: Dict[str, List[Tuple[str, float]]] = {}
    for source, neighbors in scores.items():
        source_norm = np.sqrt(max(item_counts.get(source, 1), 1))
        normalized = []
        for target, score in neighbors.items():
            target_norm = np.sqrt(max(item_counts.get(target, 1), 1))
            normalized.append((target, score / (source_norm * target_norm)))
        normalized.sort(key=lambda x: x[1], reverse=True)
        result[source] = normalized[:topn]
    return result


def build_item_transitions(
    user_items: Dict[str, List[str]],
    topn: int = 200,
) -> Dict[str, List[Tuple[str, float]]]:
    """Build next-item transition probabilities from adjacent train events."""
    counts: Dict[str, Dict[str, float]] = {}
    for items in user_items.values():
        deduped = list(dict.fromkeys(items))
        for source, target in zip(deduped, deduped[1:]):
            if source == target:
                continue
            counts.setdefault(source, {})
            counts[source][target] = counts[source].get(target, 0.0) + 1.0

    result: Dict[str, List[Tuple[str, float]]] = {}
    for source, neighbors in counts.items():
        total = sum(neighbors.values()) or 1.0
        ranked = [(target, score / total) for target, score in neighbors.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        result[source] = ranked[:topn]
    return result


def build_norm_adj(
    edges: np.ndarray,
    num_users: int,
    num_items: int,
    device: str,
) -> torch.Tensor:
    if edges.size == 0:
        raise ValueError("LightGCN requires at least one positive user-item edge.")

    user_nodes = torch.from_numpy(edges[:, 0]).long()
    item_nodes = torch.from_numpy(edges[:, 1]).long() + num_users
    rows = torch.cat([user_nodes, item_nodes])
    cols = torch.cat([item_nodes, user_nodes])
    values = torch.ones(rows.size(0), dtype=torch.float32)

    degree = torch.zeros(num_users + num_items, dtype=torch.float32)
    degree.index_add_(0, rows, values)
    deg_inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
    norm_values = deg_inv_sqrt[rows] * values * deg_inv_sqrt[cols]
    indices = torch.stack([rows, cols], dim=0)
    size = (num_users + num_items, num_users + num_items)
    with torch.sparse.check_sparse_tensor_invariants(False):
        adj = torch.sparse_coo_tensor(indices, norm_values, size=size).coalesce()
    return adj.to(device)


def _build_popularity_neg_weights(
    item_indices: np.ndarray,
    item_vocab: Dict[str, int],
    all_item_ids: Sequence[str],
    df: pd.DataFrame,
    alpha: float = 0.75,
) -> np.ndarray:
    """Build popularity-smoothed negative sampling weights for LightGCN."""
    counts = df[df["label"] == 1]["item_id"].astype(str).value_counts()
    weights = np.ones(len(all_item_ids), dtype=np.float32)
    for idx, iid in enumerate(all_item_ids):
        weights[idx] = float(counts.get(iid, 1.0))
    weights = np.maximum(weights, 1.0) ** alpha
    return weights / weights.sum()


class LightGCNDataset(Dataset):
    def __init__(
        self,
        edges: np.ndarray,
        user_positive_items: Dict[int, set],
        item_indices: Sequence[int],
        num_negatives: int = 64,
        neg_sampling_weights: Optional[np.ndarray] = None,
    ):
        self.edges = edges
        self.user_positive_items = user_positive_items
        self.item_indices = np.array(list(item_indices), dtype=np.int64)
        self.num_negatives = max(1, num_negatives)
        self.neg_sampling_weights = neg_sampling_weights

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int) -> dict:
        user_idx, pos_idx = self.edges[idx]
        blocked = self.user_positive_items.get(int(user_idx), set())
        n_items = len(self.item_indices)

        # Popularity-weighted or uniform negative sampling
        if self.neg_sampling_weights is not None:
            candidates = np.random.choice(
                n_items, size=self.num_negatives * 3, replace=True,
                p=self.neg_sampling_weights,
            )
        else:
            candidates = np.random.choice(
                n_items, size=self.num_negatives * 3, replace=True,
            )

        negatives = []
        for cand in candidates:
            neg_idx = int(self.item_indices[cand])
            if neg_idx not in blocked and neg_idx != int(pos_idx):
                negatives.append(neg_idx)
                if len(negatives) >= self.num_negatives:
                    break

        # Fallback if not enough valid negatives
        while len(negatives) < self.num_negatives:
            extra = int(np.random.choice(self.item_indices))
            if extra not in blocked and extra != int(pos_idx):
                negatives.append(extra)

        return {
            "user": torch.tensor(user_idx, dtype=torch.long),
            "pos_item": torch.tensor(pos_idx, dtype=torch.long),
            "neg_item": torch.tensor(negatives, dtype=torch.long),
        }


def lightgcn_collate(batch: List[dict]) -> dict:
    return {
        "user": torch.stack([b["user"] for b in batch]),
        "pos_item": torch.stack([b["pos_item"] for b in batch]),
        "neg_item": torch.stack([b["neg_item"] for b in batch]),
    }


class LightGCNTrainer:
    def __init__(
        self,
        model,
        norm_adj: torch.Tensor,
        all_item_ids: Sequence[str],
        user_vocab: Dict[str, int],
        item_vocab: Dict[str, int],
        item_categories: Dict[str, str],
        device: str,
        model_dir: Path,
        lr: float,
        weight_decay: float,
        item_popularity: Dict[str, float] = None,
        item_cooccurrence: Optional[Dict[str, List[Tuple[str, float]]]] = None,
        item_transitions: Optional[Dict[str, List[Tuple[str, float]]]] = None,
    ):
        self.model = model.to(device)
        self.norm_adj = norm_adj
        self.all_item_ids = list(all_item_ids)
        self.user_vocab = user_vocab
        self.item_vocab = item_vocab
        self.item_categories = item_categories
        self.device = device
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.item_tensor = torch.tensor([item_vocab[iid] for iid in self.all_item_ids], dtype=torch.long, device=device)
        self.all_item_ids_np = np.array(self.all_item_ids)
        self.max_eval_k = min(len(self.all_item_ids), max(1, config.rank.final_k))
        pop_scores = np.array([float((item_popularity or {}).get(iid, 0.0)) for iid in self.all_item_ids], dtype=np.float32)
        if pop_scores.max() > pop_scores.min():
            pop_scores = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min())
        self.pop_scores = pop_scores
        self.item_to_col = {iid: idx for idx, iid in enumerate(self.all_item_ids)}
        self.item_cooccurrence = item_cooccurrence or {}
        self.item_transitions = item_transitions or {}
        self.item_category_array = np.array([self.item_categories.get(iid, "") for iid in self.all_item_ids])

    def _neighbor_scores(
        self,
        history: Sequence[str],
        neighbors: Dict[str, List[Tuple[str, float]]],
        window: int,
        decay: float,
    ) -> Optional[np.ndarray]:
        if not neighbors:
            return None
        scores = np.zeros(len(self.all_item_ids), dtype=np.float32)
        history = list(history)[-window:]
        for rank, item_id in enumerate(reversed(history)):
            recency_weight = decay ** rank
            for neighbor_id, score in neighbors.get(item_id, []):
                col = self.item_to_col.get(neighbor_id)
                if col is not None:
                    scores[col] += float(score) * recency_weight
        max_score = float(scores.max())
        if max_score <= 0:
            return None
        return scores / max_score

    def _cooccurrence_scores(self, history: Sequence[str]) -> Optional[np.ndarray]:
        return self._neighbor_scores(
            history,
            self.item_cooccurrence,
            config.model.lightgcn_cooccurrence_window,
            config.model.lightgcn_recent_history_weight,
        )

    def _transition_scores(self, history: Sequence[str]) -> Optional[np.ndarray]:
        return self._neighbor_scores(
            history,
            self.item_transitions,
            3,
            0.50,
        )

    def _category_scores(self, history: Sequence[str]) -> Optional[np.ndarray]:
        category_counts: Dict[str, float] = {}
        for rank, item_id in enumerate(reversed(list(history)[-20:])):
            category = self.item_categories.get(item_id, "")
            if category:
                category_counts[category] = category_counts.get(category, 0.0) + (0.85 ** rank)
        if not category_counts:
            return None
        scores = np.array([category_counts.get(cat, 0.0) for cat in self.item_category_array], dtype=np.float32)
        max_score = float(scores.max())
        if max_score <= 0:
            return None
        return scores / max_score

    def _diversify_ranked(self, ranked_cols: np.ndarray, k: int) -> List[str]:
        max_same = max(1, int(config.model.lightgcn_max_same_category_in_top_k))
        diversify_k = min(k, int(config.model.lightgcn_diversify_top_k))
        selected: List[int] = []
        deferred: List[int] = []
        category_counts: Dict[str, int] = {}

        for col in ranked_cols:
            item_id = self.all_item_ids[int(col)]
            category = self.item_categories.get(item_id, "")
            if len(selected) < diversify_k and category_counts.get(category, 0) >= max_same:
                deferred.append(int(col))
                continue
            selected.append(int(col))
            category_counts[category] = category_counts.get(category, 0) + 1
            if len(selected) >= k:
                break

        if len(selected) < k:
            for col in deferred:
                if col not in selected:
                    selected.append(col)
                    if len(selected) >= k:
                        break

        return [self.all_item_ids[col] for col in selected[:k]]

    def _contrastive_loss(
        self,
        user_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        temperature: float,
        label_smoothing: float = 0.05,
    ) -> torch.Tensor:
        """Hybrid InfoNCE + BPR loss with label smoothing and hard-negative emphasis.

        L = (1-λ) * L_infonce + λ * L_bpr

        Where L_infonce is cross-entropy with label smoothing and
        L_bpr focuses on the hardest negative per user.
        """
        B = user_emb.size(0)
        N = neg_emb.size(1)
        pos_scores = (user_emb * pos_emb).sum(dim=-1, keepdim=True)  # [B, 1]
        neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1)  # [B, N]
        all_scores = torch.cat([pos_scores, neg_scores], dim=1) / temperature  # [B, 1+N]
        labels = torch.zeros(B, dtype=torch.long, device=user_emb.device)

        # InfoNCE with label smoothing
        eps = float(label_smoothing)
        log_probs = F.log_softmax(all_scores, dim=1)
        nll = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        if eps > 0:
            smooth_loss = -log_probs.mean(dim=1)
            ce_loss = ((1.0 - eps) * nll + eps * smooth_loss).mean()
        else:
            ce_loss = nll.mean()

        # BPR term on hardest negative
        hardest_neg_score = neg_scores.max(dim=1).values  # [B]
        bpr_loss = -F.logsigmoid((pos_scores.squeeze(1) - hardest_neg_score) / temperature).mean()

        bpr_weight = float(config.model.lightgcn_bpr_weight)
        return (1.0 - bpr_weight) * ce_loss + bpr_weight * bpr_loss

    def train_epoch(self, loader: DataLoader, max_steps: int = None) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        temperature = float(config.model.lightgcn_temperature)
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            user_emb, item_emb = self.model.propagate(self.norm_adj)
            users = batch["user"]
            pos_items = batch["pos_item"]
            neg_items = batch["neg_item"]  # [B, N]
            u = user_emb[users]
            pos = item_emb[pos_items]
            neg = item_emb[neg_items]       # [B, N, D]
            loss = self._contrastive_loss(u, pos, neg, temperature)
            reg = (
                self.model.user_embedding(users).pow(2).sum(dim=1)
                + self.model.item_embedding(pos_items).pow(2).sum(dim=1)
                + self.model.item_embedding(neg_items).pow(2).sum(dim=(1, 2))
            ).mean()
            loss = loss + config.model.lightgcn_reg_weight * reg

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
    def evaluate(
        self,
        samples: Sequence[Union[RetrievalEvalSample]],
        k_values=(5, 10, 20),
        max_samples: Optional[int] = None,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        self.model.eval()
        metrics = {f"recall@{k}": [] for k in k_values}
        metrics.update({f"precision@{k}": [] for k in k_values})
        metrics.update({f"hit_rate@{k}": [] for k in k_values})
        metrics.update({f"ndcg@{k}": [] for k in k_values})
        metrics.update({f"diversity@{k}": [] for k in k_values})
        total_samples = len(samples)
        warm_samples = 0
        cold_targets = 0
        eval_samples = list(samples[:max_samples]) if max_samples else list(samples)

        user_emb, item_emb_all = self.model.propagate(self.norm_adj)
        item_emb = item_emb_all[self.item_tensor]
        max_k = min(max(k_values), len(self.all_item_ids))

        prepared = []
        for sample in eval_samples:
            user_idx = self.user_vocab.get(sample.user_id, 0)
            relevant = [iid for iid in relevant_items(sample) if iid in self.item_vocab]
            cold_targets += len([iid for iid in relevant_items(sample) if iid not in self.item_vocab])
            if user_idx == 0 or not relevant:
                continue
            warm_samples += 1
            prepared.append((sample, user_idx, relevant))

        for start in range(0, len(prepared), batch_size):
            batch = prepared[start:start + batch_size]
            user_indices = torch.tensor([row[1] for row in batch], dtype=torch.long, device=self.device)
            score_matrix = torch.matmul(user_emb[user_indices], item_emb.t()).cpu().numpy()
            blend = float(config.model.lightgcn_popular_blend_weight)
            if blend > 0:
                score_matrix = (1.0 - blend) * score_matrix + blend * self.pop_scores

            for row_idx, (sample, _, relevant) in enumerate(batch):
                scores = score_matrix[row_idx]
                seen_mask = np.isin(self.all_item_ids_np, list(set(sample.history)))
                co_scores = self._cooccurrence_scores(sample.history)
                tr_scores = self._transition_scores(sample.history)
                cat_scores = self._category_scores(sample.history)
                co_blend = float(config.model.lightgcn_cooccurrence_blend_weight)
                if co_scores is not None and co_blend > 0:
                    scores = (1.0 - co_blend) * scores + co_blend * co_scores
                tr_blend = float(config.model.lightgcn_transition_blend_weight)
                if tr_scores is not None and tr_blend > 0:
                    scores = (1.0 - tr_blend) * scores + tr_blend * tr_scores
                cat_blend = float(config.model.lightgcn_category_blend_weight)
                if cat_scores is not None and cat_blend > 0:
                    scores = (1.0 - cat_blend) * scores + cat_blend * cat_scores
                scores[seen_mask] = -np.inf
                if max_k < len(scores):
                    top_cols = np.argpartition(-scores, max_k - 1)[:max_k]
                    ranked_cols = top_cols[np.argsort(-scores[top_cols])]
                else:
                    ranked_cols = np.argsort(-scores)
                ranked = self._diversify_ranked(ranked_cols[: max_k * 2], max_k)
                for k in k_values:
                    metrics[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
                    metrics[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
                    metrics[f"hit_rate@{k}"].append(hit_rate_at_k(ranked, relevant, k))
                    metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked, relevant, k))
                    metrics[f"diversity@{k}"].append(diversity_at_k(ranked, self.item_categories, k))
        result = {key: float(np.mean(vals)) if vals else 0.0 for key, vals in metrics.items()}
        result["eval_total_users"] = float(total_samples)
        result["eval_sampled_users"] = float(len(eval_samples))
        result["eval_warm_users"] = float(warm_samples)
        result["eval_warm_user_ratio"] = float(warm_samples / max(total_samples, 1))
        result["eval_sampled_warm_user_ratio"] = float(warm_samples / max(len(eval_samples), 1))
        result["eval_cold_targets"] = float(cold_targets)
        return result

    def fit(
        self,
        train_loader: DataLoader,
        val_samples: Sequence[RetrievalEvalSample],
        epochs: int,
        patience: int,
        checkpoint_name: str = "lightgcn_best.pt",
        max_steps_per_epoch: int = None,
        eval_max_samples: Optional[int] = None,
        eval_batch_size: int = 256,
    ) -> Dict[str, List[float]]:
        history = {
            "train_loss": [],
            "val_precision@5": [],
            "val_precision@10": [],
            "val_recall@20": [],
            "val_hit_rate@20": [],
            "val_ndcg@20": [],
        }
        best_metric = -1.0
        bad_epochs = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs - config.model.lightgcn_warmup_epochs),
            eta_min=1e-5,
        )
        warmup_epochs = int(config.model.lightgcn_warmup_epochs)
        base_lrs = [pg["lr"] for pg in self.optimizer.param_groups]

        for epoch in range(1, epochs + 1):
            # Linear warmup
            if epoch <= warmup_epochs:
                scale = epoch / max(warmup_epochs, 1)
                for pg, base_lr in zip(self.optimizer.param_groups, base_lrs):
                    pg["lr"] = base_lr * scale

            train_loss = self.train_epoch(train_loader, max_steps=max_steps_per_epoch)
            if epoch > warmup_epochs:
                scheduler.step()

            val_metrics = self.evaluate(
                val_samples,
                k_values=(5, 10, 20),
                max_samples=eval_max_samples,
                batch_size=eval_batch_size,
            )
            # Composite metric: NDCG@20 weighted highest, plus hit_rate@20
            valid_metric = (
                2.0 * val_metrics.get("ndcg@20", 0.0)
                + val_metrics["hit_rate@20"]
                + val_metrics["recall@20"]
            )
            history["train_loss"].append(train_loss)
            history["val_precision@5"].append(val_metrics["precision@5"])
            history["val_precision@10"].append(val_metrics["precision@10"])
            history["val_recall@20"].append(val_metrics["recall@20"])
            history["val_hit_rate@20"].append(val_metrics["hit_rate@20"])
            history["val_ndcg@20"].append(val_metrics.get("ndcg@20", 0.0))

            marker = ""
            if valid_metric > best_metric:
                best_metric = valid_metric
                bad_epochs = 0
                self.save(checkpoint_name, history)
                marker = " [BEST]"
            else:
                bad_epochs += 1

            ndcg_str = f"val_ndcg@20: {val_metrics.get('ndcg@20', 0.0):.4f} | "
            print(
                f"Epoch {epoch:3d}/{epochs} | train_loss: {train_loss:.4f} | "
                f"val_precision@5: {val_metrics['precision@5']:.4f} | "
                f"val_precision@10: {val_metrics['precision@10']:.4f} | "
                f"val_recall@20: {val_metrics['recall@20']:.4f} | "
                f"val_hit_rate@20: {val_metrics['hit_rate@20']:.4f} | "
                f"{ndcg_str}"
                f"best: {best_metric:.4f}{marker}"
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
                "training_objective": "lightgcn_contrastive_graph_retrieval",
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
