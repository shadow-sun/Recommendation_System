"""LightGCN graph retrieval framework for KuaiLive."""
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import config
from .metrics import diversity_at_k, hit_rate_at_k, precision_at_k, recall_at_k
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
    indices = torch.stack([rows, cols], dim=0).to(device)
    return torch.sparse_coo_tensor(
        indices,
        norm_values.to(device),
        size=(num_users + num_items, num_users + num_items),
        device=device,
    ).coalesce()


class LightGCNDataset(Dataset):
    def __init__(
        self,
        edges: np.ndarray,
        user_positive_items: Dict[int, set],
        item_indices: Sequence[int],
        num_negatives: int = 1,
    ):
        self.edges = edges
        self.user_positive_items = user_positive_items
        self.item_indices = list(item_indices)
        self.num_negatives = max(1, num_negatives)

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int) -> dict:
        user_idx, pos_idx = self.edges[idx]
        blocked = self.user_positive_items.get(int(user_idx), set())
        negatives = []
        while len(negatives) < self.num_negatives:
            neg_idx = random.choice(self.item_indices)
            if neg_idx not in blocked:
                negatives.append(neg_idx)
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
        pop_scores = np.array([float((item_popularity or {}).get(iid, 0.0)) for iid in self.all_item_ids], dtype=np.float32)
        if pop_scores.max() > pop_scores.min():
            pop_scores = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min())
        self.pop_scores = pop_scores

    def train_epoch(self, loader: DataLoader, max_steps: int = None) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            user_emb, item_emb = self.model.propagate(self.norm_adj)
            users = batch["user"]
            pos_items = batch["pos_item"]
            neg_items = batch["neg_item"]
            u = user_emb[users]
            pos = item_emb[pos_items]
            neg = item_emb[neg_items]
            pos_scores = (u * pos).sum(dim=-1, keepdim=True)
            neg_scores = torch.bmm(neg, u.unsqueeze(-1)).squeeze(-1)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
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
    ) -> Dict[str, float]:
        self.model.eval()
        metrics = {f"recall@{k}": [] for k in k_values}
        metrics.update({f"precision@{k}": [] for k in k_values})
        metrics.update({f"hit_rate@{k}": [] for k in k_values})
        metrics.update({f"diversity@{k}": [] for k in k_values})
        total_samples = 0
        warm_samples = 0
        cold_targets = 0

        user_emb, item_emb_all = self.model.propagate(self.norm_adj)
        item_emb = item_emb_all[self.item_tensor]

        for sample in samples:
            total_samples += 1
            user_idx = self.user_vocab.get(sample.user_id, 0)
            relevant = [iid for iid in relevant_items(sample) if iid in self.item_vocab]
            cold_targets += len([iid for iid in relevant_items(sample) if iid not in self.item_vocab])
            if user_idx == 0 or not relevant:
                continue
            warm_samples += 1
            scores = torch.matmul(user_emb[user_idx], item_emb.t()).cpu().numpy()
            blend = float(config.model.lightgcn_popular_blend_weight)
            if blend > 0:
                scores = (1.0 - blend) * scores + blend * self.pop_scores
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
        result = {key: float(np.mean(vals)) if vals else 0.0 for key, vals in metrics.items()}
        result["eval_total_users"] = float(total_samples)
        result["eval_warm_users"] = float(warm_samples)
        result["eval_warm_user_ratio"] = float(warm_samples / max(total_samples, 1))
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
    ) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_precision@10": [], "val_recall@10": [], "val_hit_rate@10": []}
        best_metric = -1.0
        bad_epochs = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, max_steps=max_steps_per_epoch)
            scheduler.step()
            val_metrics = self.evaluate(val_samples, k_values=(5, 10, 20))
            valid_metric = val_metrics["precision@10"]
            history["train_loss"].append(train_loss)
            history["val_precision@10"].append(valid_metric)
            history["val_recall@10"].append(val_metrics["recall@10"])
            history["val_hit_rate@10"].append(val_metrics["hit_rate@10"])

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
                "training_objective": "lightgcn_bpr_graph_retrieval",
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
