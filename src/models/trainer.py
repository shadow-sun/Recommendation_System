"""Generic model trainer with early stopping and checkpointing."""
import json
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.config.settings import config


class EarlyStopping:
    """Stops training when monitored metric stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True  # always save first epoch as baseline

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


class Trainer:
    """Generic trainer for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        lr: float = 1e-3,
        model_dir: Optional[Path] = None,
    ):
        self.model = model
        self.device = device or config.device
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.model_dir = model_dir or config.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_auc": []}

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def train_epoch(self, dataloader: DataLoader, loss_fn: Callable) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            loss = loss_fn(self.model, batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
        compute_auc: bool = True,
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._to_device(batch)
                loss = loss_fn(self.model, batch)
                total_loss += loss.item()
                n_batches += 1

                if compute_auc:
                    logits = self.model(
                        batch["user_id"], batch["item_id"], batch["category"],
                        batch["user_history"], batch["history_mask"],
                    )
                    if isinstance(logits, tuple):
                        user_emb, item_emb = logits
                        preds = (user_emb * item_emb).sum(dim=-1).cpu().numpy()
                    else:
                        preds = logits.squeeze(-1).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(batch["label"].cpu().numpy().tolist())

        result = {"loss": total_loss / max(n_batches, 1)}
        if compute_auc and all_labels:
            try:
                result["auc"] = roc_auc_score(all_labels, all_preds)
            except ValueError:
                result["auc"] = 0.5
        return result

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        epochs: Optional[int] = None,
        patience: int = 3,
        checkpoint_name: str = "best_model.pt",
    ) -> Dict[str, list]:
        epochs = epochs or config.model.epochs
        patience = patience or config.model.early_stopping_patience
        early_stop = EarlyStopping(patience=patience, mode="min")
        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, loss_fn)
            val_metrics = self.evaluate(val_loader, loss_fn)
            val_loss = val_metrics["loss"]

            self.metrics_history["train_loss"].append(train_loss)
            self.metrics_history["val_loss"].append(val_loss)
            self.metrics_history["val_auc"].append(val_metrics.get("auc", 0))

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f}"
                + (f" | val_auc: {val_metrics.get('auc', 0):.4f}" if "auc" in val_metrics else "")
            )

            improved = early_stop.step(val_loss)
            if improved:
                self.save(checkpoint_name)
                best_val_loss = val_loss

            if early_stop.should_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        return self.metrics_history

    def save(self, name: str = "model.pt") -> Path:
        path = self.model_dir / name
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
        }, path)
        return path

    def load(self, name: str = "best_model.pt") -> None:
        path = self.model_dir / name
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        print(f"Loaded checkpoint from {path}")


def two_tower_loss_fn(model: nn.Module, batch: dict) -> torch.Tensor:
    """BCE loss on user-item cosine similarity."""
    import torch.nn.functional as F

    user_emb, item_emb = model(
        batch["user_id"], batch["item_id"], batch["category"],
        batch["user_history"], batch["history_mask"],
    )
    scores = (user_emb * item_emb).sum(dim=-1)  # [B]
    labels = batch["label"]
    return F.binary_cross_entropy_with_logits(scores, labels)


def deepfm_loss_fn(model: nn.Module, batch: dict) -> torch.Tensor:
    """BCE loss for DeepFM."""
    preds = model(batch["sparse_inputs"], batch["dense_inputs"])
    labels = batch["label"].unsqueeze(-1)
    return nn.functional.binary_cross_entropy(preds, labels)
