"""Generic model trainer with early stopping and checkpointing."""
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .config import config


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return True
        improved = score > self.best_score + self.min_delta if self.mode == "max" else score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


class Trainer:
    def __init__(self, model, device=None, lr=1e-3, model_dir=None):
        self.model = model
        self.device = device or config.device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = None  # set in fit()
        self.model_dir = model_dir or config.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = {"train_loss": [], "val_loss": [], "val_auc": []}

    def _to_device(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def train_epoch(self, dataloader, loss_fn):
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

    def evaluate(self, dataloader, loss_fn, compute_auc=True):
        """Evaluate model. Uses full similarity matrix for AUC when possible."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                batch = self._to_device(batch)
                loss = loss_fn(self.model, batch)
                total_loss += loss.item()
                n_batches += 1
                if compute_auc:
                    logits = self.model(batch["user_id"], batch["item_id"], batch["category"], batch["user_history"], batch["history_mask"])
                    if isinstance(logits, tuple):
                        user_emb, item_emb = logits
                        scores = self.model.score(user_emb, item_emb)
                        preds = torch.sigmoid(scores).cpu().numpy()
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

    def fit(self, train_loader, val_loader, loss_fn, epochs=None, patience=3, checkpoint_name="best_model.pt"):
        epochs = epochs or config.model.epochs
        patience = patience or config.model.early_stopping_patience
        early_stop = EarlyStopping(patience=patience, mode="min")
        best_auc = 0.0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, loss_fn)
            train_metrics = self.evaluate(train_loader, loss_fn, compute_auc=True)
            val_metrics = self.evaluate(val_loader, loss_fn)
            train_auc = train_metrics.get("auc", 0)
            val_loss = val_metrics["loss"]
            val_auc = val_metrics.get("auc", 0)

            self.metrics_history["train_loss"].append(train_loss)
            self.metrics_history["val_loss"].append(val_loss)
            self.metrics_history["val_auc"].append(val_auc)

            improved = early_stop.step(val_loss)
            marker = ""
            if val_auc > best_auc:
                best_auc = val_auc
                self.save(checkpoint_name)
                marker = " [BEST]"

            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"train_loss: {train_loss:.4f} | train_auc: {train_auc:.4f} | "
                f"val_loss: {val_loss:.4f} | val_auc: {val_auc:.4f} | "
                f"best_auc: {best_auc:.4f}{marker}"
            )

            if early_stop.should_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        return self.metrics_history

    def save(self, name="model.pt"):
        path = self.model_dir / name
        torch.save({"model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(), "metrics_history": self.metrics_history}, path)
        return path

    def load(self, name="best_model.pt"):
        path = self.model_dir / name
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        print(f"Loaded checkpoint from {path}")


def two_tower_loss_fn(model, batch):
    """BCE with learnable calibration: logit = cosine * scale + bias."""
    user_emb, item_emb = model(
        batch["user_id"], batch["item_id"], batch["category"],
        batch["user_history"], batch["history_mask"],
    )
    logits = model.score(user_emb, item_emb)
    return F.binary_cross_entropy_with_logits(logits, batch["label"])
