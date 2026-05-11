"""Two-Tower recall model: User Tower + Item Tower → embeddings → cosine similarity."""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.settings import config


class UserTower(nn.Module):
    """User tower: user_id embedding + history sequence pooling + MLP."""

    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        embedding_dim: int = 64,
        hidden_units: Tuple[int, ...] = (128, 64),
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(item_vocab_size, embedding_dim, padding_idx=0)
        self.max_seq_len = max_seq_len

        input_dim = embedding_dim * 2  # user_id emb + avg seq emb
        layers = []
        prev = input_dim
        for hu in hidden_units:
            layers.append(nn.Linear(prev, hu))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hu))
            prev = hu
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_units[-1] if hidden_units else input_dim

    def forward(
        self,
        user_ids: torch.Tensor,
        user_history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)  # [B, D]

        seq_emb = self.item_embedding(user_history)  # [B, S, D]
        mask = history_mask.unsqueeze(-1)  # [B, S, 1]
        seq_avg = (seq_emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # [B, D]

        combined = torch.cat([user_emb, seq_avg], dim=-1)  # [B, 2D]
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    """Item tower: item_id embedding + category embedding + MLP."""

    def __init__(
        self,
        item_vocab_size: int,
        category_vocab_size: int,
        embedding_dim: int = 64,
        category_embedding_dim: int = 16,
        hidden_units: Tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(item_vocab_size, embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(category_vocab_size, category_embedding_dim, padding_idx=0)

        input_dim = embedding_dim + category_embedding_dim
        layers = []
        prev = input_dim
        for hu in hidden_units:
            layers.append(nn.Linear(prev, hu))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hu))
            prev = hu
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_units[-1] if hidden_units else input_dim

    def forward(self, item_ids: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_embedding(item_ids)  # [B, D]
        cat_emb = self.category_embedding(categories)  # [B, D_cat]
        combined = torch.cat([item_emb, cat_emb], dim=-1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Full Two-Tower model for training and inference."""

    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        category_vocab_size: int,
        embedding_dim: int = 64,
        category_embedding_dim: int = 16,
        hidden_units: Tuple[int, ...] = (128, 64),
        temperature: float = 0.07,
    ):
        super().__init__()
        self.user_tower = UserTower(
            user_vocab_size, item_vocab_size, embedding_dim, hidden_units
        )
        self.item_tower = ItemTower(
            item_vocab_size, category_vocab_size, embedding_dim,
            category_embedding_dim, hidden_units
        )
        self.temperature = temperature

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        categories: torch.Tensor,
        user_history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_tower(user_ids, user_history, history_mask)
        item_emb = self.item_tower(item_ids, categories)
        return user_emb, item_emb

    def compute_similarity(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor
    ) -> torch.Tensor:
        return torch.matmul(user_emb, item_emb.T) / self.temperature

    def get_user_embeddings(
        self,
        user_ids: torch.Tensor,
        user_history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.user_tower(user_ids, user_history, history_mask)

    def get_item_embeddings(
        self, item_ids: torch.Tensor, categories: torch.Tensor
    ) -> torch.Tensor:
        return self.item_tower(item_ids, categories)


def sampled_softmax_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """In-batch sampled softmax loss.

    Each positive pair (user_i, item_i) is compared against
    all items in the batch as negatives.
    """
    logits = torch.matmul(user_emb, item_emb.T) / temperature  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss
