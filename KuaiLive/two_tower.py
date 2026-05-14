"""Two-Tower recall model with attention-pooled user history.

UserTower: user embedding + attention over history items → L2-normalized output.
ItemTower: item embedding + category embedding → MLP → L2-normalized output.
L2-normalized embeddings so Faiss IP = cosine similarity.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        embedding_dim: int = 64,
        hidden_units: Tuple[int, ...] = (128, 64),
        max_seq_len: int = 50,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(item_vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        input_dim = embedding_dim * 2
        layers = []
        prev = input_dim
        for hu in hidden_units:
            layers.append(nn.Linear(prev, hu))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hu))
            layers.append(nn.Dropout(dropout_rate))
            prev = hu
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_units[-1] if hidden_units else input_dim

    def forward(self, user_ids, user_history, history_mask):
        user_emb = self.user_embedding(user_ids)
        seq_emb = self.item_embedding(user_history)
        positions = torch.arange(user_history.size(1), device=user_history.device).unsqueeze(0)
        seq_emb = seq_emb + self.position_embedding(positions)

        user_expanded = user_emb.unsqueeze(1)
        attn_scores = (user_expanded * seq_emb).sum(dim=-1)
        attn_scores = attn_scores.masked_fill(history_mask < 0.5, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        seq_context = (seq_emb * attn_weights).sum(dim=1)

        combined = torch.cat([user_emb, seq_context], dim=-1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    def __init__(
        self,
        item_vocab_size: int,
        category_vocab_size: int,
        embedding_dim: int = 64,
        category_embedding_dim: int = 16,
        hidden_units: Tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.15,
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
            layers.append(nn.Dropout(dropout_rate))
            prev = hu
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_units[-1] if hidden_units else input_dim

    def forward(self, item_ids, categories):
        item_emb = self.item_embedding(item_ids)
        cat_emb = self.category_embedding(categories)
        combined = torch.cat([item_emb, cat_emb], dim=-1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        category_vocab_size: int,
        embedding_dim: int = 64,
        category_embedding_dim: int = 16,
        hidden_units: Tuple[int, ...] = (128, 64),
        temperature: float = 0.07,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        self.user_tower = UserTower(user_vocab_size, item_vocab_size, embedding_dim, hidden_units, dropout_rate=dropout_rate)
        self.item_tower = ItemTower(item_vocab_size, category_vocab_size, embedding_dim, category_embedding_dim, hidden_units, dropout_rate=dropout_rate)
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.tensor(5.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, user_ids, item_ids, categories, user_history, history_mask):
        user_emb = self.user_tower(user_ids, user_history, history_mask)
        item_emb = self.item_tower(item_ids, categories)
        return user_emb, item_emb

    def score(self, user_emb, item_emb):
        cosine = (user_emb * item_emb).sum(dim=-1)
        return cosine * self.logit_scale + self.logit_bias

    def get_user_embeddings(self, user_ids, user_history, history_mask):
        return self.user_tower(user_ids, user_history, history_mask)

    def get_item_embeddings(self, item_ids, categories):
        return self.item_tower(item_ids, categories)
