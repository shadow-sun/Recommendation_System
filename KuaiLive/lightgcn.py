"""LightGCN recall model for KuaiLive user-item graph training."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(
        self,
        user_vocab_size: int,
        item_vocab_size: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        edge_dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(item_vocab_size, embedding_dim, padding_idx=0)
        self.n_layers = n_layers
        self.output_dim = embedding_dim
        self.edge_dropout = edge_dropout
        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity()

        # Learnable layer combination weights instead of simple mean pooling
        self.layer_weights = nn.Parameter(torch.ones(n_layers + 1) / (n_layers + 1))

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        with torch.no_grad():
            self.user_embedding.weight[0].fill_(0)
            self.item_embedding.weight[0].fill_(0)

    def _dropout_adj(self, norm_adj: torch.Tensor) -> torch.Tensor:
        if not self.training or self.edge_dropout <= 0:
            return norm_adj
        indices = norm_adj._indices()
        values = norm_adj._values()
        keep_mask = torch.rand(values.size(0), device=values.device) >= self.edge_dropout
        if keep_mask.all():
            return norm_adj
        kept_values = values * keep_mask.float()
        kept_values = kept_values / (1.0 - self.edge_dropout)
        size = norm_adj.size()
        with torch.sparse.check_sparse_tensor_invariants(False):
            return torch.sparse_coo_tensor(
                indices[:, keep_mask],
                kept_values[keep_mask],
                size=size,
                device=norm_adj.device,
            ).coalesce()

    def propagate(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.emb_dropout(self.user_embedding.weight)
        item_emb = self.emb_dropout(self.item_embedding.weight)
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            adj = self._dropout_adj(norm_adj)
            all_emb = torch.sparse.mm(adj, all_emb)
            embs.append(all_emb)
        stacked = torch.stack(embs, dim=0)
        weight = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1)
        final_emb = (stacked * weight).sum(dim=0)
        users, items = torch.split(final_emb, [user_emb.size(0), item_emb.size(0)], dim=0)
        return F.normalize(users, p=2, dim=-1), F.normalize(items, p=2, dim=-1)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        user_emb, item_emb = self.propagate(norm_adj)
        return (user_emb[user_ids] * item_emb[item_ids]).sum(dim=-1)

    def get_user_embeddings(self, norm_adj: torch.Tensor) -> torch.Tensor:
        user_emb, _ = self.propagate(norm_adj)
        return user_emb

    def get_item_embeddings(self, norm_adj: torch.Tensor) -> torch.Tensor:
        _, item_emb = self.propagate(norm_adj)
        return item_emb
