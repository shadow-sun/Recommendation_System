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
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(item_vocab_size, embedding_dim, padding_idx=0)
        self.n_layers = n_layers
        self.output_dim = embedding_dim
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        with torch.no_grad():
            self.user_embedding.weight[0].fill_(0)
            self.item_embedding.weight[0].fill_(0)

    def propagate(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs.append(all_emb)
        final_emb = torch.stack(embs, dim=0).mean(dim=0)
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
