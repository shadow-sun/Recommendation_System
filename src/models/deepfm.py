"""DeepFM: Factorization Machine + DNN for CTR prediction."""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturesLinear(nn.Module):
    """First-order (linear) part of FM."""

    def __init__(self, num_features: int):
        super().__init__()
        self.linear = nn.Linear(num_features, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FeaturesEmbedding(nn.Module):
    """Embedding layer for sparse features."""

    def __init__(self, field_dims: List[int], embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim, padding_idx=0) for dim in field_dims
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_fields] — each column is a field with value as index
        embs = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embs, dim=1)  # [B, num_fields, D]


class FactorizationMachine(nn.Module):
    """Second-order FM interaction."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_fields, D]
        square_of_sum = x.sum(dim=1).pow(2)  # [B, D]
        sum_of_square = x.pow(2).sum(dim=1)  # [B, D]
        interaction = 0.5 * (square_of_sum - sum_of_square)
        return interaction.sum(dim=-1, keepdim=True)  # [B, 1]


class MultiLayerPerceptron(nn.Module):
    """DNN component."""

    def __init__(self, input_dim: int, hidden_units: List[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for hu in hidden_units:
            layers.append(nn.Linear(prev, hu))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hu))
            layers.append(nn.Dropout(dropout))
            prev = hu
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DeepFM(nn.Module):
    """DeepFM model with separate sparse + dense inputs.

    Args:
        field_dims: vocabulary sizes for each sparse feature field.
        embed_dim: dimension for FM embeddings.
        dense_dim: number of dense/continuous features.
        hidden_units: DNN hidden layer sizes.
        dropout: dropout rate in DNN.
    """

    def __init__(
        self,
        field_dims: List[int],
        embed_dim: int = 16,
        dense_dim: int = 3,
        hidden_units: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_fields = len(field_dims)

        self.linear = FeaturesLinear(self.num_fields + dense_dim)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FactorizationMachine()

        dnn_input = self.num_fields * embed_dim + dense_dim
        self.dnn = MultiLayerPerceptron(dnn_input, list(hidden_units), dropout)
        self.output = nn.Linear(hidden_units[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(
        self,
        sparse_inputs: torch.Tensor,
        dense_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sparse_inputs: [B, num_fields] — index-encoded sparse features.
            dense_inputs: [B, dense_dim] — continuous features.
        Returns:
            [B, 1] — predicted CTR.
        """
        # Linear part
        linear_concat = torch.cat([
            sparse_inputs.float(),
            dense_inputs,
        ], dim=-1)
        linear_out = self.linear(linear_concat)

        # FM part
        embs = self.embedding(sparse_inputs)  # [B, num_fields, D]
        fm_out = self.fm(embs)

        # DNN part
        dnn_input = torch.cat([
            embs.reshape(embs.size(0), -1),
            dense_inputs,
        ], dim=-1)
        dnn_out = self.dnn(dnn_input)
        dnn_out = self.output(dnn_out)

        output = linear_out + fm_out + dnn_out
        return torch.sigmoid(output)

    def predict(self, sparse_inputs: torch.Tensor, dense_inputs: torch.Tensor) -> torch.Tensor:
        """Return CTR predictions (no grad)."""
        with torch.no_grad():
            return self.forward(sparse_inputs, dense_inputs)
