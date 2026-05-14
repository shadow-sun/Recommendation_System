"""DeepFM: Factorization Machine + DNN for CTR prediction."""
from typing import List, Tuple

import torch
import torch.nn as nn


class FeaturesLinear(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.linear = nn.Linear(num_features, 1, bias=True)

    def forward(self, x):
        return self.linear(x)


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: List[int], embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim, padding_idx=0) for dim in field_dims
        ])

    def forward(self, x):
        embs = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embs, dim=1)


class FactorizationMachine(nn.Module):
    def forward(self, x):
        square_of_sum = x.sum(dim=1).pow(2)
        sum_of_square = x.pow(2).sum(dim=1)
        interaction = 0.5 * (square_of_sum - sum_of_square)
        return interaction.sum(dim=-1, keepdim=True)


class MultiLayerPerceptron(nn.Module):
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

    def forward(self, x):
        return self.mlp(x)


class DeepFM(nn.Module):
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

    def forward(self, sparse_inputs, dense_inputs):
        linear_concat = torch.cat([sparse_inputs.float(), dense_inputs], dim=-1)
        linear_out = self.linear(linear_concat)
        embs = self.embedding(sparse_inputs)
        fm_out = self.fm(embs)
        dnn_input = torch.cat([embs.reshape(embs.size(0), -1), dense_inputs], dim=-1)
        dnn_out = self.dnn(dnn_input)
        dnn_out = self.output(dnn_out)
        output = linear_out + fm_out + dnn_out
        return torch.sigmoid(output)

    def predict(self, sparse_inputs, dense_inputs):
        with torch.no_grad():
            return self.forward(sparse_inputs, dense_inputs)
