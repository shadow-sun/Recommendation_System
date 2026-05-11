"""Feature column registry for the recommendation system."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class FeatureColumn:
    name: str
    vocab_size: int
    embedding_dim: int
    dtype: str = "int"  # "int" | "float"


@dataclass
class FeatureRegistry:
    sparse_features: List[FeatureColumn] = field(default_factory=list)
    dense_features: List[str] = field(default_factory=list)
    sequence_features: List[FeatureColumn] = field(default_factory=list)

    def add_sparse(self, name: str, vocab_size: int, embedding_dim: int) -> None:
        self.sparse_features.append(FeatureColumn(name, vocab_size, embedding_dim))

    def add_dense(self, name: str) -> None:
        self.dense_features.append(name)

    def add_sequence(self, name: str, vocab_size: int, embedding_dim: int) -> None:
        self.sequence_features.append(FeatureColumn(name, vocab_size, embedding_dim))

    def total_sparse_dim(self) -> int:
        return sum(f.embedding_dim for f in self.sparse_features)

    def total_dense_dim(self) -> int:
        return len(self.dense_features)


def build_registry(
    user_vocab_size: int,
    item_vocab_size: int,
    category_vocab_size: int,
    embedding_dim: int = 64,
    category_embedding_dim: int = 16,
) -> FeatureRegistry:
    """Build a default feature registry based on vocab sizes."""
    reg = FeatureRegistry()
    reg.add_sparse("user_id", user_vocab_size, embedding_dim)
    reg.add_sparse("item_id", item_vocab_size, embedding_dim)
    reg.add_sparse("category", category_vocab_size, category_embedding_dim)
    reg.add_dense("popularity")
    reg.add_dense("avg_rating")
    reg.add_dense("num_ratings")
    reg.add_sequence("user_history", item_vocab_size, embedding_dim)
    return reg
