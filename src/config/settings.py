"""Global configuration for the recommendation system."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class DataConfig:
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    ml100k_dir: Path = ROOT / "data" / "raw" / "ml-100k"
    kualive_dir: Path = ROOT / "data" / "raw" / "kualive"
    kualive_csv_dir: Path = ROOT / "data" / "raw" / "kualive" / "KuaiLive"
    ml100k_url: str = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    kualive_sample_ratio: float = 0.3
    kualive_max_rows_per_file: int = 0
    pos_threshold_ml100k: float = 4.0
    pos_behavior_kualive: List[str] = field(default_factory=lambda: ["click", "comment", "like", "gift"])
    min_interactions_per_user: int = 10
    min_interactions_per_item: int = 30


@dataclass
class FeatureConfig:
    embedding_dim: int = 64
    category_embedding_dim: int = 16
    max_seq_len: int = 50
    user_features: List[str] = field(default_factory=lambda: ["user_id", "avg_rating", "num_ratings"])
    item_features: List[str] = field(default_factory=lambda: [
        "item_id", "category", "popularity", "avg_rating", "num_ratings"
    ])
    context_features: List[str] = field(default_factory=lambda: ["hour", "day_of_week"])


@dataclass
class ModelConfig:
    ml100k_training_route: str = "two_tower_pointwise"
    kualive_training_route: str = "lightgcn"
    embedding_dim: int = 32
    category_embedding_dim: int = 8
    hidden_units: List[int] = field(default_factory=lambda: [64, 32])
    deepfm_hidden_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    deepfm_dropout: float = 0.2
    l2_reg: float = 1e-5
    learning_rate: float = 5e-4
    batch_size: int = 1024
    epochs: int = 50
    early_stopping_patience: int = 10
    optimizer: str = "adam"
    temperature: float = 0.07


@dataclass
class RecallConfig:
    top_k: int = 500
    faiss_index_type: str = "IVF256,Flat"
    faiss_nlist: int = 256
    faiss_nprobe: int = 32


@dataclass
class RankConfig:
    final_k: int = 20
    mmr_lambda: float = 0.7
    explore_ratio: float = 0.1
    negative_penalty: float = 0.5
    category_diversity_weight: float = 0.15


@dataclass
class ServingConfig:
    host: str = "0.0.0.0"
    port: int = 3000
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 3600


@dataclass
class SimulationConfig:
    n_users: int = 100
    n_rounds: int = 5
    click_prob: float = 0.3
    neg_feedback_prob: float = 0.05
    feedback_categories: List[str] = field(default_factory=lambda: ["not_interested", "bad_quality"])


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    rank: RankConfig = field(default_factory=RankConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    model_dir: Path = ROOT / "models"
    device: str = "cpu"


config = Config()

# Auto-detect device
import torch as _torch
if _torch.cuda.is_available():
    config.device = "cuda"
