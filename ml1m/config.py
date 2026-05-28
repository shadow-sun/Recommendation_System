"""ml-1m subsystem configuration."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch as _torch

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


@dataclass
class DataConfig:
    raw_dir: Path = PROJECT_ROOT / "ml-1m"
    processed_dir: Path = ROOT / "data" / "processed"
    pos_threshold: float = 4.0
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
    embedding_dim: int = 64
    category_embedding_dim: int = 16
    hidden_units: List[int] = field(default_factory=lambda: [128, 64])
    learning_rate: float = 1e-3
    batch_size: int = 1024
    epochs: int = 100
    early_stopping_patience: int = 20
    temperature: float = 0.07
    seed: int = 2026
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    bpr_loss_weight: float = 0.25
    hard_negative_count: int = 64
    grad_clip_norm: float = 5.0
    eval_target_count: int = 20
    history_crop_prob: float = 0.2
    history_dropout_prob: float = 0.1
    dropout: float = 0.15


@dataclass
class RecallConfig:
    top_k: int = 500
    faiss_index_type: str = "Flat"
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
class KafkaConfig:
    enabled: bool = True
    bootstrap_servers: str = "localhost:9092"
    topic: str = "ml1m_events"
    consumer_group: str = "ml1m_consumer"
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    auto_offset_reset: str = "latest"


@dataclass
class ServingConfig:
    host: str = "0.0.0.0"
    port: int = 3006
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
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    model_dir: Path = ROOT / "models"
    device: str = "cuda" if _torch.cuda.is_available() else "cpu"


config = Config()
