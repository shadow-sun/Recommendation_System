"""KuaiLive subsystem configuration."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent


@dataclass
class DataConfig:
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    csv_dir: Path = ROOT / "data" / "raw" / "KuaiLive"
    sample_ratio: float = 0.3
    max_rows_per_file: int = 0
    pos_behaviors: List[str] = field(default_factory=lambda: ["click", "comment", "like", "gift"])
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
    seed: int = 2026
    embedding_dim: int = 64
    category_embedding_dim: int = 16
    hidden_units: List[int] = field(default_factory=lambda: [128, 64])
    deepfm_hidden_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    deepfm_dropout: float = 0.2
    l2_reg: float = 1e-5
    weight_decay: float = 1e-5
    learning_rate: float = 1e-3
    batch_size: int = 1024
    epochs: int = 40
    early_stopping_patience: int = 8
    optimizer: str = "adamw"
    retrieval_loss: str = "pairwise_hinge"
    temperature: float = 0.07
    hinge_margin: float = 0.25
    label_smoothing: float = 0.1
    bpr_loss_weight: float = 0.0
    hard_negative_count: int = 64
    grad_clip_norm: float = 5.0
    history_crop_prob: float = 0.2
    history_dropout_prob: float = 0.1
    dropout_rate: float = 0.15
    eval_target_count: int = 20
    max_train_samples_per_user: int = 200
    train_stage_count: int = 4
    replay_ratio: float = 0.15
    replay_max_samples: int = 200000
    num_sampled_negatives: int = 512
    per_user_negative_count: int = 16
    explicit_negative_weight: float = 1.0
    popularity_sampling_alpha: float = 0.75
    sampled_softmax_correction: bool = True
    lightgcn_layers: int = 3
    lightgcn_num_negatives: int = 4
    lightgcn_reg_weight: float = 1e-6
    lightgcn_eval_target_count: int = 1
    lightgcn_popular_blend_weight: float = 0.15


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
    port: int = 3001
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

import torch as _torch
if _torch.cuda.is_available():
    config.device = "cuda"
