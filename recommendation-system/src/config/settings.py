from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    raw_data_dir: Path = project_root / "data" / "raw"
    processed_data_dir: Path = project_root / "data" / "processed"
    model_dir: Path = project_root / "models"
    embedding_dim: int = 64
    recall_candidates: int = 500
    final_k: int = 20
    max_history: int = 50
    mmr_lambda: float = 0.7
    explore_ratio: float = 0.1
    negative_penalty: float = 0.5
    strategy_name: str = "default"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

