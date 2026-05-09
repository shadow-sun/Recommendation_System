from __future__ import annotations

import pandas as pd


def time_split(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected 0 < train_ratio < 1, 0 <= val_ratio < 1, train_ratio + val_ratio < 1")
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return ordered.iloc[:train_end].copy(), ordered.iloc[train_end:val_end].copy(), ordered.iloc[val_end:].copy()

