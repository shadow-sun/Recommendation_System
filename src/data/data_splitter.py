"""Train/val/test split strategies for recommendation data."""
from typing import Tuple

import pandas as pd


def time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by global time order.

    Returns (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    df = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def leave_last_out_split(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user leave-last-out split.

    For each user, sorts their interactions by time and assigns the last
    ``test_ratio`` fraction to test, the preceding ``val_ratio`` to val,
    and the rest to train.  This guarantees that items seen in test also
    appear in train (via other users) and avoids all-zero recall on
    high-churn datasets like live-streaming.

    Returns (train_df, val_df, test_df).
    """
    train_parts = []
    val_parts = []
    test_parts = []

    for uid, grp in df.groupby("user_id"):
        grp = grp.sort_values(timestamp_col)
        n = len(grp)
        if n < 3:
            train_parts.append(grp)
            continue
        test_n = max(1, int(n * test_ratio))
        val_n = max(1, int(n * val_ratio))
        train_n = n - test_n - val_n
        if train_n < 1:
            test_n = max(1, n - 2)
            val_n = max(1, n - test_n - 1)
            train_n = n - test_n - val_n
        if train_n < 1:
            train_parts.append(grp)
            continue
        test_parts.append(grp.iloc[-test_n:])
        val_parts.append(grp.iloc[-(test_n + val_n):-test_n])
        train_parts.append(grp.iloc[:train_n])

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=df.columns)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train, val, test


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str,
    prefix: str = "unified",
) -> None:
    """Save splits as parquet files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    train.to_parquet(f"{output_dir}/{prefix}_train.parquet", index=False)
    val.to_parquet(f"{output_dir}/{prefix}_val.parquet", index=False)
    test.to_parquet(f"{output_dir}/{prefix}_test.parquet", index=False)
    print(f"Saved splits to {output_dir}/: train={len(train)}, val={len(val)}, test={len(test)}")
