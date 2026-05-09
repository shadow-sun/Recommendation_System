from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .adapters import UNIFIED_COLUMNS


def create_sample_dataset(output_dir: str | Path, users: int = 80, items: int = 160, events: int = 2400) -> Path:
    rng = np.random.default_rng(42)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    categories = np.array(["game", "music", "movie", "sports", "food", "study"])
    item_ids = np.array([f"i{i}" for i in range(items)])
    item_categories = {item: str(categories[idx % len(categories)]) for idx, item in enumerate(item_ids)}
    user_pref = rng.integers(0, len(categories), size=users)
    rows = []
    for t in range(events):
        u = int(rng.integers(0, users))
        if rng.random() < 0.72:
            candidates = [item for item in item_ids if item_categories[item] == categories[user_pref[u]]]
            item = str(rng.choice(candidates))
            label = 1 if rng.random() < 0.82 else 0
        else:
            item = str(rng.choice(item_ids))
            label = 1 if rng.random() < 0.28 else 0
        rows.append(
            {
                "user_id": f"u{u}",
                "item_id": item,
                "item_type": "live_room",
                "timestamp": 1_700_000_000 + t,
                "behavior_type": "click" if label else "exposure",
                "label": label,
                "category": item_categories[item],
                "source_dataset": "sample",
            }
        )
    df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
    path = output / "sample_interactions.csv"
    df.to_csv(path, index=False)
    return path

