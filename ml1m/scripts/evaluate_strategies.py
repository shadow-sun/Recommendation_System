"""Offline A/B evaluation for ml1m serving strategies.

Evaluates default, diverse, and popular with the same serving logic used by
ml1m.gateway. The holdout labels come from ml1m/data/processed/test.parquet.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml1m.config import config
from ml1m import gateway
from ml100k.metrics import diversity_at_k, hit_rate_at_k, precision_at_k, recall_at_k


def load_holdout(max_users: int = None) -> dict:
    path = config.data.processed_dir / "test.parquet"
    if not path.exists():
        raise FileNotFoundError("ml1m/data/processed/test.parquet not found. Run ml1m/scripts/train.py first.")
    df = pd.read_parquet(path)
    grouped = df.groupby("user_id")["item_id"].apply(lambda s: list(dict.fromkeys(s.astype(str)))).to_dict()
    if max_users:
        grouped = dict(list(grouped.items())[:max_users])
    return grouped


def recommend_ids(user_id: str, strategy: str, k: int) -> list:
    response = gateway.recommend(user_id=str(user_id), k=k, strategy=strategy)
    return [item["item_id"] for item in response["items"]]


def evaluate_strategy(strategy: str, holdout: dict, k_values: list) -> dict:
    max_k = max(k_values)
    per_k = {
        f"recall@{k}": [] for k in k_values
    }
    per_k.update({f"precision@{k}": [] for k in k_values})
    per_k.update({f"hit_rate@{k}": [] for k in k_values})
    per_k.update({f"diversity@{k}": [] for k in k_values})
    recommended_items = set()
    latencies = []
    failures = 0

    for user_id, relevant in holdout.items():
        start = time.perf_counter()
        try:
            ranked = recommend_ids(user_id, strategy, max_k)
        except Exception:
            failures += 1
            continue
        latencies.append((time.perf_counter() - start) * 1000)
        recommended_items.update(ranked[:max_k])
        for k in k_values:
            per_k[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
            per_k[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
            per_k[f"hit_rate@{k}"].append(hit_rate_at_k(ranked, relevant, k))
            per_k[f"diversity@{k}"].append(diversity_at_k(ranked, gateway.item_categories, k))

    result = {key: float(np.mean(values)) if values else 0.0 for key, values in per_k.items()}
    result.update({
        "evaluated_users": len(holdout) - failures,
        "failures": failures,
        "catalog_coverage@max_k": len(recommended_items) / max(len(gateway.item_metadata), 1),
        "unique_recommended@max_k": len(recommended_items),
        "latency_avg_ms": float(np.mean(latencies)) if latencies else 0.0,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=["default", "diverse", "popular"])
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--max-users", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("ml1m/reports/strategy_ab.json"))
    args = parser.parse_args()

    gateway._ensure_serving_data()
    gateway._ensure_model_artifacts()
    holdout = load_holdout(max_users=args.max_users)

    results = {
        "dataset": "ml-1m",
        "max_users": args.max_users,
        "holdout_users": len(holdout),
        "k_values": args.k,
        "strategies": {},
    }
    for strategy in args.strategies:
        print(f"Evaluating strategy={strategy} on {len(holdout)} users...")
        results["strategies"][strategy] = evaluate_strategy(strategy, holdout, args.k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
