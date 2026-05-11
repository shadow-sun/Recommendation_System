"""Closed-loop simulation for testing the recommendation system end-to-end.

Flow: Request → Simulate click/feedback → Update Redis → Request again
"""
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.config.settings import config
from src.streaming.producer import BehaviorProducer, get_event_queue
from src.streaming.consumer import EventConsumer, RedisFeatureUpdater
from src.services.feature_store import FeatureStore


class RecommendationSimulator:
    """Simulates users interacting with the recommendation system in a loop."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        n_users: Optional[int] = None,
        n_rounds: Optional[int] = None,
        click_prob: Optional[float] = None,
        neg_feedback_prob: Optional[float] = None,
        seed: int = 42,
    ):
        if api_url is None:
            api_url = f"http://localhost:{config.serving.port}"
        self.api_url = api_url.rstrip("/")
        self.n_users = n_users or config.simulation.n_users
        self.n_rounds = n_rounds or config.simulation.n_rounds
        self.click_prob = click_prob or config.simulation.click_prob
        self.neg_feedback_prob = neg_feedback_prob or config.simulation.neg_feedback_prob

        np.random.seed(seed)
        random.seed(seed)

        self.user_ids = [f"sim_user_{i}" for i in range(self.n_users)]
        self.user_histories: Dict[str, List[dict]] = {uid: [] for uid in self.user_ids}
        self.stats: Dict[str, list] = {
            "round": [],
            "click_rate": [],
            "neg_feedback_rate": [],
            "latency_ms": [],
            "category_distribution": [],
        }

    def run(self) -> dict:
        """Run the full simulation loop and return results."""
        feature_store = FeatureStore()
        consumer = EventConsumer(updater=RedisFeatureUpdater())
        consumer.start()

        for round_idx in range(self.n_rounds):
            round_clicks = 0
            round_neg = 0
            round_latencies = []
            categories_seen = {}

            for uid in self.user_ids:
                try:
                    resp = requests.get(
                        f"{self.api_url}/recommend",
                        params={"user_id": uid, "k": 20},
                        timeout=5.0,
                    )
                    data = resp.json()
                    latency = data.get("latency_ms", 0)
                    round_latencies.append(latency)

                    # Simulate user behavior
                    items = data.get("items", [])
                    for item in items:
                        cat = item.get("category", "")
                        categories_seen[cat] = categories_seen.get(cat, 0) + 1

                        if random.random() < self.click_prob:
                            round_clicks += 1
                            # Send positive feedback
                            event = {
                                "user_id": uid,
                                "item_id": item["item_id"],
                                "behavior_type": "click",
                                "category": cat,
                                "timestamp": time.time(),
                            }
                            get_event_queue().put(event)

                        if random.random() < self.neg_feedback_prob:
                            round_neg += 1
                            requests.post(f"{self.api_url}/feedback", json={
                                "user_id": uid,
                                "item_id": item["item_id"],
                                "feedback_type": "not_interested",
                                "category": cat,
                            })

                except requests.RequestException:
                    pass

            total_ops = self.n_users * 20
            click_rate = round_clicks / max(total_ops, 1)
            neg_rate = round_neg / max(total_ops, 1)
            avg_latency = np.mean(round_latencies) if round_latencies else 0

            self.stats["round"].append(round_idx + 1)
            self.stats["click_rate"].append(click_rate)
            self.stats["neg_feedback_rate"].append(neg_rate)
            self.stats["latency_ms"].append(avg_latency)
            self.stats["category_distribution"].append(categories_seen)

            print(
                f"Round {round_idx+1}/{self.n_rounds} | "
                f"click_rate: {click_rate:.3f} | "
                f"neg_rate: {neg_rate:.3f} | "
                f"p95_latency: {avg_latency:.1f}ms"
            )

        consumer.stop()

        return {
            "simulation_summary": {
                "n_users": self.n_users,
                "n_rounds": self.n_rounds,
                "final_click_rate": self.stats["click_rate"][-1] if self.stats["click_rate"] else 0,
                "avg_latency_ms": float(np.mean(self.stats["latency_ms"])),
            },
            "rounds": self.stats,
        }


def run_simulation(
    api_url: Optional[str] = None,
    n_users: int = 50,
    n_rounds: int = 5,
) -> dict:
    """Convenience function to run simulation."""
    sim = RecommendationSimulator(
        api_url=api_url, n_users=n_users, n_rounds=n_rounds
    )
    return sim.run()
