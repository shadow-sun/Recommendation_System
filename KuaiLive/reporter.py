"""Evaluation reporter: generate JSON and console reports."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class EvaluationReport:
    def __init__(self, report_name: str = "evaluation"):
        self.report_name = report_name
        self.timestamp = datetime.now().isoformat()
        self.results: Dict[str, Any] = {}
        self.sections: List[dict] = []

    def add_metric(self, name: str, value: float, section: str = "default") -> None:
        self.results[f"{section}/{name}"] = round(value, 6)

    def add_section(self, title: str, metrics: Dict[str, float]) -> None:
        self.sections.append({"title": title, "metrics": metrics})
        for k, v in metrics.items():
            self.results[f"{title}/{k}"] = round(v, 6)

    def to_dict(self) -> dict:
        return {
            "report_name": self.report_name,
            "timestamp": self.timestamp,
            "results": self.results,
            "sections": [{"title": s["title"], "metrics": s["metrics"]} for s in self.sections],
        }

    def print(self) -> None:
        print(f"\n{'='*60}")
        print(f"  {self.report_name}  —  {self.timestamp}")
        print(f"{'='*60}")
        for section in self.sections:
            print(f"\n  [{section['title']}]")
            for k, v in section["metrics"].items():
                print(f"    {k}: {v:.6f}")
        print(f"\n{'='*60}\n")


def compare_reports(reports: List[EvaluationReport]) -> EvaluationReport:
    cmp = EvaluationReport("comparison")
    for i, report in enumerate(reports):
        for section in report.sections:
            for k, v in section["metrics"].items():
                cmp.add_metric(f"{k}_run{i}", v, section["title"])
    return cmp
