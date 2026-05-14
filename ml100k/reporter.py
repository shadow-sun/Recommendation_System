"""Evaluation reporter: generate console reports."""
from datetime import datetime
from typing import Any, Dict, List


class EvaluationReport:
    def __init__(self, report_name="evaluation"):
        self.report_name = report_name
        self.timestamp = datetime.now().isoformat()
        self.results: Dict[str, Any] = {}
        self.sections: List[dict] = []

    def add_section(self, title, metrics):
        self.sections.append({"title": title, "metrics": metrics})
        for key, value in metrics.items():
            self.results[f"{title}/{key}"] = round(value, 6)

    def print(self):
        print(f"\n{'=' * 60}")
        print(f"  {self.report_name}  -  {self.timestamp}")
        print(f"{'=' * 60}")
        for section in self.sections:
            print(f"\n  [{section['title']}]")
            for key, value in section["metrics"].items():
                print(f"    {key}: {value:.6f}")
        print(f"\n{'=' * 60}\n")
