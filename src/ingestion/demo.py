"""
Synthetic/demo data source for EventPulse.

This source requires no network access and is therefore the guaranteed
reproducible mode for the hackathon demo.
"""

import json
from pathlib import Path

from src.engine.schemas import Article, Portfolio, StressScenario
from src.ingestion.base import BaseSource


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = PROJECT_ROOT / "data" / "demo"


class DemoSource(BaseSource):
    """Loads the synthetic article dataset."""

    @property
    def name(self) -> str:
        return "Synthetic Demo Data"

    @property
    def source_type(self) -> str:
        return "demo"

    def fetch(self, limit: int = 20, **kwargs) -> list[Article]:
        path = DEMO_DIR / "articles.json"

        with path.open("r", encoding="utf-8") as file:
            records = json.load(file)

        records = records[:limit]

        return [Article.model_validate(record) for record in records]


def load_articles() -> list[Article]:
    """
    Backward-compatible helper used by existing tests/code.
    """
    return DemoSource().fetch(limit=10_000)


def load_portfolio() -> Portfolio:
    path = DEMO_DIR / "portfolio.json"

    with path.open("r", encoding="utf-8") as file:
        record = json.load(file)

    return Portfolio.model_validate(record)


def load_stress_scenarios() -> list[StressScenario]:
    path = DEMO_DIR / "stress_scenarios.json"

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    return [StressScenario.model_validate(record) for record in records]


if __name__ == "__main__":
    source = DemoSource()

    articles = source.fetch()

    print(f"Source: {source.name}")
    print(f"Articles loaded: {len(articles)}")
    print(f"Portfolio assets: {len(load_portfolio().assets)}")
    print(f"Stress scenarios: {len(load_stress_scenarios())}")