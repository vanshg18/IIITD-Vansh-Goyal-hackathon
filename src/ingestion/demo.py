"""
Load synthetic demo data into validated EventPulse objects.
"""

import json
from pathlib import Path

from src.engine.schemas import Article, Portfolio, StressScenario


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = PROJECT_ROOT / "data" / "demo"


def load_articles() -> list[Article]:
    path = DEMO_DIR / "articles.json"

    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    return [Article.model_validate(record) for record in records]


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
    articles = load_articles()
    portfolio = load_portfolio()
    scenarios = load_stress_scenarios()

    print(f"Articles loaded: {len(articles)}")
    print(f"Portfolio assets: {len(portfolio.assets)}")
    print(f"Stress scenarios: {len(scenarios)}")