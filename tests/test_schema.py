from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.engine.schemas import (
    Article,
    EventClassification,
    FinancialEvent,
    ImpactResult,
    SentimentResult,
)
from src.ingestion.demo import (
    load_articles,
    load_portfolio,
    load_stress_scenarios,
)


def test_demo_articles_load():
    articles = load_articles()

    assert len(articles) == 12
    assert all(isinstance(article, Article) for article in articles)


def test_demo_portfolio_load():
    portfolio = load_portfolio()

    assert len(portfolio.assets) == 12

    total_weight = sum(asset.weight for asset in portfolio.assets)

    assert abs(total_weight - 1.0) < 1e-9


def test_demo_stress_scenarios_load():
    scenarios = load_stress_scenarios()

    assert len(scenarios) == 3


def test_sentiment_bounds():
    result = SentimentResult(
        score=-0.5,
        label="negative",
        confidence=0.9,
    )

    assert -1.0 <= result.score <= 1.0


def test_invalid_sentiment_is_rejected():
    with pytest.raises(ValidationError):
        SentimentResult(
            score=2.0,
            label="positive",
            confidence=0.9,
        )


def test_invalid_impact_is_rejected():
    with pytest.raises(ValidationError):
        ImpactResult(
            score=11.0,
            confidence=0.9,
        )


def test_financial_event_creation():
    event = FinancialEvent(
        event_id="evt_test_001",
        timestamp=datetime.now(timezone.utc),
        source={
            "name": "Synthetic Source",
            "source_type": "demo",
        },
        title="Test Event",
        raw_text="A synthetic test event.",
        event_type=EventClassification(
            label="Corporate",
            confidence=0.95,
        ),
        sentiment={
            "score": -0.2,
            "label": "negative",
            "confidence": 0.9,
        },
        impact={
            "score": 5.0,
            "confidence": 0.85,
        },
    )

    assert event.event_id == "evt_test_001"
    assert event.signal.novelty == 1.0
    assert event.signal.decay == 1.0