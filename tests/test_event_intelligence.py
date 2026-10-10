from datetime import datetime, timezone

from src.engine.corroboration import (
    CorroborationEngine,
)
from src.engine.decay import TimeDecayEngine
from src.engine.impact import ImpactEngine
from src.engine.schemas import Article


def test_breadth_is_bounded():

    engine = ImpactEngine()

    assert (
        engine.calculate_breadth(0)
        == 0.0
    )

    assert (
        0.0
        < engine.calculate_breadth(1)
        < 1.0
    )

    assert (
        engine.calculate_breadth(100)
        <= 1.0
    )


def test_high_severity_event_has_high_impact():

    engine = ImpactEngine()

    result = engine.calculate(
        event_type="Credit",
        sentiment_score=-0.8,
        sentiment_confidence=0.95,
        event_confidence=0.95,
        affected_count=3,
        novelty=1.0,
    )

    assert result.score >= 7.0


def test_novelty_affects_impact():

    engine = ImpactEngine()

    first = engine.calculate(
        event_type="Earnings",
        sentiment_score=0.5,
        sentiment_confidence=0.9,
        event_confidence=0.9,
        affected_count=1,
        novelty=1.0,
    )

    repeated = engine.calculate(
        event_type="Earnings",
        sentiment_score=0.5,
        sentiment_confidence=0.9,
        event_confidence=0.9,
        affected_count=1,
        novelty=0.25,
    )

    assert first.score > repeated.score


def test_corroboration_uses_only_articles_available_at_event_time():
    timestamp = datetime(2026, 10, 10, 10, 0, tzinfo=timezone.utc)
    articles = [
        Article(
            article_id="before",
            timestamp=timestamp.replace(minute=0),
            source={"name": "Source A", "source_type": "news"},
            title="Initial report",
            raw_text="Initial report",
        ),
        Article(
            article_id="same_time",
            timestamp=timestamp,
            source={"name": "Source B", "source_type": "news"},
            title="Second report available at timestamp",
            raw_text="Second report available at timestamp",
        ),
        Article(
            article_id="future",
            timestamp=timestamp.replace(hour=11),
            source={"name": "Source C", "source_type": "news"},
            title="Later follow-up",
            raw_text="Later follow-up",
        ),
    ]

    available = CorroborationEngine.articles_available_as_of(articles, timestamp)

    assert [article.article_id for article in available] == ["before", "same_time"]
    result = CorroborationEngine.calculate(available, base_confidence=0.8)
    assert result.unique_source_count == 2


def test_corroboration_counts_unique_sources():

    articles = [
        Article(
            article_id="a1",
            timestamp=datetime.now(timezone.utc),
            source={
                "name": "Source A",
                "source_type": "demo",
            },
            title="Same event",
            raw_text="Same event",
        ),
        Article(
            article_id="a2",
            timestamp=datetime.now(timezone.utc),
            source={
                "name": "Source B",
                "source_type": "demo",
            },
            title="Same event",
            raw_text="Same event",
        ),
        Article(
            article_id="a3",
            timestamp=datetime.now(timezone.utc),
            source={
                "name": "Source A",
                "source_type": "demo",
            },
            title="Same event",
            raw_text="Same event",
        ),
    ]

    result = CorroborationEngine.calculate(
        articles=articles,
        base_confidence=0.8,
    )

    assert result.unique_source_count == 2

    assert result.adjusted_confidence > 0.8


def test_time_decay_at_half_life():

    engine = TimeDecayEngine()

    event_time = datetime(
        2026,
        10,
        1,
        12,
        0,
        tzinfo=timezone.utc,
    )

    reference_time = datetime(
        2026,
        10,
        3,
        12,
        0,
        tzinfo=timezone.utc,
    )

    # Monetary Policy half-life = 96 hours.
    # At 48 hours we therefore expect sqrt(0.5).
    decay = engine.decay(
        event_type="Monetary Policy",
        event_timestamp=event_time,
        reference_time=reference_time,
    )

    expected = 2 ** (-48 / 96)

    assert abs(decay - expected) < 1e-6


def test_future_event_has_no_decay():

    engine = TimeDecayEngine()

    event_time = datetime(
        2026,
        10,
        5,
        12,
        0,
        tzinfo=timezone.utc,
    )

    reference_time = datetime(
        2026,
        10,
        3,
        12,
        0,
        tzinfo=timezone.utc,
    )

    decay = engine.decay(
        event_type="Market",
        event_timestamp=event_time,
        reference_time=reference_time,
    )

    assert decay == 1.0