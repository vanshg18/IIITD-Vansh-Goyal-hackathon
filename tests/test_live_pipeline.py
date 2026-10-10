from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.engine.run_live_pipeline import (
    load_portfolio_with_state,
    run_live_cycle,
)
from src.engine.schemas import Article, FinancialEvent
from src.ingestion.storage import append_articles_jsonl
from src.module_a.rebalancer import load_index_portfolio


def make_article(timestamp):
    return Article(
        article_id="live_geo_001",
        timestamp=timestamp,
        source={
            "name": "Public News Wire",
            "source_type": "news",
            "url": "https://example.test/live-geo-001",
        },
        title="Geopolitical supply disruption threatens technology supply chains",
        raw_text="A geopolitical supply disruption has raised concerns across technology supply chains.",
        tickers=["NVDA"],
    )


def make_event(timestamp):
    return FinancialEvent(
        event_id="evt_live_geo_001",
        timestamp=timestamp,
        source={
            "name": "Public News Wire",
            "source_type": "news",
            "url": "https://example.test/live-geo-001",
        },
        title="Geopolitical supply disruption threatens technology supply chains",
        raw_text="A geopolitical supply disruption has raised concerns across technology supply chains.",
        tickers=["NVDA"],
        event_type={"label": "Geopolitical", "confidence": 0.92},
        sentiment={"score": -0.85, "label": "negative", "confidence": 0.94},
        impact={"score": 9.3, "confidence": 0.88},
        signal={"novelty": 1.0, "corroboration_count": 2, "decay": 1.0},
        affected_assets=["NVDA"],
        cluster_id="event_cluster_geo_001",
        cluster_size=1,
    )


class FakePipeline:
    def __init__(self, articles):
        self.articles = articles
        self.last_source_statuses = [
            SimpleNamespace(
                source_name="Fake live source",
                success=True,
                articles_returned=len(articles),
                error=None,
            )
        ]

    def fetch_all(self, **kwargs):
        self.last_kwargs = kwargs
        return list(self.articles)


class FakeEngine:
    def __init__(self, events):
        self.events = events
        self.calls = 0

    def process(self, articles, reference_time=None):
        self.calls += 1
        assert articles
        return list(self.events)


def output_paths(tmp_path):
    return {
        "articles_output": tmp_path / "live_articles.jsonl",
        "latest_events_output": tmp_path / "latest_events.jsonl",
        "events_history_output": tmp_path / "events_history.jsonl",
        "rebalance_latest_output": tmp_path / "latest_rebalance.json",
        "rebalance_history_output": tmp_path / "rebalance_history.jsonl",
        "portfolio_state_output": tmp_path / "index_state.json",
        "processed_registry_output": tmp_path / "processed_articles.json",
        "stress_latest_output": tmp_path / "latest_stress.jsonl",
        "stress_history_output": tmp_path / "stress_history.jsonl",
    }


def test_live_cycle_runs_modules_persists_weights_and_skips_duplicate_articles(tmp_path):
    now = datetime.now(timezone.utc)
    article_time = now - timedelta(minutes=30)
    article = make_article(article_time)
    event = make_event(article_time)
    source = FakePipeline([article])
    engine = FakeEngine([event])
    outputs = output_paths(tmp_path)

    first = run_live_cycle(
        source,
        engine_factory=lambda: engine,
        reference_time=now,
        **outputs,
    )

    assert first["exit_code"] == 0
    assert first["fetched_articles"] == 1
    assert first["new_articles"] == 1
    assert first["events"] == 1
    assert first["stress_tests"] == 1
    assert engine.calls == 1
    registry = json.loads(outputs["processed_registry_output"].read_text())
    assert len(registry["processed_fingerprints"]) == 1

    state_before_duplicate = json.loads(outputs["portfolio_state_output"].read_text())
    assert abs(sum(state_before_duplicate["weights"].values()) - 1.0) < 1e-6
    assert state_before_duplicate["weights"]["NVDA"] < 1 / 12
    stress_rows = [
        json.loads(line)
        for line in outputs["stress_history_output"].read_text().splitlines()
        if line.strip()
    ]
    assert len(stress_rows) == 1
    assert stress_rows[0]["scenario_id"] == "geopolitical_supply_disruption"

    # Same article from a later overlapping fetch must not be reprocessed or
    # apply a second rebalance to the same signal.
    state_text = outputs["portfolio_state_output"].read_text()
    second = run_live_cycle(
        FakePipeline([article]),
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("model should not load")),
        reference_time=now + timedelta(minutes=5),
        **outputs,
    )
    assert second["exit_code"] == 0
    assert second["new_articles"] == 0
    assert engine.calls == 1
    assert outputs["portfolio_state_output"].read_text() == state_text
    assert len(outputs["events_history_output"].read_text().splitlines()) == 1
    assert len(outputs["stress_history_output"].read_text().splitlines()) == 1



def test_article_archived_by_ingestion_only_is_still_processed(tmp_path):
    now = datetime.now(timezone.utc)
    article = make_article(now - timedelta(minutes=5))
    event = make_event(article.timestamp)
    outputs = output_paths(tmp_path)

    # Simulate running src.ingestion.run_ingestion earlier: the raw record is
    # archived, but the dedicated processed registry does not contain it.
    append_articles_jsonl([article], outputs["articles_output"])
    engine = FakeEngine([event])
    summary = run_live_cycle(
        FakePipeline([article]),
        engine_factory=lambda: engine,
        reference_time=now,
        **outputs,
    )

    assert summary["exit_code"] == 0
    assert summary["new_articles"] == 1
    assert summary["archived_articles"] == 0
    assert engine.calls == 1
    registry = json.loads(outputs["processed_registry_output"].read_text())
    assert len(registry["processed_fingerprints"]) == 1


def test_portfolio_state_restores_prior_weights(tmp_path):
    base = load_index_portfolio()
    weights = {asset.ticker: asset.weight for asset in base.assets}
    weights["NVDA"] += 0.01
    weights["AAPL"] -= 0.01
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "portfolio_id": base.portfolio_id,
        "weights": weights,
    }), encoding="utf-8")

    restored = load_portfolio_with_state(base, state_path)

    restored_weights = {asset.ticker: asset.weight for asset in restored.assets}
    assert abs(sum(restored_weights.values()) - 1.0) < 1e-6
    assert abs(restored_weights["NVDA"] - weights["NVDA"]) < 1e-9
    assert abs(restored_weights["AAPL"] - weights["AAPL"]) < 1e-9


def test_empty_live_fetch_does_not_fall_back_to_synthetic_articles(tmp_path):
    source = FakePipeline([])
    outputs = output_paths(tmp_path)
    summary = run_live_cycle(
        source,
        engine_factory=lambda: (_ for _ in ()).throw(AssertionError("engine should not load")),
        **outputs,
    )

    assert summary["exit_code"] == 2
    assert summary["events"] == 0
    assert not outputs["articles_output"].exists()
    assert not outputs["portfolio_state_output"].exists()
    assert not outputs["processed_registry_output"].exists()
