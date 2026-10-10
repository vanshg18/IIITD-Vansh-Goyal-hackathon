from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from src.engine.schemas import FinancialEvent
from src.module_a.rebalancer import SentimentRebalancer, load_index_portfolio
from src.module_b.stress_tester import PortfolioStressTester, load_banking_portfolio


def make_event(
    *,
    event_id="evt1",
    ticker="AAPL",
    sentiment=0.8,
    event_type="Market",
    impact=8.5,
    confidence=0.9,
    cluster_id=None,
):
    now = datetime.now(timezone.utc)
    return FinancialEvent(
        event_id=event_id,
        timestamp=now,
        source={"name": "Publisher", "source_type": "news", "url": "https://example.com/story"},
        title="Example market-moving event",
        raw_text="Example market-moving event details",
        tickers=[ticker],
        event_type={"label": event_type, "confidence": confidence},
        sentiment={
            "score": sentiment,
            "label": "positive" if sentiment > 0 else ("negative" if sentiment < 0 else "neutral"),
            "confidence": confidence,
        },
        impact={"score": impact, "confidence": confidence},
        signal={"novelty": 1.0, "corroboration_count": 2, "decay": 1.0},
        affected_assets=[ticker],
        cluster_id=cluster_id or event_id,
        cluster_size=1,
    )


def test_mock_index_portfolio_is_real_ticker_subset_with_valid_weights():
    portfolio = load_index_portfolio()
    tickers = {asset.ticker for asset in portfolio.assets}
    assert {"NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "AVGO", "JPM", "LLY"} <= tickers
    assert len(tickers) == 12
    assert np.isclose(sum(asset.weight for asset in portfolio.assets), 1.0)


def test_rebalancer_respects_weight_and_turnover_limits_and_sentiment_direction():
    rebalancer = SentimentRebalancer(
        min_weight=0.02,
        max_weight=0.20,
        max_one_way_turnover=0.05,
        sensitivity=1.0,
    )
    now = datetime.now(timezone.utc)
    events = [
        make_event(event_id="positive_aapl", ticker="AAPL", sentiment=0.95, event_type="Earnings", impact=9.5),
        make_event(event_id="negative_msft", ticker="MSFT", sentiment=-0.95, event_type="Regulatory", impact=9.0),
    ]
    result = rebalancer.rebalance(events, as_of=now)

    assert np.isclose(sum(result.after_weights.values()), 1.0, atol=1e-9)
    assert min(result.after_weights.values()) >= 0.02 - 1e-9
    assert max(result.after_weights.values()) <= 0.20 + 1e-9
    assert result.one_way_turnover <= 0.05 + 1e-8
    assert result.after_weights["AAPL"] > result.before_weights["AAPL"]
    assert result.after_weights["MSFT"] < result.before_weights["MSFT"]


def test_rebalancer_deduplicates_cluster_and_reports_unknown_tickers():
    rebalancer = SentimentRebalancer()
    events = [
        make_event(event_id="a1", ticker="NVDA", sentiment=0.7, cluster_id="same-news"),
        make_event(event_id="a2", ticker="NVDA", sentiment=0.8, cluster_id="same-news"),
        make_event(event_id="x1", ticker="NOT_REAL", sentiment=-0.9),
    ]
    result = rebalancer.rebalance(events, as_of=datetime.now(timezone.utc))
    assert result.processed_events == 1
    assert result.unmatched_tickers == ["NOT_REAL"]


def test_synthetic_bank_positions_sum_to_declared_total_assets():
    _, _, positions = load_banking_portfolio()
    assert np.isclose(sum(p.market_value for p in positions), 1_000_000_000.0)


def test_high_impact_geopolitical_event_triggers_stress_test():
    tester = PortfolioStressTester()
    result = tester.evaluate_events([
        make_event(
            event_id="geo_1",
            ticker="NVDA",
            sentiment=-0.8,
            event_type="Geopolitical",
            impact=9.2,
        )
    ])

    assert len(result) == 1
    scenario = result[0]
    assert scenario.scenario_id == "geopolitical_supply_disruption"
    assert scenario.portfolio_value_after < scenario.portfolio_value_before
    assert scenario.net_loss > 0
    assert scenario.net_loss_pct > 0
    assert len(scenario.positions) == 8


def test_low_impact_event_does_not_trigger_stress_test():
    tester = PortfolioStressTester()
    result = tester.evaluate_events([
        make_event(event_id="geo_low", event_type="Geopolitical", impact=5.0)
    ])
    assert result == []
