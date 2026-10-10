from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.ml.historical_impact import (
    FEATURE_COLUMNS,
    MarketReactionImpactModel,
    build_market_reaction_dataset,
    load_price_panel,
)


def test_build_dataset_uses_next_close_and_peer_benchmark(tmp_path):
    events_path = tmp_path / "events.jsonl"
    prices_path = tmp_path / "prices.csv"
    output_path = tmp_path / "training.csv"

    event = {
        "event_id": "event_1",
        # 12:00 UTC is 07:00 in New York in January, before the market opens.
        "timestamp": "2024-01-03T12:00:00Z",
        "tickers": ["AAA"],
        "affected_assets": ["AAA"],
        "sentiment": {"score": -0.8, "confidence": 0.9},
        "event_type": {"label": "Credit", "confidence": 0.85},
        "cluster_size": 2,
        "signal": {"novelty": 1.0, "corroboration_count": 2, "decay": 1.0},
    }
    events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    prices = pd.DataFrame([
        {"date": "2024-01-02", "ticker": "AAA", "close": 100.0},
        {"date": "2024-01-03", "ticker": "AAA", "close": 90.0},
        {"date": "2024-01-04", "ticker": "AAA", "close": 91.0},
        {"date": "2024-01-02", "ticker": "BBB", "close": 100.0},
        {"date": "2024-01-03", "ticker": "BBB", "close": 101.0},
        {"date": "2024-01-04", "ticker": "BBB", "close": 102.0},
    ])
    prices.to_csv(prices_path, index=False)

    frame = build_market_reaction_dataset(events_path, prices_path, output_path)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["ticker"] == "AAA"
    # The daily bar on Jan 3 wasn't observable at the 07:00 ET publication time.
    assert row["date"] == "2024-01-02"
    assert row["next_date"] == "2024-01-03"
    assert np.isclose(row["stock_forward_return"], -0.1)
    assert np.isclose(row["benchmark_forward_return"], 0.01)
    assert np.isclose(row["abnormal_return"], -0.11)
    assert np.isclose(row["sample_weight"], 1.0)
    assert output_path.exists()


def test_after_close_event_uses_same_day_close_and_next_session_target(tmp_path):
    events_path = tmp_path / "events.jsonl"
    prices_path = tmp_path / "prices.csv"
    event = {
        "event_id": "after_close",
        # 22:00 UTC is 17:00 in New York in January, after market close.
        "timestamp": "2024-01-03T22:00:00Z",
        "tickers": ["AAA"],
        "sentiment": {"score": -0.2, "confidence": 0.7},
        "event_type": {"label": "Credit", "confidence": 0.8},
        "signal": {},
    }
    events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    pd.DataFrame([
        {"date": "2024-01-02", "ticker": "AAA", "close": 100},
        {"date": "2024-01-03", "ticker": "AAA", "close": 90},
        {"date": "2024-01-04", "ticker": "AAA", "close": 81},
        {"date": "2024-01-02", "ticker": "BBB", "close": 100},
        {"date": "2024-01-03", "ticker": "BBB", "close": 101},
        {"date": "2024-01-04", "ticker": "BBB", "close": 102},
    ]).to_csv(prices_path, index=False)

    frame = build_market_reaction_dataset(events_path, prices_path, tmp_path / "out.csv")

    assert frame.iloc[0]["date"] == "2024-01-03"
    assert frame.iloc[0]["next_date"] == "2024-01-04"


def test_weekend_event_anchors_to_latest_prior_close(tmp_path):
    events_path = tmp_path / "events.jsonl"
    prices_path = tmp_path / "prices.csv"
    event = {
        "event_id": "weekend",
        "timestamp": "2024-01-06T12:00:00Z",
        "tickers": ["AAA"],
        "sentiment": {"score": -0.2, "confidence": 0.7},
        "event_type": {"label": "Market", "confidence": 0.7},
        "signal": {},
    }
    events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    pd.DataFrame([
        {"date": "2024-01-05", "ticker": "AAA", "close": 10},
        {"date": "2024-01-08", "ticker": "AAA", "close": 9},
        {"date": "2024-01-09", "ticker": "AAA", "close": 9.5},
        {"date": "2024-01-05", "ticker": "BBB", "close": 10},
        {"date": "2024-01-08", "ticker": "BBB", "close": 10},
        {"date": "2024-01-09", "ticker": "BBB", "close": 10},
    ]).to_csv(prices_path, index=False)

    frame = build_market_reaction_dataset(events_path, prices_path, tmp_path / "out.csv")

    assert frame.iloc[0]["date"] == "2024-01-05"
    assert frame.iloc[0]["next_date"] == "2024-01-08"


def test_load_price_panel_supports_per_ticker_directory(tmp_path):
    (tmp_path / "AAA.csv").write_text(
        "Date,Close\n2024-01-01,10\n2024-01-02,11\n", encoding="utf-8"
    )
    (tmp_path / "BBB.csv").write_text(
        "Date,Close\n2024-01-01,20\n2024-01-02,21\n", encoding="utf-8"
    )
    frame = load_price_panel(tmp_path, tickers={"AAA"})
    assert set(frame["ticker"]) == {"AAA"}
    assert len(frame) == 2


class FakePredictor:
    def predict(self, frame):
        assert list(frame.columns) == FEATURE_COLUMNS
        return np.full(len(frame), 0.025)


def test_impact_model_uses_only_shared_training_serving_features():
    artifact = {
        "accepted_for_live_inference": True,
        "feature_columns": FEATURE_COLUMNS,
        "pipeline": FakePredictor(),
        "calibration_abs_returns": np.array([0.01, 0.02, 0.03, 0.04]),
    }
    model = MarketReactionImpactModel(artifact)

    prediction = model.predict_event(
        sentiment_score=-0.7,
        sentiment_confidence=0.9,
        event_confidence=0.85,
        event_type="Credit",
        tickers=["AAPL", "MSFT"],
    )

    assert 1.0 <= prediction["impact_score"] <= 10.0
    assert np.isclose(prediction["predicted_abs_abnormal_return"], 0.025)


def test_impact_model_rejects_outdated_feature_schema():
    artifact = {
        "accepted_for_live_inference": True,
        "feature_columns": ["old_feature"],
        "pipeline": FakePredictor(),
        "calibration_abs_returns": np.array([0.01, 0.02]),
    }
    try:
        MarketReactionImpactModel(artifact)
    except ValueError as exc:
        assert "feature schema" in str(exc)
    else:
        raise AssertionError("Expected old feature schema to be rejected")
