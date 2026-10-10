"""Build labels from historical stock prices and train the impact model."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ml.historical_impact import (
    build_market_reaction_dataset,
    train_market_impact_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train an impact model using future abnormal returns as labels."
    )
    parser.add_argument(
        "--events",
        default="data/processed/historical_events.jsonl",
        help="JSONL produced by src.ml.score_historical_news.",
    )
    parser.add_argument(
        "--prices",
        required=True,
        help="Long price CSV, directory of per-ticker CSVs, or FNSPID price ZIP.",
    )
    parser.add_argument(
        "--dataset-output",
        default="data/processed/market_reaction_training.csv",
    )
    parser.add_argument("--model-output", default="models/impact_model.joblib")
    parser.add_argument(
        "--metrics-output",
        default="data/processed/impact_model_metrics.json",
    )
    args = parser.parse_args()

    build_market_reaction_dataset(
        events_path=args.events,
        prices_path=args.prices,
        output_csv=args.dataset_output,
    )
    report = train_market_impact_model(
        dataset_csv=args.dataset_output,
        artifact_path=args.model_output,
        metrics_path=args.metrics_output,
    )
    return 0 if report["accepted_for_live_inference"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
