"""Run EventPulse end-to-end on clearly labeled synthetic demo data."""

import json
from datetime import datetime
from pathlib import Path

from src.engine.intelligence import EventIntelligenceEngine
from src.ingestion.demo import load_articles, load_portfolio
from src.module_a.rebalancer import SentimentRebalancer
from src.module_b.stress_tester import PortfolioStressTester


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "demo_replay"

# Keep results reproducible using the timestamp from the existing demo.
REFERENCE_TIME = datetime.fromisoformat(
    "2026-10-03T10:30:00+00:00"
)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            if hasattr(record, "model_dump"):
                data = record.model_dump(mode="json")
            else:
                data = record.to_dict()

            file.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    print("=" * 65)
    print("EVENTPULSE — SYNTHETIC END-TO-END DEMO")
    print("All demo results are simulated, not live market outcomes.")
    print("=" * 65)

    articles = load_articles()
    portfolio = load_portfolio()

    # 1. Transform synthetic articles into structured events.
    engine = EventIntelligenceEngine(
        portfolio=portfolio,
        use_pretrained=True,
    )

    events = engine.process(
        articles,
        reference_time=REFERENCE_TIME,
    )

    # 2. Rebalance the synthetic index portfolio.
    rebalancer = SentimentRebalancer(
        portfolio=portfolio,
    )

    rebalance = rebalancer.rebalance(
        events,
        as_of=REFERENCE_TIME,
    )

    # 3. Evaluate scenarios against the synthetic banking portfolio.
    stress_tester = PortfolioStressTester()
    stress_results = stress_tester.evaluate_events(events)

    # 4. Save outputs separately from the live pipeline.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(OUTPUT_DIR / "events.jsonl", events)

    (OUTPUT_DIR / "rebalance.json").write_text(
        json.dumps(
            rebalance.to_dict(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_jsonl(
        OUTPUT_DIR / "stress_results.jsonl",
        stress_results,
    )

    print(f"\nSynthetic articles: {len(articles)}")
    print(f"Structured events: {len(events)}")
    print(f"Module A processed events: {rebalance.processed_events}")
    print(f"Module A one-way turnover: {rebalance.one_way_turnover:.4f}")
    print(f"Module B stress tests: {len(stress_results)}")

    for result in stress_results:
        print(
            f"\nScenario: {result.scenario_name}"
            f"\nTrigger: {result.trigger_event_type}"
            f"\nImpact: {result.trigger_impact:.2f}/10"
            f"\nPortfolio value before: "
            f"{result.portfolio_value_before:,.2f}"
            f"\nPortfolio value after: "
            f"{result.portfolio_value_after:,.2f}"
            f"\nSimulated loss: {result.net_loss:,.2f}"
        )

    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()