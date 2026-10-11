"""Run EventPulse end-to-end using explicit synthetic headlines and a 12-stock real-ticker mock index."""

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from src.engine.intelligence import EventIntelligenceEngine
from src.engine.schemas import Article
from src.engine.decay import TimeDecayEngine
from src.module_a.rebalancer import SentimentRebalancer, load_index_portfolio
from src.module_b.stress_tester import PortfolioStressTester


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_INPUT = PROJECT_ROOT / "data" / "demo" / "index_articles.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "demo_replay"
REFERENCE_TIME = datetime.fromisoformat("2026-10-03T10:30:00+00:00")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            if hasattr(record, "model_dump"):
                data = record.model_dump(mode="json")
            elif hasattr(record, "to_dict"):
                data = record.to_dict()
            else:
                data = record
            file.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def load_index_demo_articles():
    with DEMO_INPUT.open("r", encoding="utf-8") as file:
        records = json.load(file)
    return [Article.model_validate(record) for record in records]


def apply_decay_as_of(event, as_of, decay_engine):
    decay = decay_engine.decay(
        event_type=event.event_type.label,
        event_timestamp=event.timestamp,
        reference_time=as_of,
    )
    updated_signal = event.signal.model_copy(update={"decay": decay})
    return event.model_copy(update={"signal": updated_signal})


def build_rebalance_timeline(events, portfolio):
    """Apply each semantic cluster once, in timestamp order, and retain snapshots."""
    clusters = defaultdict(list)
    for event in events:
        clusters[event.cluster_id or event.event_id].append(event)

    ordered_clusters = sorted(
        clusters.values(),
        key=lambda group: min(event.timestamp for event in group),
    )
    if not ordered_clusters:
        return None, []

    # 4 progressive replay stages, without replaying an earlier cluster again.
    chunk_size = max(1, math.ceil(len(ordered_clusters) / 4))
    batches = [
        ordered_clusters[i:i + chunk_size]
        for i in range(0, len(ordered_clusters), chunk_size)
    ]

    rebalancer = SentimentRebalancer(portfolio=portfolio)
    decay_engine = TimeDecayEngine()
    history = []
    latest_result = None

    for batch in batches:
        batch_events = [event for cluster in batch for event in cluster]
        as_of = max(event.timestamp for event in batch_events) + timedelta(minutes=1)
        events_as_of = [
            apply_decay_as_of(event, as_of, decay_engine)
            for event in batch_events
        ]
        latest_result = rebalancer.rebalance(events_as_of, as_of=as_of)
        history.append(latest_result.to_dict())

    return latest_result, history


def main():
    print("=" * 68)
    print("EVENTPULSE — SYNTHETIC END-TO-END DEMO")
    print("Synthetic headlines; real ticker symbols; all outputs are simulations.")
    print("=" * 68)

    articles = load_index_demo_articles()
    portfolio = load_index_portfolio()

    # 1. Structured event intelligence.
    engine = EventIntelligenceEngine(portfolio=portfolio, use_pretrained=True)
    events = engine.process(articles, reference_time=REFERENCE_TIME)

    # 2. Progressive mock-index rebalance with time-stamped snapshots.
    rebalance, rebalance_history = build_rebalance_timeline(events, portfolio)
    if rebalance is None:
        raise RuntimeError("Synthetic index demo produced no events.")

    total_processed_events = sum(
        int(snapshot.get("processed_events", 0))
        for snapshot in rebalance_history
    )

    # 3. Evaluate the negative/high-impact events against synthetic bank scenarios.
    stress_tester = PortfolioStressTester()
    stress_results = stress_tester.evaluate_events(events)

    # 4. Persist outputs separately from the live pipeline.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "events.jsonl", events)
    write_jsonl(OUTPUT_DIR / "rebalance_history.jsonl", rebalance_history)

    rebalance_output = rebalance.to_dict()
    rebalance_output["processed_events_total"] = total_processed_events
    rebalance_output["history_snapshots"] = len(rebalance_history)
    (OUTPUT_DIR / "rebalance.json").write_text(
        json.dumps(rebalance_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_jsonl(OUTPUT_DIR / "stress_results.jsonl", stress_results)

    print(f"\nSynthetic articles: {len(articles)}")
    print(f"Structured events: {len(events)}")
    print(f"Module A real-ticker holdings: {len(portfolio.assets)}")
    print(f"Module A clusters processed over time: {total_processed_events}")
    print(f"Module A history snapshots: {len(rebalance_history)}")
    print(f"Module A latest one-way turnover: {rebalance.one_way_turnover:.4f}")
    print(f"Module B stress tests: {len(stress_results)}")

    for result in stress_results:
        print(
            f"\nScenario: {result.scenario_name}"
            f"\nTrigger: {result.trigger_event_type}"
            f"\nImpact: {result.trigger_impact:.2f}/10"
            f"\nPortfolio value before: {result.portfolio_value_before:,.2f}"
            f"\nPortfolio value after: {result.portfolio_value_after:,.2f}"
            f"\nSimulated loss: {result.net_loss:,.2f}"
        )

    print(f"\nOutputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
