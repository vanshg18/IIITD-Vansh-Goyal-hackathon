"""Incremental live-data orchestration for EventPulse.

The cycle fetches real source records, archives and deduplicates them, runs the
intelligence engine only on newly seen articles, then sends fresh signals to
Module A and Module B. Rebalancer weights persist between separate CLI runs.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from dotenv import load_dotenv

from src.engine.schemas import Article, Portfolio
from src.ingestion import IngestionPipeline, build_live_sources
from src.ingestion.normalizer import article_fingerprint
from src.ingestion.storage import (
    append_articles_jsonl,
    load_articles_jsonl,
    write_models_jsonl,
)
from src.module_a.rebalancer import (
    SentimentRebalancer,
    load_index_portfolio,
)
from src.module_b.stress_tester import PortfolioStressTester


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _record_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="json")
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if isinstance(record, dict):
        return record
    raise TypeError(f"Cannot serialize record of type {type(record).__name__}")


def append_unique_jsonl(
    records: Iterable[Any],
    path: str | Path,
    key_field: str,
) -> int:
    """Append JSON objects not already represented by a stable key field."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if target.exists():
        with target.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {target} at line {line_number}"
                    ) from exc
                value = row.get(key_field) if isinstance(row, dict) else None
                if value is not None:
                    seen.add(str(value))

    added = 0
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            row = _record_dict(record)
            value = row.get(key_field)
            if value is None:
                raise ValueError(
                    f"Record for {target} has no stable {key_field!r} field"
                )
            identity = str(value)
            if identity in seen:
                continue
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            seen.add(identity)
            added += 1
    return added


def load_portfolio_with_state(
    base_portfolio: Portfolio,
    state_path: str | Path,
) -> Portfolio:
    """Restore valid persisted weights, or use the supplied initial portfolio."""
    path = Path(state_path)
    if not path.exists():
        return base_portfolio

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("portfolio_id") != base_portfolio.portfolio_id:
        raise ValueError(
            f"Portfolio state {path} belongs to {payload.get('portfolio_id')!r}, "
            f"expected {base_portfolio.portfolio_id!r}. Move or remove the state "
            "file only if you intentionally want to restart the simulation."
        )

    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise ValueError(f"Portfolio state {path} must contain a weights object.")

    normalized_weights = {
        str(ticker).strip().upper(): value
        for ticker, value in weights.items()
    }
    expected = {asset.ticker.upper() for asset in base_portfolio.assets}
    supplied = set(normalized_weights)
    if expected != supplied:
        raise ValueError(
            "Persisted portfolio tickers do not match the current index universe. "
            f"Missing={sorted(expected - supplied)}, unexpected={sorted(supplied - expected)}."
        )

    asset_rows = []
    for asset in base_portfolio.assets:
        value = float(normalized_weights[asset.ticker.upper()])
        if not math.isfinite(value) or value <= 0 or value > 1:
            raise ValueError(f"Invalid saved weight for {asset.ticker}: {value}")
        row = asset.model_dump()
        row["weight"] = value
        asset_rows.append(row)

    portfolio_data = base_portfolio.model_dump()
    portfolio_data["assets"] = asset_rows
    return Portfolio.model_validate(portfolio_data)


def save_portfolio_state(
    portfolio_id: str,
    weights: dict[str, float],
    path: str | Path,
    updated_at: datetime,
) -> None:
    """Persist the current simulated index weights between runs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    total = float(sum(weights.values()))
    if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
        raise ValueError(f"Cannot persist index weights that sum to {total:.9f}.")
    payload = {
        "portfolio_id": portfolio_id,
        "updated_at": updated_at.astimezone(timezone.utc).isoformat(),
        "weights": {ticker: float(value) for ticker, value in weights.items()},
        "note": "Simulated weights; initial weights are equal-weighted, not official index weights.",
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _report_sources(pipeline: Any) -> None:
    for status in getattr(pipeline, "last_source_statuses", []):
        state = "OK" if status.success else "FAILED"
        detail = f" | {status.error}" if status.error else ""
        print(
            f"{state:6} | {status.source_name:20} | "
            f"returned={status.articles_returned}{detail}"
        )


def run_live_cycle(
    pipeline: Any,
    *,
    limit_per_source: int = 15,
    timespan: str = "1day",
    query: str = '("stock market" OR finance OR markets OR economy OR earnings OR "central bank")',
    articles_output: str | Path = "data/processed/live_articles.jsonl",
    latest_events_output: str | Path = "data/processed/latest_events.jsonl",
    events_history_output: str | Path = "data/processed/events_history.jsonl",
    rebalance_latest_output: str | Path = "data/processed/latest_rebalance.json",
    rebalance_history_output: str | Path = "data/processed/rebalance_history.jsonl",
    portfolio_state_output: str | Path = "data/processed/index_portfolio_state.json",
    stress_latest_output: str | Path = "data/processed/latest_stress_results.jsonl",
    stress_history_output: str | Path = "data/processed/stress_results_history.jsonl",
    engine_factory: Callable[[], Any] | None = None,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Execute one incremental live ingestion → NLP → Module A/B cycle."""
    if limit_per_source < 1:
        raise ValueError("limit_per_source must be at least 1.")
    cycle_time = reference_time or datetime.now(timezone.utc)
    if cycle_time.tzinfo is None:
        cycle_time = cycle_time.replace(tzinfo=timezone.utc)
    cycle_time = cycle_time.astimezone(timezone.utc)

    fetched = pipeline.fetch_all(
        limit_per_source=limit_per_source,
        query=query,
        timespan=timespan,
    )
    _report_sources(pipeline)

    articles_path = _resolve_path(articles_output)
    if not fetched:
        print("[ERROR] No live articles were fetched; synthetic fixtures were not used.")
        return {
            "exit_code": 2,
            "fetched_articles": 0,
            "new_articles": 0,
            "events": 0,
            "stress_tests": 0,
            "message": "No source returned usable articles.",
        }

    archived_articles = load_articles_jsonl(articles_path)
    seen = {article_fingerprint(item) for item in archived_articles}
    fresh_articles = []
    for article in fetched:
        fingerprint = article_fingerprint(article)
        if fingerprint not in seen:
            fresh_articles.append(article)
            seen.add(fingerprint)

    # Archive all fetched rows idempotently; only freshly unseen rows go through
    # the expensive transformer pipeline and trigger downstream state changes.
    saved_count = append_articles_jsonl(fetched, articles_path)
    print(f"Raw articles: fetched={len(fetched)}, new={len(fresh_articles)}, archived={saved_count}")

    if not fresh_articles:
        print("[INFO] No new articles in this window; inference and downstream actions skipped.")
        return {
            "exit_code": 0,
            "fetched_articles": len(fetched),
            "new_articles": 0,
            "events": 0,
            "stress_tests": 0,
            "message": "All fetched records were already archived; no state changed.",
        }

    if engine_factory is None:
        # Delayed import keeps parsing/state helpers and network-independent
        # tests usable without loading the transformer stack.
        from src.engine.intelligence import EventIntelligenceEngine

        engine_factory = lambda: EventIntelligenceEngine(
            portfolio=load_index_portfolio(),
            use_pretrained=True,
        )

    engine = engine_factory()
    events = engine.process(fresh_articles, reference_time=cycle_time)

    latest_events_path = _resolve_path(latest_events_output)
    events_history_path = _resolve_path(events_history_output)
    written_events = write_models_jsonl(events, latest_events_path)
    appended_events = append_unique_jsonl(events, events_history_path, "event_id")
    print(
        f"Structured events: {len(events)}; latest snapshot={latest_events_path}; "
        f"new event history records={appended_events}"
    )

    # Module A — restore prior simulated weights, then apply fresh signals only.
    base_portfolio = load_index_portfolio()
    portfolio_state_path = _resolve_path(portfolio_state_output)
    restored_portfolio = load_portfolio_with_state(base_portfolio, portfolio_state_path)
    rebalancer = SentimentRebalancer(portfolio=restored_portfolio)
    rebalance_result = rebalancer.rebalance(events, as_of=cycle_time)

    save_portfolio_state(
        portfolio_id=rebalance_result.portfolio_id,
        weights=rebalance_result.after_weights,
        path=portfolio_state_path,
        updated_at=cycle_time,
    )
    rebalance_latest_path = _resolve_path(rebalance_latest_output)
    rebalance_latest_path.parent.mkdir(parents=True, exist_ok=True)
    rebalance_latest_path.write_text(
        json.dumps(rebalance_result.to_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    append_unique_jsonl(
        [rebalance_result],
        _resolve_path(rebalance_history_output),
        "timestamp",
    )
    print(
        f"Module A: turnover={rebalance_result.one_way_turnover:.4f}; "
        f"processed_events={rebalance_result.processed_events}; "
        f"state={portfolio_state_path}"
    )

    # Module B — only configured high-impact event/scenario matches fire.
    stress_tester = PortfolioStressTester()
    stress_results = stress_tester.evaluate_events(events)
    stress_latest_path = _resolve_path(stress_latest_output)
    write_models_jsonl([result.to_dict() for result in stress_results], stress_latest_path)
    appended_stresses = append_unique_jsonl(
        stress_results,
        _resolve_path(stress_history_output),
        "run_id",
    )
    print(
        f"Module B: stress tests triggered={len(stress_results)}; "
        f"new historical records={appended_stresses}"
    )

    return {
        "exit_code": 0,
        "fetched_articles": len(fetched),
        "new_articles": len(fresh_articles),
        "archived_articles": saved_count,
        "events": written_events,
        "event_history_appended": appended_events,
        "rebalance_turnover": rebalance_result.one_way_turnover,
        "rebalance_processed_events": rebalance_result.processed_events,
        "stress_tests": len(stress_results),
        "stress_history_appended": appended_stresses,
        "message": "Live cycle completed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one incremental EventPulse live cycle: ingest, infer, rebalance and stress test."
    )
    parser.add_argument("--limit", type=int, default=15, help="Maximum articles per configured source.")
    parser.add_argument("--timespan", default="1day", help="GDELT window, e.g. 6h, 1day, 1week.")
    parser.add_argument(
        "--query",
        default='("stock market" OR finance OR markets OR economy OR earnings OR "central bank")',
        help="GDELT news query.",
    )
    parser.add_argument("--articles-output", default="data/processed/live_articles.jsonl")
    parser.add_argument("--events-output", default="data/processed/latest_events.jsonl")
    parser.add_argument("--events-history-output", default="data/processed/events_history.jsonl")
    parser.add_argument("--rebalance-output", default="data/processed/latest_rebalance.json")
    parser.add_argument("--rebalance-history-output", default="data/processed/rebalance_history.jsonl")
    parser.add_argument("--portfolio-state-output", default="data/processed/index_portfolio_state.json")
    parser.add_argument("--stress-output", default="data/processed/latest_stress_results.jsonl")
    parser.add_argument("--stress-history-output", default="data/processed/stress_results_history.jsonl")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1.")

    load_dotenv(PROJECT_ROOT / ".env")
    pipeline = IngestionPipeline(build_live_sources())
    summary = run_live_cycle(
        pipeline,
        limit_per_source=args.limit,
        timespan=args.timespan,
        query=args.query,
        articles_output=args.articles_output,
        latest_events_output=args.events_output,
        events_history_output=args.events_history_output,
        rebalance_latest_output=args.rebalance_output,
        rebalance_history_output=args.rebalance_history_output,
        portfolio_state_output=args.portfolio_state_output,
        stress_latest_output=args.stress_output,
        stress_history_output=args.stress_history_output,
    )
    print("\nEVENTPULSE LIVE CYCLE SUMMARY")
    print(json.dumps(summary, indent=2))
    return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
