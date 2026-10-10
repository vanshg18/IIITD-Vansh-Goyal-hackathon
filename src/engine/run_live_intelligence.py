"""Run the NLP intelligence engine on real articles fetched from live sources."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.engine.intelligence import EventIntelligenceEngine
from src.ingestion import IngestionPipeline, build_live_sources
from src.ingestion.storage import append_articles_jsonl, write_models_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch live articles, classify them, and write financial event signals."
    )
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--timespan", default="1day")
    parser.add_argument(
        "--query",
        default='("stock market" OR finance OR markets OR economy OR earnings OR "central bank")',
    )
    parser.add_argument(
        "--articles-output",
        default="data/processed/live_articles.jsonl",
    )
    parser.add_argument(
        "--events-output",
        default="data/processed/latest_events.jsonl",
    )
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")

    load_dotenv(PROJECT_ROOT / ".env")
    pipeline = IngestionPipeline(build_live_sources())
    articles = pipeline.fetch_all(
        limit_per_source=args.limit,
        query=args.query,
        timespan=args.timespan,
    )
    if not articles:
        print("No real articles available; stopping instead of substituting demo fixtures.")
        return 2

    articles_path = Path(args.articles_output)
    if not articles_path.is_absolute():
        articles_path = PROJECT_ROOT / articles_path
    appended = append_articles_jsonl(articles, articles_path)
    print(f"Persisted {appended} new raw articles to {articles_path.resolve()}")

    engine = EventIntelligenceEngine(use_pretrained=True)
    events = engine.process(articles)
    events_path = Path(args.events_output)
    if not events_path.is_absolute():
        events_path = PROJECT_ROOT / events_path
    written = write_models_jsonl(events, events_path)

    print(f"Processed articles: {len(articles)}")
    print(f"Structured financial events: {len(events)}")
    print(f"Latest event snapshot: {events_path.resolve()} ({written} records)")
    for status in pipeline.last_source_statuses:
        if not status.success:
            print(f"[WARN] Source unavailable: {status.source_name}: {status.error}")
    return 0 if events else 4


if __name__ == "__main__":
    raise SystemExit(main())
