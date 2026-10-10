"""CLI for fetching and persistently storing real or synthetic articles."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.ingestion import DemoSource, IngestionPipeline, build_live_sources
from src.ingestion.storage import append_articles_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest EventPulse source articles.")
    parser.add_argument(
        "--mode",
        choices=("live", "demo"),
        default="live",
        help="live uses configured external sources; demo uses only synthetic fixtures.",
    )
    parser.add_argument("--limit", type=int, default=25, help="Maximum articles per source.")
    parser.add_argument(
        "--query",
        default='("stock market" OR finance OR markets OR economy OR earnings OR "central bank")',
        help="GDELT query.",
    )
    parser.add_argument("--timespan", default="1day", help="GDELT window, e.g. 6h, 1day, 1week.")
    parser.add_argument(
        "--output",
        default="data/processed/live_articles.jsonl",
        help="JSONL storage path (relative paths are resolved from the repository root).",
    )
    parser.add_argument(
        "--require-all-sources",
        action="store_true",
        help="Return a non-zero exit code if any configured source fails.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")

    sources = (
        [DemoSource()]
        if args.mode == "demo"
        else build_live_sources()
    )
    pipeline = IngestionPipeline(sources=sources)
    articles = pipeline.fetch_all(
        limit_per_source=args.limit,
        query=args.query,
        timespan=args.timespan,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    added = append_articles_jsonl(articles, output_path)

    print()
    print("=" * 72)
    print(f"EVENTPULSE INGESTION | mode={args.mode}")
    print("=" * 72)
    print(f"Records fetched: {len(articles)}")
    print(f"New records saved: {added}")
    print(f"Storage: {output_path.resolve()}")
    for status in pipeline.last_source_statuses:
        state = "OK" if status.success else "FAILED"
        print(
            f"{state:6} | {status.source_name:20} | "
            f"returned={status.articles_returned}"
            + (f" | {status.error}" if status.error else "")
        )

    if not articles:
        print(
            "No articles were ingested. Check connectivity, source status, "
            "and SEC_USER_AGENT configuration. Synthetic data is not used as fallback."
        )
        return 2
    if args.require_all_sources and any(
        not status.success for status in pipeline.last_source_statuses
    ):
        print("At least one configured source failed and --require-all-sources was set.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
