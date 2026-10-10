"""Score a manageable slice of a historical financial-news CSV with EventPulse NLP.

This script expects a CSV with publication date, ticker and headline columns.
Common aliases used by financial-news datasets (including FNSPID) are detected
automatically. Use --tickers to restrict work to your chosen liquid universe.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from dotenv import load_dotenv

from src.engine.impact import ImpactEngine
from src.engine.schemas import (
    EventClassification,
    FinancialEvent,
    ImpactResult,
    SentimentResult,
    SignalMetadata,
    SourceInfo,
)
from src.nlp.event_router import EventClassifierRouter
from src.nlp.sentiment import FinancialSentimentAnalyzer
from src.ml.historical_news import collect_rows


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def score_historical_news(
    input_csv: str | Path,
    output_jsonl: str | Path,
    limit: int = 2000,
    tickers: set[str] | None = None,
    device: int = -1,
    batch_size: int = 16,
    date_column: str | None = None,
    ticker_column: str | None = None,
    title_column: str | None = None,
    body_column: str | None = None,
    sample_seed: int = 42,
) -> int:
    rows = collect_rows(
        input_csv=input_csv,
        limit=limit,
        tickers=tickers,
        date_column=date_column,
        ticker_column=ticker_column,
        title_column=title_column,
        body_column=body_column,
        sample_seed=sample_seed,
    )
    if not rows:
        raise ValueError(
            "No usable rows found. Check CSV columns, ticker filter and date values."
        )

    load_dotenv(PROJECT_ROOT / ".env")
    sentiment_model = FinancialSentimentAnalyzer(device=device)
    router = EventClassifierRouter()
    texts = [row["raw_text"] for row in rows]
    sentiments = []
    for start in range(0, len(texts), max(1, batch_size)):
        sentiments.extend(
            sentiment_model.predict_batch(
                texts[start : start + batch_size],
                batch_size=batch_size,
            )
        )

    impact_engine = ImpactEngine()
    events: list[FinancialEvent] = []
    for row, sentiment_pred in zip(rows, sentiments):
        try:
            event_type = router.classify(row["raw_text"])
        except Exception as exc:
            event_type = EventClassification(
                label="Unknown",
                confidence=0.0,
                evidence_terms=[f"classifier_error:{type(exc).__name__}"],
                model_name="router:error",
            )

        for ticker in row["tickers"]:
            identity = row["url"] or f"{ticker}|{row['timestamp'].isoformat()}|{row['title']}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            impact = impact_engine.calculate(
                event_type=event_type.label,
                sentiment_score=sentiment_pred.score,
                sentiment_confidence=sentiment_pred.confidence,
                event_confidence=event_type.confidence,
                affected_count=1,
                novelty=1.0,
            )
            events.append(
                FinancialEvent(
                    event_id=f"hist_{ticker}_{digest}",
                    timestamp=row["timestamp"],
                    source=SourceInfo(
                        name=row["source"] or "Historical News Dataset",
                        source_type="news",
                        url=row["url"] or None,
                    ),
                    title=row["title"],
                    raw_text=row["raw_text"],
                    entities=[],
                    tickers=[ticker],
                    event_type=event_type,
                    sentiment=SentimentResult(
                        score=sentiment_pred.score,
                        label=sentiment_pred.label,
                        confidence=sentiment_pred.confidence,
                    ),
                    impact=ImpactResult(
                        score=impact.score,
                        confidence=impact.confidence,
                    ),
                    signal=SignalMetadata(
                        novelty=1.0,
                        corroboration_count=1,
                        decay=1.0,
                    ),
                    affected_assets=[ticker],
                    cluster_id=f"hist_{ticker}_{row['timestamp'].date()}",
                    cluster_size=1,
                )
            )

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(event.model_dump_json() + "\n")
    print(
        f"Scored {len(rows):,} historical news rows into {len(events):,} "
        f"event-ticker records. Output: {output_path.resolve()}"
    )
    return len(events)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run EventPulse models over a slice of historical financial news."
    )
    parser.add_argument("--input", required=True, help="Historical news CSV, e.g. FNSPID news CSV.")
    parser.add_argument("--output", default="data/processed/historical_events.jsonl")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum news rows to score.")
    parser.add_argument(
        "--tickers",
        default="",
        help="Comma-separated ticker filter, e.g. AAPL,MSFT,NVDA,JPM. Empty means any ticker.",
    )
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0 for first CUDA GPU.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--date-column", default=None)
    parser.add_argument("--ticker-column", default=None)
    parser.add_argument("--title-column", default=None)
    parser.add_argument("--body-column", default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1.")
    tickers = {item.strip().upper() for item in args.tickers.split(",") if item.strip()}
    score_historical_news(
        input_csv=args.input,
        output_jsonl=args.output,
        limit=args.limit,
        tickers=tickers or None,
        device=args.device,
        batch_size=args.batch_size,
        date_column=args.date_column,
        ticker_column=args.ticker_column,
        title_column=args.title_column,
        body_column=args.body_column,
        sample_seed=args.sample_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
