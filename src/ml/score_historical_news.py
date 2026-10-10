"""Score a manageable slice of a historical financial-news CSV with EventPulse NLP.

This script expects a CSV with publication date, ticker and headline columns.
Common aliases used by financial-news datasets (including FNSPID) are detected
automatically. Use --tickers to restrict work to your chosen liquid universe.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _column(columns, overrides, key, aliases, required=True):
    override = overrides.get(key)
    if override:
        if override not in columns:
            raise ValueError(
                f"Column override {override!r} for {key} was not found. "
                f"Available columns: {list(columns)}"
            )
        return override
    by_key = {_key(name): name for name in columns}
    for alias in aliases:
        if _key(alias) in by_key:
            return by_key[_key(alias)]
    if required:
        raise ValueError(
            f"Could not auto-detect the {key} column. Use a --{key}-column override. "
            f"Available columns: {list(columns)}"
        )
    return None


def collect_rows(
    input_csv: str | Path,
    limit: int,
    tickers: set[str] | None,
    date_column: str | None = None,
    ticker_column: str | None = None,
    title_column: str | None = None,
    body_column: str | None = None,
) -> list[dict[str, Any]]:
    overrides = {
        "date": date_column,
        "ticker": ticker_column,
        "title": title_column,
        "body": body_column,
    }
    aliases = {
        "date": ["timestamp", "published_at", "published", "date", "datetime", "time", "news_date"],
        "ticker": ["ticker", "symbol", "stock_symbol", "stock_symbols", "tickers", "stock"],
        "title": ["headline", "title", "article_title", "news_title", "news_headline"],
        "body": ["body", "body_text", "article", "article_text", "summary", "description", "text"],
        "url": ["url", "link", "article_url", "news_url"],
        "source": ["source", "publisher", "site", "news_source"],
    }
    rows: list[dict[str, Any]] = []
    paths = Path(input_csv)
    if not paths.is_file():
        raise FileNotFoundError(paths)

    for chunk in pd.read_csv(paths, chunksize=50_000, low_memory=False):
        date_col = _column(chunk.columns, overrides, "date", aliases["date"])
        ticker_col = _column(chunk.columns, overrides, "ticker", aliases["ticker"])
        title_col = _column(chunk.columns, overrides, "title", aliases["title"])
        body_col = _column(
            chunk.columns, overrides, "body", aliases["body"], required=False
        )
        url_col = _column(chunk.columns, {}, "url", aliases["url"], required=False)
        source_col = _column(chunk.columns, {}, "source", aliases["source"], required=False)

        for record in chunk.to_dict(orient="records"):
            title = str(record.get(title_col, "") or "").strip()
            if not title or title.lower() == "nan":
                continue
            raw_tickers = str(record.get(ticker_col, "") or "").upper()
            symbols = [x.strip().upper() for x in re.split(r"[;,|]", raw_tickers) if x.strip()]
            symbols = list(dict.fromkeys(symbols))
            if tickers:
                symbols = [symbol for symbol in symbols if symbol in tickers]
            if not symbols:
                continue

            timestamp = pd.to_datetime(record.get(date_col), errors="coerce", utc=True)
            if pd.isna(timestamp):
                continue
            body = str(record.get(body_col, "") or "").strip() if body_col else ""
            if body.lower() == "nan":
                body = ""
            text = " ".join([title, body]).strip()[:8000]
            rows.append({
                "timestamp": timestamp.to_pydatetime(),
                "title": title[:1000],
                "raw_text": text or title,
                "tickers": symbols,
                "url": str(record.get(url_col, "") or "").strip() if url_col else "",
                "source": str(record.get(source_col, "") or "").strip() if source_col else "",
            })
            if len(rows) >= limit:
                return rows
    return rows


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
) -> int:
    rows = collect_rows(
        input_csv=input_csv,
        limit=limit,
        tickers=tickers,
        date_column=date_column,
        ticker_column=ticker_column,
        title_column=title_column,
        body_column=body_column,
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
