"""Utilities for preparing reproducible historical-news samples.

A bounded bottom-k sample keyed by URL/content identity avoids taking only the
first rows of a file, which are often sorted by time or source. Sampling is
deterministic and memory is proportional to the requested sample size.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _select_column(columns, aliases) -> str | None:
    by_key = {_key(column): column for column in columns}
    for alias in aliases:
        if _key(alias) in by_key:
            return by_key[_key(alias)]
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    result = str(value).strip()
    return "" if result.lower() == "nan" else result


def collect_rows(
    input_csv: str | Path,
    limit: int,
    tickers: set[str] | None,
    date_column: str | None = None,
    ticker_column: str | None = None,
    title_column: str | None = None,
    body_column: str | None = None,
    sample_seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample usable news rows across the entire CSV, returning oldest first.

    The sampler keeps the lowest deterministic SHA-256 priorities. URL is the
    preferred identity; rows lacking URLs use publication time + source +
    normalized headline. Duplicate selected identities merge ticker lists.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    path = Path(input_csv)
    if not path.is_file():
        raise FileNotFoundError(path)

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
    desired_tickers = {str(item).strip().upper() for item in tickers} if tickers else None

    # The heap contains the worst (highest hash) selected item at its root.
    heap: list[tuple[int, str]] = []
    selected: dict[str, dict[str, Any]] = {}

    for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
        date_col = date_column or _select_column(chunk.columns, aliases["date"])
        ticker_col = ticker_column or _select_column(chunk.columns, aliases["ticker"])
        title_col = title_column or _select_column(chunk.columns, aliases["title"])
        body_col = body_column or _select_column(chunk.columns, aliases["body"])
        url_col = _select_column(chunk.columns, aliases["url"])
        source_col = _select_column(chunk.columns, aliases["source"])

        for record in chunk.to_dict(orient="records"):
            title = _text(record.get(title_col)) if title_col else ""
            if not title:
                continue

            raw_tickers = _text(record.get(ticker_col)).upper() if ticker_col else ""
            symbols = []
            for part in re.split(r"[;,|]", raw_tickers):
                symbol = part.strip().strip("[]'").strip('"').upper()
                if symbol:
                    symbols.append(symbol)
            symbols = list(dict.fromkeys(symbols))
            if desired_tickers:
                symbols = [symbol for symbol in symbols if symbol in desired_tickers]
            if not symbols:
                continue

            timestamp = pd.to_datetime(record.get(date_col), errors="coerce", utc=True) if date_col else pd.NaT
            if pd.isna(timestamp):
                continue
            body = _text(record.get(body_col)) if body_col else ""
            url = _text(record.get(url_col)) if url_col else ""
            source = _text(record.get(source_col)) if source_col else ""
            timestamp_value = timestamp.to_pydatetime()

            identity = (
                url.strip().lower()
                if url
                else "|".join([
                    timestamp_value.isoformat(),
                    source.casefold(),
                    re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip(),
                ])
            )
            priority = int(
                hashlib.sha256(f"{sample_seed}|{identity}".encode("utf-8")).hexdigest(),
                16,
            )

            candidate = {
                "timestamp": timestamp_value,
                "title": title[:1000],
                "raw_text": " ".join([title, body]).strip()[:8000] or title,
                "tickers": symbols,
                "url": url,
                "source": source,
            }

            # Merge ticker mentions for a duplicate identity that is in sample.
            if identity in selected:
                existing = selected[identity]
                existing["tickers"] = list(dict.fromkeys(existing["tickers"] + symbols))
                if not existing["raw_text"] and candidate["raw_text"]:
                    existing["raw_text"] = candidate["raw_text"]
                continue

            if len(selected) < limit:
                selected[identity] = candidate
                heapq.heappush(heap, (-priority, identity))
            else:
                worst_priority, worst_identity = -heap[0][0], heap[0][1]
                if priority < worst_priority:
                    heapq.heapreplace(heap, (-priority, identity))
                    selected.pop(worst_identity, None)
                    selected[identity] = candidate

    rows = sorted(
        selected.values(),
        key=lambda row: (row["timestamp"], tuple(row["tickers"]), row["title"]),
    )
    print(
        f"Scanned usable news rows; retained {len(rows):,} unique sampled articles "
        f"(requested limit={limit}, sample_seed={sample_seed})."
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a reproducible sample of historical news for a ticker universe."
    )
    parser.add_argument("--input", required=True, help="Historical news CSV.")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--tickers", default="")
    parser.add_argument("--sample-seed", type=int, default=42,
                        help="Reserved for a future sampler version; selection is currently hash-deterministic.")
    args = parser.parse_args()
    tickers = {item.strip().upper() for item in args.tickers.split(",") if item.strip()}
    rows = collect_rows(args.input, args.limit, tickers or None, sample_seed=args.sample_seed)
    print(f"Sample spans {rows[0]['timestamp']} through {rows[-1]['timestamp']}" if rows else "No matching records.")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
