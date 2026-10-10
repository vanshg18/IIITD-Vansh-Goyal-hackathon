from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

from src.ml.historical_news import collect_rows


def write_news_csv(path, rows):
    fieldnames = ["timestamp", "stock", "headline", "body_text", "url", "source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_bottom_k_sampling_is_order_independent_and_spans_history(tmp_path):
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(260):
        published = start + timedelta(days=7 * index)
        rows.append({
            "timestamp": published.isoformat(),
            "stock": "AAPL",
            "headline": f"AAPL financial headline {index}",
            "body_text": "Quarterly business update.",
            "url": f"https://news.example.test/article/{index}",
            "source": "NewsWire",
        })
    first_path = tmp_path / "news.csv"
    reversed_path = tmp_path / "news_reversed.csv"
    write_news_csv(first_path, rows)
    write_news_csv(reversed_path, list(reversed(rows)))

    first = collect_rows(first_path, limit=12, tickers={"AAPL"}, sample_seed=42)
    second = collect_rows(reversed_path, limit=12, tickers={"AAPL"}, sample_seed=42)

    first_ids = [row["url"] for row in first]
    second_ids = [row["url"] for row in second]
    assert first_ids == second_ids
    assert len(first) == 12
    assert first[0]["timestamp"] < first[-1]["timestamp"]
    assert (first[-1]["timestamp"] - first[0]["timestamp"]).days > 365
    assert first_ids != [f"https://news.example.test/article/{index}" for index in range(12)]


def test_duplicate_urls_merge_ticker_mentions(tmp_path):
    path = tmp_path / "duplicate_news.csv"
    rows = [
        {
            "timestamp": "2022-01-01T10:00:00Z",
            "stock": "AAPL",
            "headline": "Company issues earnings update",
            "body_text": "",
            "url": "https://news.example.test/shared-story",
            "source": "NewsWire",
        },
        {
            "timestamp": "2022-01-01T10:00:00Z",
            "stock": "MSFT",
            "headline": "Company issues earnings update",
            "body_text": "",
            "url": "https://news.example.test/shared-story",
            "source": "NewsWire",
        },
        {
            "timestamp": "2022-01-02T10:00:00Z",
            "stock": "AAPL",
            "headline": "Another company headline",
            "body_text": "",
            "url": "https://news.example.test/another-story",
            "source": "NewsWire",
        },
    ]
    write_news_csv(path, rows)

    sampled = collect_rows(path, limit=10, tickers={"AAPL", "MSFT"})

    assert len(sampled) == 2
    shared = next(row for row in sampled if row["url"].endswith("shared-story"))
    assert shared["tickers"] == ["AAPL", "MSFT"]


def test_sampling_rejects_nonpositive_limit(tmp_path):
    path = tmp_path / "news.csv"
    write_news_csv(path, [])
    try:
        collect_rows(path, limit=0, tickers={"AAPL"})
    except ValueError as exc:
        assert "at least 1" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for limit=0")
