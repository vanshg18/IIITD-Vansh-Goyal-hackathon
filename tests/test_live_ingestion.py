from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.engine.schemas import Article
from src.ingestion.gdelt import GDELTSource
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.sec_edgar import SECEdgarSource
from src.ingestion.storage import append_articles_jsonl, load_articles_jsonl


class FakeResponse:
    def __init__(self, *, payload=None, text=""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response_by_url):
        self.response_by_url = response_by_url
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.response_by_url[url]
        if isinstance(response, Exception):
            raise response
        return response


def test_gdelt_parses_real_api_shape_and_clamps_limit():
    session = FakeSession({
        GDELTSource.BASE_URL: FakeResponse(payload={
            "articles": [
                {
                    "title": "Markets fall after unexpected rate decision",
                    "url": "https://publisher.example/article-1",
                    "seendate": "20261010123000",
                    "domain": "publisher.example",
                },
                {"title": "", "url": "https://publisher.example/blank"},
            ]
        })
    })
    articles = GDELTSource(session=session).fetch(limit=500)

    assert len(articles) == 1
    assert articles[0].article_id.startswith("gdelt_")
    assert articles[0].title == "Markets fall after unexpected rate decision"
    assert articles[0].raw_text == articles[0].title
    assert articles[0].timestamp == datetime(2026, 10, 10, 12, 30, tzinfo=timezone.utc)
    assert session.calls[0][1]["params"]["maxrecords"] == 250


def test_gdelt_rejects_unexpected_payload():
    session = FakeSession({
        GDELTSource.BASE_URL: FakeResponse(payload={"message": "rate limit"})
    })
    with pytest.raises(RuntimeError, match="rate limit"):
        GDELTSource(session=session).fetch(limit=10)


def test_sec_requires_contact_user_agent():
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        SECEdgarSource(user_agent="").fetch(limit=5)


def test_sec_parses_filing_and_maps_ticker():
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>8-K - Current report</title>
        <link rel="alternate" href="https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/aapl-20261010.htm" />
        <updated>2026-10-10T10:30:00-04:00</updated>
        <summary type="html">Filed: 2026-10-10 &amp; current report</summary>
        <category term="8-K" />
      </entry>
    </feed>"""
    session = FakeSession({
        SECEdgarSource.BASE_URL: FakeResponse(text=feed),
        SECEdgarSource.COMPANY_TICKERS_URL: FakeResponse(payload={
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
        }),
    })

    articles = SECEdgarSource(
        user_agent="EventPulse tests tests@example.com",
        session=session,
    ).fetch(limit=5)

    assert len(articles) == 1
    article = articles[0]
    assert article.source.source_type == "official"
    assert article.tickers == ["AAPL"]
    assert article.entities[0].name == "Apple Inc."
    assert "Form 8-K" in article.raw_text
    assert article.timestamp == datetime(2026, 10, 10, 14, 30, tzinfo=timezone.utc)


def test_pipeline_exposes_source_failure_status():
    class BrokenSource:
        @property
        def name(self):
            return "Broken"

        @property
        def source_type(self):
            return "news"

        def fetch(self, limit=20, **kwargs):
            raise RuntimeError("upstream unavailable")

    pipeline = IngestionPipeline([BrokenSource()])
    assert pipeline.fetch_all(limit_per_source=2) == []
    assert len(pipeline.last_source_statuses) == 1
    assert pipeline.last_source_statuses[0].success is False
    assert "upstream unavailable" in pipeline.last_source_statuses[0].error


def test_jsonl_storage_appends_deduplicated_articles(tmp_path):
    article = Article(
        article_id="a1",
        timestamp=datetime(2026, 10, 10, tzinfo=timezone.utc),
        source={"name": "Publisher", "source_type": "news", "url": "https://example.test/1"},
        title="Company reports strong earnings",
        raw_text="Company reports strong earnings",
        url="https://example.test/1",
    )
    path = tmp_path / "nested" / "articles.jsonl"

    assert append_articles_jsonl([article], path) == 1
    assert append_articles_jsonl([article], path) == 0
    loaded = load_articles_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0].article_id == "a1"
