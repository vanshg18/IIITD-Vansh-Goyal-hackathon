"""
GDELT DOC 2.0 ingestion adapter.

GDELT returns article metadata and URLs rather than guaranteed full article
bodies. We preserve provenance and clearly treat the headline as the text
when no summary/body is supplied by the API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

import requests

from src.engine.schemas import Article, SourceInfo
from src.ingestion.base import BaseSource
from src.ingestion.normalizer import clean_text


class GDELTSource(BaseSource):
    """Fetch recent news articles from the public GDELT DOC API."""

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    @property
    def name(self) -> str:
        return "GDELT"

    @property
    def source_type(self) -> str:
        return "news"

    def fetch(
        self,
        limit: int = 20,
        query: str = '("stock market" OR finance OR markets OR economy OR earnings)',
        timespan: str = "1day",
        timeout: int = 20,
        **kwargs: Any,
    ) -> list[Article]:
        if limit < 1:
            return []

        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": min(int(limit), 250),
            "timespan": timespan,
            "sort": "datedesc",
            "format": "json",
        }
        response = self.session.get(
            self.BASE_URL,
            params=params,
            timeout=timeout,
            headers={"User-Agent": "EventPulse/1.0 (educational financial-risk research)"},
        )
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict):
            raise RuntimeError("GDELT returned a non-object JSON response.")

        raw_articles = payload.get("articles")
        if raw_articles is None:
            detail = payload.get("message") or payload.get("error") or str(payload)[:300]
            raise RuntimeError(f"Unexpected GDELT response: {detail}")
        if not isinstance(raw_articles, list):
            raise RuntimeError("GDELT 'articles' field was not a list.")

        articles: list[Article] = []
        for item in raw_articles:
            if not isinstance(item, dict):
                continue

            title = clean_text(item.get("title"))
            if not title:
                continue

            url = clean_text(item.get("url")) or None
            timestamp = self._parse_timestamp(item.get("seendate"))
            domain = clean_text(item.get("domain")) or "Unknown publisher"
            # GDELT's article-list endpoint may not provide article body text.
            # A title-only record is explicit and still useful for headline NLP.
            raw_text = title
            stable_identity = url or f"{domain}|{title.lower()}"
            digest = sha256(stable_identity.encode("utf-8")).hexdigest()[:20]

            articles.append(
                Article(
                    article_id=f"gdelt_{digest}",
                    timestamp=timestamp,
                    source=SourceInfo(
                        name=domain,
                        source_type="news",
                        url=url,
                    ),
                    title=title,
                    raw_text=raw_text,
                    entities=[],
                    tickers=[],
                    url=url,
                )
            )

        return articles[: min(int(limit), 250)]

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime:
        """Parse GDELT's YYYYMMDDHHMMSS timestamp into an aware UTC datetime."""
        if not value:
            return datetime.now(timezone.utc)
        try:
            return datetime.strptime(value, "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except (TypeError, ValueError):
            return datetime.now(timezone.utc)
