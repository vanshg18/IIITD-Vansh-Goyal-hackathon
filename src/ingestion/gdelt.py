"""
GDELT DOC 2.0 ingestion adapter.

GDELT DOC provides searchable news coverage and article-list output.
"""

from datetime import datetime, timezone

import requests

from src.engine.schemas import Article, SourceInfo
from src.ingestion.base import BaseSource
from src.ingestion.normalizer import clean_text


class GDELTSource(BaseSource):
    """
    GDELT DOC 2.0 source adapter.
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    @property
    def name(self) -> str:
        return "GDELT"

    @property
    def source_type(self) -> str:
        return "news"

    def fetch(
        self,
        limit: int = 20,
        query: str = "finance OR markets OR economy",
        timespan: str = "1day",
        timeout: int = 20,
        **kwargs,
    ) -> list[Article]:

        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": min(limit, 250),
            "timespan": timespan,
            "sort": "datedesc",
            "format": "json",
        }

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=timeout,
        )

        response.raise_for_status()

        payload = response.json()

        raw_articles = payload.get("articles", [])

        articles: list[Article] = []

        for index, item in enumerate(raw_articles):

            title = clean_text(item.get("title"))

            if not title:
                continue

            timestamp = self._parse_timestamp(
                item.get("seendate")
            )

            url = item.get("url")

            source_domain = item.get("domain") or "Unknown"

            # GDELT article-list responses can primarily expose headline
            # metadata rather than full article bodies. We therefore use
            # the headline as the initial text representation. A later
            # enrichment stage can fetch article content when available.
            raw_text = title

            article = Article(
                article_id=f"gdelt_{timestamp.strftime('%Y%m%d%H%M%S')}_{index}",
                timestamp=timestamp,
                source=SourceInfo(
                    name=source_domain,
                    source_type="news",
                    url=url,
                ),
                title=title,
                raw_text=raw_text,
                entities=[],
                tickers=[],
                url=url,
            )

            articles.append(article)

        return articles

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime:
        """
        Parse GDELT's timestamp representation.

        Typical format:
        YYYYMMDDHHMMSS
        """

        if not value:
            return datetime.now(timezone.utc)

        try:
            return datetime.strptime(
                value,
                "%Y%m%d%H%M%S",
            ).replace(tzinfo=timezone.utc)

        except ValueError:
            return datetime.now(timezone.utc)