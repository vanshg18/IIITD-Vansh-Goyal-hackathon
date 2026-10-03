"""
Alpha Vantage News & Sentiment ingestion adapter.
"""

import os
from datetime import datetime, timezone

import requests

from src.engine.schemas import Article, SourceInfo
from src.ingestion.base import BaseSource
from src.ingestion.normalizer import clean_text


class AlphaVantageSource(BaseSource):
    """
    Alpha Vantage NEWS_SENTIMENT source.

    API key is read from the ALPHAVANTAGE_API_KEY environment variable.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    @property
    def name(self) -> str:
        return "Alpha Vantage"

    @property
    def source_type(self) -> str:
        return "news"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")

    def fetch(
        self,
        limit: int = 20,
        tickers: str | None = None,
        topics: str | None = None,
        timeout: int = 20,
        **kwargs,
    ) -> list[Article]:

        if not self.api_key:
            raise RuntimeError(
                "ALPHAVANTAGE_API_KEY is not configured."
            )

        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "sort": "LATEST",
            "limit": min(limit, 1000),
        }

        if tickers:
            params["tickers"] = tickers

        if topics:
            params["topics"] = topics

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=timeout,
        )

        response.raise_for_status()

        payload = response.json()

        # Alpha Vantage can return informational/error messages instead
        # of a feed. Detect those explicitly.
        if "feed" not in payload:

            message = (
                payload.get("Information")
                or payload.get("Note")
                or payload.get("Error Message")
            )

            raise RuntimeError(
                message or "Unexpected Alpha Vantage response."
            )

        articles: list[Article] = []

        for index, item in enumerate(payload["feed"]):

            title = clean_text(item.get("title"))

            if not title:
                continue

            summary = clean_text(item.get("summary"))

            # Use summary when present, otherwise title.
            raw_text = summary or title

            timestamp = self._parse_timestamp(
                item.get("time_published")
            )

            source_name = (
                item.get("source")
                or "Unknown"
            )

            url = item.get("url")

            tickers_from_feed = self._extract_tickers(item)

            articles.append(
                Article(
                    article_id=(
                        f"alphavantage_"
                        f"{timestamp.strftime('%Y%m%d%H%M%S')}_"
                        f"{index}"
                    ),
                    timestamp=timestamp,
                    source=SourceInfo(
                        name=source_name,
                        source_type="news",
                        url=url,
                    ),
                    title=title,
                    raw_text=raw_text,
                    entities=[],
                    tickers=tickers_from_feed,
                    url=url,
                )
            )

        return articles

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime:
        """
        Parse Alpha Vantage publication timestamp.

        Expected representation:
        YYYYMMDDTHHMMSS
        """

        if not value:
            return datetime.now(timezone.utc)

        try:
            return datetime.strptime(
                value,
                "%Y%m%dT%H%M%S",
            ).replace(tzinfo=timezone.utc)

        except ValueError:
            return datetime.now(timezone.utc)

    @staticmethod
    def _extract_tickers(item: dict) -> list[str]:
        """
        Extract ticker symbols from Alpha Vantage ticker sentiment metadata.
        """

        ticker_sentiment = item.get(
            "ticker_sentiment",
            [],
        )

        tickers: list[str] = []

        for ticker_info in ticker_sentiment:

            ticker = ticker_info.get("ticker")

            if not ticker:
                continue

            ticker = ticker.strip().upper()

            if ticker and ticker not in tickers:
                tickers.append(ticker)

        return tickers