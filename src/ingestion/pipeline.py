"""
Unified ingestion pipeline.

Multiple sources -> common Article representation -> deduplication, with
per-source status so live runs never silently look fully successful.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.engine.schemas import Article
from src.ingestion.base import BaseSource
from src.ingestion.normalizer import deduplicate_articles


@dataclass(frozen=True)
class SourceFetchStatus:
    source_name: str
    success: bool
    articles_returned: int
    error: str | None = None


class IngestionPipeline:
    """Orchestrates independent data sources and records their outcomes."""

    def __init__(self, sources: list[BaseSource]):
        if not sources:
            raise ValueError("At least one source is required.")
        self.sources = sources
        self.last_source_statuses: list[SourceFetchStatus] = []

    def fetch_all(self, limit_per_source: int = 20, **kwargs) -> list[Article]:
        all_articles: list[Article] = []
        statuses: list[SourceFetchStatus] = []

        for source in self.sources:
            try:
                articles = source.fetch(limit=limit_per_source, **kwargs)
                if not isinstance(articles, list):
                    raise TypeError(
                        f"{source.name}.fetch() must return list[Article], "
                        f"got {type(articles).__name__}"
                    )
                invalid = [item for item in articles if not isinstance(item, Article)]
                if invalid:
                    raise TypeError(
                        f"{source.name}.fetch() returned {len(invalid)} non-Article records"
                    )
                all_articles.extend(articles)
                status = SourceFetchStatus(
                    source_name=source.name,
                    success=True,
                    articles_returned=len(articles),
                )
                print(f"[OK] {source.name}: {len(articles)} articles")
            except Exception as exc:
                status = SourceFetchStatus(
                    source_name=source.name,
                    success=False,
                    articles_returned=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
                print(f"[WARN] {source.name} failed: {status.error}")
            statuses.append(status)

        self.last_source_statuses = statuses
        unique_articles = deduplicate_articles(all_articles)
        print(
            f"[INFO] Ingestion total: {len(all_articles)} records, "
            f"{len(unique_articles)} unique after deduplication."
        )
        return unique_articles
