"""
Unified ingestion pipeline.

Multiple sources -> common Article representation -> deduplication.
"""

from src.engine.schemas import Article
from src.ingestion.base import BaseSource
from src.ingestion.normalizer import deduplicate_articles


class IngestionPipeline:
    """
    Orchestrates multiple independent data sources.
    """

    def __init__(self, sources: list[BaseSource]):
        if not sources:
            raise ValueError("At least one source is required.")

        self.sources = sources

    def fetch_all(
        self,
        limit_per_source: int = 20,
        **kwargs,
    ) -> list[Article]:

        all_articles: list[Article] = []

        for source in self.sources:

            try:
                articles = source.fetch(
                    limit=limit_per_source,
                    **kwargs,
                )

                all_articles.extend(articles)

                print(
                    f"[OK] {source.name}: "
                    f"{len(articles)} articles"
                )

            except Exception as exc:
                # One failed source must not kill the entire pipeline.
                print(
                    f"[WARN] {source.name} failed: {exc}"
                )

        return deduplicate_articles(all_articles)