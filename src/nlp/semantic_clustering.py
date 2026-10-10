"""
Semantic financial event clustering.

Uses finance-specific sentence embeddings instead of lexical TF-IDF
similarity.

We still incorporate financial metadata such as event category, ticker
overlap and timestamp proximity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sklearn.metrics.pairwise import cosine_similarity

from src.engine.schemas import Article
from src.nlp.event_classifier import (
    FinancialEventClassifier,
)
from src.nlp.finance_embeddings import FinanceEmbedder


@dataclass
class SemanticEventCluster:
    cluster_id: str
    article_ids: list[str]


class SemanticEventClusterer:

    def __init__(
        self,
        embedder: FinanceEmbedder | None = None,
        event_classifier=None,
        semantic_threshold: float = 0.60,
        max_hours_without_ticker: float = 24.0,
        max_hours_with_ticker: float = 48.0,
        semantic_threshold_with_ticker: float = 0.52,
    ):

        self.embedder = (
            embedder or FinanceEmbedder()
        )

        self.event_classifier = (
            event_classifier
            or FinancialEventClassifier()
        )

        self.semantic_threshold = (
            semantic_threshold
        )

        self.max_hours_without_ticker = (
            max_hours_without_ticker
        )

        self.max_hours_with_ticker = (
            max_hours_with_ticker
        )
        # A shared ticker is useful evidence, but does not prove two articles
        # describe the same event. Use a lower similarity threshold for
        # ticker-linked stories while still preventing obvious over-merging.
        self.semantic_threshold_with_ticker = float(
            semantic_threshold_with_ticker
        )
        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be in [0, 1].")
        if not 0.0 <= self.semantic_threshold_with_ticker <= 1.0:
            raise ValueError("semantic_threshold_with_ticker must be in [0, 1].")
        if self.max_hours_without_ticker < 0 or self.max_hours_with_ticker < 0:
            raise ValueError("Clustering time windows must be non-negative.")

    @staticmethod
    def _hours_between(
        a: datetime,
        b: datetime,
    ) -> float:

        if a.tzinfo is None:
            a = a.replace(
                tzinfo=timezone.utc
            )

        if b.tzinfo is None:
            b = b.replace(
                tzinfo=timezone.utc
            )

        return abs(
            (a - b).total_seconds()
        ) / 3600.0

    @staticmethod
    def _ticker_overlap(
        a: set[str],
        b: set[str],
    ) -> float:

        if not a or not b:
            return 0.0

        intersection = a & b
        union = a | b

        return (
            len(intersection)
            / len(union)
        )

    def _same_event(
        self,
        same_event_type: bool,
        semantic_similarity: float,
        ticker_overlap: float,
        hours_apart: float,
    ) -> bool:

        if not same_event_type:
            return False

        # Shared ticker and broad event label can still refer to distinct events.
        if ticker_overlap > 0:
            return (
                semantic_similarity >= self.semantic_threshold_with_ticker
                and hours_apart <= self.max_hours_with_ticker
            )

        # Macro/system-level event without a ticker.
        return (
            semantic_similarity
            >= self.semantic_threshold
            and
            hours_apart
            <= self.max_hours_without_ticker
        )

    def cluster(
        self,
        articles: list[Article],
    ) -> list[SemanticEventCluster]:

        if not articles:
            return []

        texts = [
            f"{article.title} "
            f"{article.raw_text}"
            for article in articles
        ]

        embeddings = self.embedder.encode(
            texts
        )

        similarity_matrix = (
            cosine_similarity(
                embeddings
            )
        )

        event_labels = [
            self.event_classifier.classify(
                text
            ).label
            for text in texts
        ]

        ticker_sets = [
            {
                ticker.upper()
                for ticker in article.tickers
            }
            for article in articles
        ]

        n = len(articles)

        parent = list(range(n))

        def find(x: int) -> int:

            while parent[x] != x:

                parent[x] = parent[
                    parent[x]
                ]

                x = parent[x]

            return x

        def union(
            a: int,
            b: int,
        ) -> None:

            root_a = find(a)
            root_b = find(b)

            if root_a != root_b:
                parent[root_b] = root_a

        for i in range(n):

            for j in range(i + 1, n):

                same_event_type = (
                    event_labels[i]
                    == event_labels[j]
                )

                semantic_similarity = float(
                    similarity_matrix[i, j]
                )

                ticker_overlap = (
                    self._ticker_overlap(
                        ticker_sets[i],
                        ticker_sets[j],
                    )
                )

                hours_apart = (
                    self._hours_between(
                        articles[i].timestamp,
                        articles[j].timestamp,
                    )
                )

                if self._same_event(
                    same_event_type=(
                        same_event_type
                    ),
                    semantic_similarity=(
                        semantic_similarity
                    ),
                    ticker_overlap=(
                        ticker_overlap
                    ),
                    hours_apart=(
                        hours_apart
                    ),
                ):
                    union(i, j)

        groups: dict[int, list[int]] = {}

        for i in range(n):

            root = find(i)

            groups.setdefault(
                root,
                [],
            ).append(i)

        clusters: list[SemanticEventCluster] = []

        for cluster_number, indices in enumerate(
            groups.values(),
            start=1,
        ):

            indices.sort(
                key=lambda idx:
                articles[idx].timestamp
            )

            clusters.append(
                SemanticEventCluster(
                    cluster_id=(
                        f"semantic_event_"
                        f"{cluster_number:03d}"
                    ),
                    article_ids=[
                        articles[i].article_id
                        for i in indices
                    ],
                )
            )

        return clusters