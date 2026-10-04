"""
Baseline financial event clustering.

Articles are grouped when they are likely to describe the same
underlying financial event.

The current baseline uses:

1. Event-category agreement
2. Ticker/entity overlap
3. Text similarity
4. Temporal proximity

Later this can be upgraded to sentence embeddings + ANN retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.engine.schemas import Article
from src.nlp.event_classifier import (
    FinancialEventClassifier,
)


@dataclass
class EventCluster:
    """A group of articles representing one underlying event."""

    cluster_id: str
    article_ids: list[str]


class EventClusterer:

    def __init__(
        self,
        similarity_threshold: float = 0.20,
        max_hours_without_ticker: float = 24.0,
        max_hours_with_ticker: float = 48.0,
    ):
        self.similarity_threshold = (
            similarity_threshold
        )

        self.max_hours_without_ticker = (
            max_hours_without_ticker
        )

        self.max_hours_with_ticker = (
            max_hours_with_ticker
        )

        self.classifier = (
            FinancialEventClassifier()
        )

    @staticmethod
    def _hours_between(
        a: datetime,
        b: datetime,
    ) -> float:
        """
        Calculate absolute time difference in hours.
        """

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
        tickers_a: set[str],
        tickers_b: set[str],
    ) -> float:

        if not tickers_a or not tickers_b:
            return 0.0

        intersection = (
            tickers_a & tickers_b
        )

        union = (
            tickers_a | tickers_b
        )

        if not union:
            return 0.0

        return (
            len(intersection)
            / len(union)
        )

    def _should_cluster(
        self,
        *,
        same_event_type: bool,
        text_similarity: float,
        ticker_overlap: float,
        hours_apart: float,
        has_ticker_information: bool,
    ) -> bool:
        """
        Decide whether two articles should represent the same event.

        Financially meaningful metadata is deliberately prioritized
        over pure lexical similarity.
        """

        # Different event categories are strong evidence that these are
        # distinct events.
        if not same_event_type:
            return False

        # Same known ticker + same event category is strong evidence.
        if ticker_overlap > 0:
            return (
                hours_apart
                <= self.max_hours_with_ticker
            )

        # No ticker information.
        #
        # We require both:
        #   1. reasonable semantic/textual overlap
        #   2. close temporal proximity
        #
        # For example:
        #
        # "Central bank announces rate increase"
        # "Markets react to surprise rate hike"
        #
        # can now be recognized as the same macro event.
        return (
            text_similarity
            >= self.similarity_threshold
            and
            hours_apart
            <= self.max_hours_without_ticker
        )

    def cluster(
        self,
        articles: list[Article],
    ) -> list[EventCluster]:

        if not articles:
            return []

        if len(articles) == 1:
            return [
                EventCluster(
                    cluster_id="event_cluster_001",
                    article_ids=[
                        articles[0].article_id
                    ],
                )
            ]

        texts = [
            f"{article.title} {article.raw_text}"
            for article in articles
        ]

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )

        matrix = vectorizer.fit_transform(
            texts
        )

        text_similarity = cosine_similarity(
            matrix
        )

        # Determine event category once for each article.
        event_labels = [
            self.classifier.classify(
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

        # ---------------------------------------------------------------
        # Union-Find
        # ---------------------------------------------------------------

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
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

        # ---------------------------------------------------------------
        # Pairwise event similarity
        # ---------------------------------------------------------------

        for i in range(n):

            for j in range(i + 1, n):

                same_event_type = (
                    event_labels[i]
                    == event_labels[j]
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

                text_score = float(
                    text_similarity[i, j]
                )

                should_cluster = (
                    self._should_cluster(
                        same_event_type=(
                            same_event_type
                        ),
                        text_similarity=(
                            text_score
                        ),
                        ticker_overlap=(
                            ticker_overlap
                        ),
                        hours_apart=(
                            hours_apart
                        ),
                        has_ticker_information=(
                            bool(
                                ticker_sets[i]
                                or ticker_sets[j]
                            )
                        ),
                    )
                )

                if should_cluster:
                    union(i, j)

        # ---------------------------------------------------------------
        # Build final groups
        # ---------------------------------------------------------------

        groups: dict[int, list[int]] = {}

        for i in range(n):

            root = find(i)

            groups.setdefault(
                root,
                [],
            ).append(i)

        clusters: list[EventCluster] = []

        for cluster_number, indices in enumerate(
            groups.values(),
            start=1,
        ):

            # Preserve chronological order inside clusters.
            indices.sort(
                key=lambda idx:
                articles[idx].timestamp
            )

            clusters.append(
                EventCluster(
                    cluster_id=(
                        f"event_cluster_"
                        f"{cluster_number:03d}"
                    ),
                    article_ids=[
                        articles[i].article_id
                        for i in indices
                    ],
                )
            )

        return clusters