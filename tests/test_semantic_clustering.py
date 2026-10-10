from datetime import datetime, timedelta, timezone

import numpy as np

from src.engine.schemas import Article, EventClassification
from src.nlp.semantic_clustering import SemanticEventClusterer


class FakeEmbedder:
    """Return deterministic embeddings without downloading a transformer."""

    def encode(self, texts, batch_size=32):
        rows = []
        for text in texts:
            lowered = text.lower()
            if "rate hike" in lowered or "interest rate" in lowered:
                rows.append([1.0, 0.0, 0.0])
            elif "fraud investigation" in lowered:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.99, 0.01, 0.0])
        return np.asarray(rows, dtype=np.float32)


class FakeClassifier:
    def classify(self, text):
        return EventClassification(label="Credit", confidence=0.9, model_name="test")


def make_article(article_id, title, hours_after=0):
    return Article(
        article_id=article_id,
        timestamp=datetime(2026, 10, 10, tzinfo=timezone.utc) + timedelta(hours=hours_after),
        source={"name": f"publisher-{article_id}", "source_type": "news"},
        title=title,
        raw_text=title,
        tickers=["ACME"],
    )


def test_same_ticker_and_event_label_do_not_force_unrelated_articles_into_one_cluster():
    articles = [
        make_article("a1", "ACME fraud investigation expands", 0),
        make_article("a2", "ACME interest rate outlook improves", 2),
    ]
    clusterer = SemanticEventClusterer(
        embedder=FakeEmbedder(),
        event_classifier=FakeClassifier(),
        semantic_threshold_with_ticker=0.52,
    )
    clusters = clusterer.cluster(articles)
    assert len(clusters) == 2
    assert {tuple(cluster.article_ids) for cluster in clusters} == {("a1",), ("a2",)}


def test_semantically_related_ticker_stories_are_still_clustered():
    articles = [
        make_article("a1", "ACME unexpected rate hike announced", 0),
        make_article("a2", "ACME interest rate outlook update", 3),
    ]
    clusterer = SemanticEventClusterer(
        embedder=FakeEmbedder(),
        event_classifier=FakeClassifier(),
        semantic_threshold_with_ticker=0.52,
    )
    clusters = clusterer.cluster(articles)
    assert len(clusters) == 1
    assert set(clusters[0].article_ids) == {"a1", "a2"}


def test_ticker_linked_stories_outside_time_window_do_not_cluster():
    articles = [
        make_article("a1", "ACME rate hike announced", 0),
        make_article("a2", "ACME interest rate update", 60),
    ]
    clusterer = SemanticEventClusterer(
        embedder=FakeEmbedder(),
        event_classifier=FakeClassifier(),
        semantic_threshold_with_ticker=0.52,
        max_hours_with_ticker=48.0,
    )
    assert len(clusterer.cluster(articles)) == 2
