from src.ingestion import (
    DemoSource,
    IngestionPipeline,
)
from src.ingestion.normalizer import (
    deduplicate_articles,
    normalize_title,
)


def test_demo_source():
    source = DemoSource()

    articles = source.fetch(limit=5)

    assert len(articles) == 5
    assert all(article.raw_text for article in articles)


def test_pipeline():
    pipeline = IngestionPipeline(
        sources=[
            DemoSource(),
        ]
    )

    articles = pipeline.fetch_all(
        limit_per_source=5,
    )

    assert len(articles) == 5


def test_title_normalization():
    title = "  Rate   Hike!!!  "

    assert normalize_title(title) == "rate hike"


def test_deduplication():
    source = DemoSource()

    articles = source.fetch(limit=5)

    duplicated = articles + articles

    unique = deduplicate_articles(
        duplicated
    )

    assert len(unique) == 5