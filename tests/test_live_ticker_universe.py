from datetime import datetime, timezone

from src.engine.schemas import Article
from src.module_a.rebalancer import load_index_portfolio
from src.nlp.entity_extractor import FinancialEntityResolver


def test_company_name_mentions_resolve_to_real_tickers():
    resolver = FinancialEntityResolver(load_index_portfolio())
    article = Article(
        article_id="live_msft",
        timestamp=datetime(2026, 10, 10, 12, tzinfo=timezone.utc),
        source={"name": "Live News", "source_type": "news", "url": "https://example.test/msft"},
        title="Microsoft shares slide after new regulatory inquiry",
        raw_text="Microsoft shares slide after new regulatory inquiry.",
    )

    enriched = resolver.enrich_article(article)

    assert "MSFT" in enriched.tickers
    assert any(
        entity.name == "Microsoft" and entity.entity_type == "ORGANIZATION"
        for entity in enriched.entities
    )


def test_ticker_symbol_and_source_metadata_are_preserved():
    resolver = FinancialEntityResolver(load_index_portfolio())
    article = Article(
        article_id="live_aapl",
        timestamp=datetime(2026, 10, 10, 12, tzinfo=timezone.utc),
        source={"name": "SEC EDGAR", "source_type": "official"},
        title="Form 8-K filed",
        raw_text="Current report",
        tickers=["AAPL"],
    )

    enriched = resolver.enrich_article(article)

    assert "AAPL" in enriched.tickers

def test_common_issuer_aliases_map_to_canonical_tickers():
    resolver = FinancialEntityResolver(load_index_portfolio())
    examples = [
        ("Alphabet announces a new corporate update", "GOOGL"),
        ("Google announces a new corporate update", "GOOGL"),
        ("JP Morgan reports quarterly earnings", "JPM"),
        ("Meta shares respond to earnings", "META"),
        ("ExxonMobil announces an investment", "XOM"),
    ]

    for title, expected_ticker in examples:
        article = Article(
            article_id=f"alias_{expected_ticker}_{len(title)}",
            timestamp=datetime(2026, 10, 10, 12, tzinfo=timezone.utc),
            source={"name": "Live News", "source_type": "news"},
            title=title,
            raw_text=title,
        )
        enriched = resolver.enrich_article(article)
        assert expected_ticker in enriched.tickers, (title, enriched.tickers)
