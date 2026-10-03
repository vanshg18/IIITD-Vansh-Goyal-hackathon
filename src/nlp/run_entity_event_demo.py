from src.ingestion.demo import load_articles, load_portfolio
from src.nlp.entity_extractor import FinancialEntityResolver
from src.nlp.event_classifier import FinancialEventClassifier


def main():

    articles = load_articles()
    portfolio = load_portfolio()

    entity_resolver = FinancialEntityResolver(
        portfolio
    )

    event_classifier = FinancialEventClassifier()

    enriched_articles = entity_resolver.enrich_articles(
        articles
    )

    print()
    print("=" * 100)
    print("EVENTPULSE — ENTITY + EVENT INTELLIGENCE")
    print("=" * 100)

    for article in enriched_articles:

        text = f"{article.title} {article.raw_text}"

        event = event_classifier.classify(text)

        print()
        print("-" * 100)

        print(
            f"ID:          {article.article_id}"
        )

        print(
            f"Title:       {article.title}"
        )

        print(
            f"Entities:    "
            f"{[entity.name for entity in article.entities]}"
        )

        print(
            f"Tickers:     {article.tickers}"
        )

        print(
            f"Event:       {event.label}"
        )

        print(
            f"Confidence:  {event.confidence:.3f}"
        )

        print(
            f"Evidence:    {event.evidence_terms}"
        )


if __name__ == "__main__":
    main()