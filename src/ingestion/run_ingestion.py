from src.ingestion import DemoSource, IngestionPipeline


def main():
    pipeline = IngestionPipeline(
        sources=[
            DemoSource(),
        ]
    )

    articles = pipeline.fetch_all(
        limit_per_source=5,
    )

    print()
    print("=" * 70)
    print("EVENTPULSE INGESTION TEST")
    print("=" * 70)

    for article in articles:

        print()
        print(f"ID:       {article.article_id}")
        print(f"Source:   {article.source.name}")
        print(f"Time:     {article.timestamp}")
        print(f"Title:    {article.title}")
        print(f"Tickers:  {article.tickers}")
        print(f"URL:      {article.url}")


if __name__ == "__main__":
    main()