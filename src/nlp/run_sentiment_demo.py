from src.ingestion.demo import load_articles
from src.nlp.sentiment import FinancialSentimentAnalyzer


def main():

    articles = load_articles()

    analyzer = FinancialSentimentAnalyzer()

    predictions = analyzer.predict_batch(
        [article.raw_text for article in articles],
        batch_size=4,
    )

    print()
    print("=" * 90)
    print("EVENTPULSE — FINANCIAL SENTIMENT")
    print("=" * 90)

    for article, prediction in zip(
        articles,
        predictions,
    ):

        print()
        print(f"ID:         {article.article_id}")
        print(f"Title:      {article.title}")
        print(f"Sentiment:  {prediction.score:+.4f}")
        print(f"Label:      {prediction.label}")
        print(f"Confidence: {prediction.confidence:.4f}")


if __name__ == "__main__":
    main()