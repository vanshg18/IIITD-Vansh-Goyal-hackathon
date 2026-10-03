from src.nlp.sentiment import FinancialSentimentAnalyzer


def main():

    analyzer = FinancialSentimentAnalyzer()

    examples = [
        "The company reported record profits and raised its annual guidance.",
        "The company warned that revenue is expected to decline sharply.",
        "The central bank maintained its policy rate unchanged.",
    ]

    for text in examples:

        prediction = analyzer.predict(text)

        print()
        print("-" * 70)
        print(text)
        print(
            f"Label:      {prediction.label}"
        )
        print(
            f"Score:      {prediction.score:.4f}"
        )
        print(
            f"Confidence: {prediction.confidence:.4f}"
        )


if __name__ == "__main__":
    main()