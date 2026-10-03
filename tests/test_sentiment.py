from src.nlp.sentiment import (
    FinancialSentimentAnalyzer,
)


def test_positive_sentiment_conversion():

    prediction = (
        FinancialSentimentAnalyzer
        .probabilities_to_prediction(
            {
                "positive": 0.85,
                "negative": 0.05,
                "neutral": 0.10,
            }
        )
    )

    assert prediction.label == "positive"
    assert abs(prediction.score - 0.80) < 1e-6


def test_negative_sentiment_conversion():

    prediction = (
        FinancialSentimentAnalyzer
        .probabilities_to_prediction(
            {
                "positive": 0.10,
                "negative": 0.75,
                "neutral": 0.15,
            }
        )
    )

    assert prediction.label == "negative"
    assert abs(prediction.score + 0.65) < 1e-6


def test_neutral_sentiment_conversion():

    prediction = (
        FinancialSentimentAnalyzer
        .probabilities_to_prediction(
            {
                "positive": 0.31,
                "negative": 0.29,
                "neutral": 0.40,
            }
        )
    )

    assert prediction.label == "neutral"
    assert abs(prediction.score - 0.02) < 1e-6