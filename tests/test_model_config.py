from src.nlp.model_config import (
    EMBEDDING_MODEL,
    EVENT_ZERO_SHOT_MODEL,
    NER_MODEL,
    SENTIMENT_MODEL,
)


def test_model_identifiers():

    assert SENTIMENT_MODEL == (
        "ProsusAI/finbert"
    )

    assert NER_MODEL == (
        "whataboutyou-ai/financial_bert"
    )

    assert EMBEDDING_MODEL == (
        "BAAI/bge-small-en-v1.5"
    )

    assert EVENT_ZERO_SHOT_MODEL == (
        "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
    )