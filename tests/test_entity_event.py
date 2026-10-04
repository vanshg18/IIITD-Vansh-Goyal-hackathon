from src.ingestion.demo import load_portfolio
from src.nlp.entity_extractor import (
    FinancialEntityResolver,
    normalize_company_name,
    normalize_text,
)
from src.nlp.event_classifier import (
    FinancialEventClassifier,
)


def test_text_normalization():

    assert (
        normalize_text("Nova-Bank, Inc.")
        == "nova bank inc"
    )


def test_company_normalization():

    assert (
        normalize_company_name(
            "NovaBank Corporation"
        )
        == "novabank"
    )


def test_entity_resolution():

    portfolio = load_portfolio()

    resolver = FinancialEntityResolver(
        portfolio
    )

    text = (
        "NovaBank announced changes to its "
        "lending practices."
    )

    entities = resolver.extract_entities(text)

    names = [
        entity.name
        for entity in entities
    ]

    assert "NovaBank" in names


def test_ticker_resolution():

    portfolio = load_portfolio()

    resolver = FinancialEntityResolver(
        portfolio
    )

    tickers = resolver.extract_tickers(
        "$NVBK announced stronger guidance."
    )

    assert "NVBK" in tickers


def test_event_classifier_rate_hike():

    classifier = FinancialEventClassifier()

    result = classifier.classify(
        "The central bank unexpectedly announced "
        "a major rate hike."
    )

    assert result.label == "Monetary Policy"

    assert (
        "rate hike"
        in result.evidence_terms
    )

    assert result.confidence > 0.35


def test_event_classifier_regulatory():

    classifier = FinancialEventClassifier()

    result = classifier.classify(
        "The regulator launched an investigation "
        "into the company's compliance practices."
    )

    assert result.label == "Regulatory"

    assert "investigation" in result.evidence_terms


def test_event_classifier_credit():

    classifier = FinancialEventClassifier()

    result = classifier.classify(
        "The company received a credit downgrade "
        "following rising defaults."
    )

    assert result.label == "Credit"


def test_event_classifier_other():

    classifier = FinancialEventClassifier()

    result = classifier.classify(
        "A completely unrelated sentence."
    )

    assert result.label == "Other"

    assert result.confidence == 0.35

def test_software_does_not_trigger_geopolitical():

    classifier = FinancialEventClassifier()

    result = classifier.classify(
        "CloudMatrix secured a major enterprise "
        "contract that will expand its software revenue."
    )

    assert result.label != "Geopolitical"

    assert "war" not in result.evidence_terms

def test_war_triggers_geopolitical():

    classifier = FinancialEventClassifier()

    result = classifier.classify(
        "The region entered a major war."
    )

    assert result.label == "Geopolitical"

    assert "war" in result.evidence_terms