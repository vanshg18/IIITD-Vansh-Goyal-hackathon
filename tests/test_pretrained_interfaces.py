from src.nlp.pretrained_event import (
    EVENT_LABEL_DESCRIPTIONS,
)


def test_event_ontology_is_complete():

    expected = {
        "Monetary Policy",
        "Earnings",
        "Credit",
        "Regulatory",
        "Geopolitical",
        "Supply Chain",
        "Corporate Action",
        "Product/Business",
        "Market",
        "Other",
    }

    assert (
        set(EVENT_LABEL_DESCRIPTIONS)
        == expected
    )


def test_event_ontology_has_no_empty_descriptions():

    assert all(
        description.strip()
        for description
        in EVENT_LABEL_DESCRIPTIONS.values()
    )