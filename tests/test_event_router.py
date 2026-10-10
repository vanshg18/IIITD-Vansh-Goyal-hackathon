
from src.engine.schemas import EventClassification
from src.nlp.event_router import EventClassifierRouter
from src.nlp.financial_topic_classifier import TopicPrediction


class FakeTopicClassifier:
    def __init__(self, result):
        self.result = result

    def predict(self, text):
        return self.result


class FakeFallbackClassifier:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def classify(self, text):
        self.calls += 1
        return self.result


def test_models_agree():
    primary = FakeTopicClassifier(
        TopicPrediction(
            source_topic="Fed | Central Banks",
            mapped_event="Monetary Policy",
            score=0.95,
        )
    )

    fallback = FakeFallbackClassifier(
        EventClassification(
            label="Monetary Policy",
            confidence=0.91,
            model_name="zero-shot-test",
            scores={
                "Monetary Policy": 0.91,
                "Other": 0.04,
            },
        )
    )

    router = EventClassifierRouter(
        topic_classifier=primary,
        fallback_classifier=fallback,
    )

    result = router.classify(
        "The central bank unexpectedly raised rates."
    )

    assert result.label == "Monetary Policy"
    assert result.model_name == "router:consensus"
    assert fallback.calls == 1


def test_strong_zero_shot_overrules_wrong_topic():
    primary = FakeTopicClassifier(
        TopicPrediction(
            source_topic="Markets",
            mapped_event="Market",
            score=0.94,
        )
    )

    fallback = FakeFallbackClassifier(
        EventClassification(
            label="Monetary Policy",
            confidence=0.64,
            model_name="zero-shot-test",
            scores={
                "Monetary Policy": 0.64,
                "Market": 0.30,
                "Other": 0.03,
            },
        )
    )

    router = EventClassifierRouter(
        topic_classifier=primary,
        fallback_classifier=fallback,
    )

    result = router.classify(
        "Markets react to a surprise rate hike."
    )

    assert result.label == "Monetary Policy"
    assert result.model_name == "router:zero-shot-adjudicated"


def test_strong_zero_shot_overrules_product_misclassification():
    primary = FakeTopicClassifier(
        TopicPrediction(
            source_topic="Company | Product News",
            mapped_event="Product/Business",
            score=0.96,
        )
    )

    fallback = FakeFallbackClassifier(
        EventClassification(
            label="Regulatory",
            confidence=0.82,
            model_name="zero-shot-test",
            scores={
                "Regulatory": 0.82,
                "Other": 0.11,
                "Corporate Action": 0.02,
            },
        )
    )

    router = EventClassifierRouter(
        topic_classifier=primary,
        fallback_classifier=fallback,
    )

    result = router.classify(
        "NovaBank denies major compliance concerns."
    )

    assert result.label == "Regulatory"


def test_topic_model_wins_when_zero_shot_is_ambiguous():
    primary = FakeTopicClassifier(
        TopicPrediction(
            source_topic="Company | Product News",
            mapped_event="Product/Business",
            score=0.96,
        )
    )

    fallback = FakeFallbackClassifier(
        EventClassification(
            label="Other",
            confidence=0.37,
            model_name="zero-shot-test",
            scores={
                "Other": 0.37,
                "Product/Business": 0.31,
                "Market": 0.11,
            },
        )
    )

    router = EventClassifierRouter(
        topic_classifier=primary,
        fallback_classifier=fallback,
    )

    result = router.classify(
        "CloudMatrix secures a major enterprise contract."
    )

    assert result.label == "Product/Business"
    assert result.model_name == "router:finance-topic-selected"


def test_unmapped_topic_uses_discriminative_zero_shot():
    primary = FakeTopicClassifier(
        TopicPrediction(
            source_topic="Energy | Oil",
            mapped_event=None,
            score=0.90,
        )
    )

    fallback = FakeFallbackClassifier(
        EventClassification(
            label="Supply Chain",
            confidence=0.40,
            model_name="zero-shot-test",
            scores={
                "Supply Chain": 0.40,
                "Market": 0.25,
                "Other": 0.15,
            },
        )
    )

    router = EventClassifierRouter(
        topic_classifier=primary,
        fallback_classifier=fallback,
    )

    result = router.classify(
        "Global supply disruptions push energy prices higher."
    )

    assert result.label == "Supply Chain"
    assert result.model_name == "router:zero-shot-adjudicated"


def test_router_caches_identical_text():
    primary = FakeTopicClassifier(
        TopicPrediction(
            source_topic="Earnings",
            mapped_event="Earnings",
            score=0.90,
        )
    )

    fallback = FakeFallbackClassifier(
        EventClassification(
            label="Earnings",
            confidence=0.80,
            model_name="zero-shot-test",
            scores={
                "Earnings": 0.80,
                "Other": 0.10,
            },
        )
    )

    router = EventClassifierRouter(
        topic_classifier=primary,
        fallback_classifier=fallback,
    )

    text = "The company reports record quarterly earnings."

    router.classify(text)
    router.classify(text)

    assert fallback.calls == 1
