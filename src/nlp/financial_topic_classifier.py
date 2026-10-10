"""
Finance-specific pretrained topic classifier.

Primary model:
    leonas5555/finnews-topic-single-classify

The model was fine-tuned on the Twitter Financial News Topic dataset.
We use it as a finance-domain event classification component and map
its topics into EventPulse's broader event ontology.

Topics that cannot be mapped safely are handed to the zero-shot
fallback classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import pipeline

from src.engine.schemas import EventClassification
from src.nlp.model_config import (
    EVENT_TOPIC_MODEL,
    get_device_index,
)


# ---------------------------------------------------------------------------
# MODEL TOPIC → EVENTPULSE EVENT
# ---------------------------------------------------------------------------

TOPIC_TO_EVENT = {
    "Fed | Central Banks": "Monetary Policy",

    "Company | Product News": "Product/Business",

    "Treasuries | Corporate Debt": "Credit",

    "Earnings": "Earnings",

    "Legal | Regulation": "Regulatory",

    "M&A | Investments": "Corporate Action",

    "Dividend": "Corporate Action",

    "IPO": "Corporate Action",

    "Markets": "Market",

    "Stock Movement": "Market",

    "Currencies": "Market",

    "Politics": "Geopolitical",

    "Personnel Change": "Corporate Action",
}


@dataclass(frozen=True)
class TopicPrediction:
    """Internal prediction from the finance-topic model."""

    source_topic: str
    mapped_event: str | None
    score: float


class FinancialTopicClassifier:
    """
    Pretrained finance-specific topic classifier.
    """

    MODEL_NAME = EVENT_TOPIC_MODEL

    def __init__(
        self,
        device: int | None = None,
    ):

        if device is None:
            device = get_device_index()

        self.classifier = pipeline(
            task="text-classification",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            device=device,
            top_k=None,
        )

    @staticmethod
    def _unwrap(
        result,
    ) -> list[dict]:
        """
        Normalize Transformers output across versions.
        """

        if not result:
            return []

        if isinstance(result[0], list):
            result = result[0]

        return result

    def predict(
        self,
        text: str,
    ) -> TopicPrediction:

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        raw = self.classifier(
            text,
            truncation=True,
        )

        predictions = self._unwrap(raw)

        if not predictions:
            raise RuntimeError(
                "Financial topic model returned no prediction."
            )

        best = max(
            predictions,
            key=lambda item: float(
                item["score"]
            ),
        )

        source_topic = best["label"]

        mapped_event = TOPIC_TO_EVENT.get(
            source_topic
        )

        return TopicPrediction(
            source_topic=source_topic,
            mapped_event=mapped_event,
            score=float(best["score"]),
        )

    def classify(
        self,
        text: str,
        minimum_confidence: float = 0.70,
    ) -> EventClassification | None:
        """
        Convert a finance-topic prediction into an EventPulse event.

        Returns None when:
        - topic is not safely mappable
        - model confidence is below the routing threshold

        Those cases are sent to the zero-shot fallback.
        """

        prediction = self.predict(text)

        if (
            prediction.mapped_event is None
            or prediction.score < minimum_confidence
        ):
            return None

        return EventClassification(
            label=prediction.mapped_event,
            confidence=prediction.score,
            evidence_terms=[
                prediction.source_topic
            ],
            model_name=self.MODEL_NAME,
            scores={
                prediction.mapped_event:
                    prediction.score
            },
        )