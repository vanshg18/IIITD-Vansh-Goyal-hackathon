"""
Zero-shot financial event classifier.

Uses a pretrained NLI-based zero-shot Transformer rather than
hand-written event keyword rules.

Current baseline model:
    MoritzLaurer/ModernBERT-base-zeroshot-v2.0
"""

from __future__ import annotations

from transformers import pipeline

from src.engine.schemas import EventClassification
from src.nlp.model_config import (
    EVENT_ZERO_SHOT_MODEL,
    get_device_index,
)


# ---------------------------------------------------------------------------
# FINANCIAL EVENT ONTOLOGY
# ---------------------------------------------------------------------------

EVENT_LABEL_DESCRIPTIONS = {
    "Monetary Policy":
        "monetary policy, central bank decisions, "
        "interest rates, inflation policy and rate changes",

    "Earnings":
        "company earnings, revenue, profit, quarterly results, "
        "financial guidance and earnings outlook",

    "Credit":
        "credit ratings, credit downgrades, defaults, debt quality, "
        "loan losses and borrower credit risk",

    "Regulatory":
        "regulation, regulators, investigations, compliance, "
        "legal actions and regulatory reviews",

    "Geopolitical":
        "wars, military conflicts, sanctions, tariffs and "
        "geopolitical tensions between countries",

    "Supply Chain":
        "supply shortages, logistics disruptions, shipping problems, "
        "supplier disruptions and production interruptions",

    "Corporate Action":
        "mergers, acquisitions, buybacks, divestitures, "
        "takeovers and major corporate restructuring",

    "Product/Business":
        "new products, product launches, partnerships, contracts, "
        "business expansion and commercial activity",

    "Market":
        "broad financial market movements, equity markets, "
        "bond markets and trading conditions",

    "Other":
        "financial news that does not fit the above event categories",
}


class PretrainedEventClassifier:
    """
    Zero-shot Transformer event classifier.

    This is our immediate pretrained baseline. It can later be replaced
    by a fine-tuned classifier implementing the same interface.
    """

    MODEL_NAME = EVENT_ZERO_SHOT_MODEL

    def __init__(
        self,
        device: int | None = None,
        multi_label: bool = False,
    ):

        if device is None:
            device = get_device_index()

        self.multi_label = multi_label

        self.classifier = pipeline(
            task="zero-shot-classification",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            device=device,
        )

        self.labels = list(
            EVENT_LABEL_DESCRIPTIONS.keys()
        )

        self.label_descriptions = list(
            EVENT_LABEL_DESCRIPTIONS.values()
        )

        self.description_to_label = {
            description: label
            for label, description
            in EVENT_LABEL_DESCRIPTIONS.items()
        }

    def classify(
        self,
        text: str,
    ) -> EventClassification:

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        result = self.classifier(
            text,
            candidate_labels=(
                self.label_descriptions
            ),
            hypothesis_template=(
                "This financial news article "
                "is primarily about {}."
            ),
            multi_label=self.multi_label,
        )

        scores: dict[str, float] = {}

        for description, score in zip(
            result["labels"],
            result["scores"],
        ):

            label = (
                self.description_to_label[
                    description
                ]
            )

            scores[label] = float(score)

        top_label = result["labels"][0]

        top_class = (
            self.description_to_label[
                top_label
            ]
        )

        top_score = float(
            result["scores"][0]
        )

        # Keep the strongest alternatives for explainability.
        evidence_terms = [
            f"{label}:{score:.3f}"
            for label, score in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        ]

        return EventClassification(
            label=top_class,
            confidence=top_score,
            evidence_terms=evidence_terms,
            model_name=self.MODEL_NAME,
            scores=scores,
        )