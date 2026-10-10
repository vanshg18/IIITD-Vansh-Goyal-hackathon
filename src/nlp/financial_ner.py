"""
Financial Named Entity Recognition using a pretrained financial BERT model.

Primary model:
    whataboutyou-ai/financial_bert

The model is fine-tuned on the BUSTER dataset for financial NER.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import pipeline

from src.engine.schemas import Entity
from src.nlp.model_config import (
    NER_MODEL,
    get_device_index,
)


@dataclass(frozen=True)
class NERPrediction:
    """Internal NER prediction representation."""

    text: str
    entity_type: str
    confidence: float


class FinancialNER:
    """
    Pretrained financial NER extractor.
    """

    MODEL_NAME = NER_MODEL

    def __init__(
        self,
        device: int | None = None,
    ):

        if device is None:
            device = get_device_index()

        self.pipeline = pipeline(
            task="token-classification",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            aggregation_strategy="simple",
            device=device,
        )

    @staticmethod
    def _map_entity_type(
        entity_type: str,
    ) -> str:

        entity_type = entity_type.upper()

        if entity_type in {
            "ORG",
            "ORGANIZATION",
            "COMPANY",
            "CORPORATION",
        }:
            return "ORGANIZATION"

        if entity_type in {
            "PER",
            "PERSON",
        }:
            return "PERSON"

        if entity_type in {
            "LOC",
            "LOCATION",
            "GPE",
        }:
            return "LOCATION"

        if entity_type in {
            "PRODUCT",
        }:
            return "PRODUCT"

        return "OTHER"

    def extract(
        self,
        text: str,
    ) -> list[Entity]:

        if not text or not text.strip():
            return []

        predictions = self.pipeline(
            text,
        )

        entities: list[Entity] = []

        seen: set[tuple[str, str]] = set()

        for prediction in predictions:

            word = (
                prediction.get("word")
                or ""
            ).strip()

            if not word:
                continue

            entity_type = (
                prediction.get(
                    "entity_group",
                    prediction.get(
                        "entity",
                        "OTHER",
                    ),
                )
            )

            mapped_type = self._map_entity_type(
                entity_type
            )

            key = (
                word.lower(),
                mapped_type,
            )

            if key in seen:
                continue

            seen.add(key)

            entities.append(
                Entity(
                    name=word,
                    entity_type=mapped_type,
                )
            )

        return entities