"""
Explainable financial event classifier.

This is the first baseline implementation.

The classifier is intentionally deterministic and explainable so that we
can benchmark it before replacing/augmenting it with a trained model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from src.engine.schemas import EventClassification
import re

@dataclass(frozen=True)
class EventRule:
    label: str
    keywords: dict[str, float]


EVENT_RULES = [
    EventRule(
        label="Monetary Policy",
        keywords={
            "central bank": 4.0,
            "interest rate": 4.0,
            "policy rate": 4.0,
            "rate hike": 5.0,
            "rate cut": 5.0,
            "rate increase": 4.0,
            "rate decrease": 4.0,
            "inflation": 2.0,
        },
    ),

    EventRule(
        label="Earnings",
        keywords={
            "earnings": 4.0,
            "revenue": 3.0,
            "profit": 4.0,
            "quarterly": 2.0,
            "guidance": 3.0,
            "forecast": 2.0,
            "outlook": 2.0,
        },
    ),

    EventRule(
        label="Corporate Action",
        keywords={
            "merger": 5.0,
            "acquisition": 5.0,
            "acquires": 5.0,
            "takeover": 5.0,
            "buyback": 3.0,
            "spin-off": 4.0,
            "spinoff": 4.0,
        },
    ),

    EventRule(
        label="Regulatory",
        keywords={
            "regulator": 4.0,
            "regulatory": 4.0,
            "investigation": 5.0,
            "investigated": 5.0,
            "compliance": 3.0,
            "lawsuit": 4.0,
            "legal action": 4.0,
            "regulatory review": 5.0,
        },
    ),

    EventRule(
        label="Credit",
        keywords={
            "credit downgrade": 5.0,
            "downgrade": 5.0,
            "downgraded":5.0,
            "downgrades":5.0,
            "default": 5.0,
            "rating downgrade": 5.0,
            "credit rating": 3.0,
            "consumer loan": 2.0,
            "loan defaults": 4.0,
            "rising defaults": 4.0,
        },
    ),

    EventRule(
        label="Geopolitical",
        keywords={
            "geopolitical": 5.0,
            "war": 5.0,
            "sanctions": 5.0,
            "sanction": 5.0,
            "tariff": 4.0,
            "conflict": 4.0,
            "military": 3.0,
            "trade restriction": 4.0,
        },
    ),

    EventRule(
        label="Supply Chain",
        keywords={
            "supply disruption": 5.0,
            "supply chain": 4.0,
            "shortage": 4.0,
            "disruption": 2.0,
            "shipping disruption": 5.0,
            "shipping": 3.0,
        },
    ),

    EventRule(
        label="Product/Business",
        keywords={
            "product launch": 5.0,
            "launches new": 4.0,
            "new processor": 3.0,
            "new product": 4.0,
            "enterprise contract": 4.0,
            "major contract": 4.0,
            "partnership": 3.0,
        },
    ),

    EventRule(
        label="Market",
        keywords={
            "equity markets": 4.0,
            "stock market": 4.0,
            "shares": 2.0,
            "stocks": 2.0,
            "market": 1.0,
            "markets": 1.0,
        },
    ),
]


class FinancialEventClassifier:
    """
    Explainable rule-based event classifier.

    Later this interface can be backed by a fine-tuned transformer.
    """

    OTHER_LABEL = "Other"

    def __init__(
        self,
        rules: list[EventRule] | None = None,
    ):
        self.rules = rules or EVENT_RULES

    @staticmethod
    def _normalise(text: str) -> str:
        """
        Normalize text while preserving word boundaries.
        """

        text = text.lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _keyword_present(
        text: str,
        keyword: str,
    ) -> bool:
        """
        Check whether a keyword/phrase occurs as a complete token/phrase.

        This prevents:
            "war" matching "software"
            "market" matching "marketing"

        while still allowing multi-word expressions such as:
            "rate hike"
            "credit downgrade"
        """

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(keyword.lower())
            + r"(?![a-z0-9])"
        )

        return re.search(
            pattern,
            text,
        ) is not None

    @staticmethod
    def _calculate_confidence(
        top_score: float,
        second_score: float,
    ) -> float:
        """
        Convert rule evidence into a conservative confidence score.

        This is a baseline confidence, NOT a calibrated probability.
        """

        if top_score <= 0:
            return 0.35

        evidence_component = (
            1.0 - exp(-top_score / 5.0)
        )

        margin = max(
            0.0,
            top_score - second_score,
        )

        margin_component = (
            1.0 - exp(-margin / 3.0)
        )

        confidence = (
            0.35
            + 0.40 * evidence_component
            + 0.20 * margin_component
        )

        return min(
            0.95,
            max(0.35, confidence),
        )

    def classify(
        self,
        text: str,
    ) -> EventClassification:

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        text = self._normalise(text)

        scores: dict[str, float] = {}

        evidence: dict[str, list[str]] = {}

        for rule in self.rules:

            score = 0.0
            matched_terms: list[str] = []

            for keyword, weight in rule.keywords.items():

                if self._keyword_present(text,keyword,):
                    score += weight
                    matched_terms.append(keyword)

            scores[rule.label] = score
            evidence[rule.label] = matched_terms

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_label, top_score = ranked[0]

        second_score = (
            ranked[1][1]
            if len(ranked) > 1
            else 0.0
        )

        # No meaningful evidence.
        if top_score <= 0:

            return EventClassification(
                label=self.OTHER_LABEL,
                confidence=0.35,
                evidence_terms=[],
            )

        confidence = self._calculate_confidence(
            top_score=top_score,
            second_score=second_score,
        )

        return EventClassification(
            label=top_label,
            confidence=confidence,
            evidence_terms=evidence[top_label],
        )