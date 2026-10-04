"""
Event impact scoring.

This module turns interpretable NLP outputs into a 1-10 financial
impact estimate.

This is a baseline scoring engine. Its coefficients are intentionally
explicit and configurable so that they can later be calibrated on data.
"""

from __future__ import annotations

from dataclasses import dataclass


EVENT_SEVERITY = {
    "Monetary Policy": 0.95,
    "Credit": 0.95,
    "Geopolitical": 0.90,
    "Regulatory": 0.80,
    "Supply Chain": 0.80,
    "Corporate Action": 0.70,
    "Earnings": 0.65,
    "Market": 0.55,
    "Product/Business": 0.50,
    "Other": 0.25,
}


@dataclass(frozen=True)
class ImpactPrediction:
    """Result produced by the impact engine."""

    score: float
    confidence: float

    severity_component: float
    sentiment_component: float
    breadth_component: float
    confidence_component: float
    novelty_component: float


class ImpactEngine:
    """
    Explainable financial event impact scorer.
    """

    def __init__(
        self,
        severity_priors: dict[str, float] | None = None,
    ):
        self.severity_priors = (
            severity_priors
            or EVENT_SEVERITY.copy()
        )

    @staticmethod
    def calculate_breadth(
        affected_count: int,
    ) -> float:
        """
        Convert affected asset/entity count into [0, 1].

        Uses diminishing returns so that the first few affected assets
        matter more than the 50th.
        """

        if affected_count <= 0:
            return 0.0

        import math

        return 1.0 - math.exp(
            -affected_count / 2.0
        )

    @staticmethod
    def clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        return max(
            minimum,
            min(maximum, value),
        )

    def calculate(
        self,
        event_type: str,
        sentiment_score: float,
        sentiment_confidence: float,
        event_confidence: float,
        affected_count: int,
        novelty: float,
    ) -> ImpactPrediction:
        """
        Calculate interpretable event impact.
        """

        severity = self.severity_priors.get(
            event_type,
            self.severity_priors["Other"],
        )

        sentiment_strength = abs(
            self.clamp(
                sentiment_score,
                -1.0,
                1.0,
            )
        )

        breadth = self.calculate_breadth(
            affected_count
        )

        model_confidence = (
            sentiment_confidence
            + event_confidence
        ) / 2.0

        novelty = self.clamp(
            novelty,
            0.0,
            1.0,
        )

        raw = (
            0.45 * severity
            + 0.15 * sentiment_strength
            + 0.15 * breadth
            + 0.15 * model_confidence
            + 0.10 * novelty
        )

        score = 1.0 + 9.0 * raw

        score = self.clamp(
            score,
            1.0,
            10.0,
        )

        # Impact confidence reflects confidence in the inputs.
        confidence = (
            0.60 * model_confidence
            + 0.20 * novelty
            + 0.20 * min(1.0, breadth + 0.25)
        )

        confidence = self.clamp(
            confidence,
            0.0,
            0.99,
        )

        return ImpactPrediction(
            score=score,
            confidence=confidence,
            severity_component=severity,
            sentiment_component=sentiment_strength,
            breadth_component=breadth,
            confidence_component=model_confidence,
            novelty_component=novelty,
        )