"""
Cross-source corroboration.

An underlying event becomes more trustworthy when independent sources
report substantially related information.

For the initial implementation, source independence is represented by
distinct source names.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from src.engine.schemas import Article


@dataclass(frozen=True)
class CorroborationResult:
    """Corroboration statistics for an event cluster."""

    unique_source_count: int
    corroboration_boost: float
    adjusted_confidence: float


class CorroborationEngine:
    """
    Calculate confidence reinforcement from independent sources.
    """

    @staticmethod
    def calculate(
        articles: list[Article],
        base_confidence: float,
    ) -> CorroborationResult:

        if not articles:
            return CorroborationResult(
                unique_source_count=0,
                corroboration_boost=0.0,
                adjusted_confidence=0.0,
            )

        source_names = {
            article.source.name.strip().lower()
            for article in articles
            if article.source.name.strip()
        }

        source_count = len(source_names)

        # No boost for the first source.
        additional_sources = max(
            0,
            source_count - 1,
        )

        boost = (
            1.0 - exp(
                -0.5 * additional_sources
            )
        )

        adjusted_confidence = min(
            0.99,
            base_confidence + 0.15 * boost,
        )

        return CorroborationResult(
            unique_source_count=source_count,
            corroboration_boost=boost,
            adjusted_confidence=adjusted_confidence,
        )