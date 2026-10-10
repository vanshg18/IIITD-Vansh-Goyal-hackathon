
"""
Confidence-aware event-classification arbitration.

The finance-topic model provides domain-specific topic predictions.
ModernBERT zero-shot provides an independent event-category assessment.

Both outputs are compared because raw model confidence scores are not
necessarily calibrated or directly comparable across models.
"""

from __future__ import annotations

from src.engine.schemas import EventClassification
from src.nlp.financial_topic_classifier import (
    FinancialTopicClassifier,
)
from src.nlp.pretrained_event import (
    PretrainedEventClassifier,
)


class EventClassifierRouter:
    """
    Arbitrate between a finance-topic model and a zero-shot classifier.

    The thresholds below are initial routing parameters, not learned
    financial facts. They should be validated against labelled examples.
    """

    def __init__(
        self,
        topic_classifier=None,
        fallback_classifier=None,
        topic_confidence_threshold: float = 0.85,
        zero_shot_strong_score: float = 0.60,
        zero_shot_strong_margin: float = 0.15,
        zero_shot_moderate_score: float = 0.35,
        zero_shot_moderate_margin: float = 0.12,
    ):
        self.topic_classifier = (
            topic_classifier
            or FinancialTopicClassifier()
        )

        self.fallback_classifier = fallback_classifier

        self.topic_confidence_threshold = (
            topic_confidence_threshold
        )
        self.zero_shot_strong_score = (
            zero_shot_strong_score
        )
        self.zero_shot_strong_margin = (
            zero_shot_strong_margin
        )
        self.zero_shot_moderate_score = (
            zero_shot_moderate_score
        )
        self.zero_shot_moderate_margin = (
            zero_shot_moderate_margin
        )

        # Prevent repeated model inference when the same text is passed
        # through clustering and event construction.
        self._cache: dict[str, EventClassification] = {}

    def _get_fallback(self):
        """Lazy-load ModernBERT on the first classification request."""

        if self.fallback_classifier is None:
            self.fallback_classifier = (
                PretrainedEventClassifier()
            )

        return self.fallback_classifier

    @staticmethod
    def _zero_shot_margin(
        result: EventClassification,
    ) -> float:
        """Difference between the top two zero-shot class scores."""

        ranked_scores = sorted(
            result.scores.values(),
            reverse=True,
        )

        if len(ranked_scores) < 2:
            return result.confidence

        return ranked_scores[0] - ranked_scores[1]

    @staticmethod
    def _selected_result(
        result: EventClassification,
        model_name: str,
        reason: str,
    ) -> EventClassification:
        """Preserve the result while recording the routing decision."""

        return result.model_copy(
            update={
                "model_name": model_name,
                "evidence_terms": [
                    reason,
                    *result.evidence_terms,
                ],
            }
        )

    def classify(
        self,
        text: str,
    ) -> EventClassification:

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        cache_key = " ".join(text.split())

        if cache_key in self._cache:
            return self._cache[cache_key]

        # ---------------------------------------------------------------
        # 1. Finance-domain topic prediction
        # ---------------------------------------------------------------

        primary = self.topic_classifier.predict(text)

        # ---------------------------------------------------------------
        # 2. Independent zero-shot assessment
        # ---------------------------------------------------------------

        fallback = self._get_fallback().classify(text)

        zero_score = fallback.confidence
        zero_margin = self._zero_shot_margin(fallback)

        mapped_event = primary.mapped_event
        topic_score = primary.score

        # ---------------------------------------------------------------
        # 3. Agreement: accept the shared category
        # ---------------------------------------------------------------

        if (
            mapped_event is not None
            and mapped_event == fallback.label
        ):
            result = EventClassification(
                label=mapped_event,

                # Conservative agreement score, not a calibrated
                # probability.
                confidence=min(
                    topic_score,
                    zero_score,
                ),

                evidence_terms=[
                    f"models_agree:{mapped_event}",
                    f"finance_topic:{primary.source_topic}",
                    f"zero_shot_margin:{zero_margin:.3f}",
                ],

                model_name="router:consensus",
                scores=fallback.scores,
            )

        # ---------------------------------------------------------------
        # 4. Strong zero-shot evidence: use its event interpretation
        # ---------------------------------------------------------------

        elif (
            zero_score >= self.zero_shot_strong_score
            and zero_margin >= self.zero_shot_strong_margin
        ):
            result = self._selected_result(
                fallback,
                model_name="router:zero-shot-adjudicated",
                reason=(
                    f"model_disagreement:finance_topic="
                    f"{primary.source_topic}->{mapped_event};"
                    f"zero_shot_margin={zero_margin:.3f}"
                ),
            )

        # ---------------------------------------------------------------
        # 5. Finance-topic prediction is strong, while zero-shot is
        #    ambiguous. This preserves the useful CloudMatrix result.
        # ---------------------------------------------------------------

        elif (
            mapped_event is not None
            and topic_score >= self.topic_confidence_threshold
            and not (
                zero_score >= self.zero_shot_moderate_score
                and zero_margin >= self.zero_shot_moderate_margin
            )
        ):
            result = EventClassification(
                label=mapped_event,
                confidence=topic_score,
                evidence_terms=[
                    f"finance_topic:{primary.source_topic}",
                    (
                        f"zero_shot_ambiguous:"
                        f"{fallback.label}:{zero_score:.3f}"
                    ),
                    f"zero_shot_margin:{zero_margin:.3f}",
                ],
                model_name="router:finance-topic-selected",
                scores={
                    "finance_topic_score": topic_score,
                    "zero_shot_top_score": zero_score,
                    "zero_shot_margin": zero_margin,
                },
            )

        # ---------------------------------------------------------------
        # 6. Moderate but discriminative zero-shot evidence
        # ---------------------------------------------------------------

        elif (
            zero_score >= self.zero_shot_moderate_score
            and zero_margin >= self.zero_shot_moderate_margin
        ):
            result = self._selected_result(
                fallback,
                model_name="router:zero-shot-adjudicated",
                reason=(
                    f"moderate_zero_shot_evidence:"
                    f"margin={zero_margin:.3f}"
                ),
            )

        # ---------------------------------------------------------------
        # 7. Neither model provides decisive evidence.
        #    Preserve a candidate but do not overstate confidence.
        # ---------------------------------------------------------------

        elif mapped_event is not None:
            result = EventClassification(
                label=mapped_event,
                confidence=min(
                    topic_score,
                    0.60,
                ),
                evidence_terms=[
                    f"low_confidence_review:{primary.source_topic}",
                    f"zero_shot_alternative:{fallback.label}",
                ],
                model_name="router:low-confidence",
                scores={
                    "finance_topic_score": topic_score,
                    "zero_shot_top_score": zero_score,
                    "zero_shot_margin": zero_margin,
                },
            )

        else:
            result = self._selected_result(
                fallback,
                model_name="router:low-confidence-zero-shot",
                reason=(
                    f"unmapped_finance_topic:"
                    f"{primary.source_topic}"
                ),
            )

        self._cache[cache_key] = result

        return result
