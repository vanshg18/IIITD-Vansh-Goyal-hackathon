"""
Financial sentiment analysis using FinBERT.

The model produces positive / negative / neutral probabilities.
We convert those probabilities into a continuous sentiment score [-1, 1].
"""

from dataclasses import dataclass

from transformers import pipeline


@dataclass
class SentimentPrediction:
    """Internal sentiment prediction representation."""

    score: float
    label: str
    confidence: float


class FinancialSentimentAnalyzer:
    """
    Financial sentiment analyzer backed by ProsusAI/finbert.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: int = -1):
        """
        Parameters
        ----------
        device:
            -1 = CPU
             0 = first CUDA GPU
             1 = second CUDA GPU
        """

        self.classifier = pipeline(
            task="text-classification",
            model=self.MODEL_NAME,
            tokenizer=self.MODEL_NAME,
            device=device,
            top_k=None,
        )

    @staticmethod
    def _normalize_single_result(result) -> list[dict]:
        """
        Normalize Hugging Face pipeline output for a single text.

        Depending on the Transformers version and top_k configuration,
        a single prediction can be returned either as:

            [dict, dict, dict]

        or:

            [[dict, dict, dict]]

        This method handles both forms.
        """

        if not result:
            raise ValueError("Model returned an empty result.")

        # Unwrap one level if the result is nested.
        if isinstance(result[0], list):
            result = result[0]

        if not all(isinstance(item, dict) for item in result):
            raise TypeError(
                f"Unexpected model output format: {result}"
            )

        return result

    @staticmethod
    def probabilities_to_prediction(
        probabilities: dict[str, float],
    ) -> SentimentPrediction:
        """
        Convert class probabilities into our canonical sentiment result.
        """

        positive = probabilities.get("positive", 0.0)
        negative = probabilities.get("negative", 0.0)

        # Continuous sentiment:
        #
        # +1 => strongly positive
        #  0 => neutral
        # -1 => strongly negative
        #
        # The neutral probability naturally reduces the magnitude because
        # positive and negative probabilities become closer to each other.
        score = positive - negative

        label = max(
            probabilities,
            key=probabilities.get,
        )

        confidence = probabilities[label]

        return SentimentPrediction(
            score=float(score),
            label=label,
            confidence=float(confidence),
        )

    def predict(self, text: str) -> SentimentPrediction:
        """
        Predict financial sentiment for one piece of text.
        """

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        raw_result = self.classifier(
            text,
            truncation=True,
        )

        result = self._normalize_single_result(
            raw_result
        )

        probabilities = {
            item["label"].lower(): float(item["score"])
            for item in result
        }

        return self.probabilities_to_prediction(
            probabilities
        )

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[SentimentPrediction]:
        """
        Predict sentiment for multiple texts.
        """

        if not texts:
            return []

        if any(
            not text or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "All texts must be non-empty."
            )

        raw_results = self.classifier(
            texts,
            truncation=True,
            batch_size=batch_size,
        )

        predictions: list[SentimentPrediction] = []

        for raw_result in raw_results:

            result = self._normalize_single_result(
                raw_result
            )

            probabilities = {
                item["label"].lower(): float(item["score"])
                for item in result
            }

            predictions.append(
                self.probabilities_to_prediction(
                    probabilities
                )
            )

        return predictions