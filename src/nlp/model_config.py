"""
Central configuration for EventPulse pretrained models.

Keeping model identifiers and inference settings in one place allows us
to benchmark/replace models without modifying the rest of the pipeline.
"""

from __future__ import annotations

import os

import torch


# ---------------------------------------------------------------------------
# MODEL IDENTIFIERS
# ---------------------------------------------------------------------------

SENTIMENT_MODEL = "ProsusAI/finbert"

NER_MODEL = "whataboutyou-ai/financial_bert"

EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"
)

EVENT_ZERO_SHOT_MODEL = (
    "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
)
EVENT_TOPIC_MODEL = (
    "leonas5555/finnews-topic-single-classify"
)

# ---------------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------------

def get_device_index() -> int:
    """
    Device index used by Hugging Face Transformers pipelines.

    Environment variable:
        EVENTPULSE_DEVICE=cpu
        EVENTPULSE_DEVICE=cuda

    Default:
        CUDA when available, otherwise CPU.
    """

    requested = os.getenv(
        "EVENTPULSE_DEVICE",
        "auto",
    ).lower()

    if requested == "cpu":
        return -1

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "EVENTPULSE_DEVICE=cuda was requested, "
                "but CUDA is not available."
            )

        return 0

    return 0 if torch.cuda.is_available() else -1


def get_sentence_transformer_device() -> str:
    """
    Device string expected by SentenceTransformer.
    """

    requested = os.getenv(
        "EVENTPULSE_DEVICE",
        "auto",
    ).lower()

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "EVENTPULSE_DEVICE=cuda was requested, "
                "but CUDA is not available."
            )

        return "cuda"

    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )