"""
Semantic embeddings for EventPulse.

Uses a public Sentence Transformer model so the prototype does not
depend on gated Hugging Face repositories.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.nlp.model_config import (
    EMBEDDING_MODEL,
    get_sentence_transformer_device,
)


class FinanceEmbedder:
    """
    Generate semantic embeddings for financial/news text.

    The model is configurable through model_config.py.
    """

    MODEL_NAME = EMBEDDING_MODEL

    def __init__(
        self,
        device: str | None = None,
    ):
        if device is None:
            device = get_sentence_transformer_device()

        self.model = SentenceTransformer(
            self.MODEL_NAME,
            device=device,
        )

        self.dimension = (
            self.model.get_embedding_dimension()
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode text into normalized semantic embeddings.
        """

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )