"""
Embedding pipeline with L2 normalization and overflow protection.

Uses sentence-transformers/all-MiniLM-L6-v2 for MVP.
All vectors are L2-normalized before FAISS insertion (required for IndexFlatIP cosine sim).

CRITICAL: MiniLM silently truncates inputs beyond 256 tokens.
This module handles overflow by logging a warning.
"""

from __future__ import annotations

import numpy as np

from quantscribe.schemas.etl import TextChunk
from quantscribe.config import get_settings
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.embeddings")


class EmbeddingPipeline:
    """Handles embedding with proper normalization and overflow protection."""

    def __init__(self, model_name: str | None = None):
        """
        Initialize the embedding pipeline.

        Args:
            model_name: HuggingFace model name. Defaults to config value.
        """
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.max_tokens = settings.embedding_max_tokens
        self.dimension = settings.embedding_dimension

        logger.info("embedding_pipeline_init", model=self.model_name, dim=self.dimension)

    def embed_chunks(
        self,
        chunks: list[TextChunk],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """
        Embed a list of TextChunks with overflow protection.

        Returns L2-normalized vectors ready for FAISS IndexFlatIP.

        Args:
            chunks: List of TextChunk objects to embed.
            batch_size: Batch size for encoding (default from config).

        Returns:
            numpy array of shape (n_chunks, dimension), float32, L2-normalized.
        """
        settings = get_settings()
        bs = batch_size or settings.embedding_batch_size

        texts: list[str] = []
        for chunk in chunks:
            # Warn if chunk exceeds max tokens (MiniLM truncates silently)
            approx_tokens = len(chunk.content.split())
            if approx_tokens > self.max_tokens:
                logger.warn(
                    "chunk_exceeds_max_tokens",
                    chunk_id=chunk.metadata.chunk_id,
                    approx_tokens=approx_tokens,
                    max_tokens=self.max_tokens,
                )
            texts.append(chunk.content)

        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2 normalization for FAISS IndexFlatIP
        )

        logger.info("chunks_embedded", count=len(texts), shape=embeddings.shape)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns L2-normalized vector for FAISS search.

        Args:
            query: Query text.

        Returns:
            numpy array of shape (1, dimension), float32, L2-normalized.
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)
