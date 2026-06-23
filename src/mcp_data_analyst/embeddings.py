"""Embedding backends and an in-memory vector index for semantic search.

Design goal: the *mechanics* of vector search (encode -> L2-normalise ->
cosine top-k) are identical regardless of how vectors are produced, so the
backend is pluggable behind a small interface.

Two backends ship:

- :class:`SentenceTransformerBackend` — real dense semantic embeddings
  (``all-MiniLM-L6-v2`` by default). Used automatically when the optional
  ``sentence-transformers`` dependency is installed.
- :class:`HashingEmbeddingBackend` — a dependency-free, deterministic
  hashing/bag-of-words vectoriser. It is lexical rather than semantic, but it
  keeps the project runnable offline and makes tests fast and reproducible.

:func:`get_default_backend` picks the best available backend.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .text_store import tokenize


class EmbeddingBackend(ABC):
    """Turn a list of texts into a matrix of L2-normalised row vectors."""

    name: str
    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim), rows normalised."""


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbeddingBackend(EmbeddingBackend):
    """Deterministic hashing vectoriser (no external model required).

    Each token is hashed into a fixed-width vector with a signed bucket, giving
    a stable bag-of-words representation. Similarity therefore reflects shared
    vocabulary. Good enough to demonstrate the search pipeline and to test it.
    """

    def __init__(self, dim: int = 512) -> None:
        self.name = "hashing"
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text, drop_stopwords=True):
                digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                bucket = digest % self.dim
                sign = 1.0 if (digest >> 8) & 1 else -1.0
                out[row, bucket] += sign
        return _l2_normalize(out)


class SentenceTransformerBackend(EmbeddingBackend):
    """Dense semantic embeddings via the sentence-transformers library."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy, optional

        self._model = SentenceTransformer(model)
        self.name = f"sentence-transformers:{model}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


def get_default_backend() -> EmbeddingBackend:
    """Return the best backend available in the current environment."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return HashingEmbeddingBackend()
    return SentenceTransformerBackend()


@dataclass
class SearchHit:
    doc_id: int
    score: float
    text: str


class EmbeddingIndex:
    """A tiny brute-force cosine-similarity index over a document set."""

    def __init__(self, backend: EmbeddingBackend, documents: list[str]) -> None:
        self.backend = backend
        self.documents = documents
        self.matrix = backend.encode(documents)  # (n_docs, dim), normalised

    @property
    def dim(self) -> int:
        return self.backend.dim

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        """Return the ``top_k`` documents most similar to ``query``."""
        query_vec = self.backend.encode([query])[0]
        scores = self.matrix @ query_vec  # cosine, since rows are normalised
        order = np.argsort(-scores)[:top_k]
        return [
            SearchHit(doc_id=int(i), score=round(float(scores[i]), 4), text=self.documents[i])
            for i in order
        ]
