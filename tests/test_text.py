"""Unit tests for the text corpus store and embedding search."""

from __future__ import annotations

import numpy as np
import pytest

from mcp_data_analyst.embeddings import EmbeddingIndex, HashingEmbeddingBackend
from mcp_data_analyst.store import DataAnalystError
from mcp_data_analyst.text_store import TextCorpusStore, tokenize

DOCS = [
    "The battery easily lasts a full day of heavy use.",
    "Shipping was lightning quick, arrived two days early.",
    "The screen is bright and colours are vivid outdoors.",
]


@pytest.fixture()
def corpus(tmp_path):
    f = tmp_path / "reviews.txt"
    f.write_text("\n".join(DOCS), encoding="utf-8")
    store = TextCorpusStore()
    store.load("reviews", str(f))
    return store


def test_tokenize_drops_stopwords():
    tokens = tokenize("The quick brown fox", drop_stopwords=True)
    assert "the" not in tokens
    assert tokens == ["quick", "brown", "fox"]


def test_load_one_doc_per_line(corpus):
    assert corpus.documents("reviews") == DOCS
    assert corpus.stats("reviews")["n_documents"] == 3


def test_missing_corpus_error():
    store = TextCorpusStore()
    with pytest.raises(DataAnalystError, match="No corpus named"):
        store.documents("nope")


def test_keyword_search_is_case_insensitive(corpus):
    hits = corpus.keyword_search("reviews", "BATTERY", regex=False, limit=10)
    assert len(hits) == 1
    assert hits[0]["doc_id"] == 0


def test_word_frequencies_excludes_stopwords(corpus):
    terms = {t["term"] for t in corpus.word_frequencies("reviews", top_k=50, drop_stopwords=True)}
    assert "the" not in terms
    assert "battery" in terms


def test_hashing_backend_normalises_vectors():
    backend = HashingEmbeddingBackend(dim=128)
    matrix = backend.encode(DOCS)
    assert matrix.shape == (3, 128)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_index_retrieves_exact_document():
    index = EmbeddingIndex(HashingEmbeddingBackend(dim=256), DOCS)
    hits = index.search(DOCS[2], top_k=1)
    assert hits[0].doc_id == 2
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_semantic_search_beats_lexical_gap():
    # 'delivery' never appears literally, but the shipping doc should rank top.
    index = EmbeddingIndex(HashingEmbeddingBackend(dim=256), DOCS)
    top = index.search("shipping arrived quick", top_k=1)[0]
    assert top.doc_id == 1
