"""In-memory store for text corpora plus shared NLP helpers.

A corpus is just an ordered list of documents (strings). The store loads
corpora from disk, exposes lightweight classic-NLP operations (keyword search,
word frequencies, summary stats) and is the text counterpart to
:class:`~mcp_data_analyst.store.DataSessionStore`.

The tokeniser and stopword list defined here are shared with the embeddings
module so lexical and vector views of the text stay consistent.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .store import DataAnalystError

# Documents come from these file types when loading a directory.
_TEXT_SUFFIXES = {".txt", ".md", ".log"}

_TOKEN_RE = re.compile(r"[a-z][a-z']+")

# A compact English stopword list (avoids an nltk dependency).
STOPWORDS = frozenset(
    """
    a an and are as at be been being but by for from had has have he her his i in
    into is it its me my no not of on or our so that the their them they this to
    too very was we were what when which who will with you your it's i'm
    """.split()
)


def tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    """Lowercase, regex-tokenise text and optionally drop stopwords."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens


@dataclass
class CorpusInfo:
    name: str
    source: str
    n_documents: int

    def to_dict(self) -> dict:
        return {"name": self.name, "source": self.source, "n_documents": self.n_documents}


class TextCorpusStore:
    """Hold named text corpora and run classic NLP queries over them."""

    def __init__(self) -> None:
        self._docs: dict[str, list[str]] = {}
        self._info: dict[str, CorpusInfo] = {}

    # -- loading -----------------------------------------------------------
    def load(self, name: str, path: str) -> CorpusInfo:
        """Load a corpus from a directory (one doc per file) or a text file.

        - Directory: every ``.txt``/``.md``/``.log`` file becomes one document.
        - Single text file: each non-empty line becomes one document (handy for
          logs and line-oriented record files).
        """
        from .store import DataSessionStore

        DataSessionStore.validate_name(name)
        p = Path(path).expanduser()
        if not p.exists():
            raise DataAnalystError(f"Path not found: '{path}'.")

        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in _TEXT_SUFFIXES)
            if not files:
                raise DataAnalystError(
                    f"No text files ({', '.join(sorted(_TEXT_SUFFIXES))}) in '{path}'."
                )
            docs = [f.read_text(encoding="utf-8", errors="replace").strip() for f in files]
        else:
            if p.suffix.lower() not in _TEXT_SUFFIXES:
                raise DataAnalystError(
                    f"Unsupported text file '{p.suffix}'. Use one of "
                    f"{', '.join(sorted(_TEXT_SUFFIXES))} or a directory."
                )
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            docs = [ln.strip() for ln in lines if ln.strip()]

        if not docs:
            raise DataAnalystError(f"Corpus '{name}' is empty after loading '{path}'.")

        self._docs[name] = docs
        info = CorpusInfo(name=name, source=str(p), n_documents=len(docs))
        self._info[name] = info
        return info

    # -- access ------------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self._docs

    def documents(self, name: str) -> list[str]:
        if name not in self._docs:
            raise DataAnalystError(
                f"No corpus named '{name}'. Loaded corpora: "
                f"{', '.join(self._docs) or '(none)'}. Load one with load_text_corpus."
            )
        return self._docs[name]

    def list(self) -> list[CorpusInfo]:
        return list(self._info.values())

    # -- analysis ----------------------------------------------------------
    def stats(self, name: str) -> dict:
        docs = self.documents(name)
        token_counts = [len(tokenize(d)) for d in docs]
        vocab = {t for d in docs for t in tokenize(d)}
        total = sum(token_counts)
        return {
            "name": name,
            "n_documents": len(docs),
            "total_tokens": total,
            "vocabulary_size": len(vocab),
            "avg_tokens_per_doc": round(total / len(docs), 2) if docs else 0,
            "shortest_doc_tokens": min(token_counts) if token_counts else 0,
            "longest_doc_tokens": max(token_counts) if token_counts else 0,
        }

    def keyword_search(
        self, name: str, query: str, *, regex: bool, limit: int
    ) -> list[dict]:
        docs = self.documents(name)
        try:
            pattern = re.compile(query if regex else re.escape(query), re.IGNORECASE)
        except re.error as exc:
            raise DataAnalystError(f"Invalid regex: {exc}") from exc
        hits = [
            {"doc_id": i, "text": doc}
            for i, doc in enumerate(docs)
            if pattern.search(doc)
        ]
        return hits[:limit]

    def word_frequencies(self, name: str, *, top_k: int, drop_stopwords: bool) -> list[dict]:
        docs = self.documents(name)
        counter: Counter[str] = Counter()
        for doc in docs:
            counter.update(tokenize(doc, drop_stopwords=drop_stopwords))
        return [{"term": term, "count": count} for term, count in counter.most_common(top_k)]
