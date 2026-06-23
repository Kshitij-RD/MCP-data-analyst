"""``data_analyst_mcp`` — an MCP server that exposes data-analysis tools.

The server keeps a single :class:`~mcp_data_analyst.store.DataSessionStore` for
the lifetime of the process. Each tool declares its inputs with
``Annotated[..., Field(...)]`` so FastMCP generates a clean, flat JSON schema
and validates arguments with Pydantic before the body runs. Tools return a
compact JSON string an LLM can reason over.

Run directly for a local stdio server::

    python -m mcp_data_analyst.server
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Optional

import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .embeddings import EmbeddingBackend, EmbeddingIndex, get_default_backend
from .store import DataAnalystError, DataSessionStore
from .text_store import TextCorpusStore

mcp = FastMCP("data_analyst_mcp")

# Single session store shared by every tool in this process.
STORE = DataSessionStore()

# Text corpora and their lazily built vector indices.
TEXT_STORE = TextCorpusStore()
_EMBED_BACKEND: EmbeddingBackend | None = None
_INDICES: dict[str, EmbeddingIndex] = {}


def _backend() -> EmbeddingBackend:
    """Return the process-wide embedding backend, created on first use.

    Initialisation is deferred so importing the server never loads a heavy
    embedding model unless a semantic-search tool is actually called.
    """
    global _EMBED_BACKEND
    if _EMBED_BACKEND is None:
        _EMBED_BACKEND = get_default_backend()
    return _EMBED_BACKEND


def _get_index(name: str) -> EmbeddingIndex:
    """Return a corpus's vector index, building it on first request."""
    if name not in _INDICES:
        _INDICES[name] = EmbeddingIndex(_backend(), TEXT_STORE.documents(name))
    return _INDICES[name]

# Where plot_chart writes PNG files.
OUTPUT_DIR = Path(os.environ.get("DATA_ANALYST_OUTPUT_DIR", "./charts")).expanduser()

# Read-only annotations reused by the analytical tools.
_READ_ONLY = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
# Tools that create/replace session or filesystem state are not idempotent.
_MUTATING = {**_READ_ONLY, "idempotentHint": False}


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _ok(payload: dict[str, Any]) -> str:
    """Serialise a successful result as compact JSON."""
    return json.dumps({"ok": True, **payload}, default=str, indent=2)


def _err(message: str) -> str:
    """Serialise a recoverable error so the agent can adjust and retry."""
    return json.dumps({"ok": False, "error": message}, indent=2)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.select_dtypes(include="number").columns]


class OutlierMethod(str, Enum):
    IQR = "iqr"
    ZSCORE = "zscore"


class ChartType(str, Enum):
    HISTOGRAM = "histogram"
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    BOX = "box"


# --------------------------------------------------------------------------
# Tabular tools
# --------------------------------------------------------------------------
@mcp.tool(name="load_dataset", annotations={"title": "Load Dataset", **_MUTATING})
def load_dataset(
    name: Annotated[
        str,
        Field(
            description="Identifier to register the dataset under (e.g. 'sales'). "
            "Letters, digits, underscores; cannot start with a digit.",
            min_length=1,
            max_length=64,
        ),
    ],
    path: Annotated[
        str,
        Field(description="Path to a .csv, .tsv, .parquet or .json file.", min_length=1),
    ],
) -> str:
    """Load a tabular file into the session and register it for SQL queries.

    Reads a .csv/.tsv/.parquet/.json file from disk, stores it under ``name``
    and makes it queryable by every other tool. Call this before analysing.

    Returns:
        str: JSON {"ok": bool, "dataset": {name, source, rows, n_columns,
            columns}} or {"ok": false, "error": str}.
    """
    try:
        info = STORE.load(name, path)
        return _ok({"dataset": info.to_dict()})
    except DataAnalystError as exc:
        return _err(str(exc))


@mcp.tool(name="list_datasets", annotations={"title": "List Datasets", **_READ_ONLY})
def list_datasets() -> str:
    """List every dataset currently loaded in the session.

    Returns:
        str: JSON {"ok": true, "count": int, "datasets": [{name, source,
            rows, n_columns, columns}, ...]}.
    """
    datasets = [info.to_dict() for info in STORE.list()]
    return _ok({"count": len(datasets), "datasets": datasets})


@mcp.tool(name="describe_dataset", annotations={"title": "Describe Dataset", **_READ_ONLY})
def describe_dataset(
    name: Annotated[str, Field(description="Name of a loaded dataset.", min_length=1)],
) -> str:
    """Profile a dataset: dtypes, missing values and summary statistics.

    Reports per-column dtype, null counts and cardinality, numeric summary
    statistics (count/mean/std/min/quartiles/max) and the most frequent
    values for categorical columns.

    Returns:
        str: JSON {"ok": true, "name": str, "rows": int, "columns": [
            {name, dtype, nulls, unique}], "numeric_summary": {...},
            "categorical_summary": {...}}.
    """
    try:
        df = STORE.get(name)
    except DataAnalystError as exc:
        return _err(str(exc))

    columns = [
        {
            "name": str(col),
            "dtype": str(df[col].dtype),
            "nulls": int(df[col].isna().sum()),
            "unique": int(df[col].nunique(dropna=True)),
        }
        for col in df.columns
    ]

    numeric = df.select_dtypes(include="number")
    numeric_summary = (
        json.loads(numeric.describe().round(4).to_json()) if not numeric.empty else {}
    )

    categorical_summary: dict[str, Any] = {}
    for col in df.select_dtypes(exclude="number").columns:
        top = df[col].value_counts().head(5)
        categorical_summary[str(col)] = {str(k): int(v) for k, v in top.to_dict().items()}

    return _ok(
        {
            "name": name,
            "rows": int(df.shape[0]),
            "columns": columns,
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
        }
    )


@mcp.tool(name="run_sql", annotations={"title": "Run SQL Query", **_READ_ONLY})
def run_sql(
    query: Annotated[
        str,
        Field(
            description="A single read-only SQL query (SELECT/WITH). Loaded "
            "datasets are tables, e.g. SELECT region, SUM(revenue) FROM sales "
            "GROUP BY region.",
            min_length=1,
        ),
    ],
    limit: Annotated[
        int,
        Field(description="Maximum rows to return.", ge=1, le=1000),
    ] = 100,
) -> str:
    """Run a read-only SQL query (DuckDB) across loaded datasets.

    Loaded datasets are addressable as tables by their name, so you can
    filter, aggregate and join them. Only SELECT/WITH style statements are
    permitted; any write/DDL statement is rejected.

    Returns:
        str: JSON {"ok": true, "row_count": int, "returned": int,
            "truncated": bool, "columns": [...], "rows": [{...}, ...]}.
    """
    try:
        result = STORE.sql(query)
    except DataAnalystError as exc:
        return _err(str(exc))

    total = int(result.shape[0])
    head = result.head(limit)
    return _ok(
        {
            "row_count": total,
            "returned": int(head.shape[0]),
            "truncated": total > limit,
            "columns": [str(c) for c in result.columns],
            "rows": json.loads(head.to_json(orient="records")),
        }
    )


@mcp.tool(
    name="compute_correlations",
    annotations={"title": "Compute Correlations", **_READ_ONLY},
)
def compute_correlations(
    name: Annotated[str, Field(description="Name of a loaded dataset.", min_length=1)],
    columns: Annotated[
        Optional[list[str]],
        Field(description="Numeric columns to correlate. Defaults to all numeric columns."),
    ] = None,
    method: Annotated[
        str,
        Field(description="Correlation method: 'pearson', 'spearman' or 'kendall'."),
    ] = "pearson",
) -> str:
    """Compute a correlation matrix over numeric columns.

    Returns the full matrix plus the strongest pairwise correlations ranked
    by absolute value, which is usually what the user actually wants.

    Returns:
        str: JSON {"ok": true, "method": str, "columns": [...],
            "matrix": {...}, "top_pairs": [{"a","b","correlation"}, ...]}.
    """
    try:
        df = STORE.get(name)
    except DataAnalystError as exc:
        return _err(str(exc))

    if method not in {"pearson", "spearman", "kendall"}:
        return _err("method must be one of: pearson, spearman, kendall.")

    cols = columns or _numeric_columns(df)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return _err(f"Columns not found: {missing}. Available: {list(df.columns)}.")
    numeric = df[cols].select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return _err("Need at least two numeric columns to correlate.")

    matrix = numeric.corr(method=method).round(4)
    pairs = []
    cols_list = list(matrix.columns)
    for i, a in enumerate(cols_list):
        for b in cols_list[i + 1 :]:
            value = matrix.loc[a, b]
            if pd.notna(value):
                pairs.append({"a": a, "b": b, "correlation": float(value)})
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

    return _ok(
        {
            "method": method,
            "columns": cols_list,
            "matrix": json.loads(matrix.to_json()),
            "top_pairs": pairs[:10],
        }
    )


@mcp.tool(name="detect_outliers", annotations={"title": "Detect Outliers", **_READ_ONLY})
def detect_outliers(
    name: Annotated[str, Field(description="Name of a loaded dataset.", min_length=1)],
    column: Annotated[str, Field(description="Numeric column to scan.", min_length=1)],
    method: Annotated[
        OutlierMethod,
        Field(description="'iqr' (1.5*IQR fences) or 'zscore' (|z| > threshold)."),
    ] = OutlierMethod.IQR,
    threshold: Annotated[
        float,
        Field(description="Z-score threshold (used when method='zscore').", gt=0),
    ] = 3.0,
) -> str:
    """Flag outliers in a numeric column using the IQR or z-score method.

    IQR flags values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]; z-score flags
    values whose standardised score exceeds ``threshold``.

    Returns:
        str: JSON {"ok": true, "method": str, "column": str, "bounds": {...},
            "outlier_count": int, "outlier_fraction": float,
            "examples": [{...}, ...]}.
    """
    try:
        df = STORE.get(name)
    except DataAnalystError as exc:
        return _err(str(exc))

    if column not in df.columns:
        return _err(f"Column '{column}' not found. Available: {list(df.columns)}.")
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return _err(f"Column '{column}' has no numeric values.")

    if method is OutlierMethod.IQR:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < low) | (series > high)
        bounds = {"lower": float(low), "upper": float(high), "q1": float(q1), "q3": float(q3)}
    else:
        std = series.std(ddof=0)
        if std == 0:
            return _err("Standard deviation is zero; cannot compute z-scores.")
        z = (series - series.mean()) / std
        mask = z.abs() > threshold
        bounds = {"mean": float(series.mean()), "std": float(std), "threshold": threshold}

    outliers = df.loc[mask.index[mask]]
    return _ok(
        {
            "method": method.value,
            "column": column,
            "bounds": bounds,
            "outlier_count": int(mask.sum()),
            "outlier_fraction": round(float(mask.mean()), 4),
            "examples": json.loads(outliers.head(10).to_json(orient="records")),
        }
    )


@mcp.tool(name="plot_chart", annotations={"title": "Plot Chart", **_MUTATING})
def plot_chart(
    name: Annotated[str, Field(description="Name of a loaded dataset.", min_length=1)],
    chart_type: Annotated[ChartType, Field(description="Kind of chart to render.")],
    x: Annotated[
        Optional[str],
        Field(description="x-axis column (or value column for histogram/box)."),
    ] = None,
    y: Annotated[
        Optional[str],
        Field(description="y-axis column (bar/line/scatter)."),
    ] = None,
    title: Annotated[Optional[str], Field(description="Optional chart title.")] = None,
) -> str:
    """Render a chart to a PNG file and return its path.

    Supported chart types: histogram, bar, line, scatter, box. For histogram
    and box set ``x`` to the value column; for bar/line/scatter set both
    ``x`` and ``y``.

    Returns:
        str: JSON {"ok": true, "chart_type": str, "path": str, "title": str}.
    """
    try:
        df = STORE.get(name)
    except DataAnalystError as exc:
        return _err(str(exc))

    import matplotlib

    matplotlib.use("Agg")  # headless rendering
    import matplotlib.pyplot as plt

    def _require(*cols: Optional[str]) -> Optional[str]:
        for col in cols:
            if col is None:
                return "This chart type requires the relevant x/y columns."
            if col not in df.columns:
                return f"Column '{col}' not found. Available: {list(df.columns)}."
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if chart_type in (ChartType.HISTOGRAM, ChartType.BOX):
            if err := _require(x):
                return _err(err)
            data = pd.to_numeric(df[x], errors="coerce").dropna()
            if chart_type is ChartType.HISTOGRAM:
                ax.hist(data, bins=30, color="#4C72B0", edgecolor="white")
                ax.set_xlabel(x)
                ax.set_ylabel("count")
            else:
                ax.boxplot(data, vert=True)
                ax.set_ylabel(x)
        elif chart_type is ChartType.BAR:
            if err := _require(x, y):
                return _err(err)
            grouped = df.groupby(x)[y].sum().sort_values(ascending=False)
            ax.bar(grouped.index.astype(str), grouped.values, color="#4C72B0")
            ax.set_xlabel(x)
            ax.set_ylabel(f"sum({y})")
            plt.xticks(rotation=45, ha="right")
        elif chart_type is ChartType.LINE:
            if err := _require(x, y):
                return _err(err)
            ordered = df.sort_values(x)
            ax.plot(ordered[x], ordered[y], color="#4C72B0")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        elif chart_type is ChartType.SCATTER:
            if err := _require(x, y):
                return _err(err)
            ax.scatter(df[x], df[y], alpha=0.6, color="#4C72B0")
            ax.set_xlabel(x)
            ax.set_ylabel(y)

        chart_title = title or f"{chart_type.value} of {name}"
        ax.set_title(chart_title)
        fig.tight_layout()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{name}_{chart_type.value}.png"
        fig.savefig(out_path, dpi=120)
        return _ok({"chart_type": chart_type.value, "path": str(out_path), "title": chart_title})
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------
# Text / NLP tools
# --------------------------------------------------------------------------
@mcp.tool(name="load_text_corpus", annotations={"title": "Load Text Corpus", **_MUTATING})
def load_text_corpus(
    name: Annotated[
        str,
        Field(
            description="Identifier for the corpus (e.g. 'reviews'). Letters, "
            "digits, underscores; cannot start with a digit.",
            min_length=1,
            max_length=64,
        ),
    ],
    path: Annotated[
        str,
        Field(
            description="A directory of .txt/.md/.log files (one document per "
            "file) or a single text file (one document per non-empty line).",
            min_length=1,
        ),
    ],
) -> str:
    """Load a text corpus for keyword and semantic search.

    Returns:
        str: JSON {"ok": bool, "corpus": {name, source, n_documents}} or
            {"ok": false, "error": str}.
    """
    try:
        info = TEXT_STORE.load(name, path)
        _INDICES.pop(name, None)  # invalidate any stale index for this name
        return _ok({"corpus": info.to_dict()})
    except DataAnalystError as exc:
        return _err(str(exc))


@mcp.tool(name="list_corpora", annotations={"title": "List Corpora", **_READ_ONLY})
def list_corpora() -> str:
    """List every text corpus currently loaded in the session.

    Returns:
        str: JSON {"ok": true, "count": int, "corpora": [{name, source,
            n_documents}, ...]}.
    """
    corpora = [info.to_dict() for info in TEXT_STORE.list()]
    return _ok({"count": len(corpora), "corpora": corpora})


@mcp.tool(name="corpus_stats", annotations={"title": "Corpus Statistics", **_READ_ONLY})
def corpus_stats(
    name: Annotated[str, Field(description="Name of a loaded corpus.", min_length=1)],
) -> str:
    """Summarise a corpus: document count, tokens, vocabulary and lengths.

    Returns:
        str: JSON {"ok": true, "name": str, "n_documents": int,
            "total_tokens": int, "vocabulary_size": int,
            "avg_tokens_per_doc": float, "shortest_doc_tokens": int,
            "longest_doc_tokens": int}.
    """
    try:
        return _ok(TEXT_STORE.stats(name))
    except DataAnalystError as exc:
        return _err(str(exc))


@mcp.tool(name="keyword_search", annotations={"title": "Keyword Search", **_READ_ONLY})
def keyword_search(
    name: Annotated[str, Field(description="Name of a loaded corpus.", min_length=1)],
    query: Annotated[
        str,
        Field(description="Literal substring (or regex) to find.", min_length=1),
    ],
    regex: Annotated[
        bool,
        Field(description="Treat the query as a regular expression."),
    ] = False,
    limit: Annotated[
        int,
        Field(description="Maximum matching documents to return.", ge=1, le=200),
    ] = 20,
) -> str:
    """Find documents containing a literal phrase or regex (case-insensitive).

    This is exact lexical matching; use semantic_search to match by meaning.

    Returns:
        str: JSON {"ok": true, "query": str, "match_count": int,
            "matches": [{"doc_id": int, "text": str}, ...]}.
    """
    try:
        hits = TEXT_STORE.keyword_search(name, query, regex=regex, limit=limit)
    except DataAnalystError as exc:
        return _err(str(exc))
    return _ok({"query": query, "match_count": len(hits), "matches": hits})


@mcp.tool(name="word_frequencies", annotations={"title": "Word Frequencies", **_READ_ONLY})
def word_frequencies(
    name: Annotated[str, Field(description="Name of a loaded corpus.", min_length=1)],
    top_k: Annotated[
        int,
        Field(description="Number of top terms to return.", ge=1, le=200),
    ] = 20,
    drop_stopwords: Annotated[
        bool,
        Field(description="Exclude common English stopwords."),
    ] = True,
) -> str:
    """Return the most frequent terms across the corpus.

    Returns:
        str: JSON {"ok": true, "name": str, "top_terms": [{"term": str,
            "count": int}, ...]}.
    """
    try:
        terms = TEXT_STORE.word_frequencies(name, top_k=top_k, drop_stopwords=drop_stopwords)
    except DataAnalystError as exc:
        return _err(str(exc))
    return _ok({"name": name, "top_terms": terms})


@mcp.tool(name="semantic_search", annotations={"title": "Semantic Search", **_READ_ONLY})
def semantic_search(
    name: Annotated[str, Field(description="Name of a loaded corpus.", min_length=1)],
    query: Annotated[
        str,
        Field(description="Natural-language query to match by meaning.", min_length=1),
    ],
    top_k: Annotated[
        int,
        Field(description="Number of most similar documents to return.", ge=1, le=50),
    ] = 5,
) -> str:
    """Rank documents by embedding similarity to the query (vector search).

    Embeds the query and every document, then returns the closest documents by
    cosine similarity. The corpus is embedded once and cached. Unlike
    keyword_search this matches meaning, so 'how long does the charge last'
    can surface a review that only mentions 'battery'.

    Returns:
        str: JSON {"ok": true, "query": str, "backend": str, "dim": int,
            "results": [{"doc_id": int, "score": float, "text": str}, ...]}.
    """
    try:
        index = _get_index(name)
        hits = index.search(query, top_k)
    except DataAnalystError as exc:
        return _err(str(exc))
    return _ok(
        {
            "query": query,
            "backend": index.backend.name,
            "dim": index.dim,
            "results": [
                {"doc_id": h.doc_id, "score": h.score, "text": h.text} for h in hits
            ],
        }
    )


def main() -> None:
    """Entry point: run the server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
