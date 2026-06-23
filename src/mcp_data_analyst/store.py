"""In-memory dataset session store backed by DuckDB and pandas.

A single :class:`DataSessionStore` instance holds every dataset the user has
loaded during a session. Datasets are kept as pandas DataFrames and also
registered as DuckDB views so they can be queried with SQL and joined against
one another. The store is the single source of truth shared by all MCP tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

# Loaders keyed by file extension. Each maps a path to a DataFrame.
_LOADERS = {
    ".csv": lambda p: pd.read_csv(p),
    ".tsv": lambda p: pd.read_csv(p, sep="\t"),
    ".parquet": lambda p: pd.read_parquet(p),
    ".json": lambda p: pd.read_json(p),
}

# A valid dataset name doubles as a SQL identifier.
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Statements that must never run against the read-only analytics connection.
_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|"
    r"ATTACH|COPY|EXPORT|INSTALL|LOAD|PRAGMA|SET|CALL)\b",
    re.IGNORECASE,
)
_ALLOWED_PREFIXES = ("SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SUMMARIZE", "SHOW")


class DataAnalystError(Exception):
    """Raised for user-facing, recoverable errors (bad input, missing data)."""


@dataclass
class DatasetInfo:
    """Lightweight metadata about a loaded dataset."""

    name: str
    source: str
    rows: int
    columns: list[str]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "rows": self.rows,
            "n_columns": len(self.columns),
            "columns": self.columns,
        }


class DataSessionStore:
    """Holds loaded datasets and exposes them to pandas and DuckDB."""

    def __init__(self) -> None:
        self._con = duckdb.connect(database=":memory:")
        self._frames: dict[str, pd.DataFrame] = {}
        self._info: dict[str, DatasetInfo] = {}

    # -- lifecycle ---------------------------------------------------------
    @staticmethod
    def validate_name(name: str) -> str:
        """Ensure a dataset name is a safe SQL identifier."""
        if not _NAME_RE.match(name):
            raise DataAnalystError(
                f"Invalid dataset name '{name}'. Use letters, digits and "
                "underscores only, and do not start with a digit "
                "(e.g. 'sales', 'q1_orders')."
            )
        return name

    def load(self, name: str, path: str) -> DatasetInfo:
        """Load a file from disk and register it under ``name``."""
        self.validate_name(name)
        file_path = Path(path).expanduser()
        if not file_path.exists():
            raise DataAnalystError(
                f"File not found: '{path}'. Provide an absolute path or a path "
                "relative to the directory the server was started in."
            )
        loader = _LOADERS.get(file_path.suffix.lower())
        if loader is None:
            raise DataAnalystError(
                f"Unsupported file type '{file_path.suffix}'. Supported types: "
                f"{', '.join(sorted(_LOADERS))}."
            )
        try:
            df = loader(file_path)
        except Exception as exc:  # noqa: BLE001 - surface a clean message
            raise DataAnalystError(f"Failed to parse '{path}': {exc}") from exc

        self._frames[name] = df
        self._con.register(name, df)
        info = DatasetInfo(
            name=name,
            source=str(file_path),
            rows=int(df.shape[0]),
            columns=[str(c) for c in df.columns],
        )
        self._info[name] = info
        return info

    # -- access ------------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self._frames

    def get(self, name: str) -> pd.DataFrame:
        if name not in self._frames:
            raise DataAnalystError(
                f"No dataset named '{name}'. Loaded datasets: "
                f"{', '.join(self._frames) or '(none)'}. "
                "Load one first with load_dataset."
            )
        return self._frames[name]

    def info(self, name: str) -> DatasetInfo:
        self.get(name)  # raises with a helpful message if missing
        return self._info[name]

    def list(self) -> list[DatasetInfo]:
        return list(self._info.values())

    # -- sql ---------------------------------------------------------------
    def assert_read_only(self, query: str) -> None:
        """Reject anything that is not a read-only query."""
        stripped = query.strip().rstrip(";").strip()
        if not stripped:
            raise DataAnalystError("Empty SQL query.")
        if ";" in stripped:
            raise DataAnalystError(
                "Only a single statement is allowed; remove extra ';'."
            )
        first = stripped.split(None, 1)[0].upper()
        if first not in _ALLOWED_PREFIXES:
            raise DataAnalystError(
                f"Only read-only queries are allowed (must start with one of "
                f"{', '.join(_ALLOWED_PREFIXES)}). Got '{first}'."
            )
        if _FORBIDDEN_SQL.search(stripped):
            raise DataAnalystError(
                "Query contains a forbidden keyword. This tool is read-only; "
                "use SELECT/WITH to analyse data."
            )

    def sql(self, query: str) -> pd.DataFrame:
        """Run a validated read-only SQL query and return the result frame."""
        self.assert_read_only(query)
        try:
            return self._con.execute(query).fetch_df()
        except duckdb.Error as exc:
            raise DataAnalystError(f"SQL error: {exc}") from exc
