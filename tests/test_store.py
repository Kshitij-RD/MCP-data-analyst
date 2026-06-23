"""Unit tests for the DataSessionStore (no network, no API key)."""

from __future__ import annotations

import pandas as pd
import pytest

from mcp_data_analyst.store import DataAnalystError, DataSessionStore


@pytest.fixture()
def store(tmp_path):
    csv = tmp_path / "t.csv"
    pd.DataFrame(
        {"region": ["N", "S", "N"], "revenue": [10, 20, 30]}
    ).to_csv(csv, index=False)
    s = DataSessionStore()
    s.load("sales", str(csv))
    return s


def test_load_registers_dataset(store):
    info = store.info("sales")
    assert info.rows == 3
    assert info.columns == ["region", "revenue"]
    assert store.has("sales")


def test_missing_dataset_raises_helpful_error():
    s = DataSessionStore()
    with pytest.raises(DataAnalystError, match="No dataset named"):
        s.get("nope")


def test_invalid_name_rejected():
    s = DataSessionStore()
    with pytest.raises(DataAnalystError, match="Invalid dataset name"):
        s.validate_name("1bad")


def test_unsupported_extension(tmp_path):
    bad = tmp_path / "x.xml"
    bad.write_text("<a/>")
    s = DataSessionStore()
    with pytest.raises(DataAnalystError, match="Unsupported file type"):
        s.load("x", str(bad))


def test_sql_aggregation(store):
    df = store.sql("SELECT region, SUM(revenue) AS total FROM sales GROUP BY region ORDER BY region")
    assert list(df["region"]) == ["N", "S"]
    assert list(df["total"]) == [40, 20]


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "INSERT INTO sales VALUES ('E', 1)",
        "SELECT 1; SELECT 2",
        "PRAGMA database_list",
    ],
)
def test_read_only_enforcement(store, query):
    with pytest.raises(DataAnalystError):
        store.sql(query)
