"""End-to-end test of the MCP server over a real stdio connection.

This launches the server as a subprocess and drives it through the same
MCPClient the agent uses, so it exercises the full protocol path: handshake,
tool discovery and tool invocation. No Groq API key is required.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from mcp_data_analyst.mcp_client import MCPClient

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "sample_sales.csv"
CORPUS = REPO_ROOT / "data" / "reviews.txt"

pytestmark = pytest.mark.asyncio


def _client() -> MCPClient:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return MCPClient(
        command=sys.executable,
        args=["-m", "mcp_data_analyst.server"],
        env=env,
    )


async def test_tool_discovery():
    async with _client() as mcp:
        names = {t.name for t in await mcp.list_tools()}
    assert {
        "load_dataset",
        "list_datasets",
        "describe_dataset",
        "run_sql",
        "compute_correlations",
        "detect_outliers",
        "plot_chart",
        "load_text_corpus",
        "list_corpora",
        "corpus_stats",
        "keyword_search",
        "word_frequencies",
        "semantic_search",
    } <= names


async def test_load_and_query_roundtrip():
    async with _client() as mcp:
        loaded = json.loads(
            await mcp.call_tool("load_dataset", {"name": "sales", "path": str(SAMPLE)})
        )
        assert loaded["ok"] is True
        assert loaded["dataset"]["rows"] == 600

        result = json.loads(
            await mcp.call_tool(
                "run_sql",
                {
                    "query": "SELECT region, SUM(revenue) AS total FROM sales "
                    "GROUP BY region ORDER BY total DESC",
                    "limit": 10,
                },
            )
        )
        assert result["ok"] is True
        assert result["row_count"] == 4
        assert "region" in result["columns"]


async def test_read_only_rejected_over_mcp():
    async with _client() as mcp:
        await mcp.call_tool("load_dataset", {"name": "sales", "path": str(SAMPLE)})
        result = json.loads(
            await mcp.call_tool("run_sql", {"query": "DROP TABLE sales"})
        )
        assert result["ok"] is False
        assert "read-only" in result["error"].lower()


async def test_text_corpus_and_semantic_search_over_mcp():
    async with _client() as mcp:
        loaded = json.loads(
            await mcp.call_tool(
                "load_text_corpus", {"name": "reviews", "path": str(CORPUS)}
            )
        )
        assert loaded["ok"] is True
        assert loaded["corpus"]["n_documents"] == 20

        result = json.loads(
            await mcp.call_tool(
                "semantic_search",
                {"name": "reviews", "query": "how long does the charge last", "top_k": 3},
            )
        )
        assert result["ok"] is True
        assert len(result["results"]) == 3
        assert "backend" in result and "dim" in result
