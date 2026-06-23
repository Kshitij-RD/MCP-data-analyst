"""MCP-powered data analysis assistant.

Public API:
    - :class:`~mcp_data_analyst.store.DataSessionStore`
    - :class:`~mcp_data_analyst.text_store.TextCorpusStore`
    - :class:`~mcp_data_analyst.embeddings.EmbeddingIndex`
    - :class:`~mcp_data_analyst.mcp_client.MCPClient`
    - :class:`~mcp_data_analyst.agent.DataAnalystAgent`
"""

from .agent import DataAnalystAgent
from .embeddings import (
    EmbeddingBackend,
    EmbeddingIndex,
    HashingEmbeddingBackend,
    get_default_backend,
)
from .mcp_client import MCPClient
from .store import DataAnalystError, DatasetInfo, DataSessionStore
from .text_store import CorpusInfo, TextCorpusStore

__version__ = "0.2.0"

__all__ = [
    "DataAnalystAgent",
    "MCPClient",
    "DataSessionStore",
    "DatasetInfo",
    "TextCorpusStore",
    "CorpusInfo",
    "EmbeddingBackend",
    "EmbeddingIndex",
    "HashingEmbeddingBackend",
    "get_default_backend",
    "DataAnalystError",
]
