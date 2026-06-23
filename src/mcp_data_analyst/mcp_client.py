"""A thin async wrapper around an MCP stdio client session.

:class:`MCPClient` launches an MCP server as a subprocess, performs the
protocol handshake, and exposes two convenience methods the agent needs:
``list_tools`` and ``call_tool``. It also converts MCP tool definitions into
the OpenAI/Groq function-calling schema the agent passes to the model.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """Manage the lifecycle of a single MCP stdio connection."""

    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None):
        self._params = StdioServerParameters(command=command, args=args, env=env)
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCPClient is not connected. Use 'async with'.")
        return self._session

    async def __aenter__(self) -> "MCPClient":
        read, write = await self._stack.enter_async_context(stdio_client(self._params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()
        self._session = None

    async def list_tools(self) -> list[Any]:
        """Return the raw MCP tool definitions advertised by the server."""
        return (await self.session.list_tools()).tools

    async def to_openai_tools(self) -> list[dict[str, Any]]:
        """Convert MCP tools into OpenAI/Groq function-calling definitions."""
        tools = await self.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke an MCP tool and flatten its result into plain text."""
        result = await self.session.call_tool(name, arguments)
        parts: list[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else str(block))
        body = "\n".join(parts).strip()
        if result.isError:
            return f"Tool error: {body}"
        return body or "(tool returned no content)"
