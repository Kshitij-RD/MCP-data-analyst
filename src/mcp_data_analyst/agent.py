"""The agent: a Groq-powered tool-use loop wired to MCP tools.

:class:`DataAnalystAgent` is the "host" in MCP terms. It owns the conversation,
sends the MCP server's tools to the model on every turn, and runs the loop:

    model proposes tool calls  ->  MCPClient executes them  ->
    results are fed back  ->  model continues until it produces a final answer.

The LLM is Groq's OpenAI-compatible Chat Completions API (free tier), so tools
use the ``{"type": "function", ...}`` schema and tool calls come back on
``message.tool_calls``. The loop is written out explicitly (rather than hidden
behind a framework) so the control flow is easy to read and reason about.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

from groq import AsyncGroq
from .mcp_client import MCPClient

DEFAULT_MODEL = os.environ.get("DATA_ANALYST_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS = 2048
MAX_TOOL_ITERATIONS = 12  # safety bound on a single turn's tool loop

SYSTEM_PROMPT = (
    "You are a data analysis assistant. You help the user explore tabular "
    "datasets and text corpora using the available tools. Always load a "
    "dataset before analysing it, and inspect its schema with describe_dataset "
    "before writing SQL so you reference real column names. Prefer run_sql for "
    "aggregations and filtering. For text, use keyword_search for exact matches "
    "and semantic_search to match by meaning. When you report findings, be "
    "concise and quantitative, and cite the numbers the tools returned. If a "
    "tool returns an error, read it and adjust instead of repeating the call."
    "Only state facts that appear in the tool results. Never invent counts, "
    "rows, or document text; quote documents exactly as the tool returned them, "
    "and if a tool returns no matches, say so."
)

# Callback invoked when a tool is about to run, for live CLI feedback.
ToolHook = Callable[[str, dict[str, Any]], Awaitable[None] | None]


class DataAnalystAgent:
    """Drive a multi-turn conversation backed by MCP tools via Groq."""

    def __init__(
        self,
        mcp_client: MCPClient,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ):
        self._mcp = mcp_client
        self._model = model
        self._client = AsyncGroq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._tools: list[dict[str, Any]] | None = None

    def reset(self) -> None:
        """Clear the conversation history (keep the system prompt and tools)."""
        self._messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def _ensure_tools(self) -> list[dict[str, Any]]:
        if self._tools is None:
            self._tools = await self._mcp.to_openai_tools()
        return self._tools

    async def chat(self, user_message: str, on_tool: ToolHook | None = None) -> str:
        """Send one user message and return the assistant's final text.

        Args:
            user_message: The user's input for this turn.
            on_tool: Optional callback ``(tool_name, tool_input)`` fired each
                time a tool is invoked, useful for streaming progress to a UI.

        Returns:
            The assistant's final natural-language response for this turn.
        """
        tools = await self._ensure_tools()
        self._messages.append({"role": "user", "content": user_message})

        for _ in range(MAX_TOOL_ITERATIONS):
            response = await self._client.chat.completions.create(
                model=self._model,
                max_tokens=MAX_TOKENS,
                temperature=0,
                messages=self._messages,
                tools=tools,
                tool_choice="auto",
            )
            message = response.choices[0].message
            self._messages.append(_assistant_message(message))

            if not message.tool_calls:
                return (message.content or "").strip()

            for call in message.tool_calls:
                name = call.function.name
                arguments = _parse_arguments(call.function.arguments)
                if on_tool is not None and isinstance(arguments, dict):
                    maybe = on_tool(name, arguments)
                    if maybe is not None:
                        await maybe
                output = (
                    arguments  # an error string from _parse_arguments
                    if isinstance(arguments, str)
                    else await self._mcp.call_tool(name, arguments)
                )
                self._messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output}
                )

        return (
            "Reached the tool-call limit for this turn without a final answer. "
            "Try narrowing the question."
        )


def _assistant_message(message: Any) -> dict[str, Any]:
    """Serialise a Groq assistant message back into the messages list."""
    payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ]
    return payload


def _parse_arguments(raw: str | None) -> dict[str, Any] | str:
    """Parse tool-call argument JSON, returning an error string on failure."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"Tool error: could not parse arguments as JSON ({exc})."
