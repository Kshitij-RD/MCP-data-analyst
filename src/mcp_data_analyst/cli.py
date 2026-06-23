"""Interactive command-line chat for the data-analysis assistant.

Launches the ``data_analyst_mcp`` server as a subprocess, connects to it over
stdio, and runs a REPL where the user chats with the model. The model uses the
MCP tools to load and analyse datasets.

Usage::

    export GROQ_API_KEY=gsk_...
    python -m mcp_data_analyst.cli            # or: data-analyst
"""

from __future__ import annotations

import asyncio
import os
import sys

from .agent import DEFAULT_MODEL, DataAnalystAgent
from .mcp_client import MCPClient
from dotenv import find_dotenv, load_dotenv


BANNER = """\
==============================================
  MCP Data Analyst  (powered by MCP + Groq)
==============================================
Type a question in plain English. The assistant
will use MCP tools to load and analyse your data.

Commands:  /help   /tools   /reset   /quit
A sample dataset lives at data/sample_sales.csv
A sample text corpus lives at data/reviews.txt
"""

HELP = """\
Ask things like:
  - Load data/sample_sales.csv as 'sales' and describe it
  - Which region has the highest total revenue?
  - Are unit_price and revenue correlated?
  - Find outliers in the units column
  - Plot a bar chart of revenue by category
  - Load data/reviews.txt as 'reviews' and show corpus stats
  - Semantically search reviews for "how long does the charge last"
"""


def _print_tool_call(name: str, tool_input: dict) -> None:
    preview = ", ".join(f"{k}={v!r}" for k, v in list(tool_input.items())[:4])
    print(f"  \033[2m> {name}({preview})\033[0m")


async def run() -> int:
    load_dotenv(find_dotenv(usecwd=True))
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY is not set.", file=sys.stderr)
        print("  Get a free key at https://console.groq.com/keys", file=sys.stderr)
        print("  export GROQ_API_KEY=gsk_...", file=sys.stderr)
        return 1

    # Launch the MCP server as a subprocess that shares this interpreter.
    client = MCPClient(
        command=sys.executable,
        args=["-m", "mcp_data_analyst.server"],
        env=dict(os.environ),
    )

    print(BANNER)
    async with client as mcp:
        agent = DataAnalystAgent(mcp, model=DEFAULT_MODEL)
        tools = await mcp.list_tools()
        print(f"Connected. {len(tools)} tools available. Model: {DEFAULT_MODEL}\n")

        loop = asyncio.get_event_loop()
        while True:
            try:
                user = (await loop.run_in_executor(None, input, "you > ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue

            cmd = user.lower()
            if cmd in ("/quit", "/exit", "/q"):
                break
            if cmd == "/help":
                print(HELP)
                continue
            if cmd == "/reset":
                agent.reset()
                print("(conversation reset)\n")
                continue
            if cmd == "/tools":
                for tool in tools:
                    print(f"  - {tool.name}: {(tool.description or '').splitlines()[0]}")
                print()
                continue

            try:
                answer = await agent.chat(user, on_tool=_print_tool_call)
                print(f"\nassistant > {answer}\n")
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive
                print(f"\n[error] {type(exc).__name__}: {exc}\n", file=sys.stderr)

    print("Goodbye.")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
