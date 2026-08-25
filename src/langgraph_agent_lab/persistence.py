"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    MemorySaver is the base implementation; SQLite is supported as an extension.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError("Install SQLite support with: pip install -e '.[sqlite]'") from exc
        target = database_url or "outputs/langgraph_checkpoints.sqlite"
        if target.startswith("sqlite:///"):
            target = target.removeprefix("sqlite:///")
        connection = sqlite3.connect(target, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.commit()
        return SqliteSaver(conn=connection)
    if kind == "postgres":
        raise NotImplementedError(
            "TODO(student): implement Postgres checkpointer (optional extension)"
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
