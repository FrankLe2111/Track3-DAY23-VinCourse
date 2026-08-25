"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

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
            raise RuntimeError("Install: pip install -e '.[sqlite]'") from exc
        connection = sqlite3.connect(database_url or "checkpoints.db", check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.commit()
        return SqliteSaver(conn=connection)
    if kind == "postgres":
        raise RuntimeError(
            "Postgres is optional; install the postgres extra and add a dedicated "
            "adapter if needed."
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
