"""Shared pytest setup.

Load the project .env so LLM-dependent tests see the configured API key
without requiring manual environment exports.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
