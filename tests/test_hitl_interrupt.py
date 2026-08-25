"""Extension test: real human-in-the-loop approval via interrupt()/resume.

With LANGGRAPH_INTERRUPT=true, approval_node calls langgraph.types.interrupt()
instead of the mock auto-approval. The graph pauses before the risky action is
executed; a reviewer decision is injected with Command(resume=...).

These tests require a configured LLM because classify_node uses a real LLM call.
"""

import os

import pytest

pytestmark = [
    pytest.mark.skipif(
        __import__("importlib.util", fromlist=["util"]).find_spec("langgraph") is None,
        reason="langgraph not installed",
    ),
    pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY")
        and not os.getenv("OPENAI_API_KEY")
        and not os.getenv("ANTHROPIC_API_KEY"),
        reason="No LLM API key configured (set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)",
    ),
]

from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state

RISKY_QUERY = "Refund this customer and send confirmation email"


@pytest.fixture
def interrupt_env(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")


def _invoke_risky(graph):
    scenario = Scenario(id="hitl", query=RISKY_QUERY, expected_route=Route.RISKY)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    return state, config, graph.invoke(state, config=config)


def test_approval_interrupt_pauses_before_tool(interrupt_env):
    """Graph must pause at approval and expose an interrupt payload."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    _, _, result = _invoke_risky(graph)

    interrupts = result.get("__interrupt__", [])
    assert interrupts, "expected approval_node to interrupt for human review"
    payload = interrupts[0].value
    assert payload["type"] == "approval_required"
    assert RISKY_QUERY in payload["query"]
    # The risky tool action must NOT have executed yet.
    assert not result.get("tool_results")
    assert result.get("final_answer") is None


def test_approval_resume_approved_completes(interrupt_env):
    """Resuming with an approval decision runs the tool and finalizes."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    state, config, _ = _invoke_risky(graph)

    decision = {
        "approved": True,
        "reviewer": "test-reviewer",
        "comment": "Approved in HITL test.",
    }
    result = graph.invoke(Command(resume=decision), config=config)

    assert result["approval"]["approved"] is True
    assert result["approval"]["reviewer"] == "test-reviewer"
    assert result["tool_results"], "tool should run after approval"
    assert result["final_answer"]
    finalize_events = [e for e in result["events"] if e.get("node") == "finalize"]
    assert finalize_events, "resumed run must terminate at finalize"


def test_approval_resume_rejected_goes_to_clarify(interrupt_env):
    """Resuming with a rejection routes to clarification, never to the tool."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    _, config, _ = _invoke_risky(graph)

    decision = {
        "approved": False,
        "reviewer": "test-reviewer",
        "comment": "Rejected in HITL test.",
    }
    result = graph.invoke(Command(resume=decision), config=config)

    assert result["approval"]["approved"] is False
    assert not result["tool_results"], "rejected action must not execute the tool"
    assert result["pending_question"], "rejection should ask for clarification"
