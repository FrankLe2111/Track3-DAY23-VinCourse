"""Node functions for the LangGraph support-ticket workflow."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


class Classification(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    risk_level: Literal["low", "high"] = "low"
    rationale: str = Field(default="")


def _content(response: object) -> str:
    value = getattr(response, "content", response)
    if isinstance(value, list):
        return " ".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in value)
    return str(value)


def intake_node(state: AgentState) -> dict:
    query = state.get("query", "").strip()
    return {"query": query, "messages": [f"intake:{query[:40]}"], "events": [make_event("intake", "completed", "query normalized")]}


def classify_node(state: AgentState) -> dict:
    """Classify intent with structured LLM output, never by scenario ID."""
    prompt = f"""Classify this support ticket into exactly one route.
Routes: risky (side effects such as refund, delete, cancel, send email),
tool (lookup/search/retrieve information), missing_info (too vague to act),
error (timeout, crash, unavailable service), simple (ordinary answer).
Priority when several apply: risky > tool > missing_info > error > simple.
Set risk_level high only for risky. Do not infer a route from scenario IDs.
Ticket: {state.get('query', '')}"""
    result = get_llm().with_structured_output(Classification).invoke(prompt)
    route = str(result.route)
    risk = result.risk_level or ("high" if route == "risky" else "low")
    return {"route": route, "risk_level": risk, "events": [make_event("classify", "completed", f"classified as {route}", risk_level=risk)]}


def tool_node(state: AgentState) -> dict:
    attempt = int(state.get("attempt", 0))
    if state.get("route") == "error" and attempt < 2:
        result, event_type = f"ERROR: transient tool failure on attempt {attempt + 1}", "error"
    else:
        result, event_type = f"Mock tool success for query: {state.get('query', '')}", "completed"
    return {"tool_results": [result], "events": [make_event("tool", event_type, result, attempt=attempt)]}


def evaluate_node(state: AgentState) -> dict:
    latest = (state.get("tool_results") or [""])[-1]
    result = "needs_retry" if "ERROR" in latest.upper() else "success"
    return {"evaluation_result": result, "events": [make_event("evaluate", "completed", result)]}


def answer_node(state: AgentState) -> dict:
    context = {"query": state.get("query", ""), "tool_results": state.get("tool_results", []), "approval": state.get("approval")}
    prompt = f"""You are a careful support agent. Answer the user's query using only the context below.
Do not claim an action happened unless the tool result and approval context support it.
If context is insufficient, say what is unknown and what the user should provide. Be concise and helpful.
Query and context: {context}"""
    answer = _content(get_llm().invoke(prompt)).strip()
    return {"final_answer": answer, "events": [make_event("answer", "completed", "grounded answer generated")]}


def ask_clarification_node(state: AgentState) -> dict:
    prompt = f"Write one specific clarification question for this incomplete support request: {state.get('query', '')}"
    question = _content(get_llm().invoke(prompt)).strip()
    if not question.endswith(("?", "？")):
        question += "?"
    return {"pending_question": question, "final_answer": question, "events": [make_event("clarify", "completed", "clarification requested")]}


def risky_action_node(state: AgentState) -> dict:
    action = f"Proposed action based on ticket: {state.get('query', '')}"
    return {"proposed_action": action, "events": [make_event("risky_action", "completed", "action prepared", action=action)]}


def approval_node(state: AgentState) -> dict:
    if os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true":
        from langgraph.types import interrupt
        value = interrupt({"type": "approval_required", "proposed_action": state.get("proposed_action")})
        decision = value if isinstance(value, dict) else {"approved": bool(value)}
    else:
        decision = {"approved": True, "reviewer": "mock-reviewer", "comment": "CI mock approval"}
    decision.setdefault("reviewer", "mock-reviewer")
    decision.setdefault("comment", "")
    return {"approval": decision, "events": [make_event("approval", "completed", "approval recorded", approved=bool(decision.get("approved")))]}


def retry_or_fallback_node(state: AgentState) -> dict:
    attempt = int(state.get("attempt", 0)) + 1
    message = f"retry attempt {attempt}/{state.get('max_attempts', 3)}"
    return {"attempt": attempt, "errors": [message], "events": [make_event("retry", "completed", message, attempt=attempt)]}


def dead_letter_node(state: AgentState) -> dict:
    message = f"Unable to complete request after {state.get('attempt', 0)} attempts; sent to dead letter handling."
    return {"final_answer": message, "events": [make_event("dead_letter", "completed", message)]}


def finalize_node(state: AgentState) -> dict:
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
