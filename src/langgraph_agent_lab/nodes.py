"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


class Classification(BaseModel):
    """Structured intent returned by the classifier model."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    risk_level: Literal["low", "high"] = Field(default="low")


def _model_name() -> str:
    return os.getenv("LLM_MODEL", "gpt-5.6-sol")


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip()


def _invoke_text(prompt: str) -> str:
    response = get_llm(model=_model_name()).invoke(prompt)
    text = _content_text(getattr(response, "content", response))
    if not text:
        raise RuntimeError("LLM returned an empty response")
    return text


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    prompt = f"""
You classify support tickets for a deterministic workflow.
Return only the requested structured fields.

Allowed route values:
- risky: any side effect such as refund, deletion, cancellation, sending an email,
  changing an account, or other action that needs approval.
- tool: an information lookup, order/status/tracking search, or data retrieval.
- missing_info: vague or incomplete request without enough actionable context.
- error: timeout, crash, failure, unavailable service, or recovery problem.
- simple: a general informational question answerable without a tool or side effect.

Priority when signals overlap: risky > tool > missing_info > error > simple.
Set risk_level to high only for risky, otherwise low.

Support ticket:
{query}
""".strip()
    classifier = get_llm(model=_model_name()).with_structured_output(Classification)
    result = classifier.invoke(prompt)
    classification = (
        result if isinstance(result, Classification) else Classification.model_validate(result)
    )
    route = classification.route
    risk_level = "high" if route == "risky" else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classified:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {route}",
                route=route,
                risk_level=risk_level,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    route = state.get("route", "")
    if route == "error" and attempt < 2:
        result = f"ERROR: transient tool failure on attempt {attempt}"
        event_type = "error"
    else:
        result = f"SUCCESS: mock tool completed for query: {state.get('query', '')}"
        event_type = "completed"
    return {
        "tool_results": [result],
        "events": [make_event("tool", event_type, result, attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    results = state.get("tool_results", [])
    latest = results[-1] if results else ""
    evaluation = "needs_retry" if not latest or "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": evaluation,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"tool result evaluated as {evaluation}",
                evaluation=evaluation,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    context = {
        "query": state.get("query", ""),
        "route": state.get("route", ""),
        "tool_results": state.get("tool_results", []),
        "approval": state.get("approval"),
    }
    prompt = f"""
You are a careful customer-support agent. Answer the user's ticket using only
the supplied workflow context. Do not invent account, order, refund, or system
details. If the context contains a tool error, say that the request could not
be completed and explain the next safe step. Keep the answer concise and
helpful. Return only the answer text.

Workflow context:
{json.dumps(context, ensure_ascii=False)}
""".strip()
    answer = _invoke_text(prompt)
    return {
        "final_answer": answer,
        "messages": ["answer:generated"],
        "events": [make_event("answer", "completed", "grounded answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    prompt = f"""
The support ticket below is too vague to act on safely. Ask one specific,
useful clarification question. Do not guess what the user means. Return only
the question.

Ticket: {state.get('query', '')}
""".strip()
    question = _invoke_text(prompt)
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": ["clarify:question_generated"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    proposed_action = (
        f"Review and, only after approval, perform the requested side-effecting action: {query}"
    )
    return {
        "proposed_action": proposed_action,
        "messages": ["risky_action:awaiting_approval"],
        "events": [make_event("risky_action", "completed", "risky action prepared")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return approval data and an audit event.
    """
    decision: object = {
        "approved": True,
        "reviewer": "mock-reviewer",
        "comment": "Approved by offline lab default.",
    }
    event_type = "completed"
    if os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt(
            {
                "type": "approval_required",
                "action": state.get("proposed_action", ""),
                "query": state.get("query", ""),
            }
        )
        event_type = "interrupt_resumed"
    approval = ApprovalDecision.model_validate(decision).model_dump()
    return {
        "approval": approval,
        "events": [
            make_event(
                "approval",
                event_type,
                "approval decision recorded",
                approved=approval["approved"],
                reviewer=approval["reviewer"],
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0) + 1
    max_attempts = state.get("max_attempts", 3)
    latest_error = (state.get("tool_results") or ["classified or transient failure"])[-1]
    error = f"retry {attempt}/{max_attempts}: {latest_error}"
    return {
        "attempt": attempt,
        "errors": [error],
        "events": [make_event("retry", "scheduled", error, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    answer = (
        f"The request could not be completed after {attempt} attempt(s). "
        "It has been recorded for support follow-up."
    )
    return {
        "final_answer": answer,
        "messages": ["dead_letter:escalated"],
        "events": [
            make_event(
                "dead_letter",
                "completed",
                "retry limit exhausted",
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
