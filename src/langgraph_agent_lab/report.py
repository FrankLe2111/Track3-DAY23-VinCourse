"""Report generation helper.

Report rendering from MetricsReport data and the lab report template.
"""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    Generate a report that includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan

    Use reports/lab_report_template.md as your guide.

    Return: formatted markdown string
    """
    lines = [
        "# Day 08 Lab Report", "", "## 1. Metrics summary", "",
        f"- Total scenarios: **{metrics.total_scenarios}**",
        f"- Success rate: **{metrics.success_rate:.2%}**",
        f"- Average nodes visited: **{metrics.avg_nodes_visited:.2f}**",
        f"- Total retries: **{metrics.total_retries}**",
        f"- Total interrupts/approvals: **{metrics.total_interrupts}**",
        f"- Resume/history evidence: **{metrics.resume_success}**", "",
        "| Scenario | Expected | Actual | Success | Nodes | Retries | Interrupts | Errors |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for item in metrics.scenario_metrics:
        errors = "; ".join(item.errors) or "-"
        lines.append(f"| {item.scenario_id} | {item.expected_route} | {item.actual_route or '-'} | {item.success} | {item.nodes_visited} | {item.retry_count} | {item.interrupt_count} | {errors} |")
    lines += [
        "", "## 2. Architecture", "",
        "The graph uses `START -> intake -> classify`, then conditional routing to simple, tool, missing-info, risky, or error flows. Tool results pass through `evaluate`; transient failures go through a bounded retry loop and then dead-letter handling. Every terminal path reaches `finalize -> END`.", "",
        "State keeps current values such as `route`, `attempt`, and `final_answer` as overwrite fields. `messages`, `tool_results`, `errors`, and `events` are append-only reducers for auditability.", "",
        "## 3. Failure analysis", "",
        "1. **Transient tool failure:** `evaluate` marks an `ERROR` result as `needs_retry`; `retry` increments `attempt` and `route_after_retry` prevents an unbounded loop.",
        "2. **Risky action without approval:** the graph cannot enter the risky tool path directly; it must pass through `risky_action -> approval`. Rejection goes to clarification.", "",
        "## 4. Persistence / recovery evidence", "",
        "Each CLI run supplies a unique `thread_id` in the configurable graph context. MemorySaver supports the base lab; SQLite checkpointer can persist state across process restarts and can be paired with state-history inspection.", "",
        "## 5. Improvement plan", "",
        "Productionize real approval interrupts, replace mock tools with authenticated adapters, add tracing and latency capture, use an LLM judge for tool quality, and test hidden-style adversarial scenarios.", "",
    ]
    return "\n".join(lines) + "\n"


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
