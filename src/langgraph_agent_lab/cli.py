"""CLI for the lab."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


def _has_state_history(graph: object, run_config: Mapping[str, object]) -> bool:
    get_history = getattr(graph, "get_state_history", None)
    if get_history is None:
        return False
    try:
        next(iter(get_history(run_config)))
    except (StopIteration, ValueError):
        return False
    return True


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))  # type: ignore[no-untyped-call]
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer_kind = cfg.get("checkpointer", "memory")
    if checkpointer_kind == "sqlite" and cfg.get("database_url"):
        # Start from a clean database so append-only state (events/errors) does
        # not accumulate across repeated runs on the same thread_id.
        for suffix in ("", "-wal", "-shm"):
            db_file = Path(cfg["database_url"] + suffix)
            if db_file.exists():
                db_file.unlink()
    checkpointer = build_checkpointer(checkpointer_kind, cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    resume_success = False
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        started = perf_counter()
        final_state = graph.invoke(state, config=run_config)
        latency_ms = round((perf_counter() - started) * 1000)
        resume_success = _has_state_history(graph, run_config) or resume_success
        metrics.append(
            metric_from_state(
                final_state,
                scenario.expected_route.value,
                scenario.requires_approval,
                latency_ms=latency_ms,
            )
        )
    report = summarize_metrics(metrics, resume_success=resume_success)
    write_metrics(report, output)
    mermaid = graph.get_graph().draw_mermaid()
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"], diagram=mermaid)
    typer.echo(f"Wrote metrics to {output}")


@app.command("diagram")
def diagram(
    output: Annotated[Path, typer.Option("--output")] = Path("outputs/graph_diagram.md"),
) -> None:
    """Export the compiled graph as a Mermaid diagram (bonus extension)."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    mermaid = graph.get_graph().draw_mermaid()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"```mermaid\n{mermaid}\n```\n", encoding="utf-8")
    typer.echo(f"Wrote Mermaid diagram to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


if __name__ == "__main__":
    app()
