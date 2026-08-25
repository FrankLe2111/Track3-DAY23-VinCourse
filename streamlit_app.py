"""Streamlit demo UI for the Day 08 LangGraph support-ticket agent.

Run:
    streamlit run streamlit_app.py
    # or
    make ui

Features demonstrated live:
- LLM classification with structured output and conditional routing
- Node-by-node execution trace (streamed from the compiled graph)
- Bounded retry loop and dead-letter escalation
- Real human-in-the-loop approval: interrupt() pauses the graph,
  the reviewer approves/rejects in the UI, and the graph resumes
- Graph diagram and scenario metrics from outputs/metrics.json
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from langgraph.types import Command  # noqa: E402

from langgraph_agent_lab.graph import build_graph  # noqa: E402
from langgraph_agent_lab.persistence import build_checkpointer  # noqa: E402
from langgraph_agent_lab.scenarios import load_scenarios  # noqa: E402
from langgraph_agent_lab.state import Route, Scenario, initial_state  # noqa: E402

# ---------------------------------------------------------------- config ----

st.set_page_config(
    page_title="LangGraph Agent Demo — Day 08",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

NODE_ICONS = {
    "intake": "📥",
    "classify": "🧠",
    "tool": "🔧",
    "evaluate": "🔍",
    "answer": "💬",
    "clarify": "❓",
    "risky_action": "⚠️",
    "approval": "🧑‍⚖️",
    "retry": "🔁",
    "dead_letter": "🪦",
    "finalize": "🏁",
}

ROUTE_STYLE = {
    "simple": ("#0f766e", "Trả lời trực tiếp"),
    "tool": ("#1d4ed8", "Tra cứu bằng tool"),
    "missing_info": ("#b45309", "Thiếu thông tin — hỏi làm rõ"),
    "risky": ("#b42318", "Hành động rủi ro — cần phê duyệt"),
    "error": ("#6d28d9", "Lỗi hệ thống — retry/dead-letter"),
}

CUSTOM_OPTION = "✍️  Nhập câu hỏi tùy chọn…"

st.markdown(
    """
<style>
  .hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 60%, #0f766e 100%);
    border-radius: 14px; padding: 26px 32px; margin-bottom: 18px; color: #fff;
  }
  .hero h1 { margin: 0; font-size: 30px; }
  .hero p { margin: 6px 0 0; opacity: .92; }
  .route-badge {
    display: inline-block; color: #fff; font-weight: 700; font-size: 20px;
    border-radius: 999px; padding: 6px 22px; letter-spacing: .5px;
  }
  .trace-row {
    color: #e2e8f0; background: #1e293b; border: 1px solid #334155;
    border-left: 4px solid #60a5fa; border-radius: 8px; padding: 8px 14px;
    margin: 6px 0; font-size: 14.5px;
  }
  .trace-row b { color: #93c5fd; }
  .approval-card {
    color: #431407; background: #fff7ed; border: 2px solid #f97316;
    border-radius: 12px; padding: 18px 22px; margin: 12px 0;
  }
  .approval-card h3 { margin: 0 0 6px; color: #9a3412; }
  .approval-card b { color: #7c2d12; }
  .approval-card code {
    color: #bbf7d0; background: #172033; border-radius: 5px; padding: 2px 7px;
  }
  .answer-card {
    color: #064e3b; background: #ecfdf5; border: 2px solid #10b981;
    border-radius: 12px; padding: 18px 22px; margin: 12px 0;
  }
  .answer-card h3 { margin: 0 0 6px; color: #065f46; }
  .dl-card {
    color: #7f1d1d; background: #fef2f2; border: 2px solid #ef4444;
    border-radius: 12px; padding: 18px 22px; margin: 12px 0;
  }
  .dl-card h3 { margin: 0 0 6px; color: #991b1b; }
  div[data-testid="stMetric"] {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 8px 14px;
  }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------- graph cache ----


@st.cache_resource(show_spinner=False)
def get_graph():
    """Compile the graph once per server process (in-memory checkpointer)."""
    return build_graph(checkpointer=build_checkpointer("memory"))


# ---------------------------------------------------------------- helpers ----


def summarize_update(node: str, update: dict[str, Any]) -> str:
    """One-line human summary of a node's state update."""
    if node == "classify":
        return f"route = **{update.get('route')}** · risk = {update.get('risk_level')}"
    if node == "tool":
        results = update.get("tool_results") or []
        return str(results[-1])[:120] if results else "tool executed"
    if node == "evaluate":
        return f"evaluation_result = **{update.get('evaluation_result')}**"
    if node == "approval":
        approval = update.get("approval") or {}
        return f"approved = **{approval.get('approved')}** · reviewer = {approval.get('reviewer')}"
    if node == "retry":
        return f"attempt = **{update.get('attempt')}**"
    if node == "risky_action":
        return str(update.get("proposed_action", ""))[:120]
    if node == "clarify":
        return str(update.get("pending_question", ""))[:120]
    if node in ("answer", "dead_letter"):
        return str(update.get("final_answer", ""))[:120]
    events = update.get("events") or []
    return events[-1].get("message", "") if events else ""


def render_trace(trace: list[tuple[str, str]]) -> None:
    """Render the node-by-node execution trace."""
    for node, summary in trace:
        icon = NODE_ICONS.get(node, "▫️")
        st.markdown(
            f'<div class="trace-row">{icon} <b>{node}</b> — {summary}</div>',
            unsafe_allow_html=True,
        )


def stream_graph(graph, payload, config) -> list[tuple[str, str]]:
    """Stream graph execution, rendering each node update live. Returns the trace."""
    trace: list[tuple[str, str]] = []
    placeholder = st.container()
    for chunk in graph.stream(payload, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                continue  # pause payload is read from the checkpoint afterwards
            summary = summarize_update(node, update or {})
            trace.append((node, summary))
            with placeholder:
                render_trace(trace)
    return trace


def get_interrupt(graph, config):
    """Return the pending interrupt payload if the graph is paused, else None."""
    snapshot = graph.get_state(config)
    if not snapshot.next:
        return None
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def route_badge(route: str) -> str:
    color, label = ROUTE_STYLE.get(route, ("#475569", route))
    return (
        f'<span class="route-badge" style="background:{color}">{route.upper()}</span>'
        f'<div style="color:#475569;margin-top:6px">{label}</div>'
    )


# ------------------------------------------------------------------ header ----

st.markdown(
    '<div class="hero"><h1>🤖 LangGraph Support-Ticket Agent</h1>'
    "<p>Day 08 Lab — Agentic Orchestration: phân loại bằng LLM · định tuyến điều kiện · "
    "retry có giới hạn · phê duyệt con người (HITL) · persistence</p></div>",
    unsafe_allow_html=True,
)

graph = get_graph()

# ----------------------------------------------------------------- sidebar ----

scenarios = load_scenarios(ROOT / "data/sample/scenarios.jsonl")
labels = {f"{s.id} — {s.query[:60]}": s for s in scenarios}

with st.sidebar:
    st.header("⚙️ Cấu hình demo")
    choice = st.selectbox("Chọn scenario", [CUSTOM_OPTION, *labels.keys()])
    if choice == CUSTOM_OPTION:
        query = st.text_area(
            "Câu hỏi của khách hàng",
            "Refund this customer and send confirmation email",
        )
        scenario = Scenario(id="custom", query=query, expected_route=Route.SIMPLE)
    else:
        scenario = labels[choice]
        st.caption(f"Route mong đợi: `{scenario.expected_route.value}`")

    hitl = st.toggle(
        "🧑‍⚖️ HITL thật (interrupt/resume)",
        value=True,
        help="Bật: graph dừng tại approval và chờ người duyệt. Tắt: tự động duyệt (mock).",
    )
    reviewer = st.text_input("Tên người duyệt", "giang-vien")
    run_clicked = st.button("🚀 Chạy workflow", type="primary", use_container_width=True)

    st.divider()
    st.caption(f"Model LLM: `{os.getenv('LLM_MODEL', 'gpt-5.6-sol')}`")
    st.caption("Checkpointer: in-memory (session demo)")

os.environ["LANGGRAPH_INTERRUPT"] = "true" if hitl else "false"

# --------------------------------------------------------------- demo logic ----

if "pending" not in st.session_state:
    st.session_state.pending = None  # {"config": ..., "payload": ..., "trace": [...]}


def start_run():
    state = initial_state(scenario)
    state["thread_id"] = f"ui-{scenario.id}-{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": state["thread_id"]}}
    return state, config


tab_demo, tab_diagram, tab_metrics = st.tabs(["🚀 Demo chạy", "🗺️ Sơ đồ graph", "📊 Metrics"])

with tab_demo:
    st.subheader("Kịch bản")
    st.info(f"**Query:** {scenario.query}")

    # --- pending HITL approval takes over the page -------------------------
    pending = st.session_state.pending
    if pending is not None:
        payload = pending["payload"] or {}
        st.markdown(
            '<div class="approval-card"><h3>⏸️ Graph đang DỪNG — chờ phê duyệt của con người</h3>'
            f"<b>Loại:</b> <code>{payload.get('type', 'approval_required')}</code><br>"
            f"<b>Hành động đề xuất:</b> {payload.get('action', '')}<br>"
            f"<b>Query gốc:</b> {payload.get('query', '')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Trace trước khi dừng:**")
        render_trace(pending["trace"])
        comment = st.text_input("Nhận xét của người duyệt", "")
        col_ok, col_no, _ = st.columns([1, 1, 3])
        approved = col_ok.button("✅ Phê duyệt", type="primary", use_container_width=True)
        rejected = col_no.button("❌ Từ chối", use_container_width=True)

        if approved or rejected:
            decision = {"approved": bool(approved), "reviewer": reviewer, "comment": comment}
            with st.status(
                "▶️ Đang resume graph sau quyết định của người duyệt…", expanded=True
            ) as status:
                trace = pending["trace"] + stream_graph(
                    graph, Command(resume=decision), pending["config"]
                )
                status.update(label="✅ Workflow hoàn tất sau khi resume", state="complete")
            final = graph.get_state(pending["config"]).values
            st.session_state.pending = None
            st.session_state.last_result = {"trace": trace, "final": final, "decision": decision}
            st.rerun()

    # --- fresh run ----------------------------------------------------------
    elif run_clicked:
        st.session_state.last_result = None
        state, config = start_run()
        try:
            started = time.perf_counter()
            with st.status("⚙️ Đang thực thi workflow…", expanded=True) as status:
                trace = stream_graph(graph, state, config)
                latency = round((time.perf_counter() - started) * 1000)
                interrupt_payload = get_interrupt(graph, config)
                if interrupt_payload is not None:
                    status.update(label="⏸️ Graph dừng tại approval — chờ quyết định", state="error")
                else:
                    status.update(label=f"✅ Hoàn tất trong {latency} ms", state="complete")

            if interrupt_payload is not None:
                st.session_state.pending = {
                    "config": config,
                    "payload": interrupt_payload,
                    "trace": trace,
                }
                st.rerun()
            final = graph.get_state(config).values
            st.session_state.last_result = {"trace": trace, "final": final, "latency": latency}
            st.rerun()
        except Exception as exc:  # noqa: BLE001 — demo UI must surface any failure
            st.error(f"Lỗi khi chạy graph: {exc}")

    # --- show last result ----------------------------------------------------
    result = st.session_state.get("last_result")
    if pending is None and result:
        final = result["final"]
        st.markdown("### Kết quả")
        col_route, col_latency = st.columns([2, 1])
        with col_route:
            st.markdown(route_badge(final.get("route", "")), unsafe_allow_html=True)
        with col_latency:
            if result.get("latency"):
                st.metric("⏱️ Độ trễ", f"{result['latency']} ms")
            st.metric("🔁 Retry", final.get("attempt", 0))

        st.markdown("**Trace thực thi:**")
        render_trace(result["trace"])

        if result.get("decision"):
            decision = result["decision"]
            st.markdown(
                f"🧑‍⚖️ **Quyết định của người duyệt:** "
                f"{'✅ Phê duyệt' if decision['approved'] else '❌ Từ chối'} "
                f"bởi `{decision['reviewer']}`"
                + (f" — “{decision['comment']}”" if decision.get("comment") else "")
            )

        if final.get("final_answer"):
            is_dead_letter = (
                final.get("route") == "error"
                and final.get("attempt", 0) >= final.get("max_attempts", 3)
            )
            card = "dl-card" if is_dead_letter else "answer-card"
            title = (
                "🪦 Dead letter — không thể hoàn thành"
                if is_dead_letter
                else "💬 Câu trả lời cuối cùng (LLM)"
            )
            st.markdown(
                f'<div class="{card}"><h3>{title}</h3>{final["final_answer"]}</div>',
                unsafe_allow_html=True,
            )
        elif final.get("pending_question"):
            st.markdown(
                '<div class="answer-card"><h3>❓ Câu hỏi làm rõ (LLM)</h3>'
                f'{final["pending_question"]}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("🔎 Xem state cuối cùng (AgentState)"):
            st.json({k: v for k, v in final.items() if k != "__interrupt__"})

# ---------------------------------------------------------------- diagram ----

with tab_diagram:
    st.subheader("Sơ đồ graph đã compile")
    st.caption("Sinh trực tiếp từ `graph.get_graph()` — cạnh đứt là cạnh điều kiện.")
    try:
        png = graph.get_graph().draw_mermaid_png()
        st.image(png, caption="Compiled StateGraph", width=520)
    except Exception:  # noqa: BLE001 — offline fallback
        st.code(graph.get_graph().draw_mermaid(), language="text")
        st.caption("Không tải được mermaid.ink — hiển thị mã Mermaid thay thế.")

# ---------------------------------------------------------------- metrics ----

with tab_metrics:
    st.subheader("Metrics từ lần chạy `make run-scenarios` gần nhất")
    metrics_path = ROOT / "outputs/metrics.json"
    if not metrics_path.exists():
        st.warning("Chưa có outputs/metrics.json — hãy chạy `make run-scenarios` trước.")
    else:
        import json

        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        cols = st.columns(6)
        cols[0].metric("Scenarios", data["total_scenarios"])
        cols[1].metric("Success rate", f"{data['success_rate']:.0%}")
        cols[2].metric("Nodes TB", f"{data['avg_nodes_visited']:.2f}")
        cols[3].metric("Retries", data["total_retries"])
        cols[4].metric("Interrupts", data["total_interrupts"])
        cols[5].metric("Resume", "✅" if data["resume_success"] else "❌")
        st.dataframe(
            [
                {
                    "Scenario": s["scenario_id"],
                    "Mong đợi": s["expected_route"],
                    "Thực tế": s["actual_route"],
                    "Thành công": "✅" if s["success"] else "❌",
                    "Nodes": s["nodes_visited"],
                    "Retry": s["retry_count"],
                    "HITL": s["interrupt_count"],
                    "Độ trễ (ms)": s["latency_ms"],
                }
                for s in data["scenario_metrics"]
            ],
            use_container_width=True,
            hide_index=True,
        )
