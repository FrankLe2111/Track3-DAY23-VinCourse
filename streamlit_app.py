"""Interactive Streamlit demo for the LangGraph Agentic Orchestration lab.

Run from the repository root with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.metrics import metric_from_state
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import Route, Scenario, initial_state


ROOT = Path(__file__).resolve().parent
SCENARIO_PATH = ROOT / "data" / "sample" / "scenarios.jsonl"
CHECKPOINT_PATH = ROOT / "outputs" / "streamlit_checkpoints.sqlite"

ROUTE_COLORS = {
    "simple": "#00B2FF",
    "tool": "#55649E",
    "missing_info": "#646F79",
    "risky": "#F06A6A",
    "error": "#F06A6A",
}


def inject_design_system() -> None:
    """Asana-inspired visual tokens from the requested DesignMD source."""
    st.markdown(
        """
        <style>
        :root {
          --asana-ink: #0D0E10;
          --asana-muted: #646F79;
          --asana-blue: #00B2FF;
          --asana-blue-soft: #C8EBFC;
          --asana-coral: #F06A6A;
          --asana-bg: #F5F4F3;
          --asana-white: #FFFFFF;
          --asana-border: #DBDBDB;
          --asana-focus: rgba(0, 178, 255, .16);
        }
        html, body, [class*="css"], .stApp { font-family: "TWK Lausanne", -apple-system,
          BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--asana-bg) !important; color: var(--asana-ink) !important; }
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] strong,
        [data-testid="stMarkdownContainer"] span, label, [data-testid="stWidgetLabel"] p,
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: var(--asana-ink) !important; }
        [data-testid="stAppViewContainer"] * { border-color: var(--asana-border); }
        [data-testid="stHeader"] { background: rgba(255,255,255,.94); border-bottom: 1px solid var(--asana-border); }
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: var(--asana-white) !important; color: var(--asana-ink) !important; border-right: 1px solid var(--asana-border); }
        [data-testid="stSidebar"] > div:first-child { padding-top: 24px; }
        .block-container { max-width: 1240px; padding: 32px 48px 80px; }
        .asana-topbar { display:flex; align-items:center; justify-content:space-between; min-height:68px;
          margin:-32px -48px 40px; padding:12px 48px; background:var(--asana-white);
          border-bottom:1px solid var(--asana-border); }
        .asana-brand { display:flex; align-items:center; gap:12px; color:var(--asana-ink); font-size:16px; font-weight:500; }
        .asana-mark { width:28px; height:28px; border-radius:50%; background:var(--asana-ink); position:relative; display:inline-block; }
        .asana-mark:before, .asana-mark:after { content:""; position:absolute; background:white; border-radius:50%; width:7px; height:7px; top:6px; }
        .asana-mark:before { left:6px; box-shadow:9px 0 0 white; }
        .asana-mark:after { left:10px; top:15px; }
        .asana-nav { display:flex; gap:28px; color:var(--asana-muted); font-size:14px; }
        .asana-nav span:first-child { color:var(--asana-ink); font-weight:500; }
        .eyebrow { color:var(--asana-muted); text-transform:uppercase; letter-spacing:.08em; font-size:12px; font-weight:500; margin-bottom:8px; }
        h1, h2, h3 { color:var(--asana-ink); letter-spacing:normal; }
        h1 { font-size:36px !important; line-height:36px !important; font-weight:400 !important; margin:0 0 8px !important; }
        h2 { font-size:24px !important; line-height:29px !important; font-weight:500 !important; }
        h3 { font-size:20px !important; line-height:30px !important; font-weight:400 !important; }
        p, label, .stMarkdown { color:var(--asana-ink); line-height:1.6; }
        .stCaption, [data-testid="stCaptionContainer"] { color:var(--asana-muted) !important; }
        [data-testid="stButton"] button { min-height:44px; border-radius:100px; padding:8px 24px;
          font-size:14px; font-weight:500; border:1px solid var(--asana-ink); transition:all .15s ease; }
        [data-testid="stButton"] button, [data-testid="stButton"] button p { color:var(--asana-ink) !important; background:var(--asana-white); }
        [data-testid="stButton"] button[kind="primary"], [data-testid="stButton"] button[kind="primary"] p { background:var(--asana-ink) !important; color:white !important; border-color:var(--asana-ink); }
        [data-testid="stButton"] button:disabled { opacity:.30; cursor:not-allowed; }
        [data-testid="stButton"] button:hover { transform:translateY(-1px); box-shadow:0 4px 12px rgba(0,0,0,.08); }
        [data-testid="stButton"] button:focus-visible { outline:2px solid var(--asana-blue); outline-offset:4px; }
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-testid="stChatInput"] textarea { min-height:50px; border:1px solid #CFCFCF !important; border-radius:4px;
          background:var(--asana-white); color:var(--asana-ink); padding:12px 16px; font-size:16px; }
        [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus,
        [data-testid="stChatInput"] textarea:focus { border-color:var(--asana-blue); box-shadow:0 0 0 3px var(--asana-focus); }
        [data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder,
        [data-testid="stChatInput"] textarea::placeholder { color:var(--asana-muted) !important; opacity:1; }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div { min-height:40px; border-radius:4px; border:1px solid #C1C1C1 !important; background:var(--asana-white) !important; }
        [data-testid="stSelectbox"] [data-baseweb="select"] *, [data-baseweb="popover"] * { color:var(--asana-ink) !important; }
        [data-baseweb="popover"], [data-baseweb="menu"] { background:var(--asana-white) !important; box-shadow:0 8px 24px rgba(0,0,0,.12) !important; }
        [data-baseweb="menu"] li:hover { background:rgba(0,0,0,.05) !important; }
        [data-testid="stTabs"] [role="tab"] { color:var(--asana-muted); font-size:14px; font-weight:500; padding:12px 16px; }
        [data-testid="stTabs"] [aria-selected="true"] { color:var(--asana-ink); }
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--asana-blue); height:2px; }
        [data-testid="stChatMessage"] { border:1px solid var(--asana-border); border-radius:16px; background:var(--asana-white) !important; padding:20px 24px; margin:12px 0; }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { font-size:16px; }
        [data-testid="stExpander"], [data-testid="stExpander"] details, [data-testid="stExpander"] summary { border:1px solid var(--asana-border); border-radius:16px; background:var(--asana-white) !important; color:var(--asana-ink) !important; }
        [data-testid="stAlert"] { border-radius:16px; border-left-width:4px; color:var(--asana-ink) !important; }
        [data-testid="stMetric"] { background:var(--asana-white) !important; border:1px solid var(--asana-border); border-radius:16px; padding:16px; }
        [data-testid="stDataFrame"] { border:1px solid var(--asana-border); border-radius:16px; overflow:hidden; }
        .live-flow-card { background:var(--asana-white); border:1px solid var(--asana-border); border-radius:16px; padding:18px 20px; margin:0 0 18px; }
        .live-flow-head { display:flex; justify-content:space-between; align-items:center; color:var(--asana-ink); font-size:13px; font-weight:600; margin-bottom:14px; }
        .live-status { color:var(--asana-blue); font-size:12px; font-weight:500; }
        .live-flow-track { display:flex; align-items:center; gap:8px; overflow-x:auto; padding:4px 0 10px; }
        .live-node, .live-start, .live-end { flex:0 0 auto; min-width:92px; min-height:48px; padding:7px 10px; border:1px solid var(--asana-border); border-radius:8px; background:var(--asana-bg); text-align:center; display:flex; flex-direction:column; justify-content:center; }
        .live-node b, .live-start, .live-end { color:var(--asana-ink); font-size:12px; }
        .live-node span { color:var(--asana-muted); font-size:10px; margin-top:2px; }
        .live-node.done { background:#eef9f2; border-color:#9bcbb0; }
        .live-node.active { background:var(--asana-blue-soft); border-color:var(--asana-blue); box-shadow:0 0 0 3px var(--asana-focus); animation:live-pulse .8s ease-in-out infinite alternate; }
        .live-node.error { background:#fff6f6; border-color:var(--asana-coral); }
        .live-arrow { color:var(--asana-muted); font-size:18px; flex:0 0 auto; }
        .live-flow-caption { color:var(--asana-muted); font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; border-top:1px solid var(--asana-border); padding-top:10px; }
        @keyframes live-pulse { from { transform:translateY(0); } to { transform:translateY(-2px); } }
        .graph-canvas-card { background:var(--asana-white); border:1px solid var(--asana-border); border-radius:16px; padding:18px 20px; margin:0 0 18px; }
        .graph-canvas-head { display:flex; justify-content:space-between; color:var(--asana-ink); font-size:13px; margin-bottom:6px; }
        .graph-canvas-head span { color:var(--asana-blue); font-size:12px; }
        .graph-canvas { width:100%; min-height:300px; display:block; overflow:visible; }
        .graph-canvas text { font:500 12px "TWK Lausanne", -apple-system, sans-serif; }
        .graph-node-active { animation:graph-node-pulse .8s ease-in-out infinite alternate; }
        @keyframes graph-node-pulse { from { opacity:.72; } to { opacity:1; } }
        .graph-canvas-legend { display:flex; gap:18px; flex-wrap:wrap; color:var(--asana-muted); font-size:11px; border-top:1px solid var(--asana-border); padding-top:10px; }
        .red-line { color:var(--asana-coral); }.gold-line { color:#B27A1D; }
        code { color:var(--asana-ink); background:var(--asana-blue-soft); border-radius:4px; padding:2px 6px; }
        .route-card { border:1px solid var(--asana-border); border-radius:16px; background:var(--asana-white); padding:24px 28px; }
        .section-gap { height:24px; }
        [data-testid="stCodeBlock"] pre, [data-testid="stJson"] { background:var(--asana-white) !important; color:var(--asana-ink) !important; border:1px solid var(--asana-border); border-radius:8px; }
        :focus-visible { outline:2px solid var(--asana-blue) !important; outline-offset:4px !important; }
        @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration:.01ms !important; transition-duration:.01ms !important; scroll-behavior:auto !important; } }
        @media (max-width: 767px) {
          .block-container { padding:24px 16px 48px; }
          .asana-topbar { margin:-24px -16px 32px; padding:12px 16px; }
          .asana-nav { display:none; }
          h1 { font-size:32px !important; line-height:36px !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session() -> None:
    defaults: dict[str, Any] = {
        "graph": None,
        "checkpointer": None,
        "state": None,
        "scenario": None,
        "config": None,
        "interrupted": False,
        "interrupt_payload": None,
        "last_duration_ms": 0,
        "run_count": 0,
        "trace": [],
        "tour_results": [],
        "chat_history": [],
        "chat_pending": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def build_demo_graph(persistence: str):
    if persistence == "SQLite":
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        checkpointer = build_checkpointer("sqlite", str(CHECKPOINT_PATH))
    else:
        checkpointer = build_checkpointer("memory")
    return checkpointer, build_graph(checkpointer=checkpointer)


def get_interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__") or result.get("__interrupts__")
    if not interrupts:
        return None
    item = interrupts[0] if isinstance(interrupts, (list, tuple)) else interrupts
    value = getattr(item, "value", item)
    return value if isinstance(value, dict) else {"message": str(value)}


GRAPH_POSITIONS = {
    "START": (42, 182), "intake": (125, 182), "classify": (220, 182),
    "answer": (360, 72), "tool": (360, 182), "clarify": (360, 330),
    "risky_action": (360, 292), "approval": (505, 292), "evaluate": (505, 182),
    "retry": (650, 182), "dead_letter": (650, 330), "finalize": (805, 182), "END": (940, 182),
}
GRAPH_EDGES = [
    ("START", "intake", "normal"), ("intake", "classify", "normal"),
    ("classify", "answer", "normal"), ("classify", "tool", "normal"),
    ("classify", "clarify", "normal"), ("classify", "risky_action", "risky"),
    ("classify", "retry", "retry"), ("risky_action", "approval", "risky"),
    ("approval", "tool", "risky"), ("approval", "clarify", "risky"),
    ("tool", "evaluate", "normal"), ("evaluate", "answer", "normal"),
    ("evaluate", "retry", "retry"), ("retry", "tool", "retry"),
    ("retry", "dead_letter", "retry"), ("answer", "finalize", "normal"),
    ("clarify", "finalize", "normal"), ("dead_letter", "finalize", "retry"),
    ("finalize", "END", "normal"),
]


def render_graph_canvas(slot: Any, active: str | None, completed: set[str], pending: bool = False) -> None:
    """Render the complete graph topology; only visual state changes per event."""
    def edge_line(source: str, target: str, kind: str) -> str:
        x1, y1 = GRAPH_POSITIONS[source]
        x2, y2 = GRAPH_POSITIONS[target]
        color = "#F06A6A" if kind == "risky" else "#D89A32" if kind == "retry" else "#B5BDC4"
        dash = " stroke-dasharray='6 5'" if kind != "normal" else ""
        return f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='2' marker-end='url(#arrow)'{dash}/>"

    edges = "".join(edge_line(*edge) for edge in GRAPH_EDGES)
    nodes = []
    for name, (x, y) in GRAPH_POSITIONS.items():
        is_active = name == active
        is_done = name in completed
        if is_active:
            fill, stroke, text = "#C8EBFC", "#00B2FF", "#0D0E10"
        elif name in {"risky_action", "approval"}:
            fill, stroke, text = "#FFF6F6", "#F06A6A", "#0D0E10"
        elif name in {"retry", "dead_letter"}:
            fill, stroke, text = "#FFF9EE", "#D89A32", "#0D0E10"
        elif is_done:
            fill, stroke, text = "#EEF9F2", "#9BCBB0", "#0D0E10"
        else:
            fill, stroke, text = "#FFFFFF", "#DBDBDB", "#646F79"
        width = 82 if len(name) < 10 else 108
        pulse = " class='graph-node-active'" if is_active else ""
        nodes.append(f"<g{pulse}><rect x='{x-width/2}' y='{y-22}' width='{width}' height='44' rx='8' fill='{fill}' stroke='{stroke}' stroke-width='2'/><text x='{x}' y='{y+4}' text-anchor='middle' fill='{text}'>{name}</text></g>")
    status = "Waiting for human approval" if pending else (f"Running: {active}" if active else "Ready")
    slot.markdown(
        f"""<div class='graph-canvas-card'><div class='graph-canvas-head'><b>Graph flow</b><span>{status}</span></div>
        <svg class='graph-canvas' viewBox='0 0 982 390' role='img' aria-label='LangGraph workflow'>{'<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#B5BDC4"/></marker></defs>'}{edges}{''.join(nodes)}</svg>
        <div class='graph-canvas-legend'><span>— fixed edge</span><span class='red-line'>- - risky / approval</span><span class='gold-line'>- - retry / dead letter</span></div></div>""",
        unsafe_allow_html=True,
    )


def state_trace(config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        history = list(st.session_state["graph"].get_state_history(config))
        return [item.values for item in reversed(history)]
    except Exception:
        return []


def run_graph(
    state: dict[str, Any],
    config: dict[str, Any],
    animation_slot: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, int]:
    started = time.perf_counter()
    graph = st.session_state["graph"]
    completed: set[str] = set()
    interrupt_payload: dict[str, Any] | None = None
    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue
        possible_interrupt = get_interrupt_payload(chunk)
        if possible_interrupt is not None:
            interrupt_payload = possible_interrupt
            break
        node_names = [name for name in chunk if not str(name).startswith("__")]
        for node_name in node_names:
            completed.add(str(node_name))
            if animation_slot is not None:
                render_graph_canvas(animation_slot, str(node_name), completed, pending=False)
                time.sleep(0.18)
    result = dict(graph.get_state(config).values)
    if interrupt_payload is not None:
        result["__interrupt__"] = (interrupt_payload,)
    duration = int((time.perf_counter() - started) * 1000)
    st.session_state["trace"] = state_trace(config) or [result]
    if animation_slot is not None:
        active = next((event.get("node") for event in reversed(result.get("events", []) or [])), None)
        render_graph_canvas(animation_slot, active, completed, pending=interrupt_payload is not None)
    return result, interrupt_payload, duration


def event_rows(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not state:
        return []
    rows = []
    for index, event in enumerate(state.get("events", []) or [], start=1):
        row = dict(event)
        row["#"] = index
        rows.append(row)
    return rows


def render_graph() -> None:
    st.subheader("Graph flow")
    st.graphviz_chart(
        """digraph {
          rankdir=LR; graph [bgcolor="#F5F4F3", pad="0.2"];
          node [shape=box, style="rounded,filled", fontname="Arial", fontcolor="#0D0E10", color="#DBDBDB", fillcolor="#FFFFFF"];
          edge [color="#00B2FF", penwidth=1.5];
          START -> intake -> classify;
          classify -> answer [label="simple"];
          classify -> tool [label="tool"];
          classify -> clarify [label="missing_info"];
          classify -> risky_action [label="risky", color="#dc2626"];
          classify -> retry [label="error", color="#d97706"];
          risky_action -> approval [color="#dc2626"];
          approval -> tool [label="approved"];
          approval -> clarify [label="rejected"];
          tool -> evaluate;
          evaluate -> answer [label="success"];
          evaluate -> retry [label="needs_retry", color="#d97706"];
          retry -> tool [label="attempt < max"];
          retry -> dead_letter [label="max reached", color="#d97706"];
          answer -> finalize; clarify -> finalize; dead_letter -> finalize -> END;
        }""",
        use_container_width=True,
    )


def render_status(state: dict[str, Any] | None, interrupted: bool) -> None:
    if interrupted:
        st.warning("Workflow đang tạm dừng để chờ human approval.")
        return
    if not state:
        st.info("Chưa có run. Chọn scenario hoặc nhập query rồi bấm Run graph.")
        return
    route = str(state.get("route") or "pending")
    final = state.get("final_answer") or state.get("pending_question") or "Đang xử lý..."
    color = ROUTE_COLORS.get(route, "#64748b")
    st.markdown(
        f"<div style='border-left:8px solid {color};padding:12px 16px;background:#f8fafc;border-radius:8px'>"
        f"<b>Route:</b> <span style='color:{color};font-size:1.2rem'>{route}</span>"
        f"<br><b>Attempt:</b> {state.get('attempt', 0)} / {state.get('max_attempts', 3)}"
        f"<br><b>Final output:</b> {final}</div>",
        unsafe_allow_html=True,
    )


def render_timeline(state: dict[str, Any] | None) -> None:
    rows = event_rows(state)
    if not rows:
        return
    st.subheader("Execution timeline")
    cards = []
    for row in rows:
        node = row.get("node", "unknown")
        event_type = row.get("event_type", "event")
        cards.append(
            f"<div style='display:flex;gap:12px;align-items:flex-start;margin:8px 0'>"
            f"<div style='min-width:30px;height:30px;border-radius:50%;background:#2563eb;color:white;text-align:center;padding-top:3px'>{row['#']}</div>"
            f"<div><b>{node}</b> <code>{event_type}</code><br><span>{row.get('message','')}</span></div></div>"
        )
    st.markdown("<div style='border-left:3px solid #bfdbfe;padding-left:14px'>" + "".join(cards) + "</div>", unsafe_allow_html=True)
    with st.expander("Raw state / audit events"):
        st.json(state)


def render_state_trace(trace: list[dict[str, Any]] | None) -> None:
    """Show the state after each checkpoint/node for classroom explanation."""
    if not trace:
        return
    st.subheader("Node-by-node state trace")
    rows = []
    for index, snapshot in enumerate(trace, start=1):
        events = snapshot.get("events", []) or []
        last_event = events[-1] if events else {}
        rows.append({
            "Step": index,
            "Node": last_event.get("node", "initial"),
            "Event": last_event.get("event_type", "initial"),
            "Route": snapshot.get("route", ""),
            "Attempt": f"{snapshot.get('attempt', 0)}/{snapshot.get('max_attempts', 3)}",
            "Evaluation": snapshot.get("evaluation_result", ""),
            "Approval": (snapshot.get("approval") or {}).get("approved", "") if isinstance(snapshot.get("approval"), dict) else "",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("State snapshots (sau từng checkpoint)"):
        for index, snapshot in enumerate(trace, start=1):
            events = snapshot.get("events", []) or []
            node = events[-1].get("node", "initial") if events else "initial"
            st.markdown(f"**Step {index}: `{node}`**")
            st.json({key: snapshot.get(key) for key in (
                "query", "route", "risk_level", "attempt", "evaluation_result",
                "pending_question", "proposed_action", "approval", "tool_results", "final_answer",
            )})


def run_tour(scenarios: list[Scenario], animation_slot: Any | None = None) -> None:
    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        state = initial_state(scenario)
        config = {"configurable": {"thread_id": f"tour-{scenario.id}"}}
        result, interrupt_payload, duration = run_graph(state, config, animation_slot=animation_slot)
        auto_approved = False
        if interrupt_payload is not None:
            auto_approved = True
            result = resume_graph(config, {
                    "approved": True,
                    "reviewer": "classroom-tour",
                    "comment": "Auto-approved for continuous tour",
                }, animation_slot)
            trace = state_trace(config)
        else:
            trace = st.session_state.get("trace", [])
        metric = metric_from_state(result, scenario.expected_route.value, scenario.requires_approval)
        results.append({"scenario": scenario, "state": result, "trace": trace, "metric": metric, "duration": duration, "auto_approved": auto_approved})
    st.session_state["tour_results"] = results


def render_tour_results() -> None:
    results = st.session_state.get("tour_results", [])
    if not results:
        return
    st.subheader("Classroom tour results")
    summary = []
    for item in results:
        scenario = item["scenario"]
        metric = item["metric"]
        summary.append({
            "Scenario": scenario.id,
            "Query": scenario.query,
            "Expected": metric.expected_route,
            "Actual": metric.actual_route,
            "Result": "PASS" if metric.success else "FAIL",
            "Nodes": metric.nodes_visited,
            "Retries": metric.retry_count,
            "Approvals": metric.interrupt_count,
        })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    for item in results:
        scenario = item["scenario"]
        metric = item["metric"]
        label = f"{scenario.id} — {metric.actual_route} — {'PASS' if metric.success else 'FAIL'}"
        with st.expander(label, expanded=True):
            st.write(f"**Query:** {scenario.query}")
            if item["auto_approved"]:
                st.info("Risky scenario đã được auto-approve để tour chạy liên tục. Demo HITL thủ công ở tab Run scenarios.")
            render_state_trace(item["trace"])
            st.json(item["state"])


def render_metrics(state: dict[str, Any] | None, scenario: Scenario | None) -> None:
    if not state or not scenario or st.session_state.get("interrupted"):
        return
    metric = metric_from_state(state, scenario.expected_route.value, scenario.requires_approval)
    st.subheader("Run metrics")
    cols = st.columns(5)
    cols[0].metric("Success", "PASS" if metric.success else "FAIL")
    cols[1].metric("Actual route", metric.actual_route or "-")
    cols[2].metric("Nodes", metric.nodes_visited)
    cols[3].metric("Retries", metric.retry_count)
    cols[4].metric("Approvals", metric.interrupt_count)
    st.dataframe([metric.model_dump()], use_container_width=True, hide_index=True)


def scenario_selector() -> Scenario | None:
    scenarios = load_scenarios(SCENARIO_PATH)
    labels = [f"{item.id} — {item.expected_route.value} — {item.query}" for item in scenarios]
    selected = st.selectbox("Sample scenario", labels)
    return scenarios[labels.index(selected)]


def start_run(scenario: Scenario, animation_slot: Any | None = None) -> None:
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    result, interrupt_payload, duration = run_graph(state, config, animation_slot=animation_slot)
    st.session_state.update(
        state=result,
        scenario=scenario,
        config=config,
        interrupted=interrupt_payload is not None,
        interrupt_payload=interrupt_payload,
        last_duration_ms=duration,
        run_count=st.session_state.get("run_count", 0) + 1,
    )


def resume_graph(config: dict[str, Any], decision: dict[str, Any], animation_slot: Any | None = None) -> dict[str, Any]:
    from langgraph.types import Command

    graph = st.session_state["graph"]
    completed: set[str] = set()
    for chunk in graph.stream(Command(resume=decision), config=config, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue
        for node_name in chunk:
            if str(node_name).startswith("__"):
                continue
            completed.add(str(node_name))
            if animation_slot is not None:
                render_graph_canvas(animation_slot, str(node_name), completed, pending=False)
                time.sleep(0.18)
    result = dict(graph.get_state(config).values)
    st.session_state["trace"] = state_trace(config) or [result]
    if animation_slot is not None:
        active = next((event.get("node") for event in reversed(result.get("events", []) or [])), None)
        render_graph_canvas(animation_slot, active, completed, pending=False)
    return result


def resume_approval(approved: bool, animation_slot: Any | None = None) -> None:
    decision = {
        "approved": approved,
        "reviewer": "classroom-demo",
        "comment": "Approved during demo" if approved else "Rejected during demo",
    }
    started = time.perf_counter()
    result = resume_graph(st.session_state["config"], decision, animation_slot)
    st.session_state.update(
        state=result,
        interrupted=False,
        interrupt_payload=None,
        last_duration_ms=int((time.perf_counter() - started) * 1000),
    )


def submit_chat_query(query: str, expected_route: str | None = None, animation_slot: Any | None = None) -> None:
    query = query.strip()
    if not query:
        return
    route = expected_route or "simple"
    scenario = Scenario(
        id=f"chat-{int(time.time() * 1000)}",
        query=query,
        expected_route=Route(route),
        requires_approval=route == "risky",
    )
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    result, interrupt_payload, duration = run_graph(state, config, animation_slot=animation_slot)
    trace = st.session_state.get("trace", [])
    content = (
        "⏸️ Tôi đã phân loại đây là hành động cần phê duyệt. "
        "Vui lòng xem proposed action và chọn Approve hoặc Reject."
        if interrupt_payload
        else result.get("final_answer") or result.get("pending_question") or "Workflow đã hoàn tất."
    )
    st.session_state["chat_history"].append({
        "role": "user", "content": query,
    })
    st.session_state["chat_history"].append({
        "role": "assistant", "content": content, "state": result, "trace": trace,
        "scenario": scenario, "config": config, "interrupt_payload": interrupt_payload,
        "duration": duration,
    })
    st.session_state["chat_pending"] = len(st.session_state["chat_history"]) - 1 if interrupt_payload else None


def resume_chat(approved: bool, animation_slot: Any | None = None) -> None:
    index = st.session_state.get("chat_pending")
    if index is None:
        return
    message = st.session_state["chat_history"][index]
    started = time.perf_counter()
    result = resume_graph(message["config"], {
            "approved": approved,
            "reviewer": "chat-user",
            "comment": "Approved in chatbot" if approved else "Rejected in chatbot",
        }, animation_slot)
    trace = state_trace(message["config"])
    message.update({
        "content": result.get("final_answer") or result.get("pending_question") or "Workflow đã hoàn tất.",
        "state": result,
        "trace": trace,
        "interrupt_payload": None,
        "duration": int((time.perf_counter() - started) * 1000),
    })
    st.session_state["chat_pending"] = None


def render_chatbot() -> None:
    st.header("💬 Support-ticket chatbot")
    st.caption("Nhập câu hỏi như người dùng thật. Mỗi câu trả lời có thể mở để xem flow, node và state.")
    animation_slot = st.empty()

    scenarios = load_scenarios(SCENARIO_PATH)
    preset_map = {item.id: item for item in scenarios}
    preset_id = st.selectbox(
        "Câu hỏi tạo sẵn theo scenario",
        [item.id for item in scenarios],
        format_func=lambda item_id: f"{item_id} — {preset_map[item_id].query}",
    )
    preset_cols = st.columns([3, 1])
    preset_cols[0].caption(f"Query mẫu: {preset_map[preset_id].query}")
    if preset_cols[1].button("Dùng câu hỏi mẫu", use_container_width=True):
        submit_chat_query(preset_map[preset_id].query, preset_map[preset_id].expected_route.value, animation_slot)
        st.rerun()

    for index, message in enumerate(st.session_state["chat_history"]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                state = message.get("state", {})
                route = state.get("route", "pending")
                st.caption(
                    f"Route: `{route}` · Nodes: {len(message.get('trace', []))} · "
                    f"Duration: {message.get('duration', 0)} ms"
                )
                pending = message.get("interrupt_payload")
                if pending:
                    st.json(pending)
                    approve, reject = st.columns(2)
                    if approve.button("✅ Approve", key=f"chat-approve-{index}", use_container_width=True):
                        resume_chat(True, animation_slot)
                        st.rerun()
                    if reject.button("❌ Reject", key=f"chat-reject-{index}", use_container_width=True):
                        resume_chat(False, animation_slot)
                        st.rerun()
                with st.expander("🔎 Xem flow & trạng thái từng bước", expanded=False):
                    render_state_trace(message.get("trace", []))
                    st.markdown("**Final state**")
                    st.json(state)

    typed_query = st.chat_input("Nhập support ticket, ví dụ: Refund this customer")
    if typed_query:
        submit_chat_query(typed_query, animation_slot=animation_slot)
        st.rerun()


def render_demo_script() -> None:
    st.header("Demo script — nói theo từng bước")
    st.markdown(
        """
### 1. Giới thiệu bài toán
“Đây là support-ticket agent. Điểm quan trọng không phải chỉ trả lời, mà là điều phối workflow có state, route, retry, approval và audit.”

### 2. Chạy simple route
Chọn **S01_simple**. Nói: “Ticket này không cần tool hay side effect. LLM phân loại `simple`, graph đi `answer → finalize → END`.”

### 3. Chạy tool route
Chọn **S02_tool**. Nói: “Lookup cần tool. Kết quả tool đi qua `evaluate`; chỉ khi success mới sang answer.”

### 4. Chạy missing information
Chọn **S03_missing**. Nói: “Agent không hallucinate. Nó dừng ở `clarify` và hỏi lại thông tin còn thiếu.”

### 5. Chạy risky + HITL
Chọn **S04_risky**. Bấm **Run graph**, sau đó giải thích: “Graph đã chuẩn bị action nhưng dừng tại approval. Tool chưa được phép chạy khi chưa có reviewer.” Bấm **Approve**, rồi chỉ ra flow tiếp tục qua tool/evaluate/answer.

### 6. Chạy retry và dead letter
Chọn **S05_error**. Nói: “Tool giả lập lỗi transient. `evaluate` trả `needs_retry`, retry tăng attempt và quay lại tool. Sau khi thành công, graph trả lời.”

Chọn **S07_dead_letter**. Nói: “max_attempts = 1, nên retry bị chặn và request đi `dead_letter → finalize`, tránh loop vô hạn.”

### 7. Chốt bằng evidence
Mở **Execution timeline**, **Raw state**, metrics và graph. Nhấn mạnh: route, retry count, approval count, node count và finalize event đều có bằng chứng trong state.
        """
    )


def main() -> None:
    st.set_page_config(page_title="LangGraph Lab Demo", page_icon="🧭", layout="wide")
    init_session()
    inject_design_system()

    st.markdown(
        """
        <div class="asana-topbar">
          <div class="asana-brand"><span class="asana-mark"></span><span>Track 3 / Agent Lab</span></div>
          <div class="asana-nav"><span>Overview</span><span>Chatbot</span><span>Execution</span><span>Evidence</span></div>
        </div>
        <div class="eyebrow">Support operations · classroom workspace</div>
        """,
        unsafe_allow_html=True,
    )
    st.title("Agentic workflow, made visible")
    st.caption("Run a support ticket through classification, routing, retry, approval and grounded response.")

    with st.sidebar:
        st.markdown("<div class='eyebrow'>Workspace</div>", unsafe_allow_html=True)
        st.header("Demo controls")
        st.caption("Configure the execution layer, then use the workspace tabs to run and inspect the agent.")
        persistence = st.radio("Persistence", ["Memory", "SQLite"], index=1)
        os.environ["LANGGRAPH_INTERRUPT"] = "true"
        if st.button("Initialize / reset graph", use_container_width=True):
            checkpointer, graph = build_demo_graph(persistence)
            st.session_state.update(graph=graph, checkpointer=checkpointer, state=None, scenario=None, interrupted=False, interrupt_payload=None)
            st.success(f"Graph ready ({persistence})")
        st.caption("HITL interrupt được bật để demo approval thật.")
        if st.session_state.get("graph") is not None:
            st.success("Graph initialized")
        st.divider()
        st.write("Run count:", st.session_state.get("run_count", 0))
        st.write("Last duration:", f"{st.session_state.get('last_duration_ms', 0)} ms")

    if st.session_state.get("graph") is None:
        checkpointer, graph = build_demo_graph("SQLite")
        st.session_state.update(graph=graph, checkpointer=checkpointer)

    tab_chat, tab_demo, tab_tour, tab_custom, tab_script = st.tabs(["💬 Chatbot", "Run scenarios", "Classroom tour", "Custom ticket", "Demo script"])
    with tab_chat:
        render_chatbot()

    with tab_demo:
        demo_animation_slot = st.empty()
        left, right = st.columns([2, 1])
        with left:
            selected_scenario = scenario_selector()
        with right:
            st.write("")
            st.write("")
            if st.button("▶ Run graph", type="primary", use_container_width=True):
                start_run(selected_scenario, demo_animation_slot)
        if st.session_state.get("interrupted"):
            payload = st.session_state.get("interrupt_payload") or {}
            st.warning("⏸ Human approval required")
            st.json(payload)
            approve, reject = st.columns(2)
            if approve.button("✅ Approve action", use_container_width=True):
                resume_approval(True, demo_animation_slot)
                st.rerun()
            if reject.button("❌ Reject action", use_container_width=True):
                resume_approval(False, demo_animation_slot)
                st.rerun()

    with tab_tour:
        tour_animation_slot = st.empty()
        st.subheader("3–5 scenario classroom tour")
        st.caption("Tour chạy liên tiếp và hiển thị state sau từng node. Risky route được auto-approve để không chặn tour.")
        tour_scenarios = load_scenarios(SCENARIO_PATH)
        tour_labels = [f"{item.id} — {item.expected_route.value} — {item.query}" for item in tour_scenarios]
        selected_labels = st.multiselect(
            "Chọn 3–5 scenario",
            tour_labels,
            default=tour_labels[:5],
        )
        if len(selected_labels) < 3 or len(selected_labels) > 5:
            st.warning("Hãy chọn từ 3 đến 5 scenario.")
        elif st.button("▶ Run classroom tour", type="primary"):
            run_tour([tour_scenarios[tour_labels.index(label)] for label in selected_labels], tour_animation_slot)
        render_tour_results()

    with tab_custom:
        query = st.text_area("Support ticket", value="Please lookup order status for order 12345", height=100)
        expected = st.selectbox("Expected route for classroom check", [route.value for route in Route if route in {Route.SIMPLE, Route.TOOL, Route.MISSING_INFO, Route.RISKY, Route.ERROR}])
        if st.button("Run custom ticket", type="primary"):
            custom = Scenario(id=f"ui-{int(time.time())}", query=query, expected_route=Route(expected), requires_approval=expected == "risky")
            start_run(custom)

    with tab_script:
        render_demo_script()

    st.divider()
    render_graph()
    if st.session_state.get("interrupted"):
        render_status(st.session_state.get("state"), True)
    else:
        render_status(st.session_state.get("state"), False)
        render_metrics(st.session_state.get("state"), st.session_state.get("scenario"))
    render_timeline(st.session_state.get("state"))
    render_state_trace(st.session_state.get("trace"))


if __name__ == "__main__":
    main()
