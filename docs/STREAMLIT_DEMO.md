# Streamlit classroom demo

## Install and run

From the repository root:

```powershell
python -m pip install -e ".[ui,openai,sqlite]"
streamlit run streamlit_app.py
```

The app loads `.env`, builds the same LangGraph used by the CLI, enables real `interrupt()` for the approval node, and uses SQLite at `outputs/streamlit_checkpoints.sqlite` by default.

## Recommended classroom sequence

### Fast tour: 3–5 scenarios

Open **Classroom tour**, keep the default five scenarios, and press **Run classroom tour**. The UI produces a summary table and an expandable panel for each scenario. Each panel shows the state snapshot after every checkpoint: node, route, attempt, evaluation result, approval, tool results, and final answer.

Use this mode when you want to show the whole workflow in one continuous classroom demo. Risky scenarios are auto-approved in this tour so the run can continue; use the individual **Run scenarios** tab for the manual Approve/Reject HITL demonstration.

1. Run `S01_simple` and show `classify → answer → finalize`.
2. Run `S02_tool` and show `tool → evaluate → answer`.
3. Run `S03_missing` and show the clarification question.
4. Run `S04_risky`; pause at approval, explain why the action cannot proceed yet, then press **Approve**.
5. Run `S05_error`; show the two retry events and eventual recovery.
6. Run `S07_dead_letter`; show `max_attempts=1` prevents an infinite loop.
7. Open the timeline, raw state, metrics and graph to close with evidence.

The **Demo script** tab contains suggested words for each step.

## Chatbot mode

The **💬 Chatbot** tab is the user-facing demo. Choose one of the preset scenario questions or type a free-form support ticket in the chat input. Each assistant message shows its route and node count, and the expandable **Xem flow & trạng thái từng bước** panel exposes every checkpoint state. Risky queries pause inside the conversation and provide **Approve/Reject** buttons; the same thread then resumes through the graph.

During a run, the **Graph flow** is a fixed topology containing every node and edge in the LangGraph. The UI updates it in real time from `graph.stream(..., stream_mode="updates")`: the active node is highlighted blue, completed nodes turn green, errors are coral, retry/dead-letter edges are gold, and the canvas pauses at approval.
