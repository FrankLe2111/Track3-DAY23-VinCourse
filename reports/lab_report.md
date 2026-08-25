# Day 08 Lab Report

## 1. Metrics summary

- Total scenarios: **7**
- Success rate: **100.00%**
- Average nodes visited: **6.43**
- Total retries: **3**
- Total interrupts/approvals: **2**
- Resume/history evidence: **True**

| Scenario | Expected | Actual | Success | Nodes | Retries | Interrupts | Errors |
|---|---|---|---:|---:|---:|---:|---|
| S01_simple | simple | simple | True | 4 | 0 | 0 | - |
| S02_tool | tool | tool | True | 6 | 0 | 0 | - |
| S03_missing | missing_info | missing_info | True | 4 | 0 | 0 | - |
| S04_risky | risky | risky | True | 8 | 0 | 1 | - |
| S05_error | error | error | True | 10 | 2 | 0 | retry attempt 1/3; retry attempt 2/3 |
| S06_delete | risky | risky | True | 8 | 0 | 1 | - |
| S07_dead_letter | error | error | True | 5 | 1 | 0 | retry attempt 1/1 |

## 2. Architecture

The graph uses `START -> intake -> classify`, then conditional routing to simple, tool, missing-info, risky, or error flows. Tool results pass through `evaluate`; transient failures go through a bounded retry loop and then dead-letter handling. Every terminal path reaches `finalize -> END`.

State keeps current values such as `route`, `attempt`, and `final_answer` as overwrite fields. `messages`, `tool_results`, `errors`, and `events` are append-only reducers for auditability.

## 3. Failure analysis

1. **Transient tool failure:** `evaluate` marks an `ERROR` result as `needs_retry`; `retry` increments `attempt` and `route_after_retry` prevents an unbounded loop.
2. **Risky action without approval:** the graph cannot enter the risky tool path directly; it must pass through `risky_action -> approval`. Rejection goes to clarification.

## 4. Persistence / recovery evidence

Each CLI run supplies a unique `thread_id` in the configurable graph context. MemorySaver supports the base lab; SQLite checkpointer can persist state across process restarts and can be paired with state-history inspection.

## 5. Improvement plan

Productionize real approval interrupts, replace mock tools with authenticated adapters, add tracing and latency capture, use an LLM judge for tool quality, and test hidden-style adversarial scenarios.

