# Codex task — Repair and finish the Incident Intelligence Agent MVP

Act as the lead Python engineer working **inside this currently open repository**. You have permission to inspect and edit project files, run terminal commands, install project-local Python dependencies as appropriate, and add focused tests. **Make the changes; don't stop at giving me a plan.** Read `AGENTS.md` first.

## Current situation

We already built an Olist analytics project with DuckDB and a weekday-aware statistical anomaly detector. The DuckDB file is normally `data/incident_intelligence.duckdb`; it contains `orders`, `order_items`, `products`, and `detected_anomalies` (221 flagged days in the last tested run). SQL analytics work. We added an MVP ZIP containing `src/agent_tools.py`, `src/incident_agent.py`, `src/app.py`, `requirements-mvp.txt`, and `README-MVP.md`.

The Streamlit dashboard launches, but **the local AI investigation is not working as expected**. We do not yet know the reason. Do not assume it is a specific bug; inspect and reproduce it. My goal is a working portfolio demo quickly, not another round of SQL exercises.

## Outcome I need

In Streamlit, select an anomaly (preferably `2017-11-24`), click **Investigate with local AI**, and see:
- An actual Ollama/Qwen-backed investigation report, grounded in database facts.
- The analytical tools called by the agent, with input arguments and returned evidence.
- A useful message if the Ollama service/model is not available, rather than hanging or silently failing.

Use a local model (`qwen3:4b` by default), on a Linux laptop with 16 GB RAM and ~6 GB usable NVIDIA VRAM. Keep it responsive and lightweight. Preserve the existing Olist data and analytical functionality.

## Steps — execute in order

### 1. Inspect and diagnose
- Check the repo tree, `.gitignore`, dependency files, Python entry points, and relevant scripts (`src/app.py`, `src/incident_agent.py`, `src/agent_tools.py`, `src/explore_data.py`, `src/detect_anomalies.py`, `src/analytics.py`).
- Confirm how the Streamlit button calls `investigate()`, import paths, DB location, expected schema, Ollama model configuration, and Python package versions.
- Verify the local service/model when accessible using `ollama list`, `ollama ps`, and/or a small API smoke test. If the service is absent, don't claim local inference worked.
- Reproduce the error via the CLI (`python src/incident_agent.py 2017-11-24`) and/or Streamlit; collect and interpret any traceback. Look for tool-call serialization, model response handling, import/module path problems, Ollama connection issues, and UI state/rendering problems.

### 2. Fix the shortest path to a working local agent
- Reuse existing parameterized read-only tools and SQL; adjust implementations only as needed.
- Ensure the complete loop works: select anomaly -> query baseline evidence -> call Ollama -> model selects tool(s) -> Python executes tool(s) -> tool results return to model -> final report -> UI shows trace.
- Preserve a clear distinction between seeded lookups and model-initiated tool calls. If no actual tool call occurred, report that rather than displaying a misleading success state.
- Add sensible timeouts, round caps, validation, and observable error handling for nonresponsive Ollama, missing model, bad tool arguments, or unexpected responses.
- Keep the model configurable via environment variables. Avoid adding LangGraph, Docker, cloud LLMs, or a second orchestration stack unless absolutely necessary to repair the current approach.
- Persist the last report/trace in Streamlit session state if ordinary widget reruns would otherwise make them disappear. Provide a small indication that the investigation is executing and whether it completed or failed.

### 3. Validate, test, and document
- Add targeted automated tests using a tiny temporary DuckDB dataset and mocked Ollama chat responses. Test tool dispatch, argument passing, multiple tool steps, correct final report, and failure paths.
- Run tests and fix failures. If actual Ollama is locally available, run a true end-to-end investigation for `2017-11-24` and inspect the report and tool trace.
- Sanity-check known historical metrics: `2017-11-24` should have approximately **R$149,916.58 item sales** and **1,147 distinct delivered orders** if the existing Olist data is intact.
- Update the README with verified Linux commands for activating `.venv`, installing dependencies, starting/checking Ollama, pulling the model, running a CLI smoke test, and launching Streamlit.
- Add/fix `.gitignore` so the repo does not track `.venv`, raw CSVs, generated DuckDB files, models, credentials, or caches.
- Preserve all existing user code and data unless a change is essential. Do not delete, regenerate, or overwrite the user's database merely to pass tests.

## Done means

A working dashboard can select a stored anomaly, run an actual local LLM investigation using one or more **model-selected** read-only tools, show grounded findings and an audit trail, and explain failures clearly. Automated tests pass. Document any environment-related blockers honestly. A GUI screenshot or a chart by itself is **not** sufficient evidence that the agent works.

## Final reply after you implement

Tell me:
1. The root cause(s) you found.
2. Which files you changed.
3. Exactly what you tested and whether each test passed (distinguish mocked from live-model tests).
4. The exact terminal commands I should run on my Linux machine.
5. Any remaining issues worth fixing after the MVP.

Do not ask me a question unless proceeding would be unsafe or fundamentally impossible; make reasonable engineering decisions and implement them.
