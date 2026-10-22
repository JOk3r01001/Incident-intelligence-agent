# Incident Intelligence Agent — repository instructions for Codex

## Project goal
Ship a small, working, local-first portfolio MVP that investigates anomalies in historical Brazilian e-commerce (Olist) data. A Streamlit user asks the local Ollama-hosted LLM to discover and investigate anomalies using **read-only, parameterized Python/DuckDB tools**, with concise evidence-based answers and a transparent tool-call log. Manual date exploration remains optional; selecting an anomaly must not be required for chat or discovery.

## Existing implementation (inspect, don't assume)
Likely project files:
- `src/prepare_data.py` — safely imports CSVs and detects anomalies for new databases; checks existing databases read-only. Replaces the original exploratory importer and detector.
- `src/agent_tools.py` — database tools exposed to the agent.
- `src/incident_agent.py` — Ollama chat/tool-call loop.
- `src/model_clients.py` and `src/langchain_client.py` — native and optional LangChain model adapters.
- `src/app.py` — Streamlit UI.
- `requirements-mvp.txt` and `README-MVP.md` — current MVP setup.

Some files might not exist or may differ: **inspect the real repository first**, and treat actual code as authoritative.

Expected DuckDB database: `data/incident_intelligence.duckdb`.
Expected tables: `orders`, `order_items`, `products`, `detected_anomalies`.
Olist CSVs may be directly in `data/`, not `data/raw/`; do not assume otherwise. Preserve existing data.

Example known-good source observation for 2017-11-24 (delivered orders, sum of order-item prices): **R$149,916.58** and **1,147 distinct orders**. Historical item sales are not recognized accounting revenue or profit. A reported anomaly is not automatically a business incident.

## Technology and constraints
- Python 3 virtual environment, DuckDB, Streamlit, Ollama with `qwen3:4b` by default.
- Target laptop is Linux with 16 GB system RAM and about 6 GB usable NVIDIA VRAM; minimize model context, unnecessary dependencies, and startup cost.
- Keep this MVP **local**. No cloud API keys, cloud deployment, mandatory Docker, model training, or multi-agent framework required.
- Prefer the existing Ollama native tool-calling implementation for speed. Do not add LangGraph unless genuinely necessary to make the current workflow work.
- Preserve current files, names, database schema, and features where feasible; make surgical fixes rather than a wholesale rewrite.

## Safety and data correctness
- Expose only narrowly scoped, read-only analytical tools. Open DuckDB in read-only mode for agent queries; use parameterized SQL. Do not give an LLM arbitrary SQL execution, shell execution, filesystem write, or eval tools.
- Validate tool arguments, especially dates, pagination/limits, and missing anomalies. Cap tool turns and result sizes.
- Do not allow user- or model-originated text to alter system instructions or execute arbitrary commands.
- Separate **verified database observations**, **plausible hypotheses**, and **unknowns** in the report. Avoid fabricated metrics or unsupported causal claims (e.g., promotions causing an increase).
- Distinguish the initial deterministic evidence lookups from *additional tools selected by the model*. Do not misrepresent seeded lookups as evidence of autonomous tool calling.
- Include informative and actionable errors for absent DB/tables, unavailable Ollama, missing model, model timeout, or invalid tool calls. Do not display a fake success report on failure.
- Do not delete or overwrite raw datasets or generated DB files. Never commit `.venv/`, large CSVs, DBs, downloaded models, credentials, or secrets. Keep unrelated Git changes intact.

## Engineering workflow
1. Inspect repository structure, relevant source files, dependency files, current Git status, and errors before editing.
2. Reproduce the reported problem using available commands; record the actual failure and root cause when possible.
3. Make the smallest coherent set of changes that delivers the end-to-end behavior.
4. Add a fast offline test path with mocks/fake LLM responses and a tiny temporary DuckDB fixture, so core behavior is testable even without Ollama.
5. If the local Ollama service and model are available, execute a live smoke test; if not, report the exact blocker rather than asserting successful live inference.
6. Keep docs and dependency installation instructions synchronized with working code.
7. At the end, report changed files, verified commands/results, remaining limitations, and exact instructions to run the MVP.

## MVP acceptance criteria
- From the repo root, the Streamlit app starts and lists actual stored anomalies (when database is present).
- The default chat can find actual stored anomalies through a model-selected search and investigate a returned result without the user supplying or selecting its date.
- Clicking **Investigate with local AI** runs the backend `investigate(selected_date)` and displays its report and trace, or an actionable error.
- The agent can execute at least one **model-selected** read-only tool and incorporate its results into the report. A pre-populated trace alone does not qualify.
- Investigating `2017-11-24` yields correct metrics (R$149,916.58 item sales, 1,147 distinct delivered orders) and an appropriately cautious discussion of the Black Friday spike.
- The app works without hardcoded absolute machine-specific paths.
- Key tests run without requiring a live Ollama server; actual LLM availability is verified separately.

## Coding conventions
- Prefer standard library and existing dependencies, type hints where helpful, short reusable functions, and explicit error messages.
- Use `pathlib` for repository-relative paths.
- Don't silently swallow exceptions. Don't overengineer architecture or styling before the workflow passes.
- Prefer simple README instructions and one reliable startup command to unnecessary new infrastructure.
