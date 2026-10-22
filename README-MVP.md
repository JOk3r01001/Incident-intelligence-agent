# Incident Intelligence Agent — local MVP

Investigate historical Olist sales anomalies with Streamlit, read-only DuckDB tools,
and a local Ollama model. No cloud API keys, Docker, training, or orchestration
framework is required.

See [README.md](README.md) for the portfolio overview and
[docs/LANGCHAIN.md](docs/LANGCHAIN.md) for the prepared model-adapter contract and
migration plan. The current runtime remains native Ollama; LangChain is not yet
installed or active.

## Launch on Linux

Run these commands from the repository root. The existing prepared database is
`data/incident_intelligence.duckdb`; launching the app does not rebuild it.

```bash
# Only create the environment if .venv does not already exist:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mvp.txt

ollama list
ollama pull qwen3:4b-instruct
ollama ps

export INCIDENT_MODEL=qwen3:4b-instruct
python src/incident_agent.py 2017-11-24
python -m streamlit run src/app.py --server.address 127.0.0.1
```

Open <http://localhost:8501>, select `2017-11-24`, and click
**Investigate with local AI**. The spinner remains visible while Ollama works.
Completed reports display the elapsed time and model used. Completed reports and
failed investigations retain their audit trails across
ordinary widget reruns in the same browser session. Results are shown for their
selected date and are not saved to disk.

If `ollama list` cannot connect, start the service in a separate terminal:

```bash
ollama serve
```

Then rerun `ollama list` and the launch commands. If Ollama is already running,
use that service rather than starting another copy. Install Ollama first if the
command is unavailable: <https://docs.ollama.com/linux>.

## Model configuration and responsiveness

The launch commands explicitly select the tested **`qwen3:4b-instruct`** variant.
Keep `INCIDENT_MODEL` exported in the terminal that starts Streamlit; if Streamlit
was already running with the old model, stop it with Ctrl+C and relaunch with that
environment variable. With no environment override the original repository default
remains `qwen3:4b`.

Model tags can change: on the tested machine `qwen3:4b`
tag resolved to **Qwen3-4B-Thinking-2507**, whose mandatory thinking ignores
`think=False` and `/no_think`. Its template also opens a thinking block
unconditionally. This explained the original long waits and leaked deliberation.
The repaired agent first requests a focused tool call, then supplies the full
baseline evidence for the assessment, allows up to 3,072 generated tokens per
request, and removes reasoning from the displayed report and subsequent messages.
The model now generates only Possible explanations and Confidence and next checks
(requested under 120 words); Python supplies Findings. This avoids asking the model
to generate numerical findings that would then be discarded.

The workflow remains native Ollama tool calling. A final report requires at least
one successful **model-selected** tool call. Three initial database lookups are
explicitly marked `source: seeded`; additional calls use `source: model` and
include arguments, returned evidence, status, and round number. Invalid calls are
returned to the model for correction, and do not count as successful tool use.
Empty reports, truncated responses, missing models, timeouts, and exhausted
budgets produce failures with the evidence collected so far.

The instruct variant passed the actual model-selected tool-call and Streamlit
report/rerun test in **6.1 seconds**, including a cold model load, and **6.3 seconds**
in the final repeat after the prompt/UI refinements. The earlier
thinking-model run with the same shorter assessment prompt took **42.5 seconds**.
These are individual local measurements, not a controlled benchmark or latency
guarantee. Both model tags remain installed; no alias or weights were overwritten.

Launch in one command after activating `.venv`:

```bash
INCIDENT_MODEL=qwen3:4b-instruct python -m streamlit run src/app.py --server.address 127.0.0.1
```

Environment variables are read by the application process; restart Streamlit after
changing them:

| Variable | Default | Meaning |
| --- | --- | --- |
| `INCIDENT_MODEL` | `qwen3:4b` | Installed Ollama model tag |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Address of your local Ollama service |
| `INCIDENT_TIMEOUT_SECONDS` | `90` | Request timeout, from 1 to 300 seconds |

The investigation checks an overall time budget of twice the request timeout
between rounds and limits each new request to the remaining budget. It permits
at most seven model rounds by default, eight model tool calls, 8,000 characters
per tool result, and a 4,096-token context. Connect timeout is three seconds;
proxy environment variables are ignored for local Ollama connections. A cold
model load can take longer than a warm investigation.

## Data and interpretation

Required tables: `orders`, `order_items`, `products`, `detected_anomalies`. The
application resolves the database relative to its source directory, independently
of the shell's working directory. Queries are fixed, parameterized, and open
DuckDB in read-only mode. Dates and limits are strictly validated. The model has
no arbitrary SQL, shell, file-writing, or evaluation tool.

The existing database contains 221 anomaly dates. For `2017-11-24`, delivered
orders grouped by purchase date yield **R$149,916.58 item sales**, **1,147 distinct
orders**, **1,345 items**, and **R$130.70 average order value**. The detector's
expected sales are R$21,581.32 (median of four prior matching weekdays), giving
+594.66%. Prior matching weekdays averaged 153 orders; their weighted average
order value was R$148.14. Expected sales divided by current orders is not a valid
historical order-value baseline.

Python renders the Findings section directly from seeded and model-selected tool
results, including the detector baseline and additional category/status evidence.
The model writes Possible explanations and Confidence and next checks after
receiving the actual results. Its proposed Findings are discarded: live testing
reproduced invented percentages even when the underlying counts were correct.
The report separates database observations, plausible explanations, and unknowns.
Black Friday is supplied as calendar context for November 24, 2017, not database
proof of a promotion. Item sales exclude freight and are neither recognized
accounting revenue nor profit. Statistical anomalies are not automatically
business incidents. Model prose can still be wrong; inspect the trace, especially
causal claims and qualitative comparisons.

**Preserve your data:** do not run the existing exploratory importer
`src/explore_data.py` against a prepared database to troubleshoot Ollama. That
original script replaces source tables and expects anomaly records later in its
exploration. Restore/copy a prepared database for a new checkout. If source tables
are intact and only `detected_anomalies` is missing, the existing
`python src/detect_anomalies.py` command creates it (and replaces existing anomaly
records if present). Neither setup script is run automatically by the MVP.
Raw CSVs may live directly in `data/`. They are not needed at application startup.

## Verification

Fast offline tests require no Ollama service and only create temporary databases:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m pip check
```

Tests cover actual analytical SQL, distinct delivered-order counting, tool argument
validation, read-only enforcement, missing database/tables/anomalies, multiple
model-selected tool steps and batched calls, error recovery, response handling,
timeouts, budgets, and the real Streamlit button with mocked Ollama responses.
They verify report/error rendering, truthful trace labels, and session persistence.

The separate opt-in test uses the actual prepared Olist database and actual
Ollama model, without mocking the LLM:

```bash
INCIDENT_MODEL=qwen3:4b-instruct python tests/live_smoke.py
```

It drives the Streamlit button using Streamlit's AppTest harness, requires a
successful model-selected tool and final report, checks the known November 24
metrics and report headings, reruns the app to check persistence, and compares
the database SHA256 before and after. It prints the actual report, trace, and
elapsed time, or fails with the observed error. AppTest exercises Streamlit's
Python runtime; it does not test browser layout.

Recorded checks on October 22, 2026: **26 offline tests passed** in about 2.5
seconds with `INCIDENT_MODEL=qwen3:4b-instruct`, including injected model adapters,
matching tool-call IDs, and allowlist enforcement across the adapter boundary.
The instruct-model Streamlit
integration passed twice (**6.1 and 6.3 seconds**), executing a model-selected
category tool, rendering a report, preserving it on rerun, and leaving the database
SHA256 unchanged. The adapter refactor also passed a real Streamlit/model test in
**6.3 seconds** with the same unchanged-database check. The shorter-assessment thinking-model integration previously
passed in **42.5 seconds** with a model-selected status tool. The previous longer
prompt took 55.7 seconds in its last recorded thinking-model run; these are
individual measurements, not a controlled benchmark. The CLI investigation passed;
Streamlit started on localhost and its `/_stcore/health` endpoint returned `ok`.
`python -m pip check`, compilation, and `git diff --check` also passed.

Verified environment: Python 3.14.8, DuckDB 1.5.6, ollama Python client 0.6.3,
Streamlit 1.66.1, pandas 2.3.3, httpx 0.28.1; Ollama server 0.33.2 with CLI 0.40.8.
The original analytics, detector, raw CSVs, and database schema are preserved.
No model files, databases, datasets, credentials, virtual environments, or caches
belong in Git; `.gitignore` excludes them.
