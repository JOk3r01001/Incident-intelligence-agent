# Incident Intelligence Agent — local MVP

Investigate historical Olist sales anomalies with Streamlit, read-only DuckDB tools,
and a local Ollama model. No cloud API keys, Docker, training, or orchestration
framework is required.

See [README.md](README.md) for the portfolio overview and
[docs/LANGCHAIN.md](docs/LANGCHAIN.md) for the optional, implemented LangChain
backend. Native Ollama remains the default and does not require LangChain.

## Launch on Linux

Run these commands from the repository root. The existing prepared database is
`data/incident_intelligence.duckdb`; launching the app does not rebuild it.

```bash
# Only create the environment if .venv does not already exist:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mvp.txt
python src/prepare_data.py

ollama list
ollama pull qwen3:4b-instruct
ollama ps

export INCIDENT_MODEL=qwen3:4b-instruct
python -m streamlit run src/app.py --server.address 127.0.0.1
```

Open <http://localhost:8501> and use **Ask the agent**, the default workspace.
No anomaly selection is required. The spinner remains visible while Ollama works;
replies and errors retain their audit trails across ordinary widget reruns.

## Discover anomalies by asking

Try this conversation:

1. “Find the three sales spikes with the highest item sales in November 2017.”
2. “Investigate the first anomaly in that list.”
3. “Could a promotion explain that? What evidence would test it?”

The agent chooses `find_anomalies`, which searches the existing detector's stored
flags using read-only, parameterized SQL. It can filter inclusive date ranges,
spikes or drops, sort by absolute percentage deviation, actual item sales, or
newest date, and paginate with at most 10 results per call. The default is the
five largest absolute percentage deviations; this ranking is not business impact.
Each result page reports its total matching count, offset, filters, ranking, and
whether more results are available. Ask “show more” to request another page.

Python displays the returned dates, actual sales, expected sales, and deviations
in an evidence table. The model can then select daily metrics, weekday history,
category, and status tools for a discovered date. “Investigate the first one”
uses the saved search results rather than asking you to choose a date.
When daily metrics and weekday history are available, Python calculates the
historical weighted average order value and displays its comparison separately.
Total sales growth does not imply higher average order value.
This searches precomputed statistical flags; it does not rerun detection, mutate
the database, monitor live sales, or establish that a flag is a business incident.

One dataset-summary lookup is marked `source: seeded`; discovery must include a
successful **model-selected** `find_anomalies` call. A metadata count or generated
list alone does not qualify. Later explanatory replies may reuse saved evidence
with no new model-selected calls. Empty searches return an explicit no-match
result; an empty anomaly table still supports chat. Errors retain the evidence
collected before failure. A malformed chat answer gets at most one formatting
retry within the same round/time limits, reusing collected results.

**New conversation** clears this workspace's history. Model/backend changes start
a fresh conversation. The latest successful search among the 12 retained
exchanges remains available even when its older conversational wording leaves
the recent-message window. The independent **Explore a date** workspace retains the original
dashboard: select a date there and click **Investigate with local AI** when you
want to inspect a known day. No such selection is needed in **Ask the agent**.

## Discuss an investigation

In the optional **Explore a date** workspace, once a report completes, type into
**Discuss this investigation**. For example:

- “Could Black Friday explain this spike?”
- “Compare category mix with the previous Friday.”
- “I suspect a promotion. What evidence would test that?”
- “What should we check next?”

The agent receives the selected anomaly's refreshed database baseline, a bounded
selection of saved tool results, and recent successful exchanges. It can answer
from that evidence or select additional read-only tools, including queries for
other dates. Each reply shows **Answer**, **Unknowns and next checks**, Python-rendered
**Database evidence**, elapsed time, and its own audit trail. Three seeded lookups
remain labeled separately from model-selected calls. An explanatory answer may
have zero new model-selected calls; an initial investigation still requires one.
When category results cover both dates, Python also calculates and displays
same-category sales comparisons. A category's lower rank does not mean its
absolute sales declined; categories missing from a limited result remain unknown.

User suggestions remain hypotheses. Promotions, campaign records, and other
unavailable data cannot be verified by these tools. Model interpretations still
need review against the supplied evidence. Discussion does not automatically
rewrite the initial report.

Each workspace displays at most 12 exchanges. Each model request uses at most the
three most recent successful exchanges within 2,000 characters of conversational
text, plus a selection of evidence within 3,500 characters. Questions are limited
to 1,000 characters. A 12,000-character request guard complements the model's
4,096-token context; it is not an exact token counter. Oversized conversations
fail explicitly with guidance to clear the chat or narrow the question.
The same timeout, round, tool-count, argument-validation, and read-only rules apply.

Failures retain their traces and stay visible; failed exchanges are excluded from
model memory. **Clear conversation** resets discussion only. Switching dates hides
the current investigation's chat; switching back restores it until another
investigation replaces it. Running **Investigate with local AI** again resets both
report and conversation. Changing model/backend requires a new investigation.
Chats live only in the current browser session and are lost when that session ends.
Switching workspaces preserves their independent histories.

If `ollama list` cannot connect, start the service in a separate terminal:

```bash
ollama serve
```

Then rerun `ollama list` and the launch commands. If Ollama is already running,
use that service rather than starting another copy. Install Ollama first if the
command is unavailable: <https://docs.ollama.com/linux>.

## Model configuration and responsiveness

To use LangChain with the same read-only tools and report/trace contract:

```bash
python -m pip install -r requirements-langchain.txt
export INCIDENT_BACKEND=langchain
export INCIDENT_MODEL=qwen3:4b-instruct
python -m streamlit run src/app.py --server.address 127.0.0.1
```

Set `INCIDENT_BACKEND=native` to return to the native client. The optional adapter
uses `ChatOllama.bind_tools()` and converts LangChain messages/tool IDs into the
existing interface. Python still validates and executes every tool. It disables
remote tracing and streaming, honors the same timeouts, and preserves evidence on
errors. Missing optional dependencies produce an installation instruction rather
than a silent backend change.

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

Both backends use Ollama's local tool calling. A date-specific final report requires
at least one successful **model-selected** tool call. Its three initial lookups are
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
| `INCIDENT_BACKEND` | `native` | `native` or `langchain`; restart Streamlit after changing |
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
For example, the latest live responses referred to "typical Black Friday trends"
without historical category comparisons. That interpretation is not verified by
the tool results; the numerical Findings remain rendered directly from evidence.
During chat verification Qwen also occasionally mislabeled the categories used
for a same-category comparison as the selected date's top three. Python's evidence
and comparison values remain accurate; model summaries are explicitly labeled as
interpretations. The live test verifies tools, evidence, and UI behavior, not all
semantic claims in generated prose.

### Prepare data safely

```bash
python src/prepare_data.py
```

If the database already exists, this command checks its four required tables and
prints row counts using a read-only connection. It leaves the file unchanged and
does not require CSVs. If it is absent, the command imports these files from
`data/`: `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`, and
`olist_products_dataset.csv`. It preserves all source columns and applies the
original detector: median of the previous four observed matching weekdays, at
least 30% absolute deviation. Missing sales days are not filled with zeros.

The import builds a temporary database and publishes it only after completion.
An existing destination is never replaced, including one created concurrently.
An incomplete existing database produces an error; restore a prepared copy or
build a separate database:

```bash
python src/prepare_data.py --data-dir data --db data/rebuilt.duckdb
```

The app continues to read `data/incident_intelligence.duckdb`; `--db` selects only
the preparation destination. To use a rebuilt copy, stop the app, move the old
database to a backup path, and move the prepared replacement to the app's default
path. Keep the backup. Preparation never runs automatically at app startup.
Raw CSVs are not needed once the database is prepared.

The superseded `explore_data.py`, `detect_anomalies.py`, and `analytics.py` scripts
were removed. Their required functionality now lives in `prepare_data.py` and
the read-only `agent_tools.py`. Earlier versions remain available in Git history.

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
Chat tests cover cross-date tool calls, bounded history, evidence reuse, failed
reply recovery, date isolation, clearing, session limits, and both model adapters.
Discovery tests additionally cover default chat access without a selector,
model-selected searches, reference-based investigations, date/direction filters,
sorting, pagination, no matches, SQL injection rejection, preserved searches,
bounded formatting repair, and unchanged database contents.
With the optional requirements installed, the suite also tests real LangChain
message types with fake model responses, batched calls and tool-call IDs, invalid
calls, tracing isolation, and the LangChain-selected Streamlit button. These tests
skip if optional packages are absent; the native tests still run.
Preparation tests also check fresh CSV imports, unchanged existing databases,
missing files, import failure cleanup, concurrent destination creation, empty
anomaly results, and the original detector's weekday baselines and thresholds.

The separate opt-in tests use the actual prepared Olist database and actual
Ollama model, without mocking the LLM:

```bash
INCIDENT_BACKEND=native INCIDENT_MODEL=qwen3:4b-instruct python tests/live_smoke.py
INCIDENT_BACKEND=langchain INCIDENT_MODEL=qwen3:4b-instruct python tests/live_smoke.py
INCIDENT_BACKEND=native INCIDENT_MODEL=qwen3:4b-instruct python tests/live_discovery.py
INCIDENT_BACKEND=langchain INCIDENT_MODEL=qwen3:4b-instruct python tests/live_discovery.py
```

`live_smoke.py` drives the Streamlit button using Streamlit's AppTest harness, requires a
successful model-selected tool and final report, checks the known November 24
metrics and report headings, asks for a cross-date category comparison requiring
new model-selected evidence, asks a second hypothesis follow-up, reruns the app
to check report/chat persistence, and compares
the database SHA256 before and after. It prints the actual report, trace, and
elapsed time, or fails with the observed error. AppTest exercises Streamlit's
Python runtime; it does not test browser layout. `live_discovery.py` tests the
default chat workspace: it asks for November's top three spikes by actual sales,
checks the model-selected search and table, asks to investigate “the first anomaly”
without supplying its date, verifies real daily/history/category calls and known
metrics, asks a hypothesis follow-up, and checks rerun persistence and DB hash.

The optional CLI still supports a known-date smoke test:

```bash
python src/incident_agent.py 2017-11-24
```

Recorded checks on October 22, 2026: **88 offline tests passed** in about 11
seconds with optional dependencies installed, including native and LangChain
adapters, matching tool-call IDs, allowlist enforcement, error recovery, and
tracing isolation. The new chat tests include a category falling in rank while
its absolute sales increase, checking that Python supplies the correct comparison.
Discovery tests also verify the calculated weighted AOV direction, rather than
inferring average order value from total sales or the detector's sales baseline.
A fresh import of the three real Olist CSVs into a temporary
database reproduced all 221 stored anomalies (dates and numerical values within
0.000001) and the known November 24 metrics. The existing database was checked
read-only and its SHA256 remained unchanged. The complete live chat-first
Streamlit/AppTest workflow passed with `qwen3:4b-instruct` using **LangChain (15.4
seconds total)** and **native Ollama (12.6 seconds total)**. Each discovered the
November top-three spikes through a model-selected search, investigated the first
result without date selection, fetched daily/history/category evidence, and
answered a hypothesis follow-up. Both retained chat on rerun and left the
database hash unchanged. The existing date-explorer/report/comparison/chat live
regression also passed through native Ollama (11.7 seconds); earlier runs passed
through LangChain. These are individual measurements, not a controlled comparison
or latency guarantee. Model prose still sometimes invents typical promotional
behavior or suggests checks the available tools cannot establish; tests verify
the tool workflow and numerical evidence, not every semantic claim.
Earlier report-only and CLI investigations also passed;
Streamlit started on localhost and its `/_stcore/health` endpoint returned `ok`.
`python -m pip check`, compilation, and `git diff --check` also passed.

Verified environment: Python 3.14.8, DuckDB 1.5.6, ollama Python client 0.6.3,
Streamlit 1.66.1, pandas 2.3.3, httpx 0.28.1; Ollama server 0.33.2 with CLI 0.40.8.
Optional integration: langchain-ollama 1.1.0, langchain-core 1.6.13, langsmith 0.14.12.
The detector calculation, raw CSVs, and database schema are preserved; duplicate
original scripts were replaced with the safe setup entry point described above.
No model files, databases, datasets, credentials, virtual environments, or caches
belong in Git; `.gitignore` excludes them.
