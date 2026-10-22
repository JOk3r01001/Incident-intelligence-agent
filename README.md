# Incident Intelligence Agent

Incident Intelligence Agent is a local application for discovering and
investigating unusual sales patterns in historical Brazilian Olist e-commerce
data. Users can ask questions, compare dates, explore product categories, and
request charts through a conversational Streamlit interface.

A LangChain agent uses a local Ollama-hosted Qwen model to choose from a defined
set of analytical tools.
Python queries DuckDB, calculates numerical findings, and returns the evidence
for the model's assessment. Reports distinguish database observations from
hypotheses and include a tool-call log showing how the evidence was collected.
Data processing and model inference run locally, without cloud API keys.

**Status: MVP under active development.** The current release focuses on the
complete discovery, investigation, and follow-up workflow. Features, interfaces,
and implementation details may change as the application develops.
The longer-term goal is a business intelligence agent that works across configured
business datasets through controlled SQL tools. See [Future direction](#future-direction).

## Current capabilities

- Ask the agent to discover stored anomalies, filter/rank them, and investigate
  a result without selecting a date manually.
- Collect deterministic baseline evidence, then execute additional tools selected
  by the model, with the two sources clearly labeled.
- Render numeric findings from database results and ask the model for a short,
  cautious assessment of explanations and next checks.
- Retain the report, tool results, model name, and elapsed time across UI reruns.
- Discuss the investigation in chat: test a hypothesis, compare dates, and inspect
  each reply's additional tool calls and database evidence.
- Request database-backed graphs of daily sales/orders/AOV, category sales, or
  order statuses; keep them with the reply across reruns.
- Report unavailable models, timeouts, invalid calls, and missing data explicitly.

## Run locally

From the repository root, with a prepared Olist database or the three source CSVs
in `data/`, Python 3, and Ollama installed:

```bash
# Create the environment if it does not already exist.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mvp.txt
python src/prepare_data.py
ollama pull qwen3:4b-instruct
export INCIDENT_MODEL=qwen3:4b-instruct
export INCIDENT_BACKEND=langchain
python -m streamlit run src/app.py --server.address 127.0.0.1
```

Ollama must be running before pulling or using the model. If its service is not
running, start `ollama serve` in another terminal. Data preparation checks an
existing database read-only; it imports CSVs and calculates anomaly flags only
when the destination database is absent.

Open <http://localhost:8501>. **Ask the agent** is the default workspace. Try:

1. “Find the three biggest sales spikes by item sales in November 2017.”
2. “Investigate the first one.”
3. “Could a promotion explain that? What evidence would test it?”

For graphs, ask “Plot daily sales for November 2017,” then “Graph the top five
categories on the day with the highest sales.” The model selects `create_graph`;
Streamlit plots actual query results. Graph requests also work as the first message.
Open **About the agent and its tools** in the app for a quick explanation.

The model selects the read-only `find_anomalies` tool, then analytical tools for
the dates it discovers. Python renders the actual search results and numerical
evidence; each reply includes a tool audit trail. **New conversation** clears the
discussion. The agent searches the existing statistical detector's stored flags;
it does not regenerate the database or run live monitoring.

**Explore a date** retains the optional dashboard and report workflow. Known
observations for `2017-11-24` are **R$149,916.58 item sales** and **1,147 distinct
delivered orders**. Questions can use saved evidence or trigger more read-only
tool calls; no additional call is claimed when the agent reuses context.

Raw datasets, the generated database, and model weights are intentionally excluded
from Git.
The [complete agent guide](docs/AGENT_GUIDE.md) explains the detector, every tool,
graph options, model/runtime responsibilities, memory, safety limits, and testing.

## Prepare the data

The application reads `data/incident_intelligence.duckdb`, resolved relative to
the repository. It requires the tables `orders`, `order_items`, `products`, and
`detected_anomalies`.

```bash
python src/prepare_data.py
```

For an existing database, this command checks the required tables and prints row
counts through a read-only connection. CSVs are not required in that case, and
the database remains unchanged.

For a new database, place these source files directly in `data/`:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_products_dataset.csv`

Preparation imports the source columns and calculates anomaly flags in a temporary
database before publishing the completed file. It never replaces an existing
destination. Missing sales days are not filled with zeros.

If an existing database is incomplete, restore a prepared copy or build a separate
one without overwriting it:

```bash
python src/prepare_data.py --data-dir data --db data/rebuilt.duckdb
```

The `--db` option changes only the preparation destination. To use the rebuilt
copy, stop the application, move the original database to a backup path, then
move the completed replacement to `data/incident_intelligence.duckdb`. Keep the
backup. Application startup does not run preparation automatically.

## LangChain runtime

LangChain is the only supported backend and is installed with
`requirements-mvp.txt`. Its `create_agent` runtime manages model calls, tool
routing, and continuation until a validated answer is ready. Application
middleware enforces argument validation, budgets, evidence requirements, and the
tool-call audit trail. The former custom Ollama loop and direct native client
have been removed.

`ChatOllama` connects LangChain to the local Ollama service that hosts Qwen.
Ollama is still required for model inference. LangChain's agent runtime includes
LangGraph as an internal dependency; no separate graph service or cloud API is
needed. LangSmith tracing is disabled around the complete agent run.
`requirements-langchain.txt` remains a compatibility alias for the main
requirements file. An old `INCIDENT_BACKEND=native` override produces an
instruction to remove it or set it to `langchain`; there is no native fallback.

## Configuration and session limits

Set environment variables in the terminal that launches Streamlit. Restart the
application after changing them.

| Variable | Default | Purpose |
| --- | --- | --- |
| `INCIDENT_MODEL` | `qwen3:4b` | Installed Ollama model tag; the launch instructions select the tested `qwen3:4b-instruct` variant |
| `INCIDENT_BACKEND` | `langchain` | Only supported backend; an override is unnecessary unless replacing an old setting |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Local Ollama service address |
| `INCIDENT_TIMEOUT_SECONDS` | `90` | Timeout per model request, from 1 to 300 seconds |

The LangChain runtime permits seven model rounds by default and at most eight model tool
attempts, with an 8,000-character limit per tool result and a 4,096-token model
context. It checks an overall budget of twice the request timeout between rounds
and limits new requests to the remaining time. Cold model loading may take longer
than a warm request.

Each workspace retains at most 12 exchanges in the current session. Model requests
use up to three recent successful exchanges within 2,000 characters, plus bounded
evidence within 3,500 characters. Questions are limited to 1,000 characters; a
12,000-character request guard complements the model context limit.

**New conversation** clears the main chat. In **Explore a date**, **Clear
conversation** clears follow-ups; running a new investigation resets both its
report and discussion. Model/backend changes require a fresh conversation or
investigation. Switching workspaces preserves their separate session histories.

## Architecture

```mermaid
flowchart LR
    UI[Streamlit] --> Agent[LangChain create_agent]
    Agent --> LC[ChatOllama]
    LC --> Ollama
    Agent --> Policy[Validation and evidence middleware]
    Policy --> Tools[Validated analytical tools]
    Tools --> DB[(Read-only DuckDB)]
    Agent --> Report[Verified findings + model assessment + audit trail]
    Report --> UI
```

`src/langchain_runtime.py` builds the LangChain agent and applies the per-run
validation, timeout, tool-count, and report-completion policies.
`src/langchain_client.py` handles ChatOllama requests and actionable model errors;
`src/model_clients.py` defines the reply contract and selects the single supported
client. The contract also allows offline tests to inject fake replies into the
real LangChain runtime. `src/incident_agent.py` prepares bounded context and
renders numerical findings from the database evidence.
`ask_agent()` handles chat-first discovery; `find_anomalies()` supports date ranges,
spikes/drops, sorting, and pagination. Percentage deviation is the default ranking;
asking for highest item sales selects the sales ranking instead.

`src/prepare_data.py` imports the three Olist CSVs and calculates anomaly flags
for a new database. The detector compares daily item sales against the median of
the previous four matching weekdays and flags absolute deviations of at least
30%. Existing databases are checked read-only. `src/agent_tools.py` provides the
fixed analytical queries used during investigations; the model has no arbitrary
SQL, shell, or filesystem-write tool.

Setup, the message contract, and implementation details are in
[docs/LANGCHAIN.md](docs/LANGCHAIN.md).

## Verify the installation

```bash
python -m pip check
python src/incident_agent.py 2017-11-24
```

Run these commands from the activated environment with the prepared database and
Ollama available. The CLI runs an investigation for the supplied date and prints
the report and tool-call trace. With the original Olist data, the November 24
check should show the item sales and delivered-order count listed above.
The separate development test suite is kept locally and is not distributed in
this repository.

## Troubleshooting

- **Ollama is unavailable:** run `ollama list` to check connectivity. Start
  `ollama serve` in another terminal if the service is stopped. Ollama installation
  instructions are at <https://docs.ollama.com/linux>.
- **The model is missing:** run `ollama pull qwen3:4b-instruct`, export
  `INCIDENT_MODEL=qwen3:4b-instruct`, and restart Streamlit.
- **Replies time out or expose reasoning:** use the instruct variant selected by
  the launch instructions. The tested `qwen3:4b` tag used a thinking variant that
  caused long waits. Check `ollama ps`; if needed, increase
  `INCIDENT_TIMEOUT_SECONDS` within its supported range.
- **The database or tables are missing:** follow the data preparation instructions
  above. An incomplete existing database must be restored or rebuilt separately.
- **No anomalies are listed:** inspect the row counts from `prepare_data.py`.
  A valid dataset can have no deviations above the detector threshold; the chat
  can still query daily metrics and create graphs.
- **An investigation fails:** inspect its retained tool-call trace. Invalid tool
  arguments, exhausted budgets, and model failures produce explicit errors with
  any evidence already collected.

## MVP scope and limitations

The current implementation is specific to Olist. Its importer expects three named
CSV files; analytical tools assume particular tables, columns, joins, delivered
order semantics, and Brazilian real values. Prompts and the anomaly detector also
assume an e-commerce sales investigation. Replacing the data files with an
unrelated dataset will not make the agent understand or query it correctly.
Changing the model or using LangChain does not remove these assumptions.

The application investigates historical data and searches precomputed anomaly
flags. It does not provide live monitoring, automated incident response, or
confirmed root-cause analysis. Graphs support daily sales, orders, average order
value, category sales, and order statuses over ranges of up to 90 days.
Conversation history is limited to the current Streamlit session.

Item sales exclude freight and are neither recognized accounting revenue nor
profit. A statistical anomaly is not automatically a business incident. Black
Friday is calendar context, not proof of a promotion. Model-written hypotheses
still require review against the evidence.

## Future direction

### Intended product

The goal is a local-first business intelligence agent that helps people explore
their business data, investigate changes, and collaborate on follow-up questions.
Anomaly investigation would become one capability alongside metric exploration,
period comparisons, segment analysis, and chart creation.

A user would connect or import a dataset, review its structure, confirm the
business meaning of its metrics, and ask questions in plain language. The agent
would select controlled SQL-backed tools and return an explanation, supporting
tables or charts, and a trace of the queries and definitions used. When the data
cannot answer a question, it would explain what is missing or ask for clarification.

Supporting different datasets requires explicit business definitions. A column
named `amount` could represent item sales, payments, costs, or refunds; schema
inspection alone cannot establish its meaning. The intended product would make
those definitions visible and reusable rather than silently guessing them.

### Proposed development path

These are planned capabilities, not features available in the MVP. Priorities
may change as additional datasets expose new requirements.

| Stage | Planned capability | Outcome |
| --- | --- | --- |
| 1. Dataset onboarding | Inspect local CSV, Parquet, or DuckDB sources; summarize columns, types, date coverage, missing values, and candidate relationships | Understand what data is available and surface quality problems before analysis |
| 2. Business definitions | Configure approved tables and joins, metric formulas, record grain, date fields, dimensions, units, currencies, and status filters | Define what each measure means and prevent duplicate counting or invalid comparisons |
| 3. Reusable SQL tools | Build validated aggregation, filtering, grouping, time-series, and comparison tools against the approved dataset configuration | Answer questions across configured datasets without rewriting the investigation loop |
| 4. Broader investigations | Configure metrics and anomaly baselines; compare periods and segments while checking data completeness | Investigate changes beyond Olist daily sales, with assumptions and uncertainty visible |
| 5. Reusable analysis | Save dataset configurations and explicitly saved investigations; export reports, tables, and charts | Revisit an analysis with its definitions, evidence, and follow-up questions |

Additional database connectors and scheduled checks could follow once the local
workflow works reliably across multiple datasets. They are outside the current MVP.

### SQL tool boundary

The intended design keeps query execution under application control. The agent
would request a structured analysis using approved metrics, dimensions, filters,
and date ranges. Python would validate the request and construct parameterized,
read-only SQL from approved definitions, enforcing row limits and query timeouts.
The model would not receive an unrestricted SQL execution tool.

Business definitions and source identifiers would be validated separately from
filter values. Query results, metric definitions, and data coverage would remain
available in the evidence trace, while the model would explain results and propose
hypotheses. This preserves the current separation between measured observations
and generated interpretation as the range of questions expands.

### Next milestone

The next milestone is to introduce a dataset configuration layer, represent Olist
through that configuration, and validate the same analytical workflow on a second
dataset with different table and column names. That dataset should include a
different metric and record structure so the check goes beyond renaming columns.
Success means changing the dataset configuration rather than the core tools or
prompts for each source, while retaining correct totals, comparisons, charts,
and transparent evidence. A small set of independently calculated reference
answers should verify each dataset before broader onboarding is introduced.
