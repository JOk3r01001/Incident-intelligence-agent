# LangChain agent runtime

LangChain is the default and only supported backend. Its `create_agent` runtime
orchestrates the model/tool cycle. `ChatOllama` communicates with Qwen hosted by
your local Ollama service. The previous custom Ollama loop and native client
are removed.

## Install and launch

From the repository root, with Ollama installed and the Olist data available:

```bash
source .venv/bin/activate
python -m pip install -r requirements-mvp.txt
python src/prepare_data.py
ollama list
export INCIDENT_MODEL=qwen3:4b-instruct
export INCIDENT_BACKEND=langchain
python src/incident_agent.py 2017-11-24
python -m streamlit run src/app.py --server.address 127.0.0.1
```

If needed, start `ollama serve` in another terminal and pull the model with
`ollama pull qwen3:4b-instruct`. `INCIDENT_BACKEND` defaults to `langchain`;
remove any old `native` override or replace it with `langchain`. Restart Streamlit
after changing environment variables. No native fallback remains.

The main requirements include `langchain` and `langchain-ollama`.
`requirements-langchain.txt` remains a compatibility alias. LangChain installs
LangGraph as an internal runtime dependency; no graph server, cloud service, or
API key is required. LangSmith is a transitive library dependency; tracing is
explicitly disabled around the entire agent run, including model and tool calls.

## Responsibilities

| Component | Responsibility |
| --- | --- |
| `src/incident_agent.py` | Prompts, seeded evidence, bounded conversational context, Python-rendered numerical findings |
| `src/langchain_runtime.py` | `create_agent`, LangChain model boundary, and per-invocation policy middleware |
| `src/langchain_client.py` | Single-request ChatOllama transport, linked message conversion, model errors, connection cleanup |
| `src/model_clients.py` | Reply types, injectable test-client contract, and LangChain-only client selection |
| `src/agent_tools.py` | Fixed read-only, parameterized DuckDB analytics and graph data |

`investigate(day, max_rounds=7, *, client=None)` prepares three deterministic
lookups, then starts the LangChain agent. Its report requires a successful
model-selected tool call and a validated final assessment. Python supplies
numerical Findings; the model supplies possible explanations and next checks.

`ask_agent(question, *, history=None, max_rounds=7, client=None)` supplies dataset
coverage and bounded saved evidence. Fresh discovery requires a model-selected
`find_anomalies` call; graph requests require `create_graph` when their dates are
known. A graph request without dates can instead ask for clarification.

`discuss(day, question, *, investigation_trace, history=None, max_rounds=7,
client=None)` refreshes the selected date's evidence and supports follow-up
comparisons, hypotheses, and charts. Explanatory replies can reuse saved evidence
without another tool call. Seeded lookups and model-selected calls remain labeled
separately in every trace.

The `ModelClient` contract performs one request, not a loop. In production it
uses ChatOllama; tests can inject fake replies through the same boundary while
exercising the real LangChain agent runtime and real temporary-database tools.
LangChain owns routing from model proposals to its tool node and back to the
model. There is no application `for`/`while` loop repeatedly calling Ollama.

## Middleware and execution boundaries

Each invocation creates its own policy instance and transient LangChain state.
There is no persisted checkpoint or remote store.

- `before_model` enforces the model-round limit and adds newly computed evidence,
  comparison directions, or the current graph request after completed tool batches.
- `wrap_model_call` validates context size, enforces remaining time, supplies the
  appropriate system prompt, and preserves model failures with their traces.
- `after_model` rejects truncated, empty, or malformed final assessments, enforces
  required evidence, and requests bounded formatting or missing-tool corrections.
- `wrap_tool_call` validates tool names and original arguments against the fixed
  allowlist, executes the analytical function, checks result size, and returns a
  linked `ToolMessage`. Original values are checked before schema coercion can
  weaken strict date/integer validation. Unknown tools return structured errors.

Tool execution is sequential (`max_concurrency=1`), keeping call budgets and
trace order deterministic for batched proposals. Fatal tool/time limits remain
fatal even if the runtime converts a tool exception into an error message.
Duplicate tool-call IDs are rejected. Invalid calls stay in the trace and do not
qualify as successful evidence. Only read-only, parameterized queries are exposed;
there is no arbitrary SQL, shell, filesystem-write, or evaluation tool.

The model uses a 4,096-token context, at most 3,072 generated tokens per response,
and a five-minute keep-alive. Reasoning and streaming are disabled where supported.
Each synchronous ChatOllama connection uses the current request timeout, ignores
proxy environment variables, and is closed after the request. The default limits
remain seven model rounds, eight tool attempts, and 8,000 characters per result.
See [README.md](../README.md#configuration-and-session-limits) for all limits.

Graph traces retain full observations for Streamlit. The model receives metadata,
min/max, and a labeled five-point preview. Python renders charts and calculates
same-category and weighted average-order-value comparisons from actual results.
Model interpretations remain hypotheses requiring review against this evidence.

## Verification

The development test suite is kept locally and excluded from the public repository.
It exercises the actual LangChain runtime with fake model replies and temporary
DuckDB fixtures, including multi-step/batched calls, invalid arguments, missing
models, timeouts, budgets, report validation, reasoning removal, chart payloads,
conversation memory, and Streamlit rerun persistence.

Separate opt-in local-model tests cover date-specific reports and follow-ups,
chat-first discovery without anomaly selection, and reference-based chart requests.
They require the real prepared Olist database and local Qwen model, inspect actual
tool results and UI state, and compare database hashes before and after.
Streamlit AppTest checks the Python runtime and chart payloads, not browser pixels.
A passing workflow does not certify every causal or qualitative claim in model prose.

Verified on October 22, 2026 with LangChain 1.4.7, langchain-ollama 1.1.0, and
`qwen3:4b-instruct`: **120 offline tests passed**. Separate real-model Streamlit
tests passed for the date-specific report/comparison/follow-up workflow (16.4
seconds), chat-first discovery/investigation/follow-up (14.1 seconds), and three
reference-based graph requests (12.3 seconds). November 24 observations matched
R$149,916.58 item sales and 1,147 distinct delivered orders. Each live test
confirmed an unchanged database SHA256. These are individual local measurements,
not latency guarantees. Dependency consistency and Python compilation also passed.

For a fresh installation, use the CLI command in the launch instructions and
`python -m pip check`. Neither requires the private test files.

Reference: [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents)
and [custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom).
