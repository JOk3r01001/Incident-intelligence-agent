# Local LangChain backend

Status: implemented and optional. `LangChainOllamaClient` uses LangChain's
`ChatOllama` to communicate with the same local Ollama service as the native
client. The default backend remains `native`.

## Install and launch

From the repository root, with the prepared Olist database and Ollama service:

```bash
source .venv/bin/activate
python -m pip install -r requirements-langchain.txt
ollama list
export INCIDENT_MODEL=qwen3:4b-instruct
export INCIDENT_BACKEND=langchain
python src/incident_agent.py 2017-11-24
python -m streamlit run src/app.py --server.address 127.0.0.1
```

The optional requirements include `requirements-mvp.txt` and `langchain-ollama`.
No top-level `langchain` or `langgraph` package is needed. LangSmith is a transitive
library dependency, but tracing is explicitly disabled during model invocation,
even when another application enabled it in the environment. No API keys or cloud
services are required. Dependencies are imported lazily when the LangChain backend
is invoked; base installations remain usable without them.

Set `INCIDENT_BACKEND=native` to return to native Ollama. Restart Streamlit after
changing environment variables. The UI records backend, model, and elapsed time
for each investigation. An unavailable backend fails explicitly; it does not
silently fall back to the other client.

The default **Ask the agent** workspace supports discovery without date selection.
Ask for unusual days, then “investigate the first one.” The model chooses
`find_anomalies` and analytical tools; each answer has its own trace. The optional
**Explore a date** workspace retains date-specific reports and follow-up chat.

## Shared boundary

`investigate(day, max_rounds=7, *, client=None)` uses the configured backend unless
an explicit `ModelClient` is injected. Both adapters implement
`invoke(messages, *, tools, timeout) -> ModelReply` and never execute tools.

`ask_agent(question, *, history=None, max_rounds=7, client=None)` is the chat-first
entry point. It seeds only dataset coverage/counts, requires a real model-selected
anomaly search in a fresh conversation, and retains the latest successful search
for references such as “the first one.” The search tool has bounded pagination,
date/direction filters, and allowlisted sort choices; all values are parameters.
It queries stored statistical flags and never changes or regenerates the database.
When daily metrics and weekday history are available, Python computes the pooled
historical average order value and its direction relative to the investigated
day, supplying and displaying these observations separately from model prose.

Messages use plain dictionaries:

- `system` and `user`: `role` and `content`.
- `assistant`: `role`, `content`, and optional `tool_calls`, each containing
  `name`, `arguments`, and a unique `id`.
- `tool`: `role`, `content` (JSON evidence), `tool_name`, and `tool_call_id`
  matching the assistant request.

The LangChain adapter translates these to `SystemMessage`, `HumanMessage`,
`AIMessage`, and `ToolMessage`. It binds analytical callables as tool schemas,
then normalizes `AIMessage.tool_calls` into `ToolCall(name, arguments, id)` objects.
Missing IDs are generated; duplicate IDs and malformed calls fail explicitly.
Text content blocks are supported; reasoning blocks are excluded.

`ModelReply` carries `content`, tool calls, and `done_reason`. Truncated generations
use `done_reason="length"` and are rejected by the investigation loop. SDK errors
become actionable `ModelClientError` messages; the loop retains collected evidence
in `InvestigationError.trace`.

`discuss(day, question, *, investigation_trace, history=None, max_rounds=7,
client=None)` uses the same dispatch loop as `investigate`. It refreshes the three
seeded lookups, supplies bounded saved evidence, and reconstructs recent successful
question/answer pairs with application-assigned roles. Historical tool messages
are supplied as evidence objects, avoiding orphaned tool-call IDs. New assistant
proposals and their tool results retain matching IDs through both adapters.
User suggestions and generated text are never promoted to database evidence.
For category comparisons, Python computes sales direction only for categories
actually returned on both dates and supplies these observations to the model.
This separates ranking changes from absolute sales changes and treats omitted
categories as unknown. The answer displays these calculated observations separately.

## Local execution and limits

Both transports use the configured model with a 4,096-token context, at most 3,072
generated tokens per response, reasoning disabled where supported, and a five-minute
model keep-alive. `ChatOllama.invoke()` explicitly receives `stream=False`, and its
HTTP clients use the current request's timeout with proxy environment variables
ignored. The adapter owns and closes its synchronous connection pool.

The existing loop limits total time between rounds, rounds, tool calls, and result
size. It validates dates and arguments before executing only allowlisted,
parameterized Python/DuckDB tools. It continues to render Findings directly from
verified evidence, while the model writes only explanations and next checks.
Chat answers with the wrong headings get one formatting retry within the existing
round/time budget. Empty/truncated responses and persistent bad format fail with
their evidence retained.

There is no arbitrary SQL, shell, filesystem, Python execution, or evaluation
tool. The same deterministic seeds and model-selected-call labels remain in the
audit trail. A report requires a successful model-selected call followed by a
final assessment. LangChain is a model transport here, not an unrestricted agent
executor or a replacement for the safety checks.

Chat can answer from existing evidence without a new tool call. A reply that
attempts tools must obtain a successful result before it can complete. Questions,
history, evidence context, and request size are capped; the session retains at
most 12 exchanges. See [README-MVP.md](../README-MVP.md) for exact limits and reset
behavior. These limits add no new dependencies or model capabilities.

## Verification

```bash
python -m unittest discover -s tests -v
INCIDENT_BACKEND=native INCIDENT_MODEL=qwen3:4b-instruct python tests/live_smoke.py
INCIDENT_BACKEND=langchain INCIDENT_MODEL=qwen3:4b-instruct python tests/live_smoke.py
INCIDENT_BACKEND=native INCIDENT_MODEL=qwen3:4b-instruct python tests/live_discovery.py
INCIDENT_BACKEND=langchain INCIDENT_MODEL=qwen3:4b-instruct python tests/live_discovery.py
python -m pip check
```

Offline tests use temporary DuckDB fixtures and fake model replies. Optional tests
cover multiple and batched calls, request/result IDs, bad arguments, model errors,
invalid responses, tracing isolation, and report/chat persistence in Streamlit.
The live tests use the actual local model and prepared database, require a real
model-selected tool call, checks the known November 24 metrics, tests a category
comparison and a second hypothesis follow-up, and compares the
database hash before and after. AppTest tests the Streamlit Python runtime, not
browser layout. `live_discovery.py` additionally tests the default chat workspace:
model-selected anomaly search, “investigate the first one” without a supplied
date, real daily/history/category calls, a hypothesis follow-up, and persistence.

On October 22, 2026, all 88 offline tests passed (including safe data preparation,
discovery, and conversational follow-ups) with langchain-ollama 1.1.0, langchain-core 1.6.13,
and langsmith 0.14.12 on Python 3.14.8. The complete real Streamlit chat-first
workflow (discovery, reference-based investigation, hypothesis follow-up, and
reruns) completed in 15.4 seconds via LangChain and 12.6 seconds via native Ollama.
Both used qwen3:4b-instruct, selected real analytical tools, retained the discussion,
and left the database unchanged. These timings are not a controlled performance
comparison. Generated prose can still mislabel rankings or overstate evidence;
the accurate Python-rendered evidence and raw tool results remain available for review.

Reference: [LangChain's official ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama).
