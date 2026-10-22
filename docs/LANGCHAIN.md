# LangChain integration preparation

Status: the native Ollama MVP is implemented. The model interface is ready for a
future LangChain adapter; there is no active LangChain implementation or dependency.

## Stable boundary

`investigate(day, max_rounds=7, *, client=None)` accepts an optional object
implementing `ModelClient.invoke(messages, *, tools, timeout) -> ModelReply`.
Omitting it selects `OllamaModelClient(INCIDENT_MODEL)` and retains the native path.

Messages use plain dictionaries:

- `system` and `user`: `role` and `content`.
- `assistant`: `role`, `content`, and optional `tool_calls`, each containing
  `name`, `arguments`, and a unique `id`.
- `tool`: `role`, `content` (JSON evidence), `tool_name`, and `tool_call_id`
  matching the assistant request.

`ModelReply` carries `content`, normalized `ToolCall` objects, and `done_reason`.
Use `done_reason="length"` for truncated generations. Raise `ModelClientError`
with an actionable message for service, model, timeout, or response failures.
Do not put reasoning text or SDK-specific objects into the neutral message history.

The native adapter translates this contract to Ollama messages and strips internal
call IDs that its native API does not need. LangChain can preserve the same IDs in
its `AIMessage` / `ToolMessage` exchange.

## Next implementation

1. Add the optional `langchain-ollama` integration after checking its current
   compatibility with the project's Python and Ollama client versions. Keep the
   existing MVP dependency file lightweight.
2. Implement a `LangChainOllamaClient` using `ChatOllama.bind_tools()` and
   `invoke()`. Translate neutral messages into LangChain message objects; map
   `AIMessage.tool_calls` (`name`, `args`, `id`) back to `ToolCall`.
3. Pass the validated analytical callables as tool definitions. Binding tools
   exposes their schemas; the investigation loop remains responsible for actually
   executing the allowlisted functions and validating all arguments.
4. Honor the supplied per-request timeout and remaining time budget, preserve the
   model/context/token settings, and normalize invalid tool calls and exceptions.
   Do not enable remote tracing or require API keys.
5. Add an explicit backend setting only when the adapter exists. Verify native and
   LangChain implementations with the same offline and live acceptance tests.

The model may request a tool; it must not get a general SQL, shell, filesystem,
Python execution, or evaluation tool. Keep deterministic seeds distinct from
model-selected calls, preserve the audit trail on errors, and retain Python's
verified-findings renderer. Do not replace the bounded loop with an unrestricted
agent executor.

## Acceptance checks

- Native and future LangChain clients produce the same report/trace contract.
- Multiple and batched calls retain matching request/result IDs.
- Invalid tools and arguments never bypass the dispatch allowlist.
- No valid model-selected call means failure, not a seeded success report.
- Missing model/service, timeout, malformed response, and limits remain actionable.
- The Streamlit button still works, and the known Olist metrics remain correct.
- Both offline fixtures and the real local-model test leave user data intact.

Reference: [LangChain's official ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama).
