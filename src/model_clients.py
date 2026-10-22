"""Small model boundary shared by native Ollama and optional LangChain adapters.

Adapters select tools and return text. Only incident_agent dispatches allowlisted
tools, enforces budgets, and renders verified findings.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence
from uuid import uuid4

import httpx
from ollama import ChatResponse, Client, ResponseError


class ModelClientError(RuntimeError):
    """Actionable model failure, independent of a provider's SDK."""


@dataclass(frozen=True)
class ToolCall:
    """Provider-neutral proposal; the agent validates it before running any tool."""
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class ModelReply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done_reason: str | None = None


class ModelClient(Protocol):
    def invoke(self, messages: list[dict[str, Any]], *, tools: Sequence[Callable],
               timeout: float) -> ModelReply:
        """Honor timeout; never execute tools; report failures as ModelClientError."""
        ...


def create_model_client(model: str, backend: str = "native") -> ModelClient:
    if backend == "native":
        return OllamaModelClient(model)
    if backend == "langchain":
        # Native users need neither the optional packages nor their import overhead.
        from langchain_client import LangChainOllamaClient
        return LangChainOllamaClient(model)
    raise ModelClientError("INCIDENT_BACKEND must be 'native' or 'langchain'.")


class OllamaModelClient:
    """Native Ollama transport; no LangChain dependency or import is needed."""

    def __init__(self, model: str):
        self.model = model

    def invoke(self, messages: list[dict[str, Any]], *, tools: Sequence[Callable],
               timeout: float) -> ModelReply:
        native_messages = []
        # Ollama expects nested function schemas; the shared loop uses a flat shape.
        for message in messages:
            native = {key: value for key, value in message.items() if key != "tool_call_id"}
            if message.get("tool_calls"):
                native["tool_calls"] = [
                    {"function": {"name": call["name"], "arguments": call["arguments"]}}
                    for call in message["tool_calls"]
                ]
            native_messages.append(native)
        try:
            # Ignore proxy environment variables for this local-only transport.
            with Client(timeout=httpx.Timeout(timeout, connect=min(3, timeout)),
                        trust_env=False) as client:
                reply = ChatResponse.model_validate(client.chat(
                    model=self.model, messages=native_messages, tools=list(tools),
                    think=False, stream=False, keep_alive="5m",
                    # Bound context/output for the target laptop's memory and latency.
                    options={"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                             "num_ctx": 4096, "num_predict": 3072},
                ))
        except httpx.TimeoutException as exc:
            raise ModelClientError(
                f"Ollama timed out after at most {timeout:.0f}s. Check 'ollama ps'; "
                "retry after the model loads, or increase INCIDENT_TIMEOUT_SECONDS (maximum 300)."
            ) from exc
        except (ConnectionError, httpx.RequestError) as exc:
            raise ModelClientError(
                "Cannot connect to Ollama. Start 'ollama serve' if it is not running; "
                "check 'ollama list' and OLLAMA_HOST (default http://127.0.0.1:11434)."
            ) from exc
        except ResponseError as exc:
            detail = (f"Model '{self.model}' is unavailable. Run 'ollama pull {self.model}' and 'ollama list'."
                      if exc.status_code == 404 else
                      f"Ollama rejected the request ({exc.status_code}): {exc.error}. "
                      "Check 'ollama ps' and that INCIDENT_MODEL supports native tool calling.")
            raise ModelClientError(detail) from exc
        except (ValueError, TypeError) as exc:
            raise ModelClientError(f"Invalid Ollama response or configuration: {exc}. "
                                   "Retry with a tool-capable model.") from exc
        return ModelReply(
            content=reply.message.content or "",
            tool_calls=[ToolCall(call.function.name, dict(call.function.arguments or {}))
                        for call in reply.message.tool_calls or []],
            done_reason=reply.done_reason,
        )
