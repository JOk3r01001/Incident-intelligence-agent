"""Model reply types and the single supported LangChain model client.

The small protocol also allows offline tests to inject fake replies while using
the real LangChain agent runtime and analytical tools.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence
from uuid import uuid4


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


def create_model_client(model: str, backend: str = "langchain") -> ModelClient:
    if backend != "langchain":
        raise ModelClientError(
            "INCIDENT_BACKEND only supports 'langchain'. Remove the old backend "
            "override or set INCIDENT_BACKEND=langchain, then restart Streamlit."
        )
    from langchain_client import LangChainOllamaClient
    return LangChainOllamaClient(model)
