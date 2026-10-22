"""Local ChatOllama transport used by the LangChain agent runtime."""
from typing import Any, Callable, Sequence
from uuid import uuid4

import httpx
from ollama import ResponseError

from model_clients import ModelClientError, ModelReply, ToolCall


class LangChainOllamaClient:
    def __init__(self, model: str):
        self.model = model

    def invoke(self, messages: list[dict[str, Any]], *, tools: Sequence[Callable],
               timeout: float) -> ModelReply:
        # Keep an actionable error if the environment has not been upgraded.
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
            from langchain_ollama import ChatOllama
            from langsmith import tracing_context
        except ImportError as exc:
            raise ModelClientError(
                "LangChain is required. Activate .venv and run "
                "'python -m pip install -r requirements-mvp.txt'."
            ) from exc

        try:
            converted = []
            # Preserve call IDs so each ToolMessage matches its assistant proposal,
            # including when the model requests multiple tools in one response.
            for message in messages:
                role, content = message["role"], message["content"]
                if role == "system":
                    converted.append(SystemMessage(content=content))
                elif role == "user":
                    converted.append(HumanMessage(content=content))
                elif role == "assistant":
                    converted.append(AIMessage(content=content, tool_calls=[
                        {"name": call["name"], "args": call["arguments"],
                         "id": call["id"], "type": "tool_call"}
                        for call in message.get("tool_calls", [])
                    ]))
                elif role == "tool":
                    converted.append(ToolMessage(content=content, name=message["tool_name"],
                                                 tool_call_id=message["tool_call_id"]))
                else:
                    raise ValueError(f"Unsupported message role: {role}")

            # Own the synchronous connection pool; ChatOllama has no public close().
            # Disable tracing even if another application enabled it in the environment.
            with httpx.HTTPTransport(trust_env=False) as transport, tracing_context(enabled=False):
                model = ChatOllama(
                    model=self.model, reasoning=False, disable_streaming=True, cache=False,
                    keep_alive="5m", temperature=0.6, top_p=0.95, top_k=20,
                    num_ctx=4096, num_predict=3072,
                    client_kwargs={"timeout": httpx.Timeout(timeout, connect=min(3, timeout)),
                                   "trust_env": False},
                    sync_client_kwargs={"transport": transport},
                )
                # Binding exposes schemas; it never invokes analytical functions.
                response = model.bind_tools(list(tools)).invoke(
                    converted, stream=False, config={"callbacks": []},
                )
            if not isinstance(response, AIMessage):
                raise ValueError("Expected an AIMessage from ChatOllama.")
            if response.invalid_tool_calls:
                raise ValueError("ChatOllama returned malformed tool calls. Retry the investigation.")
            if isinstance(response.content, str):
                content = response.content
            else:
                # Reasoning/metadata blocks are not user-facing report text.
                content = "\n".join(
                    block if isinstance(block, str) else block["text"]
                    for block in response.content
                    if isinstance(block, str) or
                    (isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str))
                )
            calls = [ToolCall(call["name"], dict(call["args"]), call.get("id") or uuid4().hex)
                     for call in response.tool_calls]
            if len({call.id for call in calls}) != len(calls):
                raise ValueError("ChatOllama returned duplicate tool-call IDs.")
            return ModelReply(content=content, tool_calls=calls,
                              done_reason=response.response_metadata.get("done_reason"))
        except httpx.TimeoutException as exc:
            raise ModelClientError(
                f"LangChain/Ollama timed out after at most {timeout:.0f}s. Check 'ollama ps'; "
                "retry after loading the model, or increase INCIDENT_TIMEOUT_SECONDS (maximum 300)."
            ) from exc
        except (ConnectionError, httpx.RequestError) as exc:
            raise ModelClientError(
                "Cannot connect to Ollama through LangChain. Check 'ollama serve', "
                "'ollama list', and OLLAMA_HOST."
            ) from exc
        except ResponseError as exc:
            detail = (f"Model '{self.model}' is unavailable. Run 'ollama pull {self.model}'."
                      if exc.status_code == 404 else
                      f"Ollama rejected the LangChain request ({exc.status_code}): {exc.error}. "
                      "Check that the model supports native tool calling.")
            raise ModelClientError(detail) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise ModelClientError(f"Invalid LangChain response or configuration: {exc}") from exc
