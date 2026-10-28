"""LangChain owns model/tool iteration; middleware applies application policies."""
import inspect
import json
from typing import Any

from duckdb import Error as DuckDBError
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langsmith import tracing_context
from pydantic import ConfigDict

from model_clients import ModelClientError


def message_dict(message) -> dict[str, Any]:
    """Preserve tool-call links across LangChain state and the model boundary."""
    role = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}[message.type]
    result = {"role": role, "content": message.content}
    if isinstance(message, AIMessage) and message.tool_calls:
        result["tool_calls"] = [{"name": c["name"], "arguments": c["args"], "id": c["id"]}
                                for c in message.tool_calls]
    if isinstance(message, ToolMessage):
        result.update(tool_name=message.name, tool_call_id=message.tool_call_id)
    return result


def input_message(message: dict):
    if message["role"] == "user":
        return HumanMessage(content=message["content"])
    if message["role"] == "assistant":
        return AIMessage(content=message["content"])
    raise ValueError("Initial conversation must contain only user and assistant messages.")


class InvestigationModel(BaseChatModel):
    """Connect the real ChatOllama client or an injected offline fake to LangChain.

    This adapter performs one model request, never a tool loop. LangChain's agent
    binds the tools and decides when to call the model again.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Any
    analytical_tools: list[Any]
    request_timeout: float

    @property
    def _llm_type(self) -> str:
        return "incident-chatollama"

    def bind_tools(self, tools, **kwargs):
        # The transport binds the same fixed allowlist as schemas to ChatOllama.
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self.client.invoke([message_dict(m) for m in messages],
                                   tools=self.analytical_tools, timeout=self.request_timeout)
        if len({call.id for call in reply.tool_calls}) != len(reply.tool_calls):
            raise ModelClientError("Model returned duplicate tool-call IDs.")
        content = reply.content or ""
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()
        message = AIMessage(content=content, tool_calls=[
            {"name": c.name, "args": c.arguments, "id": c.id, "type": "tool_call"}
            for c in reply.tool_calls
        ], response_metadata={"done_reason": reply.done_reason})
        return ChatResult(generations=[ChatGeneration(message=message)])


def run_agent(day, messages, trace, model_client, timeout, max_rounds, *,
              report_context=None, chat_evidence=None, required_tool=None,
              current_question=None, tool_registry=None, allow_clarification=False,
              completion_check=None, tool_check=None):
    # Import at call time to avoid a module cycle and preserve patched constants
    # and analytical functions in offline tests.
    import incident_agent as agent

    registry = agent.TOOLS if tool_registry is None else tool_registry
    business_mode = tool_registry is not None
    is_chat = report_context is None
    required_tools = (required_tool,) if isinstance(required_tool, str) else required_tool or ()
    graph_request = bool(not business_mode and current_question and agent._requests_graph(current_question))
    graph_types = agent._requested_graph_types(current_question) if graph_request else set()
    graph_range = agent._requested_iso_range(current_question) if graph_request else None
    deadline = agent.monotonic() + timeout * 2

    class InvestigationPolicy(AgentMiddleware):
        """Per-invocation state; no global histories or persisted checkpoints."""
        def __init__(self):
            self.rounds = self.attempted = self.successful = self.format_retries = 0
            self.prompt = messages[0]["content"]
            self.last_comparisons = ""
            self.last_tool_count = 0
            self.assessment = None
            self.fatal_error = None
            self.call_ids = set()

        def fail(self, detail):
            raise agent.InvestigationError(detail, trace)

        def exhausted(self):
            if required_tools and not any(row["source"] == "model" and row["status"] == "ok"
                                          and row["tool"] in required_tools for row in trace):
                self.fail(f"The model did not execute a valid {' or '.join(required_tools)} call. "
                          "Retry the question; discovery and graphs require actual database results.")
            self.fail("The local model did not execute a valid model-selected tool."
                      if not self.successful else
                      "The model reached the round limit without completing a final report. Retry.")

        def before_model(self, state, runtime):
            if self.fatal_error:
                self.fail(self.fatal_error)
            if self.rounds >= max_rounds:
                self.exhausted()
            additions = []
            # LangChain routes completed tool batches back here. Supply only
            # newly computed comparisons and the current question as context.
            if self.attempted > self.last_tool_count:
                if business_mode:
                    additions.append(HumanMessage(content=(
                        f"Current question: {current_question}\n"
                        "Use only the configured metric definitions and actual tool results. "
                        "Resolve references using those observations. Select another tool if evidence is missing; "
                        "otherwise finish using '### Answer' and '### Unknowns and next checks'. "
                        "Keep the explanation qualitative; Python displays the measured values. /no_think")))
                if is_chat and self.successful and not graph_request and not business_mode:
                    combined = (chat_evidence or []) + trace
                    comparisons = {"categories": agent._category_comparisons(day, combined),
                                   "daily_metrics": agent._daily_comparisons(combined) if day is None else []}
                    serialized = json.dumps(comparisons, ensure_ascii=False)
                    if any(comparisons.values()) and serialized != self.last_comparisons:
                        additions.append(HumanMessage(content=(
                            "Comparisons computed by Python from actual tool results:\n" + serialized
                            + "\nUse these directions when comparing sales. Rank movement is not a sales decline. "
                            "Use the computed weighted AOV comparison; never infer its direction from total sales. "
                            "Categories missing from a limited list are unknown, not zero. "
                            "Now answer the question with the required sections. /no_think")))
                        self.last_comparisons = serialized
                    if day is None:
                        additions.append(HumanMessage(content=(
                            f"Current user request: {current_question or 'Use the original question'}\n"
                            "Continue satisfying this request with these actual results. "
                            "If investigating, fetch any missing daily metrics, weekday history, or category evidence. "
                            "When ready, write only '### Answer' and '### Unknowns and next checks'; "
                            "use qualitative prose, as Python displays the numerical evidence and anomaly table. "
                            + ("Calendar context: the investigated date was Black Friday. This is not proof of promotions or causation. "
                               if any(row["tool"] == "get_daily_metrics" and row["status"] == "ok"
                                      and row["arguments"].get("day") == "2017-11-24" for row in trace) else "")
                            + "/no_think")))
                if is_chat and graph_request:
                    additions.append(HumanMessage(content=(
                        f"Current graph request: {current_question}\n"
                        "Use the requested graph type and dates from actual evidence. Once create_graph succeeds, "
                        "finish under 60 words using '### Answer' and '### Unknowns and next checks'. "
                        "Do not repeat prior graphs or run an unrelated investigation. /no_think")))
                if not is_chat and self.successful and self.prompt != agent.SYSTEM_PROMPT:
                    self.prompt = agent.SYSTEM_PROMPT
                    additions.append(HumanMessage(content=(
                        "Now write the concise final report using your tool results and these verified "
                        "database observations:\n" + json.dumps(report_context, ensure_ascii=False)
                        + ("\nCalendar context (not database evidence): 2017-11-24 was Black Friday. "
                           "Mention this as a plausible explanation for volume, not proof of promotions "
                           "or causation; promotional records are unavailable." if day == "2017-11-24" else "")
                        + "\nAvoid further tool calls unless essential. /no_think")))
                self.last_tool_count = self.attempted
            return {"messages": additions} if additions else None

        def wrap_model_call(self, request, handler):
            model_messages = request.messages
            if business_mode:
                # Keep tool proposals/results linked, but omit intermediate
                # assistant prose and superseded generic reminders. The full
                # LangChain state and audit remain intact outside model context.
                compact = []
                reminders = [m for m in model_messages if isinstance(m, HumanMessage)
                             and m.content.startswith("Current question:")]
                for message in model_messages:
                    if isinstance(message, AIMessage):
                        if message.tool_calls:
                            compact.append(message.model_copy(update={"content": ""}))
                    elif (isinstance(message, HumanMessage) and message.content.startswith("Current question:")
                          and message is not reminders[-1]):
                        continue
                    else:
                        compact.append(message)
                model_messages = compact
            transcript = [{"role": "system", "content": self.prompt}]
            transcript.extend(message_dict(m) for m in model_messages)
            if is_chat and len(json.dumps(transcript, ensure_ascii=False)) > agent.MAX_CHAT_REQUEST_CHARS:
                self.fail("Chat context limit reached. Clear the conversation or ask a narrower question.")
            remaining = deadline - agent.monotonic()
            if remaining <= 0:
                self.fail("Investigation time budget reached. Retry or use a faster local model.")
            self.rounds += 1
            request.model.request_timeout = min(timeout, remaining)
            try:
                return handler(request.override(system_message=SystemMessage(content=self.prompt), messages=model_messages))
            except ModelClientError as exc:
                self.fail(f"Model round {self.rounds}: {exc}")

        @hook_config(can_jump_to=["model"])
        def after_model(self, state, runtime):
            message = state["messages"][-1]
            if message.response_metadata.get("done_reason") == "length":
                self.fail("Ollama reached the response token limit before completing. Retry with "
                          "INCIDENT_MODEL=qwen3:4b-instruct; some qwen3:4b tags use thinking-only checkpoints.")
            if message.tool_calls:
                ids = {call["id"] for call in message.tool_calls}
                if self.call_ids.intersection(ids):
                    self.fail("Model returned duplicate tool-call IDs across rounds. Retry.")
                self.call_ids.update(ids)
                return None  # LangChain routes this response to its tool node.
            if (allow_clarification and self.attempted == 0 and message.content.strip().startswith("### Clarification")
                    and "<think>" not in message.content and message.content.strip() != "### Clarification"):
                self.assessment = message.content.strip()
                return None
            needs_graph_dates = (required_tools == ("create_graph",) and not
                                 agent._graph_scope_known(current_question or "", (chat_evidence or []) + trace))
            if required_tools and not any(row["source"] == "model" and row["status"] == "ok"
                                          and row["tool"] in required_tools for row in trace) and not needs_graph_dates:
                return self.retry(
                    ("Call one of the registered business tools to obtain actual evidence before answering. "
                     "For a requested graph, call create_metric_graph. If the metric or dates are ambiguous, "
                     "ask a short question under '### Clarification' without claiming results. /no_think")
                    if business_mode else
                    "The user requested a graph. Call create_graph now using the requested metric "
                    "and dates from the question or actual evidence. Do not stop at proposing another query. /no_think"
                    if required_tools == ("create_graph",) else
                    "Call find_anomalies for discovery or create_graph for a requested graph "
                    "to retrieve actual evidence before answering. /no_think")
            if not (self.successful or (is_chat and self.attempted == 0)):
                return self.retry(f"Please call get_sales_by_category for {day} now before reporting. /no_think"
                                  if day else "Call find_anomalies to retrieve actual stored anomalies before answering. /no_think")
            if completion_check is not None:
                correction = completion_check(trace)
                if correction:
                    return self.retry(correction)
            content = message.content.strip()
            if not content or "<think>" in content:
                self.fail("Ollama returned an empty final report. Retry the investigation.")
            headings = (("### Answer", "### Unknowns and next checks") if is_chat else
                        ("### Possible explanations", "### Confidence and next checks"))
            if any(heading not in content for heading in headings):
                if is_chat and self.format_retries == 0 and self.rounds < max_rounds:
                    self.format_retries += 1
                    return self.retry("Rewrite the answer using exactly '### Answer' and "
                                      "'### Unknowns and next checks'. Use the evidence already collected. "
                                      "Do not repeat tool calls just to repair formatting. /no_think")
                self.fail("Ollama did not return the required report sections. Retry the investigation.")
            self.assessment = headings[0] + content.split(headings[0], 1)[1]
            return None  # A validated final answer ends the LangChain agent run.

        def retry(self, instruction):
            if self.rounds >= max_rounds:
                self.exhausted()
            return {"messages": [HumanMessage(content=instruction)], "jump_to": "model"}

        def wrap_tool_call(self, request, handler):
            call = request.tool_call
            name, args = call["name"], dict(call["args"])
            # Keep fatal limits sticky: the runtime may convert tool exceptions
            # into messages, but it must never turn an exhausted budget into success.
            if self.attempted >= agent.MAX_TOOL_CALLS:
                self.fatal_error = f"Model reached the {agent.MAX_TOOL_CALLS}-tool-call limit without a final report. Retry."
            if deadline - agent.monotonic() <= 0:
                self.fatal_error = "Investigation time budget reached. Retry or use a faster local model."
            if self.fatal_error:
                return ToolMessage(content=json.dumps({"error": self.fatal_error}),
                                   name=name, tool_call_id=call["id"], status="error")
            status = "ok"
            try:
                fn = registry.get(name)
                if fn is None:
                    raise ValueError(f"Unknown tool: {name}. Use one of {', '.join(registry)}.")
                inspect.signature(fn).bind(**args)
                if tool_check is not None:
                    tool_check(name, args)
                if name == "create_graph" and graph_types and args.get("graph_type") not in graph_types:
                    raise ValueError(f"The current graph request needs graph_type in {', '.join(sorted(graph_types))}; "
                                     f"received {args.get('graph_type')}. Follow the current question, not an earlier graph.")
                requested_range = (agent._requested_iso_range(current_question or "")
                                   if name == "create_metric_graph" else graph_range)
                if name in ("create_graph", "create_metric_graph") and requested_range and (
                    args.get("start_date"), args.get("end_date") or args.get("start_date")
                ) != requested_range:
                    raise ValueError(f"The requested graph range is {requested_range[0]} through {requested_range[1]}. "
                                     "Set both start_date and end_date to those dates.")
                # Invoke the allowlisted function with original values. This avoids
                # schema coercion weakening strict integer/date argument checks.
                output = fn(**args)
                if len(json.dumps(output, ensure_ascii=False)) > agent.MAX_RESULT_CHARS:
                    raise ValueError("Tool result is too large. Request a smaller limit.")
                if output.get("found") is False:
                    raise ValueError(f"No stored anomaly for {args.get('day')}.")
            except (ValueError, TypeError, OSError, RuntimeError, DuckDBError) as exc:
                status, output = "error", {"error": f"{type(exc).__name__}: {exc}"}
            trace.append({"source": "model", "round": self.rounds, "tool_call_id": call["id"],
                          "tool": name, "arguments": args, "result": output, "status": status})
            self.attempted += 1
            self.successful += status == "ok"
            result = agent._model_tool_result(name, output) if status == "ok" else output
            return ToolMessage(content=json.dumps(result, ensure_ascii=False, default=str),
                               name=name, tool_call_id=call["id"], status="success" if status == "ok" else "error")

    policy = InvestigationPolicy()
    model = InvestigationModel(client=model_client, analytical_tools=list(registry.values()),
                               request_timeout=timeout)
    # Tracing is disabled around the entire runtime, including tools and graph
    # nodes, even if another application enabled LangSmith through the environment.
    with tracing_context(enabled=False):
        runtime = create_agent(model, tools=list(registry.values()), middleware=[policy],
                               checkpointer=None, store=None)
        runtime.invoke({"messages": [input_message(m) for m in messages[1:]]},
                       config={"callbacks": [], "max_concurrency": 1,
                               "recursion_limit": max_rounds * 6 + 10})
    if policy.fatal_error:
        policy.fail(policy.fatal_error)
    if policy.assessment is None:
        policy.exhausted()
    return policy.assessment, trace
