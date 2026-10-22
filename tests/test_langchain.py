"""Optional adapter parity tests: real LangChain messages, fake chat responses."""
import copy
import importlib.util
import json
import sys
import unittest
from unittest.mock import patch

import httpx
from ollama import ResponseError
from streamlit.testing.v1 import AppTest

from test_mvp import DatabaseCase, DAY, REPORT, ROOT, agent, tools
from langchain_client import LangChainOllamaClient
from model_clients import ModelClientError, OllamaModelClient, create_model_client

HAS_LANGCHAIN = importlib.util.find_spec("langchain_ollama") is not None
if HAS_LANGCHAIN:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langsmith.run_helpers import get_tracing_context


def call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class OptionalDependencyTests(unittest.TestCase):
    def test_native_backend_does_not_import_langchain(self):
        with patch.dict(sys.modules, {"langchain_ollama": None, "langchain_core": None}):
            self.assertIsInstance(create_model_client("qwen3:4b", "native"), OllamaModelClient)

    def test_missing_optional_package_has_install_instruction(self):
        with patch.dict(sys.modules, {"langchain_ollama": None}), self.assertRaisesRegex(ModelClientError, "requirements-langchain.txt"):
            create_model_client("qwen3:4b-instruct", "langchain").invoke([], tools=[], timeout=10)

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(ModelClientError, "INCIDENT_BACKEND"):
            create_model_client("qwen3:4b", "unknown")


@unittest.skipUnless(HAS_LANGCHAIN, "Install requirements-langchain.txt for adapter tests.")
class LangChainTests(DatabaseCase):
    def mock_chat(self, responses):
        constructor_patch = patch("langchain_ollama.ChatOllama")
        self.constructor = constructor_patch.start()
        self.addCleanup(constructor_patch.stop)
        self.requests = []
        iterator = iter(responses)

        def invoke(messages, **kwargs):
            self.assertIs(get_tracing_context()["enabled"], False)
            self.requests.append((copy.deepcopy(messages), kwargs))
            response = next(iterator)
            if isinstance(response, Exception):
                raise response
            return response

        self.bound = self.constructor.return_value.bind_tools.return_value
        self.bound.invoke.side_effect = invoke

    def test_multiple_steps_batched_tools_and_id_correspondence(self):
        self.mock_chat([
            AIMessage(content="", tool_calls=[
                call("get_sales_by_category", {"day": DAY, "limit": 1}, "categories"),
                call("get_order_status_breakdown", {"day": DAY}, "statuses"),
            ]),
            AIMessage(content="", tool_calls=[call("get_weekday_history", {"day": DAY, "weeks": 1}, "history")]),
            AIMessage(content=REPORT),
        ])
        with patch.object(agent, "BACKEND", "langchain"):
            report, trace = agent.investigate(DAY)
        self.assertIn("books: R$100.00", report)
        self.assertEqual([r["source"] for r in trace], ["seeded"] * 3 + ["model"] * 3)
        self.assertEqual([r["tool_call_id"] for r in trace[3:]], ["categories", "statuses", "history"])
        messages, kwargs = self.requests[-1]
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        evidence = [m for m in messages if isinstance(m, ToolMessage)]
        self.assertEqual([m.tool_call_id for m in evidence], ["categories", "statuses", "history"])
        self.assertEqual(json.loads(evidence[0].content)["categories"][0]["item_sales"], 100)
        self.assertEqual(json.loads(evidence[-1].content)["sample_size"], 1)
        self.assertFalse(kwargs["stream"])
        self.assertEqual(kwargs["config"]["callbacks"], [])
        self.assertFalse(self.constructor.call_args.kwargs["reasoning"])
        self.assertTrue(self.constructor.call_args.kwargs["disable_streaming"])
        self.assertFalse(self.constructor.call_args.kwargs["cache"])
        self.assertFalse(self.constructor.call_args.kwargs["client_kwargs"]["trust_env"])
        self.assertEqual(self.constructor.call_args.kwargs["client_kwargs"]["timeout"].read, 90)
        self.assertEqual(self.constructor.call_args.kwargs["num_ctx"], 4096)
        self.assertEqual(self.constructor.call_args.kwargs["model"], agent.MODEL)
        self.assertEqual(self.constructor.return_value.bind_tools.call_args.args[0], list(agent.TOOLS.values()))

    def test_binding_and_normalizing_does_not_execute_a_tool(self):
        self.mock_chat([AIMessage(content="", tool_calls=[call("get_daily_metrics", {"day": DAY}, "daily")])])
        with patch.object(tools, "get_daily_metrics") as analytical:
            reply = LangChainOllamaClient("qwen3:4b-instruct").invoke([], tools=[analytical], timeout=12)
        analytical.assert_not_called()
        self.assertEqual(reply.tool_calls[0].arguments, {"day": DAY})
        self.assertEqual(reply.tool_calls[0].id, "daily")

    def test_dispatch_error_is_returned_to_model_and_can_recover(self):
        self.mock_chat([
            AIMessage(content="", tool_calls=[call("get_order_status_breakdown", {"day": DAY, "limit": 10}, "bad")]),
            AIMessage(content="", tool_calls=[call("get_order_status_breakdown", {"day": DAY}, "good")]),
            AIMessage(content=REPORT),
        ])
        _, trace = agent.investigate(DAY, client=LangChainOllamaClient(agent.MODEL))
        self.assertEqual([r["status"] for r in trace[3:]], ["error", "ok"])
        returned = next(m for m in self.requests[1][0] if isinstance(m, ToolMessage))
        self.assertEqual(returned.tool_call_id, "bad")
        self.assertIn("error", json.loads(returned.content))

    def test_unknown_tool_and_bad_dates_are_not_successful(self):
        for name, arguments in (("execute_sql", {"sql": "DROP TABLE orders"}),
                                ("get_daily_metrics", {"day": "bad"})):
            with self.subTest(name=name):
                self.mock_chat([AIMessage(content="", tool_calls=[call(name, arguments, "bad")]), AIMessage(content=REPORT)])
                with self.assertRaisesRegex(agent.InvestigationError, "valid model-selected") as caught:
                    agent.investigate(DAY, max_rounds=2, client=LangChainOllamaClient(agent.MODEL))
                self.assertEqual(caught.exception.trace[-1]["status"], "error")
        self.assertEqual(tools.get_daily_metrics(DAY)["item_sales"], 150)

    def test_no_tools_is_failure(self):
        self.mock_chat([AIMessage(content=REPORT)])
        with self.assertRaisesRegex(agent.InvestigationError, "model-selected"):
            agent.investigate(DAY, max_rounds=1, client=LangChainOllamaClient(agent.MODEL))

    def test_model_errors_preserve_seed_trace_and_never_fall_back(self):
        failures = ((httpx.ReadTimeout("slow"), "timed out"),
                    (ConnectionError("offline"), "ollama serve"),
                    (ResponseError("missing", status_code=404), "ollama pull"),
                    (ResponseError("no tools", status_code=400), "native tool calling"))
        for error, message in failures:
            with self.subTest(error=error):
                self.mock_chat([error])
                with patch.object(agent, "BACKEND", "langchain"), patch("model_clients.Client") as native:
                    with self.assertRaisesRegex(agent.InvestigationError, message) as caught:
                        agent.investigate(DAY)
                native.assert_not_called()
                self.assertEqual(len(caught.exception.trace), 3)

    def test_malformed_and_truncated_responses_fail(self):
        responses = (None,
                     AIMessage(content="", invalid_tool_calls=[{"name": "get_daily_metrics", "args": "{bad", "id": "bad", "error": "Invalid JSON"}]),
                     AIMessage(content=REPORT, response_metadata={"done_reason": "length"}),
                     AIMessage(content="", tool_calls=[call("get_daily_metrics", {"day": DAY}, "same")] * 2))
        for response in responses:
            with self.subTest(response=response):
                self.mock_chat([response])
                with self.assertRaises(agent.InvestigationError):
                    agent.investigate(DAY, client=LangChainOllamaClient(agent.MODEL))

    def test_content_blocks_exclude_reasoning(self):
        self.mock_chat([AIMessage(content=[{"type": "reasoning", "reasoning": "internal"},
                                          {"type": "text", "text": REPORT}])])
        reply = LangChainOllamaClient(agent.MODEL).invoke([], tools=[], timeout=10)
        self.assertEqual(reply.content, REPORT)
        self.assertNotIn("internal", reply.content)

    def test_tracing_environment_is_overridden_locally_and_restored(self):
        self.mock_chat([AIMessage(content=REPORT)])
        previous = get_tracing_context()["enabled"]
        with patch.dict("os.environ", {"LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true"}):
            LangChainOllamaClient(agent.MODEL).invoke([], tools=[], timeout=10)
        self.assertIs(get_tracing_context()["enabled"], previous)

    def test_invalid_backend_surfaces_as_investigation_failure(self):
        with patch.object(agent, "BACKEND", "unsupported"), self.assertRaisesRegex(agent.InvestigationError, "INCIDENT_BACKEND"):
            agent.investigate(DAY)

    def test_dashboard_selects_backend_and_persists_provenance(self):
        self.mock_chat([AIMessage(content="", tool_calls=[call("get_sales_by_category", {"day": DAY, "limit": 1}, "category")]),
                        AIMessage(content=REPORT)])
        with patch.object(agent, "BACKEND", "langchain"):
            app = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=15).run()
            app.radio[0].set_value("Explore a date").run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn("via langchain", app.success[0].value)
            self.assertEqual(app.session_state["investigation"]["backend"], "langchain")
            self.assertIn("books: R$100.00", app.markdown[0].value)
            app.run()
            self.assertIn("via langchain", app.success[0].value)
        self.assertEqual(len(self.requests), 2)

    def test_followup_chat_uses_langchain_tool_ids_and_preserves_history(self):
        from test_chat import ANSWER
        self.mock_chat([
            AIMessage(content="", tool_calls=[call("get_sales_by_category", {"day": DAY, "limit": 1}, "initial")]),
            AIMessage(content=REPORT),
            AIMessage(content="", tool_calls=[
                call("get_daily_metrics", {"day": "2017-11-17"}, "followup")]),
            AIMessage(content=ANSWER),
            AIMessage(content=ANSWER),
        ])
        with patch.object(agent, "BACKEND", "langchain"):
            app = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=15).run()
            app.radio[0].set_value("Explore a date").run()
            app.button[0].click().run()
            app.chat_input[0].set_value("Compare with the previous Friday.").run()
            self.assertFalse(app.exception)
            exchange = app.session_state["investigation"]["conversation"][0]
            self.assertEqual(exchange["backend"], "langchain")
            self.assertEqual(exchange["trace"][-1]["tool_call_id"], "followup")
            tool = next(m for m in self.requests[3][0] if isinstance(m, ToolMessage))
            self.assertEqual(tool.tool_call_id, "followup")
            self.assertEqual(json.loads(tool.content)["item_sales"], 20)
            app.chat_input[0].set_value("What would confirm that?").run()
            self.assertFalse(app.exception)
            questions = [m.content for m in self.requests[4][0] if isinstance(m, HumanMessage)]
            self.assertIn("Compare with the previous Friday.", questions)
            self.assertEqual(len(app.chat_message), 4)

    def test_chat_first_discovery_uses_model_selected_langchain_tools(self):
        from test_chat import ANSWER
        self.mock_chat([
            AIMessage(content="", tool_calls=[call("find_anomalies", {"direction": "spikes", "limit": 1}, "search")]),
            AIMessage(content=ANSWER),
            AIMessage(content="", tool_calls=[call("get_daily_metrics", {"day": DAY}, "investigate")]),
            AIMessage(content=ANSWER),
        ])
        with patch.object(agent, "BACKEND", "langchain"):
            app = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=15).run()
            self.assertFalse(app.selectbox)
            app.chat_input[0].set_value("Find the largest spike").run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("Investigate the first one").run()
            self.assertFalse(app.exception)
            conversation = app.session_state["agent_workspace"]["conversation"]
            self.assertEqual(len(conversation), 2)
            self.assertEqual(conversation[0]["trace"][-1]["tool_call_id"], "search")
            self.assertEqual(conversation[1]["trace"][-1]["tool_call_id"], "investigate")
            self.assertEqual(conversation[1]["backend"], "langchain")
            self.assertIn("R$150.00 item sales; 2 distinct delivered orders", conversation[1]["answer"])


if __name__ == "__main__":
    unittest.main()
