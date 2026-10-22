"""Fast offline verification: temporary DuckDB, real tools, fake Ollama replies."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import httpx
from ollama import ChatResponse, ResponseError
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import agent_tools as tools
import incident_agent as agent
import model_clients
from model_clients import ModelReply, ToolCall

DAY = "2017-11-24"
REPORT = """### Findings
R$150.00 item sales across 2 delivered orders; books contributed R$100.00.
### Possible explanations
Black Friday is plausible calendar context, not proof of a promotion.
### Confidence and next checks
The database verifies sales. Promotions and causality remain unknown.
"""


def reply(content="", calls=(), **kwargs):
    return ChatResponse(model="qwen3:4b", done=True, message={
        "role": "assistant", "content": content,
        "tool_calls": [{"function": {"name": name, "arguments": args}} for name, args in calls],
    }, **kwargs)


class DatabaseCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "fixture.duckdb"
        with duckdb.connect(str(self.db)) as con:
            con.execute("CREATE TABLE orders(order_id VARCHAR, order_status VARCHAR, order_purchase_timestamp TIMESTAMP)")
            con.execute("CREATE TABLE order_items(order_id VARCHAR, product_id VARCHAR, price DECIMAL(12,2))")
            con.execute("CREATE TABLE products(product_id VARCHAR, product_category_name VARCHAR)")
            con.execute("CREATE TABLE detected_anomalies(order_date DATE, actual_sales DOUBLE, expected_sales DOUBLE, deviation_pct DOUBLE)")
            con.execute("INSERT INTO detected_anomalies VALUES ('2017-11-24', 150, 30, 400)")
            con.execute("INSERT INTO products VALUES ('p1', 'books'), ('p2', 'toys')")
            con.execute("""INSERT INTO orders VALUES
                ('a', 'delivered', '2017-11-24'), ('b', 'delivered', '2017-11-24'),
                ('c', 'canceled', '2017-11-24'), ('d', 'delivered', '2017-11-17'),
                ('e', 'delivered', '2017-11-10')""")
            con.execute("""INSERT INTO order_items VALUES
                ('a','p1',60), ('a','p1',40), ('b','p2',50), ('c','p1',900),
                ('d','p1',20), ('e','p1',40)""")
        self.db_patch = patch.object(tools, "DB_PATH", self.db)
        self.db_patch.start()
        self.addCleanup(self.db_patch.stop)
        self.env_patch = patch.dict("os.environ", {"INCIDENT_TIMEOUT_SECONDS": "90"})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def mock_client(self, responses):
        client_patch = patch.object(model_clients, "Client")
        constructor = client_patch.start()
        self.addCleanup(client_patch.stop)
        client = constructor.return_value.__enter__.return_value
        self.requests = []
        iterator = iter(responses)

        def chat(**kwargs):
            self.requests.append(copy.deepcopy(kwargs))
            result = next(iterator)
            if isinstance(result, Exception):
                raise result
            return result

        client.chat.side_effect = chat
        self.constructor = constructor
        return client


class ToolTests(DatabaseCase):
    def test_metrics_count_distinct_delivered_orders(self):
        self.assertEqual(tools.get_daily_metrics(DAY), {
            "date": DAY, "item_sales": 150.0, "orders": 2, "items": 3, "average_order_value": 75.0,
        })
        self.assertEqual(tools.get_daily_metrics("2017-11-25")["orders"], 0)
        self.assertIsNone(tools.get_daily_metrics("2017-11-25")["average_order_value"])

    def test_history_categories_statuses_and_chart(self):
        history = tools.get_weekday_history(DAY, weeks=2)
        self.assertEqual(history["sample_size"], 2)
        self.assertEqual(history["median_item_sales"], 30.0)
        self.assertEqual(history["mean_orders"], 1)
        self.assertEqual(tools.get_sales_by_category(DAY, limit=1)["categories"][0]["item_sales"], 100.0)
        self.assertEqual(len(tools.get_order_status_breakdown(DAY)["statuses"]), 2)
        self.assertEqual(len(tools.get_sales_window(DAY)), 3)
        self.assertEqual(tools.list_anomalies()[0]["date"], DAY)
        self.assertFalse(tools.get_anomaly("2017-11-25")["found"])

    def test_strict_date_and_limit_validation(self):
        for day in (None, 20171124, "20171124", "2017-W47-5", "2017-02-30", "2017-11-24' OR 1=1 --"):
            with self.subTest(day=day), self.assertRaises(ValueError):
                tools.get_daily_metrics(day)
        for value in (0, 16, True, 2.5, "2", None):
            with self.subTest(limit=value), self.assertRaises(ValueError):
                tools.get_sales_by_category(DAY, value)
        for value in (0, 9, False, "2"):
            with self.subTest(weeks=value), self.assertRaises(ValueError):
                tools.get_weekday_history(DAY, value)

    def test_read_only_connection_rejects_writes(self):
        with tools._connect() as con:
            with self.assertRaises(duckdb.Error):
                con.execute("DELETE FROM orders")
        self.assertEqual(tools.get_daily_metrics(DAY)["orders"], 2)

    def test_missing_database_is_not_created(self):
        absent = Path(self.tmp.name) / "missing.duckdb"
        with patch.object(tools, "DB_PATH", absent), self.assertRaisesRegex(FileNotFoundError, "not found"):
            tools.list_anomalies()
        self.assertFalse(absent.exists())

    def test_missing_table_has_actionable_error(self):
        with duckdb.connect(str(self.db)) as con:
            con.execute("DROP TABLE detected_anomalies")
        with self.assertRaisesRegex(RuntimeError, "missing tables: detected_anomalies"):
            tools.list_anomalies()


class AgentTests(DatabaseCase):
    def test_injected_model_client_receives_linked_tool_evidence(self):
        class FakeModel:
            def __init__(self):
                self.messages = []

            def invoke(self, messages, *, tools, timeout):
                self.messages.append(copy.deepcopy(messages))
                if len(self.messages) == 1:
                    return ModelReply(tool_calls=[ToolCall("get_sales_by_category",
                                                          {"day": DAY, "limit": 1}, "call-1")])
                return ModelReply(content=REPORT)

        client = FakeModel()
        with patch.object(model_clients, "Client") as native:
            report, trace = agent.investigate(DAY, client=client)
        native.assert_not_called()
        self.assertIn("books: R$100.00", report)
        self.assertEqual(trace[-1]["tool_call_id"], "call-1")
        assistant = next(m for m in client.messages[-1] if m["role"] == "assistant")
        evidence = next(m for m in client.messages[-1] if m["role"] == "tool")
        self.assertEqual(assistant["tool_calls"][0]["id"], evidence["tool_call_id"])
        self.assertEqual(json.loads(evidence["content"])["categories"][0]["item_sales"], 100)

    def test_injected_adapter_cannot_bypass_tool_allowlist(self):
        class UnsafeModel:
            def invoke(self, messages, *, tools, timeout):
                return ModelReply(tool_calls=[ToolCall("execute_sql", {"sql": "DELETE FROM orders"})])

        with self.assertRaises(agent.InvestigationError) as caught:
            agent.investigate(DAY, max_rounds=1, client=UnsafeModel())
        self.assertEqual(caught.exception.trace[-1]["status"], "error")
        self.assertEqual(tools.get_daily_metrics(DAY)["orders"], 2)

    def test_multiple_model_tool_steps_and_evidence_returned(self):
        self.mock_client([
            reply(calls=[("get_sales_by_category", {"day": DAY, "limit": 1})]),
            reply(calls=[("get_order_status_breakdown", {"day": DAY})]),
            reply(REPORT),
        ])
        report, trace = agent.investigate(DAY)
        self.assertIn("R$150.00", report)
        self.assertIn("2 distinct delivered orders", report)
        self.assertIn("Black Friday is plausible calendar context", report)
        self.assertIn("books: R$100.00", report)
        self.assertEqual([row["source"] for row in trace], ["seeded"] * 3 + ["model"] * 2)
        self.assertEqual(trace[3]["arguments"], {"day": DAY, "limit": 1})
        self.assertEqual(trace[3]["result"]["categories"][0]["item_sales"], 100.0)
        tool_messages = [m for m in self.requests[-1]["messages"] if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 2)
        self.assertEqual(json.loads(tool_messages[0]["content"]), trace[3]["result"])
        self.assertEqual(tool_messages[0]["tool_name"], "get_sales_by_category")
        self.assertEqual(self.requests[1]["messages"][2]["tool_calls"][0]["function"]["name"], "get_sales_by_category")
        self.assertFalse(self.requests[0]["think"])
        self.assertEqual(self.requests[0]["model"], agent.MODEL)
        self.assertIn("/no_think", self.requests[0]["messages"][1]["content"])
        self.assertIn('"prior_weekdays_weighted_average_order_value": 30.0', self.requests[1]["messages"][-1]["content"])
        self.assertFalse(self.constructor.call_args.kwargs["trust_env"])
        self.assertEqual(self.constructor.call_args.kwargs["timeout"].read, 90)

    def test_multiple_tools_in_one_message(self):
        self.mock_client([reply(calls=[("get_daily_metrics", {"day": DAY}),
                                      ("get_weekday_history", {"day": DAY, "weeks": 1})]), reply(REPORT)])
        _, trace = agent.investigate(DAY)
        self.assertEqual([r["status"] for r in trace[3:]], ["ok", "ok"])
        self.assertEqual(trace[4]["result"]["sample_size"], 1)

    def test_invalid_tool_then_model_recovery(self):
        self.mock_client([
            reply(calls=[("get_order_status_breakdown", {"day": DAY, "limit": 10})]),
            reply(calls=[("get_order_status_breakdown", {"day": DAY})]), reply(REPORT),
        ])
        _, trace = agent.investigate(DAY)
        self.assertEqual(trace[3]["status"], "error")
        self.assertIn("unexpected keyword argument", trace[3]["result"]["error"])
        self.assertEqual(trace[4]["status"], "ok")
        returned_tools = [m for m in self.requests[1]["messages"] if m["role"] == "tool"]
        self.assertIn("error", returned_tools[-1]["content"])

    def test_unknown_tool_and_invalid_arguments_do_not_count_as_success(self):
        for name, args in (("execute_sql", {"sql": "DROP TABLE orders"}),
                           ("get_daily_metrics", {"day": "bad"}),
                           ("get_sales_by_category", {"day": DAY, "limit": 1000}),
                           ("get_anomaly", {"day": "2017-11-25"})):
            with self.subTest(name=name):
                self.mock_client([reply(calls=[(name, args)]), reply(REPORT)])
                with self.assertRaisesRegex(agent.InvestigationError, "did not execute a valid") as caught:
                    agent.investigate(DAY, max_rounds=2)
                self.assertEqual(caught.exception.trace[-1]["status"], "error")
        self.assertEqual(tools.get_daily_metrics(DAY)["item_sales"], 150)

    def test_no_tool_call_is_failure_and_prompts_retry(self):
        self.mock_client([reply(REPORT), reply(REPORT)])
        with self.assertRaisesRegex(agent.InvestigationError, "model-selected") as caught:
            agent.investigate(DAY, max_rounds=2)
        self.assertEqual(len(caught.exception.trace), 3)
        returned_users = [m for m in self.requests[1]["messages"] if m["role"] == "user"]
        self.assertIn("Please call get_sales_by_category", returned_users[-1]["content"])

    def test_round_limit_never_returns_intermediate_prose(self):
        self.mock_client([reply("Preliminary speculation", [("get_daily_metrics", {"day": DAY})])])
        with self.assertRaisesRegex(agent.InvestigationError, "round limit"):
            agent.investigate(DAY, max_rounds=1)

    def test_tool_call_limit(self):
        self.mock_client([reply(calls=[("get_daily_metrics", {"day": DAY})] * 9)])
        with self.assertRaisesRegex(agent.InvestigationError, "8-tool-call") as caught:
            agent.investigate(DAY)
        self.assertEqual(len(caught.exception.trace), 11)

    def test_empty_truncated_and_malformed_responses_fail(self):
        for last in (reply(), reply("I will investigate"), reply("partial", done_reason="length"), {"message": None}):
            with self.subTest(last=last):
                self.mock_client([reply(calls=[("get_daily_metrics", {"day": DAY})]), last])
                with self.assertRaises(agent.InvestigationError):
                    agent.investigate(DAY)

    def test_reasoning_is_removed_from_final_report(self):
        self.mock_client([reply(calls=[("get_daily_metrics", {"day": DAY})]),
                          reply("secret deliberation</think>\n" + REPORT)])
        report, _ = agent.investigate(DAY)
        self.assertNotIn("deliberation", report)
        self.assertIn("### Findings", report)

    def test_model_cannot_invent_findings_from_correct_tool_results(self):
        invented = REPORT.replace("R$150.00 item sales across 2 delivered orders; books contributed R$100.00.",
                                  "99.8% of 1,159 orders were delivered; sales were R$999,999.99.")
        self.mock_client([reply(calls=[("get_order_status_breakdown", {"day": DAY})]), reply(invented)])
        report, _ = agent.investigate(DAY)
        self.assertNotIn("99.8%", report)
        self.assertNotIn("999,999.99", report)
        self.assertIn("delivered: 2 orders", report)
        self.assertIn("canceled: 1 orders", report)
        self.assertIn("R$150.00", report)

    def test_model_only_generates_assessment_while_python_renders_findings(self):
        assessment = REPORT[REPORT.index("### Possible explanations"):]
        self.mock_client([reply(calls=[("get_sales_by_category", {"day": DAY, "limit": 1})]),
                          reply(assessment)])
        report, _ = agent.investigate(DAY)
        self.assertEqual(report.count("### Findings"), 1)
        self.assertIn("books: R$100.00", report)
        self.assertIn(assessment.strip(), report)
        self.assertIn("do not repeat dates, figures", self.requests[-1]["messages"][0]["content"])

    def test_connection_timeout_and_missing_model_errors_keep_trace(self):
        failures = [(ConnectionError("offline"), "ollama serve"),
                    (httpx.ReadTimeout("slow"), "timed out"),
                    (ResponseError("not found", status_code=404), "ollama pull"),
                    (ResponseError("model lacks tools", status_code=400), "native tool calling")]
        for error, text in failures:
            with self.subTest(error=error):
                self.mock_client([error])
                with self.assertRaisesRegex(agent.InvestigationError, text) as caught:
                    agent.investigate(DAY)
                self.assertEqual(len(caught.exception.trace), 3)

    def test_error_after_tool_call_keeps_model_trace(self):
        self.mock_client([reply(calls=[("get_daily_metrics", {"day": DAY})]), httpx.ReadTimeout("slow")])
        with self.assertRaises(agent.InvestigationError) as caught:
            agent.investigate(DAY)
        self.assertEqual(caught.exception.trace[-1]["source"], "model")

    def test_bad_inputs_and_missing_anomaly_never_contact_ollama(self):
        client = self.mock_client([])
        for day, rounds in (("bad", 7), ("2017-11-25", 7), (DAY, 0), (DAY, 11)):
            with self.subTest(day=day, rounds=rounds), self.assertRaises(agent.InvestigationError):
                agent.investigate(day, max_rounds=rounds)
        client.chat.assert_not_called()
        for value in ("zero", "0", "nan", "301"):
            with patch.dict("os.environ", {"INCIDENT_TIMEOUT_SECONDS": value}), self.assertRaisesRegex(agent.InvestigationError, "INCIDENT_TIMEOUT_SECONDS"):
                agent.investigate(DAY)

    def test_total_time_budget(self):
        self.mock_client([])
        with patch.object(agent, "monotonic", side_effect=[0, 181]), self.assertRaisesRegex(agent.InvestigationError, "time budget"):
            agent.investigate(DAY)


class DashboardTests(DatabaseCase):
    def app(self):
        return AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=15).run()

    def test_button_runs_real_backend_with_mock_model_and_persists_report(self):
        self.mock_client([reply(calls=[("get_sales_by_category", {"day": DAY, "limit": 1})]), reply(REPORT)])
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].value, DAY)
        self.assertEqual(app.metric[0].value, "R$ 150.00")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.success), 1)
        self.assertIn("3 seeded lookups, 1 model-selected", app.expander[0].label)
        self.assertIn("books: R$100.00", app.markdown[0].value)
        app.run()
        self.assertEqual(len(app.success), 1)
        self.assertIn("books: R$100.00", app.markdown[0].value)
        self.assertEqual(len(self.requests), 2)

    def test_failure_visible_with_trace_and_no_success(self):
        self.mock_client([ConnectionError("offline")])
        app = self.app()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.success), 0)
        self.assertIn("ollama serve", app.error[0].value)
        self.assertIn("3 seeded lookups, 0 model-selected", app.expander[0].label)
        app.run()
        self.assertIn("Investigation failed", app.error[0].value)

    def test_missing_database_is_visible(self):
        with patch.object(tools, "DB_PATH", Path(self.tmp.name) / "absent.duckdb"):
            app = self.app()
        self.assertFalse(app.exception)
        self.assertIn("not found", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
