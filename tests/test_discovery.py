"""Chat-first anomaly discovery, using real temporary DuckDB and fake models."""
from pathlib import Path
import json
from unittest.mock import patch

import duckdb
from streamlit.testing.v1 import AppTest

from test_chat import ANSWER
from test_mvp import DatabaseCase, DAY, ROOT, agent, reply, tools


class DiscoveryCase(DatabaseCase):
    def setUp(self):
        super().setUp()
        # Include rises, drops, and a day with high sales but modest deviation,
        # so percentage severity and absolute sales cannot be conflated.
        with duckdb.connect(str(self.db)) as con:
            con.execute("""INSERT INTO detected_anomalies VALUES
                ('2017-11-10', 40, 20, 100), ('2017-11-17', 20, 40, -50),
                ('2017-11-03', 200, 400, -50)""")
            con.execute("INSERT INTO orders VALUES ('f', 'delivered', '2017-11-03')")
            con.execute("INSERT INTO order_items VALUES ('f', 'p1', 200)")


class DiscoveryToolTests(DiscoveryCase):
    def test_summary_describes_actual_historical_coverage(self):
        result = tools.get_dataset_summary()
        self.assertEqual(result["first_order_date"], "2017-11-03")
        self.assertEqual(result["last_order_date"], DAY)
        self.assertEqual(result["orders"], 6)
        self.assertEqual(result["stored_anomalies"], 4)

    def test_filtering_ranking_pagination_and_inclusive_dates(self):
        result = tools.find_anomalies(limit=2)
        self.assertEqual([row["date"] for row in result["anomalies"]], [DAY, "2017-11-10"])
        self.assertEqual(result["total_matches"], 4)
        self.assertTrue(result["has_more"])
        second = tools.find_anomalies(limit=2, offset=2)
        self.assertEqual([row["date"] for row in second["anomalies"]], ["2017-11-17", "2017-11-03"])
        self.assertFalse(second["has_more"])
        self.assertEqual(tools.find_anomalies(sort_by="sales", limit=1)["anomalies"][0]["date"], "2017-11-03")
        self.assertEqual(tools.find_anomalies(sort_by="date", direction="drops")["anomalies"][0]["date"], "2017-11-17")
        spikes = tools.find_anomalies(start_date="2017-11-10", end_date=DAY, direction="spikes")
        self.assertEqual(spikes["total_matches"], 2)
        one = tools.find_anomalies(start_date="2017-11-17", end_date="2017-11-17")
        self.assertEqual(one["anomalies"][0]["date"], "2017-11-17")

    def test_invalid_filters_and_sql_injection_are_rejected(self):
        for arguments in ({"start_date": "2017-11-24' OR 1=1 --"}, {"end_date": "bad"},
                          {"start_date": DAY, "end_date": "2017-11-01"},
                          {"direction": "DELETE FROM orders"}, {"sort_by": "date; DROP TABLE orders"},
                          {"sort_by": []}, {"limit": True}, {"limit": 11}, {"limit": 0},
                          {"offset": True}, {"offset": -1}, {"offset": 10001}, {"offset": "1"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                tools.find_anomalies(**arguments)
        self.assertEqual(tools.get_daily_metrics(DAY)["orders"], 2)

    def test_no_matches_and_pagination_beyond_end_are_truthful(self):
        result = tools.find_anomalies(start_date="2025-01-01")
        self.assertEqual(result["anomalies"], [])
        self.assertEqual(result["total_matches"], 0)
        self.assertFalse(result["has_more"])
        beyond = tools.find_anomalies(offset=100)
        self.assertEqual(beyond["anomalies"], [])
        self.assertEqual(beyond["total_matches"], 4)
        self.assertFalse(beyond["has_more"])

    def test_discovery_queries_leave_database_unchanged(self):
        before = self.db.read_bytes()
        tools.find_anomalies(direction="drops")
        tools.get_dataset_summary()
        self.assertEqual(self.db.read_bytes(), before)


class DiscoveryAgentTests(DiscoveryCase):
    def test_model_selects_search_and_actual_rows_are_rendered(self):
        self.mock_client([reply(calls=[("find_anomalies", {"direction": "spikes", "limit": 2})]), reply(ANSWER)])
        answer, trace = agent.ask_agent("Find the biggest sales spikes.")
        self.assertEqual(trace[0]["tool"], "get_dataset_summary")
        self.assertEqual(trace[0]["source"], "seeded")
        self.assertEqual(trace[1]["tool"], "find_anomalies")
        self.assertEqual(trace[1]["source"], "model")
        self.assertIn("| 2017-11-24 | 150.00 | 30.00 | +400.00% |", answer)
        self.assertIn("2 matching stored flags", answer)
        self.assertNotIn("2017-11-24", self.requests[0]["messages"][0]["content"])
        returned = next(m for m in self.requests[-1]["messages"] if m["role"] == "tool")
        self.assertEqual(json.loads(returned["content"])["anomalies"][0]["date"], DAY)

    def test_model_can_discover_then_investigate_without_user_providing_date(self):
        self.mock_client([
            reply(calls=[("find_anomalies", {"direction": "spikes", "sort_by": "sales", "limit": 1})]),
            reply(calls=[("get_daily_metrics", {"day": DAY}), ("get_weekday_history", {"day": DAY}),
                         ("get_sales_by_category", {"day": DAY, "limit": 1})]), reply(ANSWER),
        ])
        answer, trace = agent.ask_agent("Find and investigate the spike with the highest item sales.")
        self.assertIn("R$150.00 item sales; 2 distinct delivered orders", answer)
        self.assertIn("books: R$100.00", answer)
        self.assertEqual([entry["tool"] for entry in trace[1:]],
                         ["find_anomalies", "get_daily_metrics", "get_weekday_history", "get_sales_by_category"])
        self.assertIn("current R$75.00; prior matching weekdays weighted AOV R$86.67; current AOV is lower", answer)
        self.assertIn('"average_order_value_is": "lower"', "\n".join(m["content"] for m in self.requests[-1]["messages"]))

    def test_discovery_requires_real_search_not_a_seed_or_fabricated_prose(self):
        self.mock_client([reply(ANSWER), reply(ANSWER)])
        with self.assertRaises(agent.InvestigationError) as caught:
            agent.ask_agent("Find anomalies", max_rounds=2)
        self.assertEqual(len(caught.exception.trace), 1)
        self.assertIn("Call find_anomalies", self.requests[1]["messages"][-1]["content"])

    def test_format_repair_reuses_evidence_without_repeating_tools(self):
        self.mock_client([reply(calls=[("find_anomalies", {"limit": 1})]),
                          reply("### Findings\nA stored spike exists.\n### Next checks\nCampaign records are unknown."),
                          reply(ANSWER)])
        answer, trace = agent.ask_agent("Find anomalies")
        self.assertIn("| 2017-11-24 |", answer)
        self.assertEqual(len(trace), 2)
        self.assertIn("Rewrite the answer", self.requests[-1]["messages"][-1]["content"])

    def test_persistent_bad_format_fails_after_one_repair_attempt(self):
        self.mock_client([reply(calls=[("find_anomalies", {"limit": 1})]),
                          reply("Wrong headings"), reply("Still wrong")])
        with self.assertRaisesRegex(agent.InvestigationError, "required report sections") as caught:
            agent.ask_agent("Find anomalies")
        self.assertEqual(len(caught.exception.trace), 2)
        self.assertEqual(len(self.requests), 3)

    def test_followup_resolves_saved_results_and_explanations_need_no_new_call(self):
        self.mock_client([reply(calls=[("find_anomalies", {"direction": "spikes", "limit": 1})]), reply(ANSWER)])
        answer, trace = agent.ask_agent("Find the largest spike")
        history = [{"question": "Find the largest spike", "answer": answer, "trace": trace}]
        self.mock_client([reply(calls=[("get_daily_metrics", {"day": DAY})]), reply(ANSWER)])
        answer2, trace2 = agent.ask_agent("Investigate the first one", history=history)
        self.assertEqual(trace2[-1]["arguments"]["day"], DAY)
        self.assertIn('"anomalies"', self.requests[0]["messages"][1]["content"])
        self.assertIn('"2017-11-24"', self.requests[0]["messages"][1]["content"])
        history.append({"question": "Investigate the first one", "answer": answer2, "trace": trace2})
        self.mock_client([reply(ANSWER)])
        _, explanation_trace = agent.ask_agent("Could a promotion explain this?", history=history)
        self.assertEqual(len(explanation_trace), 1)

    def test_invalid_search_can_be_corrected_and_no_matching_dates_is_valid(self):
        self.mock_client([reply(calls=[("find_anomalies", {"limit": 1000})]),
                          reply(calls=[("find_anomalies", {"start_date": "2025-01-01"})]), reply(ANSWER)])
        answer, trace = agent.ask_agent("Find anomalies in 2025")
        self.assertEqual([entry["status"] for entry in trace[1:]], ["error", "ok"])
        self.assertIn("No stored anomalies match these filters", answer)

    def test_latest_search_remains_evidence_after_its_text_ages_out(self):
        search = {"source": "model", "status": "ok", "tool": "find_anomalies", "arguments": {"limit": 1},
                  "result": tools.find_anomalies(limit=1)}
        history = [{"question": "Find the largest spike", "answer": ANSWER, "trace": [search]},
                   {"question": "Explain it", "answer": ANSWER + "x" * (agent.MAX_HISTORY_CHARS - 100 - len(ANSWER)),
                    "trace": [{"source": "model", "status": "ok", "tool": "get_daily_metrics",
                               "arguments": {"day": DAY}, "result": tools.get_daily_metrics(DAY)}]}]
        self.mock_client([reply(ANSWER)])
        _, trace = agent.ask_agent("What evidence would test this?", history=history)
        self.assertEqual(len(trace), 1)
        self.assertIn('"anomalies"', self.requests[0]["messages"][1]["content"])
        self.assertFalse(any(m.get("content") == "Find the largest spike" for m in self.requests[0]["messages"]))

    def test_invalid_input_and_missing_database_never_contact_model(self):
        client = self.mock_client([])
        for question in (None, " ", "x" * 1001):
            with self.assertRaises(agent.InvestigationError):
                agent.ask_agent(question)
        with self.assertRaises(agent.InvestigationError):
            agent.ask_agent("Find anomalies", max_rounds=True)
        with patch.object(tools, "DB_PATH", Path(self.tmp.name) / "absent.duckdb"):
            with self.assertRaisesRegex(agent.InvestigationError, "not found"):
                agent.ask_agent("Find anomalies")
        client.chat.assert_not_called()


class DiscoveryDashboardTests(DiscoveryCase):
    def app(self):
        return AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=15).run()

    def test_default_workspace_accepts_discovery_without_selecting_an_anomaly(self):
        self.mock_client([reply(calls=[("find_anomalies", {"direction": "spikes", "limit": 2})]), reply(ANSWER)])
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].value, "Ask the agent")
        self.assertFalse(app.selectbox)
        self.assertEqual(len(app.chat_input), 1)
        self.assertEqual(len(self.requests), 0)
        app.chat_input[0].set_value("Find the biggest sales spikes").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.selectbox)
        exchange = app.session_state["agent_workspace"]["conversation"][0]
        self.assertIsNone(exchange["error"])
        self.assertIn("| 2017-11-24 |", exchange["answer"])
        self.assertIn("1 model-selected calls", app.expander[0].label)
        app.run()
        self.assertEqual(len(app.chat_message), 2)
        self.assertEqual(len(self.requests), 2)

    def test_followup_and_new_conversation_work_without_date_widgets(self):
        self.mock_client([reply(calls=[("find_anomalies", {"direction": "spikes", "limit": 1})]), reply(ANSWER),
                          reply(calls=[("get_daily_metrics", {"day": DAY})]), reply(ANSWER)])
        app = self.app()
        app.chat_input[0].set_value("Find the largest spike").run()
        app.chat_input[0].set_value("Investigate the first one").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 4)
        self.assertFalse(app.selectbox)
        app.radio[0].set_value("Explore a date").run()
        app.radio[0].set_value("Ask the agent").run()
        self.assertEqual(len(app.chat_message), 4)
        app.button[0].click().run()
        self.assertEqual(app.session_state["agent_workspace"]["conversation"], [])
        self.assertFalse(app.chat_message)

    def test_empty_anomaly_table_does_not_block_discovery_chat(self):
        with duckdb.connect(str(self.db)) as con:
            con.execute("DELETE FROM detected_anomalies")
        self.mock_client([reply(calls=[("find_anomalies", {})]), reply(ANSWER)])
        app = self.app()
        self.assertEqual(len(app.chat_input), 1)
        app.chat_input[0].set_value("Find unusual days").run()
        self.assertFalse(app.exception)
        answer = app.session_state["agent_workspace"]["conversation"][0]["answer"]
        self.assertIn("No stored anomalies match", answer)

    def test_model_failure_stays_visible_with_trace(self):
        self.mock_client([ConnectionError("offline")])
        app = self.app()
        app.chat_input[0].set_value("Find anomalies").run()
        self.assertFalse(app.exception)
        self.assertIn("Agent reply failed", app.error[0].value)
        exchange = app.session_state["agent_workspace"]["conversation"][0]
        self.assertIsNone(exchange["answer"])
        self.assertEqual(exchange["trace"][0]["source"], "seeded")
        app.run()
        self.assertIn("ollama serve", app.error[0].value)
