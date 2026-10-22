"""Opt-in real chat-first discovery: no anomaly widget or mocked model responses."""
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import agent_tools
import incident_agent


def main():
    before = hashlib.sha256(agent_tools.DB_PATH.read_bytes()).hexdigest()
    started = monotonic()
    app = AppTest.from_file(str(ROOT / "src" / "app.py"),
                           default_timeout=2 * incident_agent._timeout_seconds() + 20).run()
    assert not app.exception, app.exception
    assert app.radio[0].value == "Ask the agent" and not app.selectbox
    app.chat_input[0].set_value(
        "Find the three sales spikes with highest item sales between 2017-11-01 and 2017-11-30. "
        "Show the results ranked by actual item sales."
    ).run()
    assert not app.exception, app.exception
    conversation = app.session_state["agent_workspace"]["conversation"]
    assert len(conversation) == 1 and not conversation[0]["error"], conversation
    searches = [row for row in conversation[0]["trace"] if row["source"] == "model"
                and row["status"] == "ok" and row["tool"] == "find_anomalies"]
    assert searches, conversation[0]
    search = searches[-1]
    assert search["result"] == agent_tools.find_anomalies(**search["arguments"])
    assert search["result"]["direction"] == "spikes" and search["result"]["sort_by"] == "sales", search
    assert search["result"]["start_date"] == "2017-11-01", search
    assert search["result"]["end_date"] == "2017-11-30", search
    assert len(search["result"]["anomalies"]) == 3, search
    discovered = search["result"]["anomalies"][0]["date"]
    assert discovered == "2017-11-24", search
    assert f"| {discovered} | 149,916.58 |" in conversation[0]["answer"]

    # The user supplies a reference to the returned list, not an anomaly date.
    app.chat_input[0].set_value(
        "Investigate the first anomaly in that list. Check its daily metrics, previous matching weekdays, "
        "and top three categories, then explain the findings and unknowns."
    ).run()
    assert not app.exception, app.exception
    conversation = app.session_state["agent_workspace"]["conversation"]
    assert len(conversation) == 2 and not conversation[-1]["error"], conversation
    calls = [row for row in conversation[-1]["trace"] if row["source"] == "model" and row["status"] == "ok"]
    for name in ("get_daily_metrics", "get_weekday_history", "get_sales_by_category"):
        assert any(row["tool"] == name and row["arguments"].get("day") == discovered for row in calls), calls
    metrics = next(row["result"] for row in calls if row["tool"] == "get_daily_metrics")
    assert metrics["item_sales"] == 149916.58 and metrics["orders"] == 1147, metrics
    assert "R$149,916.58 item sales; 1,147 distinct delivered orders" in conversation[-1]["answer"]

    app.chat_input[0].set_value("Could a promotion explain that? What evidence would test the hypothesis?").run()
    assert not app.exception, app.exception
    conversation = app.session_state["agent_workspace"]["conversation"]
    assert len(conversation) == 3 and not conversation[-1]["error"], conversation
    saved = json.dumps(conversation, ensure_ascii=False)
    app.run()
    assert len(app.chat_message) == 6 and not app.selectbox
    assert json.dumps(app.session_state["agent_workspace"]["conversation"], ensure_ascii=False) == saved
    assert hashlib.sha256(agent_tools.DB_PATH.read_bytes()).hexdigest() == before, "Database changed."
    print(json.dumps(conversation, ensure_ascii=False, indent=2))
    print(f"PASS: chat-first discovery/investigation/follow-up/rerun; discovered={discovered}; "
          f"no manual anomaly selection; database SHA256 unchanged; {monotonic() - started:.1f}s; "
          f"model={incident_agent.MODEL}; backend={incident_agent.BACKEND}")


if __name__ == "__main__":
    main()
