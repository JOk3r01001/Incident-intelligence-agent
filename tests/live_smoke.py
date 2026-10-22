"""Opt-in real Streamlit -> Ollama -> DuckDB test. Never used by offline discovery."""
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
    # This test uses the real dataset; its hash must stay unchanged end to end.
    before = hashlib.sha256(agent_tools.DB_PATH.read_bytes()).hexdigest()
    started = monotonic()
    app = AppTest.from_file(str(ROOT / "src" / "app.py"),
                           default_timeout=2 * incident_agent._timeout_seconds() + 20).run()
    app.radio[0].set_value("Explore a date").run()
    assert not app.exception, app.exception
    assert app.selectbox[0].value == "2017-11-24", "Known anomaly is missing."
    assert app.metric[0].value == "R$ 149,916.58", app.metric[0].value
    app.button[0].click().run()
    assert not app.exception, app.exception
    result = app.session_state["investigation"]
    assert result["backend"] == incident_agent.BACKEND, result
    assert not result["error"], result["error"]
    assert app.success, "No completed investigation was rendered."
    # Seeded context alone cannot qualify as an actual model/tool integration.
    selected = [row for row in result["trace"] if row["source"] == "model" and row["status"] == "ok"]
    assert selected, "No successful model-selected tool call."
    metrics = next(row["result"] for row in result["trace"] if row["tool"] == "get_daily_metrics")
    assert metrics["item_sales"] == 149916.58 and metrics["orders"] == 1147, metrics
    for heading in ("### Findings", "### Possible explanations", "### Confidence and next checks"):
        assert heading in result["report"], result["report"]
    assert "<think>" not in result["report"] and "</think>" not in result["report"]
    app.run()
    assert app.success and app.session_state["investigation"] == result, "Report disappeared on rerun."

    # A real follow-up must fetch missing comparison data, return its evidence,
    # preserve the selected anomaly, and remember the discussion on another turn.
    comparison = ("Compare the top three categories by item sales for 2017-11-24 and 2017-11-17. "
                  "Use existing category evidence where available; fetch missing category data with limit=3. "
                  "Explain what this comparison can establish.")
    app.chat_input[0].set_value(comparison).run()
    assert not app.exception, app.exception
    discussion = app.session_state["investigation"]["conversation"]
    assert len(discussion) == 1 and not discussion[0]["error"], discussion
    category_calls = [row for row in discussion[0]["trace"]
                      if row["source"] == "model" and row["status"] == "ok"
                      and row["tool"] == "get_sales_by_category"]
    assert any(row["arguments"]["day"] == "2017-11-17" for row in category_calls), category_calls
    available_categories = [row for row in selected if row["tool"] == "get_sales_by_category"] + category_calls
    assert {row["arguments"]["day"] for row in available_categories} >= {"2017-11-24", "2017-11-17"}, available_categories
    for row in category_calls:
        assert row["result"] == agent_tools.get_sales_by_category(**row["arguments"]), row
    assert app.selectbox[0].value == "2017-11-24"
    assert "### Answer" in discussion[0]["answer"] and "### Database evidence" in discussion[0]["answer"]
    app.chat_input[0].set_value("Could a promotion explain that? What records would test this hypothesis?").run()
    assert not app.exception, app.exception
    discussion = app.session_state["investigation"]["conversation"]
    assert len(discussion) == 2 and not discussion[-1]["error"], discussion
    app.run()
    assert len(app.chat_message) == 4, "Follow-up history disappeared on rerun."
    assert hashlib.sha256(agent_tools.DB_PATH.read_bytes()).hexdigest() == before, "Database changed."
    print(result["report"])
    print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
    print(json.dumps(discussion, ensure_ascii=False, indent=2))
    print(f"PASS: real dashboard/model/tools/report/chat/rerun; {len(selected)} initial model-selected calls; "
          f"{len(category_calls)} follow-up category calls; "
          f"database SHA256 unchanged; {monotonic() - started:.1f}s; model={incident_agent.MODEL}; "
          f"backend={incident_agent.BACKEND}")


if __name__ == "__main__":
    main()
