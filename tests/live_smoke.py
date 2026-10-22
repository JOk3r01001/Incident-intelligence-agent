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
    before = hashlib.sha256(agent_tools.DB_PATH.read_bytes()).hexdigest()
    started = monotonic()
    app = AppTest.from_file(str(ROOT / "src" / "app.py"),
                           default_timeout=2 * incident_agent._timeout_seconds() + 20).run()
    assert not app.exception, app.exception
    assert app.selectbox[0].value == "2017-11-24", "Known anomaly is missing."
    assert app.metric[0].value == "R$ 149,916.58", app.metric[0].value
    app.button[0].click().run()
    assert not app.exception, app.exception
    result = app.session_state["investigation"]
    assert not result["error"], result["error"]
    assert app.success, "No completed investigation was rendered."
    selected = [row for row in result["trace"] if row["source"] == "model" and row["status"] == "ok"]
    assert selected, "No successful model-selected tool call."
    metrics = next(row["result"] for row in result["trace"] if row["tool"] == "get_daily_metrics")
    assert metrics["item_sales"] == 149916.58 and metrics["orders"] == 1147, metrics
    for heading in ("### Findings", "### Possible explanations", "### Confidence and next checks"):
        assert heading in result["report"], result["report"]
    assert "<think>" not in result["report"] and "</think>" not in result["report"]
    app.run()
    assert app.success and app.session_state["investigation"] == result, "Report disappeared on rerun."
    assert hashlib.sha256(agent_tools.DB_PATH.read_bytes()).hexdigest() == before, "Database changed."
    print(result["report"])
    print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
    print(f"PASS: real dashboard/model/tools/report/rerun; {len(selected)} model-selected calls; "
          f"database SHA256 unchanged; {monotonic() - started:.1f}s; model={incident_agent.MODEL}")


if __name__ == "__main__":
    main()
