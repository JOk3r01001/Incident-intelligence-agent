"""Minimal Streamlit interface for the Incident Intelligence MVP."""
from time import monotonic

import pandas as pd
import streamlit as st

from agent_tools import get_sales_window, list_anomalies
from incident_agent import MODEL, InvestigationError, investigate

st.set_page_config(page_title="Incident Intelligence", page_icon="🔎", layout="wide")
st.title("🔎 Incident Intelligence Agent")
st.caption(f"Local anomaly investigation · DuckDB + Ollama ({MODEL})")

try:
    rows = list_anomalies()
except Exception as exc:
    st.error(f"Could not load anomalies: {exc}")
    st.info("Restore the prepared database if absent. If only anomalies are missing, run `python src/detect_anomalies.py` from the project root. See README-MVP.md before importing data.")
    st.stop()

if not rows:
    st.warning("No saved anomalies. Run `python src/detect_anomalies.py` first.")
    st.stop()

by_day = {r["date"]: r for r in rows}
dates = list(by_day)
preferred = "2017-11-24"
initial = dates.index(preferred) if preferred in dates else 0
selected = st.selectbox(
    "Select an anomaly",
    dates,
    index=initial,
    format_func=lambda d: f"{d}   ({by_day[d]['deviation_pct']:+.1f}%)",
)

anomaly = by_day[selected]
c1, c2, c3 = st.columns(3)
c1.metric("Actual item sales", f"R$ {anomaly['actual_sales']:,.2f}")
c2.metric("Expected (detector)", f"R$ {anomaly['expected_sales']:,.2f}")
c3.metric("Deviation", f"{anomaly['deviation_pct']:+.1f}%")

try:
    series = get_sales_window(selected)
except Exception as exc:
    st.error(f"Could not load the sales chart: {exc}")
    series = []
if series:
    st.subheader("Sales near the anomaly")
    df = pd.DataFrame(series)
    df["date"] = pd.to_datetime(df["date"])
    st.line_chart(df.set_index("date")["item_sales"], y_label="Item sales (R$)")

if st.button("Investigate with local AI", type="primary"):
    st.session_state["investigation"] = {
        "date": selected, "model": MODEL, "report": None, "trace": [], "error": None,
    }
    started = monotonic()
    with st.spinner("Analyzing database evidence with the local model..."):
        try:
            report, trace = investigate(selected)
            st.session_state["investigation"].update(report=report, trace=trace)
        except InvestigationError as exc:
            st.session_state["investigation"].update(error=str(exc), trace=exc.trace)
        except Exception as exc:
            st.session_state["investigation"]["error"] = f"Unexpected agent error ({type(exc).__name__}): {exc}"
        finally:
            st.session_state["investigation"]["elapsed_seconds"] = monotonic() - started

result = st.session_state.get("investigation")
if result and result["date"] == selected:
    if result["error"]:
        st.error(f"Investigation failed: {result['error']}")
    elif result["report"]:
        st.success(f"Investigation completed for {selected} in {result.get('elapsed_seconds', 0):.1f}s "
                   f"with {result.get('model', MODEL)}.")
        st.subheader("Investigation report")
        st.markdown(result["report"])
    trace = result["trace"]
    seeded = sum(row["source"] == "seeded" for row in trace)
    model_calls = sum(row["source"] == "model" for row in trace)
    with st.expander(f"Tool audit trail ({seeded} seeded lookups, {model_calls} model-selected calls)"):
        st.json(trace)

with st.expander("Anomaly records"):
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.caption("Historical Olist data · Indicative item sales, not recognized revenue · Statistical anomalies are not automatically incidents.")
