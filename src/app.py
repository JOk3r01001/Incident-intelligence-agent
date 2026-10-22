"""Minimal Streamlit interface for the Incident Intelligence MVP."""
from time import monotonic

import pandas as pd
import streamlit as st

from agent_tools import get_dataset_summary, get_sales_window, list_anomalies
from incident_agent import BACKEND, MODEL, MAX_QUESTION_CHARS, TOOLS, InvestigationError, ask_agent, discuss, investigate


def render_graphs(trace: list[dict]) -> None:
    """Render only graphs actually produced by the validated Python tool.

    Neither generated prose nor model-supplied code/specifications are rendered
    as charts. The result data lives in session state with the reply's audit log.
    """
    for entry in trace:
        if entry["source"] != "model" or entry["status"] != "ok" or entry["tool"] != "create_graph":
            continue
        graph = entry["result"]
        st.markdown(f"**{graph['title']}**")
        if not graph["data"]:
            st.info(graph["note"])
            continue
        df = pd.DataFrame(graph["data"])
        if graph["chart_type"] == "line":
            df[graph["x"]] = pd.to_datetime(df[graph["x"]])
            st.line_chart(df.set_index(graph["x"])[graph["y"]], y_label=graph["y_label"])
        else:
            st.bar_chart(df.set_index(graph["x"])[graph["y"]], y_label=graph["y_label"])
        st.caption(graph["note"] + (" Additional categories/statuses were omitted by the requested limit."
                                    if graph["truncated"] else ""))


def render_agent_guide() -> None:
    """Keep the in-app tool list synchronized with the execution allowlist."""
    with st.expander("About the agent and its tools"):
        st.write("Ask for unusual sales days, an investigation, comparisons, or a graph. "
                 "A local Qwen model chooses analytical tools; Python validates the arguments, "
                 "queries DuckDB read-only, and displays the evidence, graph, and tool audit trail.")
        st.write("Anomalies are stored flags from a detector comparing each day with the median "
                 "of four prior matching weekdays (at least 30% deviation). The model interprets "
                 "the evidence; suggested causes remain hypotheses. Sales are item prices excluding freight.")
        st.table(pd.DataFrame([{"Tool": name, "Purpose": (fn.__doc__ or "").strip().splitlines()[0]}
                               for name, fn in TOOLS.items()]))
        st.write("Graphs support daily sales, delivered orders, average order value, category sales, "
                 "and final order statuses, for up to 90 days and 15 categories/statuses. "
                 "Try: ‘Plot daily sales for November 2017’ or ‘Graph the top categories for that day.’")
        st.caption("Dataset coverage/counts are seeded context; source=model identifies tools selected "
                   "by Qwen. Conversation memory is limited and session-only. LangChain orchestrates "
                   "the local model and validated tools. Campaign causes, profit, "
                   "delivery delays, and live incidents cannot be established from these tools alone.")


def render_conversation(conversation: list[dict], error_label: str = "Follow-up failed") -> None:
    """Render session-owned messages and the evidence for each model reply."""
    for exchange in conversation:
        with st.chat_message("user"):
            st.markdown(exchange["question"])
        with st.chat_message("assistant"):
            if exchange["error"]:
                st.error(f"{error_label}: {exchange['error']}")
            else:
                st.caption("Model interpretation — check it against the database evidence below.")
                st.markdown(exchange["answer"])
            st.caption(f"{exchange['elapsed_seconds']:.1f}s · {exchange['model']} via {exchange['backend']}")
            trace = exchange["trace"]
            render_graphs(trace)
            calls = sum(row["source"] == "model" for row in trace)
            seeds = sum(row["source"] == "seeded" for row in trace)
            with st.expander(f"Reply evidence ({seeds} seeded lookups, {calls} model-selected calls)"):
                st.json(trace)


def agent_workspace() -> None:
    """Let the model discover and investigate dates through conversation."""
    st.subheader("Ask the agent")
    st.write("Ask me to find unusual sales days, investigate a spike, compare anomalies, or create a graph.")
    st.caption('Try: “Find the five biggest sales spikes in November 2017” → '
               '“Investigate the one with the highest item sales” → “Could a promotion explain it?”')
    try:
        summary = get_dataset_summary()
    except Exception as exc:
        st.error(f"Could not open the dataset: {exc}")
        st.info("Run `python src/prepare_data.py` to prepare an absent database from the Olist CSVs. "
                "Existing databases are checked read-only. See README.md.")
        st.stop()
    st.caption(f"Historical coverage: {summary['first_order_date']} to {summary['last_order_date']} · "
               f"{summary['stored_anomalies']:,} stored statistical flags · Item sales exclude freight.")
    workspace = st.session_state.get("agent_workspace")
    if not workspace or (workspace["model"], workspace["backend"]) != (MODEL, BACKEND):
        # A new model/backend gets a separate conversation; never silently mix provenance.
        workspace = {"model": MODEL, "backend": BACKEND, "conversation": []}
        st.session_state["agent_workspace"] = workspace
    conversation = workspace["conversation"]
    if st.button("New conversation", key="new_agent_conversation"):
        conversation.clear()
    question = st.chat_input("Ask me to find or investigate anomalies…", max_chars=MAX_QUESTION_CHARS,
                             key="agent_question")
    if question:
        exchange = {"question": question, "answer": None, "trace": [], "error": None,
                    "model": MODEL, "backend": BACKEND}
        started = monotonic()
        with st.spinner("Searching and checking the historical evidence..."):
            try:
                answer, trace = ask_agent(question, history=conversation)
                exchange.update(answer=answer, trace=trace)
            except InvestigationError as exc:
                exchange.update(error=str(exc), trace=exc.trace)
            except Exception as exc:
                exchange["error"] = f"Unexpected agent error ({type(exc).__name__}): {exc}"
            finally:
                exchange["elapsed_seconds"] = monotonic() - started
        conversation.append(exchange)
        del conversation[:-12]
    render_conversation(conversation, error_label="Agent reply failed")
    render_agent_guide()
    st.caption("Historical Olist data · Statistical anomalies are not automatically business incidents.")

st.set_page_config(page_title="Incident Intelligence", layout="wide")
st.title("Incident Intelligence Agent")
st.caption(f"Local anomaly investigation · DuckDB + LangChain + Ollama ({MODEL}; {BACKEND})")

# Conversation is the default entry point. The original dashboard remains an
# optional workspace for inspecting a known date, rather than a prerequisite.
workspace_mode = st.radio("Workspace", ["Ask the agent", "Explore a date"], horizontal=True)
if workspace_mode == "Ask the agent":
    agent_workspace()
    st.stop()

try:
    rows = list_anomalies()
except Exception as exc:
    st.error(f"Could not load anomalies: {exc}")
    st.info("Run `python src/prepare_data.py` to prepare an absent database from CSVs or check an existing database read-only. For an incomplete database, restore a prepared copy or build to a new path. See README.md.")
    st.stop()

if not rows:
    st.warning("No saved anomalies. A valid dataset can have no deviations above the detector threshold. Run `python src/prepare_data.py` to inspect row counts; see README.md to prepare a separate database from CSVs.")
    st.stop()

# Store actual dates as widget values; formatted labels are only for display.
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
    # Streamlit reruns the script on widget changes. Persist success or failure,
    # together with the model/backend that produced it, until another investigation.
    st.session_state["investigation"] = {
        "date": selected, "model": MODEL, "backend": BACKEND, "report": None, "trace": [], "error": None,
        "conversation": [],
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
# Never show a saved report under a different selected anomaly.
if result and result["date"] == selected:
    if result["error"]:
        st.error(f"Investigation failed: {result['error']}")
    elif result["report"]:
        st.success(f"Investigation completed for {selected} in {result.get('elapsed_seconds', 0):.1f}s "
                   f"with {result.get('model', MODEL)} via {result.get('backend', 'unknown')}.")
        st.subheader("Investigation report")
        st.markdown(result["report"])
    trace = result["trace"]
    render_graphs(trace)
    # Seeded lookups provide context; only source='model' denotes model tool choices.
    seeded = sum(row["source"] == "seeded" for row in trace)
    model_calls = sum(row["source"] == "model" for row in trace)
    with st.expander(f"Tool audit trail ({seeded} seeded lookups, {model_calls} model-selected calls)"):
        st.json(trace)

    if result["report"] and not result["error"]:
        st.subheader("Discuss this investigation")
        st.caption("Ask about the findings, suggest a hypothesis, or request another comparison. "
                   "The agent can check more data; your suggestions remain hypotheses until supported by evidence.")
        conversation = result.setdefault("conversation", [])
        if st.button("Clear conversation", key=f"clear_chat_{selected}"):
            conversation.clear()

        same_model = result.get("model") == MODEL and result.get("backend") == BACKEND
        if not same_model:
            st.info("Run the investigation again with the current model/backend to start a new conversation.")
        question = st.chat_input("Ask a follow-up about this anomaly…", max_chars=MAX_QUESTION_CHARS,
                                 key=f"chat_{selected}", disabled=not same_model)
        if question and same_model:
            exchange = {"question": question, "answer": None, "trace": [], "error": None,
                        "model": MODEL, "backend": BACKEND}
            started = monotonic()
            with st.spinner("Checking your question against the investigation evidence..."):
                try:
                    answer, followup_trace = discuss(selected, question,
                                                     investigation_trace=result["trace"], history=conversation)
                    exchange.update(answer=answer, trace=followup_trace)
                except InvestigationError as exc:
                    exchange.update(error=str(exc), trace=exc.trace)
                except Exception as exc:
                    exchange["error"] = f"Unexpected chat error ({type(exc).__name__}): {exc}"
                finally:
                    exchange["elapsed_seconds"] = monotonic() - started
            conversation.append(exchange)
            # Bound session memory as well as the smaller context sent to Qwen.
            del conversation[:-12]

        render_conversation(conversation)

with st.expander("Anomaly records"):
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

render_agent_guide()
st.caption("Historical Olist data · Indicative item sales, not recognized revenue · Statistical anomalies are not automatically incidents.")
