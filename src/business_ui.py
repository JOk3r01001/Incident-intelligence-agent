"""Streamlit view for configured business data and Python-rendered evidence."""
from time import monotonic

import pandas as pd
import streamlit as st

from business_agent import ask_business
from business_tools import BusinessAnalytics, model_tools
from dataset_config import DatasetProfile
from incident_agent import BACKEND, MODEL, MAX_QUESTION_CHARS, InvestigationError


def render_business_evidence(trace):
    """Quantitative tables always come from tools, independently of model prose."""
    for row in trace:
        if row["source"] != "model" or row["status"] != "ok":
            continue
        result, name = row["result"], row["tool"]
        if name == "query_metric":
            st.markdown(f"**Database evidence: {result['label']} ({result['unit']})**")
            st.caption(f"{result['start_date']} to {result['end_date']}")
            st.dataframe(pd.DataFrame(result["rows"]), hide_index=True)
            st.caption(f"Returned records: {result['quality']['records']}; "
                       f"records missing calculation values: {result['quality']['missing_values']}.")
            st.caption(result["note"] + (" Additional groups were omitted." if result["truncated"] else ""))
        elif name == "compare_metric_periods":
            st.markdown(f"**Period comparison: {result['label']} ({result['unit']})**")
            st.table(pd.DataFrame([{"period": label, **result[key]} for label, key in
                                  (("Current", "current"), ("Baseline", "baseline"))]))
            st.caption(f"Calculated difference: {result['difference']} {result['unit']}; "
                       f"change: {result['change_pct']}%. " + result["note"])
        elif name == "find_metric_anomalies":
            st.markdown(f"**Unusual observations: {result['label']} ({result['unit']})**")
            if result["anomalies"]:
                st.dataframe(pd.DataFrame(result["anomalies"]), hide_index=True)
            else:
                st.info("No assessed observations exceeded the requested threshold.")
            st.caption(f"{result['total_matches']} matching observations; "
                       f"{result['unassessed_observed_days']} observed days could not be assessed. "
                       + result["note"] + (" Additional matches were omitted." if result["truncated"] else ""))
        elif name == "describe_business_dataset":
            st.table(pd.DataFrame([{"metric": metric, "unit": item["unit"], "definition": item["description"]}
                                   for metric, item in result["metrics"].items()]))


def business_workspace(config_path, render_graphs):
    try:
        profile = DatasetProfile.load(config_path)
        analytics = BusinessAnalytics(profile)
        catalogue = analytics.describe()
    except Exception as exc:
        st.error(f"Could not open the configured dataset: {exc}")
        st.info("Check INCIDENT_DATASET_CONFIG, the referenced local DuckDB file, and its metric/table definitions. "
                "See README.md. Existing data is never overwritten by configuration.")
        st.stop()
    st.subheader(profile.definition.name)
    st.write(profile.definition.description)
    st.caption("Ask for a configured metric, a period comparison, unusual days, or a graph. "
               "No anomaly selection is required.")
    default = catalogue["metrics"][catalogue["default_metric"]]
    st.caption(f"Default metric: {catalogue['default_metric']} ({default['unit']}) · "
               f"Coverage: {default['first_date']} to {default['last_date']}")
    with st.expander("Dataset and metric definitions"):
        st.table(pd.DataFrame([{"metric": name, "unit": item["unit"], "aggregation": item["aggregation"],
                               "dimensions": ", ".join(item["dimensions"]), "definition": item["description"]}
                              for name, item in catalogue["metrics"].items()]))
        st.caption("Definitions are configured by the operator. The agent cannot change formulas, joins, filters, or units.")
        st.json(profile.definition.model_dump(mode="json", exclude={"database"}))
    identity = (profile.identity, MODEL, BACKEND)
    workspace = st.session_state.get("business_workspace")
    if not workspace or workspace["identity"] != identity:
        workspace = {"identity": identity, "conversation": []}
        st.session_state["business_workspace"] = workspace
    conversation = workspace["conversation"]
    if st.button("New conversation", key="new_business_conversation"):
        conversation.clear()
    question = st.chat_input("Ask about this business dataset…", max_chars=MAX_QUESTION_CHARS, key="business_question")
    if question:
        exchange = {"question": question, "answer": None, "trace": [], "error": None,
                    "model": MODEL, "backend": BACKEND, "dataset_identity": profile.identity}
        started = monotonic()
        with st.spinner("Checking the configured business data..."):
            try:
                reply = ask_business(question, analytics, history=conversation)
                exchange.update(answer=reply.answer, trace=reply.trace, model_commentary=reply.model_commentary)
            except InvestigationError as exc:
                exchange.update(error=str(exc), trace=exc.trace)
            except Exception as exc:
                exchange["error"] = f"Unexpected business agent error ({type(exc).__name__}): {exc}"
            finally:
                exchange["elapsed_seconds"] = monotonic() - started
        conversation.append(exchange)
        del conversation[:-12]
    for exchange in conversation:
        with st.chat_message("user"):
            st.markdown(exchange["question"])
        with st.chat_message("assistant"):
            if exchange["error"]:
                st.error(f"Business reply failed: {exchange['error']}")
            else:
                st.caption("Agent clarification — no calculated report yet." if exchange["answer"].startswith("### Clarification")
                           else "Verified answer — generated by Python from actual tool results.")
                st.markdown(exchange["answer"])
            st.caption(f"{exchange['elapsed_seconds']:.1f}s · {exchange['model']} via {exchange['backend']}")
            render_business_evidence(exchange["trace"])
            render_graphs(exchange["trace"])
            if exchange.get("model_commentary"):
                with st.expander("Model discussion — unverified"):
                    st.caption("Generated commentary may contradict observations or introduce unsupported explanations. "
                               "Use the verified answer and numerical tables above to assess it.")
                    st.markdown(exchange["model_commentary"])
            seeds = sum(row["source"] == "seeded" for row in exchange["trace"])
            calls = sum(row["source"] == "model" for row in exchange["trace"])
            with st.expander(f"Business evidence ({seeds} seeded lookups, {calls} model-selected calls)"):
                st.json(exchange["trace"])
    with st.expander("Business tools and limits"):
        st.table(pd.DataFrame([{"tool": name, "purpose": fn.__doc__.strip().splitlines()[0]}
                              for name, fn in model_tools(analytics).items()]))
        st.write("Read-only, parameterized SQL. Totals/comparisons/anomaly requests cover at most 366 days; "
                 "daily series and charts cover at most 90 days. Dimension results contain at most 15 groups. "
                 "Each SQL statement has a 10-second timeout. Missing values and non-positive anomaly baselines "
                 "remain unknown. Conversations live only in this session and reset when definitions or data change.")
