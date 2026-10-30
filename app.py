"""Streamlit front end for the ML-based ITDR prototype.

Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from itdr.data_gen import generate_events
from itdr.detection import DEFAULT_WEIGHTS, ITDRPipeline, evaluate
from itdr.explain import build_alert
from itdr.features import build_features
from itdr.response import audit_entry, audit_to_jsonl, decide, execute

st.set_page_config(page_title="ML-based ITDR Prototype", page_icon="🛡️", layout="wide")

# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Generating events, engineering features, training models...")
def build(seed: int, n_users: int, days: int):
    events = generate_events(n_users=n_users, days=days, seed=seed)
    feats = build_features(events)
    split = int(days * 0.7)
    pipe = ITDRPipeline(seed=seed).fit(feats, split)
    return events, feats, pipe, split


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ ITDR Prototype")
st.sidebar.caption("Machine Learning-Based Identity Threat Detection and Response")
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)
n_users = st.sidebar.slider("Users", 30, 200, 60, 10)
days = st.sidebar.slider("Days of logs", 10, 30, 14)

st.sidebar.subheader("Risk score weights")
w_rules = st.sidebar.slider("Rules", 0.0, 1.0, DEFAULT_WEIGHTS["rules"], 0.05)
w_anom = st.sidebar.slider("Anomaly (Isolation Forest)", 0.0, 1.0, DEFAULT_WEIGHTS["anomaly"], 0.05)
w_sup = st.sidebar.slider("Supervised (Logistic Regression)", 0.0, 1.0, DEFAULT_WEIGHTS["supervised"], 0.05)
medium = st.sidebar.slider("Medium-risk threshold", 0.05, 0.9, 0.35, 0.05)
high = st.sidebar.slider("High-risk threshold", 0.1, 0.99, 0.60, 0.05)
dry_run = st.sidebar.toggle("Dry-run mode (recommended)", value=True,
                            help="No real action is executed. Wire execute() in itdr/response.py to an IdP to go live.")
analyst = st.sidebar.text_input("Analyst name", "analyst1")

events, feats, pipe, split = build(int(seed), int(n_users), int(days))
scored = pipe.score(feats, {"rules": w_rules, "anomaly": w_anom, "supervised": w_sup}, medium, high)

if "audit" not in st.session_state:
    st.session_state.audit = []
if "handled" not in st.session_state:
    st.session_state.handled = {}

# --------------------------------------------------------------------------- header
st.title("Machine Learning-Based Identity Threat Detection and Response")
st.caption("Pipeline: identity sources → normalisation → features → rules + ML scoring → explainable alert "
           "→ policy-based response → analyst review → audit. Data is synthetic.")

tab_over, tab_alerts, tab_eval, tab_audit, tab_data = st.tabs(
    ["Overview", "Alert queue & investigation", "Evaluation", "Audit log", "Data & features"])

# --------------------------------------------------------------------------- overview
with tab_over:
    c = st.columns(5)
    c[0].metric("Events", f"{len(events):,}")
    c[1].metric("User-day windows", f"{len(scored):,}")
    c[2].metric("High", int((scored.tier == "high").sum()))
    c[3].metric("Medium", int((scored.tier == "medium").sum()))
    c[4].metric("Low", int((scored.tier == "low").sum()))
    left, right = st.columns(2)
    with left:
        st.subheader("Alerts per day (medium + high)")
        per_day = (scored[scored.tier != "low"].groupby(["date", "tier"]).size()
                   .unstack(fill_value=0).reindex(columns=["medium", "high"], fill_value=0))
        st.bar_chart(per_day)
    with right:
        st.subheader("Risk score distribution")
        st.bar_chart(pd.cut(scored["risk"], bins=10).value_counts().sort_index().rename(lambda i: f"{i.left:.1f}-{i.right:.1f}"))
    if not pipe.sup_ok:
        st.warning("Too few labelled attacks in the training period: supervised component is disabled.")
    st.info("The model score is decision support. High-impact actions need corroborating evidence and "
            "analyst approval (see the response policy in itdr/response.py).")

# --------------------------------------------------------------------------- alerts
with tab_alerts:
    queue = scored[scored.tier != "low"].sort_values("risk", ascending=False)
    if queue.empty:
        st.success("No medium or high alerts with the current thresholds.")
    else:
        st.subheader(f"Alert queue ({len(queue)})")
        show = queue[["alert_id", "user_id", "role", "date", "tier", "risk", "rule_score",
                      "anomaly_score", "supervised_score"]].copy()
        show["status"] = show["alert_id"].map(lambda a: st.session_state.handled.get(a, "open"))
        st.dataframe(show.round(2), hide_index=True, use_container_width=True)

        sel = st.selectbox("Open an alert", queue["alert_id"].tolist(),
                           format_func=lambda a: f"{a}  ·  {queue.set_index('alert_id').loc[a, 'tier'].upper()}  ·  "
                                                 f"risk {queue.set_index('alert_id').loc[a, 'risk']:.2f}")
        row = queue.set_index("alert_id", drop=False).loc[sel]
        alert = build_alert(row, pipe, events)
        decision = decide(row)

        st.markdown(f"### {sel}")
        st.write(alert["summary"])
        m = st.columns(4)
        m[0].metric("Risk", f"{row['risk']:.2f}")
        m[1].metric("Rules", f"{row['rule_score']:.2f}")
        m[2].metric("Anomaly", f"{row['anomaly_score']:.2f}")
        m[3].metric("Supervised", f"{row['supervised_score']:.2f}")

        a, b = st.columns(2)
        with a:
            st.markdown("**Rules fired** (observed policy matches)")
            for r in alert["rules"] or ["None"]:
                st.write(f"- {r}")
            st.markdown("**Model indicators** (what pushed the supervised score up)")
            if alert["top_indicators"].empty:
                st.write("No strong positive contributors.")
            else:
                st.bar_chart(alert["top_indicators"].set_index("label")["contribution"])
            st.caption(alert["uncertainty"])
        with b:
            st.markdown("**Deviation from typical behaviour** (robust z-score, anomaly evidence)")
            st.dataframe(alert["deviations"].head(6)[["label", "value", "typical", "robust_z"]].round(2),
                         hide_index=True, use_container_width=True)
            st.markdown("**Suggested investigation steps**")
            for s in alert["steps"]:
                st.write(f"- {s}")

        st.markdown("**Event timeline** (original evidence)")
        st.dataframe(alert["timeline"], hide_index=True, use_container_width=True)

        with st.expander("Ground truth (demo only — never available in production)"):
            st.write(f"Scenario: `{row['scenario']}` · is_attack: `{bool(row['is_attack'])}`")

        st.markdown("### Recommended response")
        st.write(f"**Action:** `{decision['action']}`  ·  **Policy:** {decision['policy_rule']}")
        st.write(f"Requires human approval: **{'yes' if decision['requires_approval'] else 'no'}**  ·  "
                 f"Corroborated: **{decision['corroborated']}**  ·  Mode: **{'dry-run' if dry_run else 'LIVE'}**")

        status = st.session_state.handled.get(sel)
        if status:
            st.success(f"This alert is already marked: {status}")
        else:
            col1, col2, col3 = st.columns(3)
            label = "Approve & execute" if decision["requires_approval"] else "Execute action"
            if col1.button(label, type="primary", key=f"go-{sel}"):
                try:
                    outcome = execute(decision["action"], row["user_id"], dry_run)
                    st.session_state.audit.append(audit_entry(
                        sel, row, decision, pipe.version, "orchestrator",
                        "executed", outcome, approver=analyst if decision["requires_approval"] else None))
                    st.session_state.handled[sel] = "executed"
                except NotImplementedError as e:
                    st.error(str(e))
                st.rerun()
            if col2.button("Reject action", key=f"no-{sel}"):
                st.session_state.audit.append(audit_entry(
                    sel, row, decision, pipe.version, analyst, "rejected", "No action taken", approver=analyst))
                st.session_state.handled[sel] = "rejected"
                st.rerun()
            fb = col3.selectbox("Analyst feedback", ["malicious", "benign", "duplicate", "inconclusive"], key=f"fb-{sel}")
            if col3.button("Save feedback", key=f"fbb-{sel}"):
                st.session_state.audit.append(audit_entry(
                    sel, row, decision, pipe.version, analyst, "feedback", "Queued for controlled model review",
                    feedback=fb))
                st.session_state.handled[sel] = f"feedback: {fb}"
                st.rerun()
            st.caption("Feedback is logged for review; it does not automatically retrain the production model.")

# --------------------------------------------------------------------------- evaluation
with tab_eval:
    st.subheader("Baseline comparison on the held-out period")
    st.write(f"Train: days 0–{split - 1} · Test: days {split}–{days - 1} (time-based split, no future leakage).")
    res = evaluate(scored, split, medium)
    st.dataframe(res.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}",
                                   "False-positive rate": "{:.3f}", "PR-AUC": "{:.2f}"}),
                 use_container_width=True)
    n_pos = int(scored[scored.day_idx >= split]["is_attack"].sum())
    st.warning(f"Only {n_pos} attack windows are in the test period and the data is synthetic. These numbers "
               "demonstrate the evaluation harness; they are NOT evidence of real-world performance. "
               "Per the article, use governed enterprise logs and realistic attack simulations for real claims.")
    st.caption("Not yet implemented from the evaluation plan: detection latency, response-time metrics, "
               "ablation by feature family, analyst explanation-usefulness study.")

# --------------------------------------------------------------------------- audit
with tab_audit:
    st.subheader("Audit trail")
    if st.session_state.audit:
        adf = pd.DataFrame(st.session_state.audit)
        st.dataframe(adf, hide_index=True, use_container_width=True)
        st.download_button("Download audit log (JSONL)", audit_to_jsonl(st.session_state.audit),
                           file_name="audit_log.jsonl", mime="application/json")
    else:
        st.write("No decisions recorded yet. Act on an alert in the investigation tab.")

# --------------------------------------------------------------------------- data
with tab_data:
    st.subheader("Normalised identity events")
    st.dataframe(events.drop(columns=["scenario", "is_attack"]).head(500), hide_index=True, use_container_width=True)
    st.subheader("Engineered features (user-day windows)")
    st.dataframe(feats.drop(columns=["scenario", "is_attack"]).head(500), hide_index=True, use_container_width=True)
    st.download_button("Download events (CSV)", events.to_csv(index=False), file_name="events.csv")
