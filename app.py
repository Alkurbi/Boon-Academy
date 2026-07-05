"""Facilitator dashboard. Reads outputs/ only - run `python -m src.pipeline` first.

    streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from src.config import DATA_DIR, OUTPUT_DIR as OUT
st.set_page_config(page_title="Boon Intervention Engine", layout="wide")

if not (OUT / "risk_scores.csv").exists():
    st.error("No outputs found - run `python -m src.pipeline` first.")
    st.stop()

scores = pd.read_csv(OUT / "risk_scores.csv")
plan = pd.read_csv(OUT / "action_plan.csv")
summary = json.loads((OUT / "run_summary.json").read_text())
insights = (pd.read_csv(OUT / "note_insights.csv")
            if (OUT / "note_insights.csv").exists() else pd.DataFrame())
daily = pd.read_csv(DATA_DIR / "student_daily_metrics.csv", parse_dates=["date"])
notes = pd.read_csv(DATA_DIR / "facilitator_notes.csv", parse_dates=["date"])

st.title("Boon Intervention Engine")
st.caption(f"Run date {summary['run_date']} - Quiz 2 on {summary['quiz2_date']}")

fac = st.sidebar.selectbox("Facilitator", ["All"] + sorted(scores.facilitator_email.unique()))
tiers = st.sidebar.multiselect("Tier", ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                               default=["CRITICAL", "HIGH"])
view = scores if fac == "All" else scores[scores.facilitator_email == fac]
view = view[view.tier.isin(tiers)] if tiers else view

c1, c2, c3, c4 = st.columns(4)
c1.metric("Failed Quiz 1", summary["failing_quiz1"])
c2.metric("Failing, never contacted", summary["failing_never_contacted"])
c3.metric("CRITICAL students", summary["tiers"].get("CRITICAL", 0))
c4.metric("Planned coverage before Quiz 2", summary["planned_coverage_critical_high"])

st.subheader("Risk tiers by campus")
st.bar_chart(scores.groupby(["campus_id", "tier"]).size().unstack(fill_value=0))

st.subheader(f"Students ({len(view)})")
st.dataframe(
    view.sort_values("risk_score", ascending=False)[
        ["student_id", "student_name", "campus_id", "tier", "risk_score", "quiz_score",
         "avg_attend_min", "avg_practice", "note_count"]],
    width="stretch", hide_index=True)

st.subheader("Student detail")
sid = st.selectbox("Pick a student", view.sort_values("risk_score", ascending=False).student_id,
                   format_func=lambda s: f"{s} - {scores.set_index('student_id').student_name[s]}")
if sid:
    r = scores[scores.student_id == sid].iloc[0]
    st.markdown(f"**{r.student_name}** - {r.tier} (risk {r.risk_score:.0f}/100) - "
                f"Quiz 1: {r.quiz_score:.0f}/100 (target {r.target_score})")
    st.markdown("Score components: " + ", ".join(
        f"{k} {r[k]:.0f}" for k in ["quiz", "attendance", "attend_decline", "practice", "contact"]))

    d = daily[daily.student_id == sid].set_index("date")
    st.line_chart(d[["session_attended_min", "practice_questions"]])

    ins = insights[insights.student_id == sid] if len(insights) else pd.DataFrame()
    if len(ins):
        i = ins.iloc[0]
        if i.get("name_mismatch") is True or str(i.get("name_mismatch")) == "True":
            st.warning("The note text uses a different name. Known data issue: "
                       "the names written in notes are unreliable.")
        st.info(f"**Contact history (LLM):** {i.summary_en}  \n"
                f"Last outcome: {i.contact_outcome} - trajectory: {i.trajectory}"
                + (f" - flags: {i.risk_flags}" if isinstance(i.risk_flags, str) and i.risk_flags else ""))
    sn = notes[notes.student_id == sid].sort_values("date")
    if len(sn) and st.toggle("Show raw notes"):
        for n in sn.itertuples():
            st.markdown(f"- `{n.date.date()}` {n.note_text}")
    elif not len(sn):
        st.warning("No facilitator notes on file - this student has never been contacted.")

    actions = plan[plan.student_id == sid]
    if len(actions):
        st.markdown("**Planned actions:** " + "; ".join(
            f"{a.date}: {a.action.replace('_', ' ')}" for a in actions.itertuples()))
