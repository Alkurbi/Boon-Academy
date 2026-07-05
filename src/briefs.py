import json

import pandas as pd

from .config import OUTPUT_DIR, QUIZ2_DATE
from .risk import score_reasons


def write_outputs(scored: pd.DataFrame, insights: dict, drafts: dict,
                  plan: pd.DataFrame, notes: pd.DataFrame,
                  run_date: str, cleaning_log: dict, llm_stats: dict):
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "briefs").mkdir(exist_ok=True)

    cols = ["student_id", "student_name", "campus_id", "facilitator_email", "grade",
            "learning_track", "quiz_score", "target_score", "avg_attend_min",
            "attend_trend", "avg_practice", "zero_practice_days_recent", "note_count",
            "quiz", "attendance", "attend_decline", "practice", "contact",
            "risk_score", "tier"]
    scored.sort_values("risk_score", ascending=False)[cols].to_csv(
        OUTPUT_DIR / "risk_scores.csv", index=False)

    if insights:
        pd.DataFrame([{"student_id": sid,
                       **{k: (", ".join(map(str, v)) if isinstance(v, list) else v)
                          for k, v in ins.items()}}
                      for sid, ins in insights.items()]).to_csv(
            OUTPUT_DIR / "note_insights.csv", index=False)

    plan.to_csv(OUTPUT_DIR / "action_plan.csv", index=False)

    tomorrow = plan.date.min() if len(plan) else run_date
    for fac, group in scored.groupby("facilitator_email"):
        _write_brief(fac, group, plan, insights, drafts, notes, tomorrow)

    failing = scored.quiz_score < 60
    summary = {
        "run_date": run_date,
        "quiz2_date": QUIZ2_DATE,
        "students": len(scored),
        "failing_quiz1": int(failing.sum()),
        "failing_never_contacted": int((failing & (scored.note_count == 0)).sum()),
        "tiers": scored.tier.value_counts().to_dict(),
        "planned_coverage_critical_high": (
            f"{plan[plan.tier.isin(['CRITICAL', 'HIGH'])].student_id.nunique()}"
            f"/{int(scored.tier.isin(['CRITICAL', 'HIGH']).sum())} before Quiz 2"),
        "cleaning_log": cleaning_log,
        **llm_stats,
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_brief(fac, group, plan, insights, drafts, notes, day):
    days_to_quiz = (pd.Timestamp(QUIZ2_DATE) - pd.Timestamp(day)).days
    tiers = group.tier.value_counts().to_dict()
    today = plan[(plan.facilitator_email == fac) & (plan.date == day)]
    by_id = group.set_index("student_id")

    lines = [f"# Daily brief - {fac.split('@')[0]} - {day}",
             f"**Quiz 2 (Verbal) in {days_to_quiz} days.** Your students: "
             + ", ".join(f"{tiers.get(t, 0)} {t}" for t in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]),
             "", "## Today's actions (do these 5 first)", ""]

    for i, a in enumerate(today.itertuples(), 1):
        r = by_id.loc[a.student_id]
        comps = {k: r[k] for k in ["quiz", "attendance", "attend_decline", "practice", "contact"]}
        ins = insights.get(a.student_id)
        draft = drafts.get(a.student_id)
        lines += [f"### {i}. {a.student_name} ({a.student_id}) - {a.tier}, "
                  f"risk {a.risk_score:.0f}/100",
                  f"- **Action:** {a.action.replace('_', ' ')}"
                  + (f" (follow-up promised {a.followup_due})" if a.followup_due else ""),
                  f"- **Why:** {score_reasons(r, comps)}",
                  f"- **Parent phone:** {r.parent_phone}"]
        if ins:
            lines.append(f"- **Contact history:** {ins['summary_en']} "
                         f"(last outcome: {ins['contact_outcome']})")
            if ins.get("name_mismatch"):
                lines.append("- **Note:** the note text uses a different name (known "
                             "data issue - names in notes are unreliable; check the "
                             "raw notes before relying on the contact history).")
        else:
            lines.append("- **Contact history:** no notes on file - first contact")
        if draft:
            lines += ["- **Draft WhatsApp to parent (review before sending):**",
                      f"  > {draft['whatsapp_ar']}",
                      "- **Talking points:** " + "; ".join(draft["talking_points"])]
        if ins and ins.get("fallback"):
            raw = notes[notes.student_id == a.student_id].sort_values("date")
            lines += ["- **Raw notes (LLM unavailable):**"] + [
                f"  - [{d.date()}] {t}" for d, t in zip(raw.date, raw.note_text)]
        lines.append("")

    lines += ["## Full roster by risk", "",
              "| Student | Tier | Risk | Quiz 1 | Avg attend | Avg practice | Notes |",
              "|---|---|---|---|---|---|---|"]
    for r in group.sort_values("risk_score", ascending=False).itertuples():
        quiz = f"{r.quiz_score:.0f}" if pd.notna(r.quiz_score) else "-"
        lines.append(f"| {r.student_name} | {r.tier} | {r.risk_score:.0f} | {quiz} "
                     f"| {r.avg_attend_min:.0f} min | {r.avg_practice:.0f} q | {r.note_count} |")

    path = OUTPUT_DIR / "briefs" / f"{day}_{fac.split('@')[0]}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
