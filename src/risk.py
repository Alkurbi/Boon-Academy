import pandas as pd

from .config import ACTIONS_PER_DAY, PASS_SCORE, QUIZ2_DATE, SESSION_MAX_MIN

TIERS = [(65, "CRITICAL"), (45, "HIGH"), (25, "MEDIUM"), (0, "LOW")]


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def score_student(row, insight: dict | None) -> dict:
    c = {}

    q = row.quiz_score
    if pd.isna(q):
        c["quiz"] = 25.0 
    elif q < PASS_SCORE:
        c["quiz"] = 25 + 15 * (PASS_SCORE - q) / PASS_SCORE
    else:
        c["quiz"] = 10 * clamp((row.target_score - q) / 40)

    avg = row.avg_attend_min if pd.notna(row.avg_attend_min) else 0.0
    c["attendance"] = 20 * (1 - avg / SESSION_MAX_MIN)
    trend = row.attend_trend if pd.notna(row.attend_trend) else 0.0
    c["attend_decline"] = 10 * clamp(-trend / 30)
    prac = row.avg_practice if pd.notna(row.avg_practice) else 0.0
    c["practice"] = min(15.0, 15 * (1 - clamp(prac / 30))
                        + (3 if row.zero_practice_days_recent >= 3 else 0))

    failing = pd.notna(q) and q < PASS_SCORE
    contact = 0.0
    if failing and row.note_count == 0:
        contact += 10 
    if insight:
        if insight.get("intervention_succeeded") is False:
            contact += 5 
        recent_success = (
            insight.get("intervention_succeeded") is True
            and pd.notna(row.days_since_last_note) and row.days_since_last_note <= 5
        )
        if recent_success:
            contact -= 5
    c["contact"] = clamp(contact, 0, 10)

    total = round(min(100.0, sum(c.values())), 1)
    tier = next(name for cutoff, name in TIERS if total >= cutoff)
    if failing and tier in ("MEDIUM", "LOW"):
        tier = "HIGH"
    if getattr(row, "absent_last2", False) and tier in ("MEDIUM", "LOW"):
        tier = "HIGH"
    return {**{k: round(v, 1) for k, v in c.items()}, "risk_score": total, "tier": tier}


def score_reasons(row, comps: dict) -> str:
    """Top-2 component reasons in plain words, for the facilitator brief."""
    texts = {
        "quiz": (f"failed Quiz 1 ({row.quiz_score:.0f}/100)" if pd.notna(row.quiz_score)
                 and row.quiz_score < PASS_SCORE else "below target score"),
        "attendance": f"low attendance (avg {row.avg_attend_min:.0f}/90 min)"
                      if pd.notna(row.avg_attend_min) else "attendance unknown",
        "attend_decline": "attendance is dropping week-over-week",
        "practice": f"little evening practice (avg {row.avg_practice:.0f} questions/day)",
        "contact": "no successful contact yet" if row.note_count else "never contacted",
    }
    top = sorted((k for k in texts), key=lambda k: -comps.get(k, 0))[:2]
    return "; ".join(texts[k] for k in top if comps.get(k, 0) > 0) or "on track"


ACTION_BY_TIER = {
    "CRITICAL": "call_parent",
    "HIGH": "one_on_one",
    "MEDIUM": "motivational_message",
    "LOW": "none",
}


def plan_actions(scored: pd.DataFrame, insights: dict, run_date: str) -> pd.DataFrame:
    days = pd.date_range(pd.Timestamp(run_date) + pd.Timedelta(days=1),
                         pd.Timestamp(QUIZ2_DATE) - pd.Timedelta(days=1))
    plans = []
    for fac, group in scored.groupby("facilitator_email"):
        queue = group.sort_values(
            ["risk_score", "days_since_last_note"], ascending=[False, False]
        )
        queue = queue[queue.tier != "LOW"]
        todo = list(queue.itertuples())
        followups = {  # student_id -> due date from LLM note analysis
            s: insights[s]["followup_due"] for s in group.student_id
            if s in insights and isinstance(insights[s].get("followup_due"), str)
            and insights[s]["followup_due"]
        }
        for day in days:
            picked, day_str = [], day.strftime("%Y-%m-%d")
            due = [t for t in todo if followups.get(t.student_id, "") <= day_str
                   and t.student_id in followups]
            for t in due + [t for t in todo if t not in due]:
                if len(picked) >= ACTIONS_PER_DAY:
                    break
                picked.append(t)
            for t in picked:
                plans.append({
                    "date": day_str, "facilitator_email": fac,
                    "student_id": t.student_id, "student_name": t.student_name,
                    "tier": t.tier, "risk_score": t.risk_score,
                    "action": ("call_parent" if getattr(t, "absent_last2", False)
                               else ACTION_BY_TIER[t.tier]),
                    "followup_due": followups.get(t.student_id, ""),
                })
                todo.remove(t)
    return pd.DataFrame(plans)


if __name__ == "__main__":
    from .config import RUN_DATE
    from .load import build_features, load_data
    daily, notes, students, _ = load_data()
    feats = build_features(daily, notes, students, RUN_DATE)
    comps = feats.apply(lambda r: score_student(r, None), axis=1, result_type="expand")
    scored = pd.concat([feats, comps], axis=1)
    plan = plan_actions(scored, {}, RUN_DATE)
    covered = plan[plan.tier.isin(["CRITICAL", "HIGH"])].student_id.nunique()
    total_ch = scored.tier.isin(["CRITICAL", "HIGH"]).sum()
    print(f"OK: tiers={scored.tier.value_counts().to_dict()}, "
          f"planned coverage of CRITICAL+HIGH before Quiz 2: {covered}/{total_ch}")
