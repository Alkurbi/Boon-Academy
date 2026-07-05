"""End-to-end pipeline: load -> analyze notes (LLM) -> score -> plan -> draft (LLM) -> render.

    python -m src.pipeline            # full run (needs ANTHROPIC_API_KEY)
    python -m src.pipeline --no-llm   # deterministic fallback run, no key needed
    python -m src.pipeline --eval     # also run the golden-set / judge eval
"""
import argparse
import json
import sys
import time

import pandas as pd

from .briefs import write_outputs
from .config import RUN_DATE
from .llm import LLM
from .load import build_features, load_data
from .risk import plan_actions, score_student


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="run with deterministic fallbacks")
    ap.add_argument("--eval", action="store_true", help="run output-quality eval after the pipeline")
    args = ap.parse_args(argv)

    t0 = time.time()
    print(f"[1/6] Loading data (run date {RUN_DATE})...")
    daily, notes, students, cleaning_log = load_data()
    feats = build_features(daily, notes, students, RUN_DATE)
    print(f"      {len(feats)} students, {len(notes)} notes; cleaned: "
          + ", ".join(f"{k}={v}" for k, v in cleaning_log.items() if v))

    llm = LLM(enabled=not args.no_llm)
    if not llm.enabled and not args.no_llm:
        print("      WARNING: no ANTHROPIC_API_KEY or OPENROUTER_API_KEY set - "
              "falling back to --no-llm behavior")

    print(f"[2/6] Analyzing facilitator notes ({notes.student_id.nunique()} students, "
          f"LLM {'on' if llm.enabled else 'OFF - fallbacks'})...")
    insights = llm.analyze_notes(notes, students)

    print("[3/6] Scoring risk (transparent rules)...")
    comps = feats.apply(lambda r: score_student(r, insights.get(r.student_id)),
                        axis=1, result_type="expand")
    scored = pd.concat([feats, comps], axis=1)
    print("      tiers: " + json.dumps(scored.tier.value_counts().to_dict()))

    print("[4/6] Planning actions until Quiz 2...")
    plan = plan_actions(scored, insights, RUN_DATE)

    day1 = plan[plan.date == plan.date.min()] if len(plan) else plan
    actioned = scored.merge(day1[["student_id", "action"]], on="student_id")
    print(f"[5/6] Drafting outreach for tomorrow's {len(actioned)} actions...")
    drafts = llm.draft_messages(actioned, insights)

    print("[6/6] Writing outputs/...")
    summary = write_outputs(scored, insights, drafts, plan, notes,
                            RUN_DATE, cleaning_log, llm.stats())
    print(f"Done in {time.time() - t0:.1f}s. Coverage: "
          f"{summary['planned_coverage_critical_high']}; "
          f"LLM calls: {summary['llm_calls']}, fallbacks: {summary['fallbacks_used']}, "
          f"est cost: ${summary['est_cost_usd']}")

    if args.eval:
        from .evaluate import run_eval
        run_eval(llm, drafts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
