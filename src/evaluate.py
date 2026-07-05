import json
import random
from pathlib import Path

from .config import OUTPUT_DIR
from .llm import LLM, NOTE_SCHEMA, NOTE_SYSTEM

GOLDENS = json.loads(  # goldens live with the code, not with the outputs
    (Path(__file__).resolve().parent.parent / "eval" / "goldens.json")
    .read_text(encoding="utf-8"))

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "integer", "description": "1-5: uses only the provided facts"},
        "tone": {"type": "integer", "description": "1-5: warm, respectful, appropriate for a Saudi parent"},
        "arabic": {"type": "integer", "description": "1-5: natural Arabic"},
        "actionable": {"type": "integer", "description": "1-5: clear next step for the parent"},
        "worst_issue": {"type": "string"},
    },
    "required": ["grounded", "tone", "arabic", "actionable", "worst_issue"],
    "additionalProperties": False,
}


def run_eval(llm: LLM, drafts: dict):
    lines = ["# Eval report", ""]

    # --- 1. golden-set accuracy ---
    lines += ["## Note-analysis goldens (field-level exact match)", ""]
    field_hits, field_total = 0, 0
    for g in GOLDENS:
        text = "\n".join(f"[{n['date']}] {n['text']}" for n in g["notes"])
        user = f"Student: {g['student_name']}\nFacilitator notes (oldest first):\n{text}"
        got = llm.call(NOTE_SYSTEM, user, NOTE_SCHEMA, max_tokens=800) or {}
        for field, want in g["expected"].items():
            ok = got.get(field) in (want if isinstance(want, list) else [want])
            field_hits += ok
            field_total += 1
            mark = "PASS" if ok else f"FAIL (got {got.get(field)!r}, want {want!r})"
            lines.append(f"- {g['student_id']}.{field}: {mark}")
    acc = field_hits / field_total if field_total else 0
    verdict = "OK" if acc >= 0.8 else "BELOW TARGET (0.8) - review prompt before shipping"
    lines += ["", f"**Accuracy: {field_hits}/{field_total} = {acc:.0%}** - {verdict}", ""]

    # --- 2. judge a sample of drafts ---
    lines += ["## Message-draft quality (LLM-as-judge, sample of 5)", ""]
    real = [d for d in drafts.values() if not d.get("fallback")]
    sample = random.Random(0).sample(real, min(5, len(real)))
    scores = []
    for d in sample:
        user = ("Score this WhatsApp draft a facilitator will send to a parent. "
                f"Draft:\n{d['whatsapp_ar']}\n\nTalking points given to the facilitator: "
                + "; ".join(d["talking_points"]))
        v = llm.call("You are a strict reviewer of parent-communication quality.",
                      user, JUDGE_SCHEMA, max_tokens=300)
        if v:
            mean = (v["grounded"] + v["tone"] + v["arabic"] + v["actionable"]) / 4
            scores.append(mean)
            lines.append(f"- {d['student_id']}: grounded {v['grounded']}, tone {v['tone']}, "
                         f"arabic {v['arabic']}, actionable {v['actionable']}"
                         + (f" - issue: {v['worst_issue']}" if min(
                             v['grounded'], v['tone'], v['arabic'], v['actionable']) <= 2 else ""))
    if scores:
        lines.append(f"\n**Mean judge score: {sum(scores)/len(scores):.1f}/5**")
    elif not real:
        lines.append("(no LLM drafts to judge - run without --no-llm)")

    (OUTPUT_DIR / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"      eval: goldens {field_hits}/{field_total} ({acc:.0%}); "
          f"report -> outputs/eval_report.md")
