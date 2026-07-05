"""Claude integration: (A) Arabic note analysis, (B) intervention message drafts.

Design rules:
- Structured output via output_config.format -> guaranteed-valid JSON, no parsing retries.
- One call per student for Job A (a bad response costs one student, not a batch);
  ThreadPoolExecutor keeps wall time low. One call per facilitator for Job B.
- Every call degrades to a deterministic fallback, so the pipeline completes
  with no API key (--no-llm uses the same path).
- Parent phone numbers never enter a prompt; they are merged into briefs locally.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .config import ANTHROPIC_API_KEY, MODEL, OPENROUTER_API_KEY, OPENROUTER_MODEL

MTOK_IN, MTOK_OUT = 1.0, 5.0  # same model either provider

NOTE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary_en": {"type": "string",
                       "description": "1-2 sentence English summary of the notes. "
                                      "Refer to people only as 'the student' / 'the "
                                      "parent' - never use a personal name (names in "
                                      "notes are unreliable in this dataset)"},
        "last_contact_method": {"type": "string",
                                "enum": ["call", "whatsapp", "in_person", "other", "none"]},
        "contact_outcome": {"type": "string",
                            "enum": ["reached", "no_answer", "promised_followup",
                                     "refused", "unknown"]},
        "intervention_succeeded": {"type": ["boolean", "null"],
                                   "description": "true only if a note shows the intervention actually landed; null if unclear"},
        "followup_due": {"type": ["string", "null"],
                         "description": "YYYY-MM-DD if a note promises a follow-up date, else null"},
        "risk_flags": {"type": "array", "items": {
            "type": "string",
            "enum": ["phone_addiction", "family_issue", "health", "disengagement", "other"]}},
        "trajectory": {"type": "string",
                       "enum": ["improving", "stable", "worsening", "unknown"]},
        "name_mismatch": {"type": "boolean",
                          "description": "true if the notes call the student by a different name than the one given (possibly misfiled notes)"},
    },
    "required": ["summary_en", "last_contact_method", "contact_outcome",
                 "intervention_succeeded", "followup_due", "risk_flags", "trajectory",
                 "name_mismatch"],
    "additionalProperties": False,
}

NOTE_SYSTEM = (
    "You analyze Saudi-dialect Arabic facilitator notes about a student in a test-prep "
    "program. Extract facts only. contact_method and contact_outcome refer to outreach "
    "to the PARENT or guardian only - the facilitator talking with the student in class "
    "is not a contact. Distinguish a contact ATTEMPT from a contact that actually "
    "succeeded (e.g. a call nobody answered is no_answer, not reached). "
    "intervention_succeeded is true only if the notes show the student's behavior or "
    "situation actually improved after an intervention, false if attempts clearly did "
    "not help, null if unclear. If the notes do not say something, answer unknown/null. "
    "Names inside the note text are unreliable in this dataset (a data-entry issue): "
    "if the notes use a DIFFERENT name than the one given, set name_mismatch true, and "
    "in summary_en refer to 'the student' / 'the parent' instead of any name."
)

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {"drafts": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "student_id": {"type": "string"},
            "whatsapp_ar": {"type": "string",
                            "description": "Warm, specific WhatsApp message to the parent in Arabic, <=80 words"},
            "talking_points": {"type": "array", "items": {"type": "string"},
                               "description": "3 short English talking points for the facilitator"},
        },
        "required": ["student_id", "whatsapp_ar", "talking_points"],
        "additionalProperties": False,
    }}},
    "required": ["drafts"],
    "additionalProperties": False,
}

DRAFT_SYSTEM = (
    "You help classroom facilitators at a Saudi test-prep academy draft parent outreach. "
    "Write WhatsApp drafts in respectful, warm Saudi Arabic addressed to the parent. "
    "Ground every draft in the specific numbers and note context provided - no generic "
    "platitudes, no invented facts. If a previous call went unanswered, do not suggest "
    "calling again; suggest WhatsApp instead. A human reviews before sending."
)


class LLM:
    def __init__(self, enabled: bool = True):
        self.provider = None
        if enabled and ANTHROPIC_API_KEY:
            import anthropic
            self.client = anthropic.Anthropic()  # SDK retries 429/5xx itself
            self.provider = "anthropic"
        elif enabled and OPENROUTER_API_KEY:
            import httpx  # already a dependency of the anthropic SDK
            self.client = httpx.Client(
                base_url="https://openrouter.ai/api/v1", timeout=90,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"})
            self.provider = "openrouter"
        self.enabled = self.provider is not None
        self.calls = 0
        self.fallbacks = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.or_cost_usd = 0.0

    def call(self, system: str, user: str, schema: dict, max_tokens: int):
        """One structured-output call; returns parsed dict or None on failure."""
        if not self.enabled:
            self.fallbacks += 1
            return None
        try:
            if self.provider == "anthropic":
                resp = self.client.messages.create(
                    model=MODEL, max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": user}],
                    output_config={"format": {"type": "json_schema", "schema": schema}},
                )
                self.calls += 1
                self.tokens_in += resp.usage.input_tokens
                self.tokens_out += resp.usage.output_tokens
                text = next(b.text for b in resp.content if b.type == "text")
                return json.loads(text)
            return self.call_openrouter(system, user, schema, max_tokens)
        except Exception as e:
            print(f"  LLM call failed ({type(e).__name__}: {e}); using fallback")
            self.fallbacks += 1
            return None

    def call_openrouter(self, system: str, user: str, schema: dict, max_tokens: int):
        body = {
            "model": OPENROUTER_MODEL, "max_tokens": max_tokens * 4 + 1500,
            "reasoning": {"effort": "low"},
            "messages": [{"role": "system", "content": system
                          + "\nRespond ONLY with a single JSON object matching this "
                            "schema, no prose:\n" + json.dumps(schema)},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "out", "strict": True, "schema": schema}},
        }
        last = None
        for attempt in range(3):
            r = self.client.post("/chat/completions", json=body)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                import time
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                last = f"OpenRouter: {data['error'].get('message', data['error'])}"
                import time
                time.sleep(2 ** attempt)
                continue
            self.calls += 1
            usage = data.get("usage", {})
            self.tokens_in += usage.get("prompt_tokens", 0)
            self.tokens_out += usage.get("completion_tokens", 0)
            self.or_cost_usd += usage.get("cost", 0)
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if text.startswith("```"):
                text = text.strip("`").removeprefix("json").strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                last = f"non-JSON content: {text[:60]!r}"  # prose answer -> retry
        raise RuntimeError(f"OpenRouter retries exhausted ({last})")

    # ---- Job A: note analysis -------------------------------------------------
    def analyze_notes(self, notes: pd.DataFrame, students: pd.DataFrame) -> dict:
        """{student_id: insight dict} for every student that has notes."""
        names = students.set_index("student_id").student_name
        jobs = []
        for sid, g in notes.sort_values("date").groupby("student_id"):
            lines = "\n".join(f"[{d.date()}] {t}" for d, t in zip(g.date, g.note_text))
            user = f"Student: {names.get(sid, sid)}\nFacilitator notes (oldest first):\n{lines}"
            jobs.append((sid, user))

        def run(job):
            sid, user = job
            return sid, self.call(NOTE_SYSTEM, user, NOTE_SCHEMA, max_tokens=800)

        with ThreadPoolExecutor(max_workers=8) as ex:
            results = dict(ex.map(run, jobs))
        return {sid: (r if r is not None else self.note_fallback(notes, sid))
                for sid, r in results.items()}

    @staticmethod
    def note_fallback(notes: pd.DataFrame, sid: str) -> dict:
        n = (notes.student_id == sid).sum()
        return {"summary_en": f"(LLM unavailable - {n} raw Arabic note(s) in brief)",
                "last_contact_method": "none", "contact_outcome": "unknown",
                "intervention_succeeded": None, "followup_due": None,
                "risk_flags": [], "trajectory": "unknown", "name_mismatch": None,
                "fallback": True}

    # ---- Job B: message drafting ----------------------------------------------
    def draft_messages(self, actioned: pd.DataFrame, insights: dict) -> dict:
        """{student_id: draft dict} for students with a day-1 action, batched per facilitator."""
        jobs = []
        for fac, g in actioned.groupby("facilitator_email"):
            blocks = []
            for r in g.itertuples():
                ins = insights.get(r.student_id, {})
                blocks.append(
                    f"- student_id {r.student_id}: {r.student_name.split()[0]}, "
                    f"grade {r.grade}, quiz {r.quiz_score:.0f}/100, "
                    f"avg attendance {r.avg_attend_min:.0f}/90 min, "
                    f"avg practice {r.avg_practice:.0f} q/day, planned action: {r.action}. "
                    f"Contact history: {ins.get('summary_en', 'no notes on file')} "
                    f"(outcome: {ins.get('contact_outcome', 'unknown')})"
                )
            user = ("Draft outreach for these students (Quiz 2 is in 6 days):\n"
                    + "\n".join(blocks))
            jobs.append((fac, [r.student_id for r in g.itertuples()], user))

        def run(job):
            fac, sids, user = job
            out = self.call(DRAFT_SYSTEM, user, DRAFT_SCHEMA,
                             max_tokens=400 + 300 * len(sids))
            return sids, out

        drafts = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for sids, out in ex.map(run, jobs):
                got = {d["student_id"]: d for d in out["drafts"]} if out else {}
                for sid in sids:
                    drafts[sid] = got.get(sid) or self.draft_fallback(actioned, sid)
        return drafts

    @staticmethod
    def draft_fallback(actioned: pd.DataFrame, sid: str) -> dict:
        r = actioned[actioned.student_id == sid].iloc[0]
        first = r.student_name.split()[0]
        return {
            "student_id": sid,
            "whatsapp_ar": (f"السلام عليكم، معكم مشرف/ة {first} من الأكاديمية. "
                            f"نتيجة {first} في الاختبار الأخير كانت {r.quiz_score:.0f} من 100، "
                            "ونحتاج دعمكم في متابعة الحضور والتدريب المسائي قبل الاختبار القادم "
                            "يوم الاثنين. نقدر تواصلكم معنا."),
            "talking_points": [
                f"Quiz 1 score {r.quiz_score:.0f}/100; Quiz 2 in 6 days",
                f"Avg attendance {r.avg_attend_min:.0f}/90 min - ask about barriers",
                "Agree one concrete evening-practice commitment",
            ],
            "fallback": True,
        }

    def stats(self) -> dict:
        cost = self.or_cost_usd or (self.tokens_in * MTOK_IN
                                    + self.tokens_out * MTOK_OUT) / 1e6
        return {"llm_enabled": self.enabled, "llm_provider": self.provider,
                "llm_calls": self.calls,
                "fallbacks_used": self.fallbacks, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out, "est_cost_usd": round(cost, 4)}
