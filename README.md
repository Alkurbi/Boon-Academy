# Boon Academy Intervention Engine

Ranks all 200 students by transparent risk rules, reads the facilitator
notes with an LLM, and gives each facilitator a 5-action daily brief so every
failing student is reached before Quiz 2.

## Run it

```
python -m venv .venv
.venv\Scripts\activate             # Windows | mac/linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # set ANTHROPIC_API_KEY or OPENROUTER_API_KEY
python -m src.pipeline             # the run command
```

Flags: `--no-llm` (fully offline, deterministic fallbacks), `--eval` (quality report).
Dashboard: `streamlit run app.py` (reads `outputs/` only, no API calls).

## What it produces (`outputs/`)

- `briefs/<date>_<facilitator>.md` - the artifact facilitators actually use
- `risk_scores.csv` - all 200 students, score components, tier
- `action_plan.csv` - day-by-day schedule until Quiz 2
- `note_insights.csv` - structured LLM analysis of the notes
- `run_summary.json` - cleaning log, coverage, LLM cost
- `eval_report.md` - golden-set + LLM-judge eval (with `--eval`)
