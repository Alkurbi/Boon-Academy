# Analysis: Boon Academy Intervention Engine

## Diagnosis

Facilitators aren't short on effort, they're short on visibility: the notes show plenty of activity, but nobody can see which of their ~25 students to reach first, and writing a note gets mistaken for intervening. The students most at risk are exactly the ones with no paper trail at all, so they stay invisible until the next quiz exposes them. More facilitator hours won't fix that; a ranked daily queue that remembers contact outcomes will.

## What I found in the data

- 67 of 200 students failed Quiz 1. Reading the 180 Arabic notes with the LLM, only 7 (10%) show evidence of receiving an intervention. 2 had attempts that never landed, 38 have notes that just describe the problem, and 20 have no notes at all. The real rate is worse than the reported 30%.
- The bottom attendance decile averages 30.6 of 90 session minutes. Attendance predicts quiz failure better than anything else in this data, so it carries 30 risk points, second only to the quiz itself.
- The data is deliberately messy: 3 null attendance values, 3 practice outliers, a quiz score dated two days before Quiz 1, session days contradicting the stated Mon-Thu schedule, and 152 of 180 notes naming the wrong student. Absence notes match the filed student's attendance 9/9, so the join is right and the system trusts no name written inside a note.

## What I built and why

- `load.py` cleans the known problems (nulls kept as missing, outliers capped, orphans dropped) and rebuilds quiz dates from raw dates, because derived columns drift.
- `risk.py` scores every student 0-100 with rules a facilitator can check by hand: quiz 40 (failed: `25 + 15*(60-score)/60`; passed: `10*(target-score)/40`), attendance 20 (`20*(1-avg_min/90)`) + decline 10, practice 15 (`15*(1-avg_q/30)`, +3 if 3+ recent zero days), contact 10 (+10 never-contacted failer, or +5 attempts never landed, -5 recent success). Tiers cut at 65/45/25; two hard rules: quiz-failers and students absent both of the last two days are at least HIGH - sudden dropout is an emergency that averages dilute.
- `llm.py` spends the LLM on the one job rules can't do, turning Saudi-dialect notes into structured facts like "called twice, mother didn't answer", with schema-enforced JSON and a deterministic fallback so the pipeline still finishes with no API key. Parent phone numbers never enter a prompt.
- The planner deals each facilitator 5 ranked actions per day (outreach doesn't need a school day), promised follow-ups jump the queue, and every note written today feeds tomorrow's rescore. All 70 at-risk students are covered before Quiz 2, for cents per run.
- A Streamlit dashboard shows raw Arabic beside every LLM summary; an eval harness gates prompt changes: hand-labeled goldens 100% field accuracy, LLM judge 4.2/5 on drafts.

## What I cut and why

- No ML risk model. One quiz gives you 200 labels and no outcome history, so a trained model would be impossible to validate and impossible to explain to the people acting on it. Rules can be argued about in a code review.
- No WhatsApp auto-send. Sending machine-written Arabic to parents unreviewed is a trust risk week one doesn't need, and delivery plumbing teaches us nothing. Drafts sit in the brief for a human to review.

## What I'd build next

A feedback loop on outcomes: one tap on the brief to record "done / reached / no answer", joined against Quiz 2 score changes. That shows which interventions move scores, gives the weights real labels, and turns the intervention rate into a measured number.
