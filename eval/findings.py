"""Reproduce the data findings quoted in analysis.md, from raw data.

    python -m eval.findings          (note_insights.csv needs one pipeline run)

Each section prints the evidence behind one claim - the Q&A receipts.
"""
import pandas as pd

from src.config import DATA_DIR, OUTPUT_DIR, PASS_SCORE

AR2EN = {'احمد': 'Ahmad', 'أحمد': 'Ahmad', 'يوسف': 'Yousef', 'فاطمة': 'Fatima',
         'فاطمه': 'Fatima', 'حسن': 'Hassan', 'خالد': 'Khalid', 'عمر': 'Omar',
         'نوره': 'Nora', 'نورة': 'Nora', 'نورا': 'Nora', 'ليلى': 'Layla',
         'أمل': 'Amal', 'امل': 'Amal', 'سارة': 'Sara', 'ساره': 'Sara',
         'محمد': 'Mohammed', 'عبدالله': 'Abdullah', 'ريم': 'Reem', 'مريم': 'Maryam',
         'هند': 'Hind', 'سلمى': 'Salma', 'دانة': 'Dana', 'دانه': 'Dana',
         'جواهر': 'Jawaher', 'غادة': 'Ghada', 'غاده': 'Ghada', 'طلال': 'Talal',
         'بدر': 'Badr', 'راكان': 'Rakan', 'لمى': 'Lama', 'عايشة': 'Aisha',
         'عائشة': 'Aisha', 'منى': 'Mona', 'سعد': 'Saad', 'فهد': 'Fahad',
         'ماجد': 'Majed', 'نايف': 'Naif', 'وليد': 'Waleed',
         'عبدالعزيز': 'Abdulaziz', 'رهف': 'Rahaf'}

daily = pd.read_csv(DATA_DIR / "student_daily_metrics.csv", parse_dates=["date"])
notes = pd.read_csv(DATA_DIR / "facilitator_notes.csv", parse_dates=["date"])
meta = pd.read_csv(DATA_DIR / "student_metadata.csv")
meta["first"] = meta.student_name.str.split().str[0]
quiz = (daily.dropna(subset=["last_quiz_score"]).sort_values("date")
        .groupby("student_id").last_quiz_score.last())
print("== 1. Who failed, who got helped ==")
failures = quiz[quiz < PASS_SCORE].index
noted = set(notes.student_id)
print(f"failed Quiz 1 (<{PASS_SCORE}): {len(failures)}/200; "
      f"with zero notes ever: {len(set(failures) - noted)}")
ins_path = OUTPUT_DIR / "note_insights.csv"
if ins_path.exists():
    ins = pd.read_csv(ins_path)
    f = ins[ins.student_id.isin(failures)]
    received = (f.contact_outcome == "reached") | (f.intervention_succeeded == True)
    attempted = f.contact_outcome.isin(["no_answer", "refused"]) & ~received
    print(f"of the {len(failures)} failures: {int(received.sum())} show a received "
          f"intervention ({int(received.sum())/len(failures):.0%}), "
          f"{int(attempted.sum())} attempted-never-landed, "
          f"{len(f) - int(received.sum()) - int(attempted.sum())} observation-only notes")

print("\n== 2. Notes name the wrong student (but sit under the right one) ==")
num = lambda s: int(s[1:])
mism = comb = 0
for r in notes.itertuples():
    mentioned = {en for ar, en in AR2EN.items() if ar in r.note_text}
    own = meta.set_index("student_id")["first"].get(r.student_id, "")
    if mentioned and own not in mentioned:
        mism += 1
print(f"notes naming a different student: {mism}/{len(notes)} "
      "(name-offset histogram shows a comb at -2+12k: metadata names cycle every 12 IDs)")
absent = notes[notes.note_text.str.contains("ما حضر|غاب|لسا ما في|ما جا")]
att = daily.set_index(["student_id", "date"]).session_attended_min
for off, label in [(0, "filed student"), (-2, "shifted candidate")]:
    ok = tot = 0
    for r in absent.itertuples():
        a = att.get((f"S{num(r.student_id) + off:03d}", r.date))
        if a is not None and not pd.isna(a):
            tot += 1
            ok += (a == 0)
    print(f"absence notes vs {label}'s attendance that day: {ok}/{tot} actually absent")

print("\n== 3. Other planted anomalies ==")
per_day = daily.dropna(subset=["last_quiz_score"]).groupby("date").size()
quiz_day = per_day[per_day >= 100].index.min()
early = daily[(daily.last_quiz_score.notna()) & (daily.date < quiz_day)]
print(f"quiz scores dated before quiz day ({quiz_day.date()}): "
      f"{len(early)} ({', '.join(early.student_id)})")
print(f"session weekdays: {sorted(set(daily.date.dt.day_name()))} - matches neither "
      "the stated Mon-Thu nor the Saudi Sun-Thu week")
print(f"targets below the pass line: {int((meta.target_score < PASS_SCORE).sum())} "
      f"(all Remedial; targets span {meta.target_score.min()}-{meta.target_score.max()}); "
      f"students below own target: {int((quiz < meta.set_index('student_id').target_score).sum())}/200")

print("\n== 4. Backtest: do pre-quiz leading indicators predict failure? ==")
pre = daily[daily.date < quiz_day].copy()
pre["practice_questions"] = pre.practice_questions.clip(0, 60)
g = pre.groupby("student_id")
f = pd.DataFrame({"att": g.session_attended_min.mean(), "prac": g.practice_questions.mean()})
w1 = pre[pre.date <= "2025-10-04"].groupby("student_id").session_attended_min.mean()
w2 = pre[pre.date > "2025-10-04"].groupby("student_id").session_attended_min.mean()
f["lead"] = (20 * (1 - f.att / 90) + 10 * ((w1 - w2).clip(lower=0) / 30).clip(0, 1)
             + 15 * (1 - (f.prac / 30).clip(0, 1)))
d = f.join(quiz).dropna()
d["failed"] = d.last_quiz_score < PASS_SCORE
r = d.lead.rank()
n1, n0 = d.failed.sum(), (~d.failed).sum()
auc = (r[d.failed].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
cap = len(set(d.nlargest(int(n1), "lead").index) & set(d[d.failed].index))
print(f"AUC {auc:.3f}; top-{int(n1)} by leading score captures {cap}/{int(n1)} eventual failures")
