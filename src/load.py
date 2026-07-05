"""Load the three CSVs, clean known data-quality issues, build per-student features.

Every cleaning action is counted in the returned `cleaning_log` so the run
summary can prove the data was handled programmatically, not silently.
"""
import pandas as pd

from .config import DATA_DIR, PRACTICE_CAP, SESSION_MAX_MIN


def load_data():
    daily = pd.read_csv(DATA_DIR / "student_daily_metrics.csv", parse_dates=["date"])
    notes = pd.read_csv(DATA_DIR / "facilitator_notes.csv", parse_dates=["date"])
    students = pd.read_csv(DATA_DIR / "student_metadata.csv")

    log = {}

    # Referential integrity: drop rows pointing at unknown students (log them).
    known = set(students.student_id)
    for name, df in (("daily", daily), ("notes", notes)):
        orphans = ~df.student_id.isin(known)
        log[f"{name}_orphan_rows_dropped"] = int(orphans.sum())
    daily = daily[daily.student_id.isin(known)].copy()
    notes = notes[notes.student_id.isin(known)].copy()

    dupes = daily.duplicated(["student_id", "date"])
    log["daily_duplicate_rows_dropped"] = int(dupes.sum())
    daily = daily[~dupes]

    # Attendance: keep nulls as NaN (don't invent attendance), clamp impossible values.
    log["attendance_nulls_kept_as_missing"] = int(daily.session_attended_min.isna().sum())
    bad_att = (daily.session_attended_min < 0) | (daily.session_attended_min > SESSION_MAX_MIN)
    log["attendance_out_of_range_clamped"] = int(bad_att.sum())
    daily.loc[:, "session_attended_min"] = daily.session_attended_min.clip(0, SESSION_MAX_MIN)

    # Practice: winsorize data-entry outliers (values >60 observed up to 120).
    outliers = daily.practice_questions > PRACTICE_CAP
    log["practice_outliers_capped"] = int(outliers.sum())
    daily.loc[:, "practice_questions"] = daily.practice_questions.clip(0, PRACTICE_CAP)

    # Quiz scores should only exist from quiz day onward. Flag rows dated before
    # the first day a majority of students have one (e.g. S042 scored on Oct 8,
    # two days before Quiz 1) - kept, since the value matches later rows, but logged.
    scored = daily.dropna(subset=["last_quiz_score"])
    per_day = scored.groupby("date").size()
    quiz_day = per_day[per_day >= 0.5 * daily.student_id.nunique()].index.min()
    log["quiz_scores_dated_before_quiz_day"] = int((scored.date < quiz_day).sum())

    notes.loc[:, "note_text"] = notes.note_text.fillna("").str.strip()
    log["empty_notes_dropped"] = int((notes.note_text == "").sum())
    notes = notes[notes.note_text != ""]

    return daily, notes, students, log


def build_features(daily: pd.DataFrame, notes: pd.DataFrame, students: pd.DataFrame,
                   run_date: str) -> pd.DataFrame:
    """One row per student with the signals risk scoring needs."""
    run_ts = pd.Timestamp(run_date)
    daily = daily[daily.date <= run_ts]
    mid = daily.date.min() + (run_ts - daily.date.min()) / 2  # split window in half for trends

    g = daily.groupby("student_id")
    feats = pd.DataFrame({
        "avg_attend_min": g.session_attended_min.mean(),
        "avg_practice": g.practice_questions.mean(),
        "days_observed": g.size(),
    })

    w1 = daily[daily.date <= mid].groupby("student_id")
    w2 = daily[daily.date > mid].groupby("student_id")
    feats["attend_trend"] = w2.session_attended_min.mean() - w1.session_attended_min.mean()
    feats["zero_practice_days_recent"] = (
        daily[daily.date > mid].assign(z=lambda d: d.practice_questions == 0)
        .groupby("student_id").z.sum()
    )

    # Quiz 1 score = last non-null last_quiz_score per student.
    quiz = (daily.dropna(subset=["last_quiz_score"])
            .sort_values("date").groupby("student_id").last_quiz_score.last())
    feats["quiz_score"] = quiz

    # Sudden dropout: absent on both of the last two recorded days.
    last2 = daily.sort_values("date").groupby("student_id").tail(2)
    feats["absent_last2"] = last2.groupby("student_id").session_attended_min.max() == 0

    note_g = notes.groupby("student_id")
    feats["note_count"] = note_g.size()
    feats["last_note_date"] = note_g.date.max()
    feats["note_count"] = feats.note_count.fillna(0).astype(int)
    feats["days_since_last_note"] = (run_ts - feats.last_note_date).dt.days

    out = students.merge(feats, left_on="student_id", right_index=True, how="left")
    out["note_count"] = out.note_count.fillna(0).astype(int)
    out["zero_practice_days_recent"] = out.zero_practice_days_recent.fillna(0).astype(int)
    out["absent_last2"] = out.absent_last2.fillna(False).astype(bool)
    return out


if __name__ == "__main__":
    from .config import RUN_DATE
    daily, notes, students, log = load_data()
    feats = build_features(daily, notes, students, RUN_DATE)
    failing = (feats.quiz_score < 60).sum()
    no_contact = ((feats.quiz_score < 60) & (feats.note_count == 0)).sum()
    print(f"OK: {len(feats)} students, {failing} failing, {no_contact} failing with zero notes")
    print("cleaning log:", log)
