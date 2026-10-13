"""Validation, cleaning and feature engineering for the Student Depression dataset.

Run:  python -m src.preprocess
Reads  data/raw/Student_Depression_Dataset.csv
Writes data/processed/cleaned_student_wellbeing.csv, outputs/data_quality_report.md, outputs/cleaning_log.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Student_Depression_Dataset.csv"
PROCESSED = ROOT / "data" / "processed" / "cleaned_student_wellbeing.csv"
REPORT = ROOT / "outputs" / "data_quality_report.md"
LOG = ROOT / "outputs" / "cleaning_log.csv"

MIN_CITY_RECORDS = 100          # city values with fewer records are data-entry errors (names, degrees, numbers)
SLEEP_ORDER = ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
DIET_ORDER = ["Unhealthy", "Moderate", "Healthy"]
AGE_ORDER = ["18-20", "21-23", "24-26", "27-30", "31+"]
PRESSURE_ORDER = ["Low (1-2)", "Moderate (3)", "High (4-5)"]
HOURS_ORDER = ["0-4 h", "5-8 h", "9-12 h"]
CGPA_ORDER = ["Below 6", "6-7.99", "8+"]
DEGREE_ORDER = ["Class 12", "Bachelor's", "Master's", "Doctorate / MD"]
STATUS = {0: "Not depressed", 1: "Depressed"}

RENAME = {"Academic Pressure": "academic_pressure", "Study Satisfaction": "study_satisfaction",
          "Financial Stress": "financial_stress", "Work/Study Hours": "study_hours", "Sleep Duration": "sleep_duration",
          "Dietary Habits": "diet", "Have you ever had suicidal thoughts ?": "suicidal_thoughts",
          "Family History of Mental Illness": "family_history", "CGPA": "cgpa", "Age": "age", "Gender": "gender",
          "City": "city", "Degree": "degree", "Depression": "depression", "id": "id"}
LABELS = {"age": "Age", "academic_pressure": "Academic pressure (1-5)", "study_satisfaction": "Study satisfaction (1-5)",
          "financial_stress": "Financial stress (1-5)", "study_hours": "Work/study hours per day", "cgpa": "CGPA (0-10)",
          "depression": "Depression indicator (0/1)"}
ORDINAL = ["academic_pressure", "study_satisfaction", "financial_stress", "study_hours", "cgpa", "age", "depression"]
FACTORS = {  # categorical factor -> (column, display name)
    "sleep_duration": "Sleep duration", "diet": "Dietary habits", "academic_pressure_level": "Academic pressure",
    "financial_stress": "Financial stress", "study_hours_band": "Work/study hours", "family_history": "Family history of mental illness",
    "suicidal_thoughts": "Ever had suicidal thoughts", "age_group": "Age group", "degree_level": "Degree level",
    "cgpa_band": "CGPA band", "gender": "Gender"}


def load_raw(path: Path = RAW) -> pd.DataFrame:
    return pd.read_csv(path)


def validate(raw: pd.DataFrame) -> dict:
    """Honest facts about the raw file."""
    s = raw[raw["Profession"] == "Student"]
    city_counts = raw["City"].value_counts()
    return {
        "rows": len(raw), "columns": raw.shape[1],
        "missing": raw.isna().sum()[lambda x: x > 0].to_dict(), "duplicates": int(raw.duplicated().sum()),
        "non_student": int((raw["Profession"] != "Student").sum()),
        "work_pressure_nonzero": int((raw["Work Pressure"] != 0).sum()),
        "job_satisfaction_nonzero": int((raw["Job Satisfaction"] != 0).sum()),
        "invalid_city_values": int((city_counts < MIN_CITY_RECORDS).sum()),
        "invalid_city_rows": int(city_counts[city_counts < MIN_CITY_RECORDS].sum()),
        "ap_zero": int((s["Academic Pressure"] == 0).sum()), "ss_zero": int((s["Study Satisfaction"] == 0).sum()),
        "cgpa_zero": int((s["CGPA"] == 0).sum()), "sleep_others": int((raw["Sleep Duration"] == "Others").sum()),
        "diet_others": int((raw["Dietary Habits"] == "Others").sum()), "age_max": float(raw["Age"].max()),
    }


MASTERS = {"M.Com", "M.Ed", "M.Pharm", "M.Tech", "MA", "MBA", "MCA", "ME", "MHM", "MSc", "LLM"}


def degree_level(d: str) -> str:
    if d == "Class 12": return "Class 12"
    if d in {"PhD", "MD"}: return "Doctorate / MD"
    if d == "Others": return "Other"
    return "Master's" if d in MASTERS else "Bachelor's"   # MBBS, LLB, B.*, BA, BSc, BCA, BBA, BHM, BE -> Bachelor's


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Returns the cleaned frame and a step-by-step log of rows/values affected."""
    log, df = [], raw.copy()

    def step(name, before, after, note):
        log.append({"step": name, "rows_before": before, "rows_after": after, "rows_removed": before - after, "note": note})

    n = len(df)
    df = df.drop_duplicates(); step("Drop exact duplicates", n, len(df), "none found")
    n = len(df); df = df[df["Profession"] == "Student"]
    step("Keep Profession == 'Student'", n, len(df), "non-students are 31 rows and are not the target population")
    df = df.drop(columns=["Profession", "Work Pressure", "Job Satisfaction"])
    log.append({"step": "Drop Profession, Work Pressure, Job Satisfaction", "rows_before": len(df), "rows_after": len(df),
                "rows_removed": 0, "note": "constant/near-constant (>99.9% zero) for students – no information"})
    df = df.rename(columns=RENAME)
    for c in ["gender", "city", "degree", "sleep_duration", "diet", "suicidal_thoughts", "family_history"]:
        df[c] = df[c].astype(str).str.strip()
    n = len(df); df = df[df["financial_stress"].notna()]
    step("Drop rows with missing Financial Stress", n, len(df), "3 missing values")
    n = len(df); df = df[(df["academic_pressure"] >= 1) & (df["study_satisfaction"] >= 1) & (df["cgpa"] > 0)]
    step("Drop rows with out-of-scale values", n, len(df), "Academic Pressure = 0, Study Satisfaction = 0 or CGPA = 0 (scales are 1-5 and 0-10; 0 is invalid)")
    counts = df["city"].value_counts()
    bad_city = df["city"].map(counts) < MIN_CITY_RECORDS
    log.append({"step": "Set invalid city values to 'Unknown'", "rows_before": len(df), "rows_after": len(df),
                "rows_removed": 0, "note": f"{int(bad_city.sum())} rows had city entries that are names/degrees/numbers (<{MIN_CITY_RECORDS} records); kept, city = Unknown"})
    df.loc[bad_city, "city"] = "Unknown"
    nso, ndo = int((df["sleep_duration"] == "Others").sum()), int((df["diet"] == "Others").sum())
    df["sleep_duration"] = df["sleep_duration"].replace("Others", np.nan)
    df["diet"] = df["diet"].replace("Others", np.nan)
    log.append({"step": "Set sleep/diet 'Others' to missing", "rows_before": len(df), "rows_after": len(df), "rows_removed": 0,
                "note": f"{nso} sleep and {ndo} diet responses were 'Others' (not interpretable); excluded from sleep/diet charts only"})
    for c in ("academic_pressure", "study_satisfaction", "financial_stress", "age", "study_hours"):
        df[c] = df[c].astype(int)
    return df.reset_index(drop=True), log


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], [17, 20, 23, 26, 30, 100], labels=AGE_ORDER)
    df["academic_pressure_level"] = pd.cut(df["academic_pressure"], [0, 2, 3, 5], labels=PRESSURE_ORDER)
    df["study_hours_band"] = pd.cut(df["study_hours"], [-1, 4, 8, 12], labels=HOURS_ORDER)
    df["cgpa_band"] = pd.cut(df["cgpa"], [0, 6, 8, 10.01], labels=CGPA_ORDER, right=False)
    df["degree_level"] = df["degree"].map(degree_level)
    df["depression_status"] = df["depression"].map(STATUS)
    return df


def write_report(rep, log, final):
    L = ["# Data quality report", "",
         f"- Raw file: **{rep['rows']:,} rows × {rep['columns']} columns**; exact duplicates: **{rep['duplicates']}**",
         f"- Missing values in raw file: {rep['missing']} (Financial Stress only)",
         f"- Non-student rows: **{rep['non_student']}**; Work Pressure non-zero: {rep['work_pressure_nonzero']}; Job Satisfaction non-zero: {rep['job_satisfaction_nonzero']} → columns dropped",
         f"- City column: **{rep['invalid_city_values']} invalid values** across {rep['invalid_city_rows']} rows (e.g. person names, degree names, numbers)",
         f"- Out-of-scale values among students: Academic Pressure=0: {rep['ap_zero']}, Study Satisfaction=0: {rep['ss_zero']}, CGPA=0: {rep['cgpa_zero']}",
         f"- 'Others' responses: sleep {rep['sleep_others']}, diet {rep['diet_others']}",
         f"- Maximum age: {rep['age_max']:.0f} (kept: adult learners are plausible)",
         f"- **Final analysis file: {len(final):,} rows × {final.shape[1]} columns**", "", "## Cleaning log", "",
         "| Step | Rows before | Rows after | Removed | Note |", "|---|---|---|---|---|"]
    for r in log:
        L.append(f"| {r['step']} | {r['rows_before']:,} | {r['rows_after']:,} | {r['rows_removed']} | {r['note']} |")
    L += ["", "## Derived columns", "", "`age_group`, `academic_pressure_level`, `study_hours_band`, `cgpa_band`, `degree_level`, `depression_status`. "
          "Degree level: Class 12; Bachelor's (incl. MBBS, B.*, BA, BSc, BCA, BBA, BHM, LLB, BE); Master's (M.Com, M.Ed, M.Pharm, M.Tech, MA, MBA, MCA, ME, MHM, MSc, LLM); Doctorate/MD (PhD, MD).",
          "", "## Interpretation caution", "", "`Depression` is a 0/1 survey indicator supplied with the dataset, not a clinical diagnosis. "
          "Results describe associations in this sample only."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    pd.DataFrame(log).to_csv(LOG, index=False)


def _order(df):
    for col, cats in {"sleep_duration": SLEEP_ORDER, "diet": DIET_ORDER, "age_group": AGE_ORDER,
                      "academic_pressure_level": PRESSURE_ORDER, "study_hours_band": HOURS_ORDER,
                      "cgpa_band": CGPA_ORDER, "degree_level": DEGREE_ORDER + ["Other"]}.items():
        df[col] = pd.Categorical(df[col], cats, ordered=True)
    return df


def load_clean() -> pd.DataFrame:
    if not PROCESSED.exists():
        run()
    return _order(pd.read_csv(PROCESSED))


def run() -> pd.DataFrame:
    raw = load_raw(); rep = validate(raw)
    cleaned, log = clean(raw); final = engineer(cleaned)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(PROCESSED, index=False)
    write_report(rep, log, final)
    print(f"Saved {PROCESSED.relative_to(ROOT)}  shape={final.shape}")
    return _order(final)


if __name__ == "__main__":
    run()
