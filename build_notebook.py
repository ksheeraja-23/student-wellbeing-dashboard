"""Generates student_wellbeing_analysis.ipynb. Run: python build_notebook.py (run_all.py also executes it)."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))

M("""# Student Lifestyle, Academic Stress and Depression Indicator – Exploratory Analysis
**Subject:** Data Visualization Using Python (AID30050) · **Institution:** MIT World Peace University
**Stack:** pandas, NumPy, Plotly, Seaborn, Streamlit

**Dataset:** *Student Depression Dataset* (27,901 survey responses from students in Indian cities; Kaggle). Variables: gender, age, city, degree,
CGPA, academic pressure, study satisfaction, financial stress, sleep duration, dietary habits, work/study hours, suicidal thoughts, family history
and a 0/1 `Depression` indicator.

**Aim:** apply preprocessing, exploratory analysis and interactive visualisation to see how lifestyle, academic pressure and background
characteristics are associated with the depression indicator.

> **Interpretation caution.** `Depression` is a survey label in the dataset, not a clinical diagnosis. This is a cross-sectional sample:
> every result is an *association*, not a cause. The suicidal-thoughts question is sensitive and is used only in aggregate.""")
C("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
import plotly.io as pio
from src import preprocess as P, charts as C
from src.stats import rate_table
sns.set_theme(style="whitegrid")
pio.renderers.default = "notebook_connected"   # use "png" if your viewer cannot run JavaScript""")
M("## 1. Load and inspect the raw data")
C("raw = P.load_raw()\nprint(raw.shape)\ndisplay(raw.head())\nraw.info()")
C("display(raw.describe().T.round(2))\nfor c in ['Gender','Profession','Sleep Duration','Dietary Habits','Degree']: print(raw[c].value_counts().head(8).to_dict())")
M("## 2. Data-quality checks (real problems found)")
C("""rep = P.validate(raw)
pd.Series({k: v for k, v in rep.items()}).to_frame('value')""")
C("""print(raw['City'].value_counts().tail(12).to_dict())   # junk entries: person names, degrees, numbers
print(raw[['Work Pressure','Job Satisfaction']].ne(0).sum().to_dict(), 'non-zero values out of', len(raw))""")
M("""**Findings.** No duplicates and only 3 missing values (Financial Stress). But the file does have genuine issues:
- 31 respondents are not students (profession column) and `Work Pressure` / `Job Satisfaction` are zero for virtually everyone → dropped.
- 22 invalid `City` values (person names, degree names, numbers) covering 26 rows → set to *Unknown*.
- Out-of-scale values: Academic Pressure = 0, Study Satisfaction = 0 and CGPA = 0 (scales are 1-5 and 0-10) → 14 rows removed.
- `Others` in sleep (18) and diet (12) is not interpretable → set to missing for those charts.""")
M("## 3. Cleaning and feature engineering")
C("""cleaned, log = P.clean(raw)
display(pd.DataFrame(log))
df = P.engineer(cleaned)
df.to_csv(P.PROCESSED, index=False)
df = P.load_clean()
print(raw.shape, '->', df.shape)
display(df.head())""")
M("""Derived columns: `age_group`, `academic_pressure_level` (Low 1-2 / Moderate 3 / High 4-5), `study_hours_band`, `cgpa_band`, `degree_level`, `depression_status`.
The raw file is untouched; the processed file is `data/processed/cleaned_student_wellbeing.csv`.""")
M("## 4. Distributions")
C("C.status_donut(df).show()")
M("**Interpretation.** 58.5 % of the 27,853 students carry the depression indicator. This is the sample's share – it is **not** a prevalence estimate for Indian students.")
C("C.sleep_distribution(df).show()")
M("**Interpretation.** Short sleep is the most common answer: 29.8 % report sleeping less than 5 hours, 26.4 % 7-8 hours, 22.2 % 5-6 hours and 21.7 % more than 8 hours.")
C("C.age_histogram(df).show()")
M("**Interpretation.** Median age is 25 and half of the students are 25 or younger, with a long tail up to 59. Young students (18-20) are a large group.")
C("C.hours_pressure_histograms(df).show()")
M("**Interpretation.** Academic pressure and financial stress are spread across the whole 1-5 range (pressure most often 3 or 5; financial stress most often 5 or 4); work/study hours are skewed to long days (10-12 h).")
M("## 5. Depression rate by factor (with 95 % confidence intervals)")
C("C.rate_bar(df, 'academic_pressure').show()")
M("**Interpretation.** The strongest gradient in the data: depression-indicator rate rises from 19 % (pressure 1) to 86 % (pressure 5).")
C("C.rate_bar(df, 'financial_stress').show()")
M("**Interpretation.** Same pattern for financial stress: 32 % at level 1 to 81 % at level 5.")
C("C.stacked_status(df, 'sleep_duration').show()")
M("**Interpretation.** Students sleeping under 5 hours show the highest rate (64.5 %) and those sleeping over 8 hours the lowest (50.9 %). The relation is not perfectly monotonic (5-6 h: 56.9 %, 7-8 h: 59.5 %), so sleep alone is only a modest separator.")
C("C.stacked_status(df, 'diet').show()")
M("**Interpretation.** Unhealthy diet 70.7 %, moderate 56.0 %, healthy 45.3 % – a clear gradient.")
C("C.hours_line(df).show()")
M("**Interpretation.** The rate rises with daily work/study hours, from 43.5 % (0-4 h) to 67.5 % (9-12 h).")
C("C.rate_bar(df, 'age_group').show(); C.rate_bar(df[df.degree_level != 'Other'], 'degree_level').show()")
M("**Interpretation.** Younger students show higher rates (18-20: 72.4 % vs 31+: 40.8 %); Class 12 students are highest by degree level (70.8 %). Age and degree level overlap strongly, so they cannot be separated here.")
C("C.rate_bar(df, 'gender').show(); C.rate_bar(df, 'family_history').show(); C.rate_bar(df, 'cgpa_band').show()")
M("**Interpretation.** Gender shows no difference (58.4 % vs 58.6 %). Family history shows a small gap (61.2 % vs 56.0 %). CGPA bands differ little (55.6 % to 60.7 %), and higher CGPA is, if anything, slightly *more* common among students with the indicator – academic marks alone do not separate groups.")
C("C.cgpa_box(df).show()")
M("**Interpretation.** The CGPA distribution is almost identical across pressure levels (median ≈ 7.8) and across depression status (median 7.85 vs 7.64).")
M("## 6. Relationships")
C("C.correlation_heatmap(df, 'spearman').show()")
M("""**Interpretation.** Spearman rank correlations with the indicator: academic pressure +0.47, financial stress +0.36, work/study hours +0.20, study satisfaction −0.17, age −0.23, CGPA ≈ 0.
The pressure and stress variables are only weakly correlated with each other (+0.15), so they add separate information.""")
C("sns.heatmap(df[P.ORDINAL].corr(method='spearman'), annot=True, fmt='.2f', cmap='RdBu', center=0, vmin=-1, vmax=1); plt.title('Spearman correlation (Seaborn)'); plt.show()")
C("C.rate_heatmap(df, 'financial_stress', 'academic_pressure_level').show()")
M("**Interpretation.** The two stressors compound: with low academic pressure and financial stress 1 the rate is 11 %; with high pressure and financial stress 5 it is 94 %.")
C("C.pressure_stress_bubble(df).show()")
C("C.rate_heatmap(df, 'diet', 'sleep_duration').show()")
M("**Interpretation.** Diet and sleep both matter: <5 h sleep with unhealthy diet = 76 % vs >8 h with healthy diet = 37 %.")
C("C.sankey(df).show()")
M("**Interpretation.** The Sankey shows how students are distributed across sleep, pressure and the indicator. High-pressure students flow mostly into the *Depressed* group. This shows grouping, not a causal path.")
C("C.city_rates(df).show()")
M("**Interpretation.** City rates range from about 52 % (Jaipur, Nashik) to 67 % (Hyderabad, Ahmedabad); many confidence intervals overlap, so avoid ranking cities.")
C("C.factor_gap(df).show()")
M("""**Interpretation.** Ranked by the gap between the highest and lowest group: suicidal thoughts (56 pts), academic pressure (54), financial stress (49), age group (32), diet (25), work/study hours (24), then smaller gaps for degree level, sleep, family history, CGPA and gender (≈0).
The suicidal-thoughts item is closely related to the indicator itself and should not be read as an independent 'risk factor'.""")
M("""## 7. Conclusions and limitations
- Academic pressure, financial stress, long work/study hours and unhealthy diet show the largest differences in the depression-indicator rate; sleep shows a smaller, non-monotonic gradient; gender and CGPA show almost none.
- Younger students and Class 12 students show higher rates, but age and education stage are intertwined.
- **Limitations:** cross-sectional survey (no causation or timing); the indicator is a label, not a diagnosis, and its collection method is not documented in the file; self-reported variables; Indian-city sample only (not representative of all students or MIT-WPU);
  the cleaned sample is 27,853 of 27,901 rows; many patterns may be confounded (e.g. age vs education stage).

**Dataset source:** Student Depression Dataset (Kaggle, widely shared). Confirm the exact author and licence before citing.""")
nbf.write(nb, "student_wellbeing_analysis.ipynb")
