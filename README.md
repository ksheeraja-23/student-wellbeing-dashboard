# Student Lifestyle, Academic Stress and Well-being – Data Visualization Project

**Subject:** Data Visualization Using Python (AID30050) · **Institution:** MIT World Peace University
**Stack:** Python, pandas, NumPy, Plotly, Seaborn, Streamlit

Exploratory analysis and an interactive dashboard on the *Student Depression Dataset* (27,901 survey responses from students in Indian
cities). It investigates how sleep, diet, academic pressure, financial stress, study hours and background characteristics are associated
with a 0/1 depression indicator.

> The depression indicator is a survey label, **not a clinical diagnosis**. The data are cross-sectional, so all findings are associations,
> not causes. The dashboard is for educational exploration only; it contains no individual risk predictor by design.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py          # interactive dashboard (opens in the browser)
python run_all.py             # optional: rebuild cleaned CSV, 19 figures and the executed notebook
```
PNG export needs Chrome/Chromium for kaleido (set `BROWSER_PATH=/path/to/chrome` if it is not found); HTML export always works.

## Structure
| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (6 pages, 8 sidebar filters, KPI tiles, auto-insights) |
| `student_wellbeing_analysis.ipynb` | Executed notebook: inspection, quality checks, cleaning, 25+ charts with interpretations |
| `src/preprocess.py` | Validation, cleaning (logged), feature engineering |
| `src/charts.py` | All Plotly figure builders (used by app, notebook and exporter) |
| `src/stats.py` | Wilson 95 % confidence intervals and rate tables |
| `src/insights.py` | Data-driven auto-insight sentences |
| `data/raw/` · `data/processed/` | Original file kept separate from `cleaned_student_wellbeing.csv` |
| `outputs/figures/` | 19 report figures as PNG + interactive HTML |
| `outputs/data_quality_report.md`, `outputs/cleaning_log.csv` | Genuine before/after cleaning results |
| `report/PROJECT_REPORT.md` | Written report: problem, method, findings, limitations, references |

## Dashboard pages
Overview · Sleep & lifestyle · Stress & academics · Demographics & geography · Relationships (Spearman/Pearson heatmap, Sankey, configurable rate heatmap) · Data quality & notes.
Filters: gender, age group, degree level, city, sleep duration, diet, family history, suicidal-thoughts item. Rate charts show 95 % confidence
intervals, groups under 30 students are hidden, and every KPI/chart/insight updates with the filters.

## What preprocessing actually did
Raw 27,901 × 18 → cleaned 27,853 × 21. No duplicates; 3 missing Financial Stress values; 31 non-students removed; Work Pressure and Job Satisfaction
dropped (≈ all zeros); 14 rows with out-of-scale values (pressure/satisfaction/CGPA = 0) removed; 22 invalid city strings (26 rows) set to *Unknown*;
sleep/diet "Others" set to missing; six derived columns added. See `outputs/data_quality_report.md`.

## Limitations
Cross-sectional self-report data; indicator is not a diagnosis and its collection method is undocumented in the file; Indian-city sample, not representative
of all students or MIT-WPU; age and education stage are confounded; small differences between cities lie within their confidence intervals.
Confirm the dataset's author and licence on its Kaggle page and cite it in your submission. Get your teacher's approval for the topic and dataset.

If you or someone you know is struggling, please talk to a counsellor or call Tele-MANAS (India): 14416.
