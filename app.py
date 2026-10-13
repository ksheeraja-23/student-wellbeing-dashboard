"""Streamlit dashboard – Student lifestyle, academic stress and depression-indicator analysis.
Run:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
from src import charts as C, insights
from src.preprocess import load_clean, FACTORS, SLEEP_ORDER, DIET_ORDER, AGE_ORDER, DEGREE_ORDER
from src.stats import MIN_N

st.set_page_config(page_title="Student Wellbeing Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def get_data():
    return load_clean()


@st.cache_data
def get_report():
    from pathlib import Path
    from src.preprocess import REPORT, LOG
    return (REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""), (pd.read_csv(LOG) if LOG.exists() else pd.DataFrame())


base = get_data()

# ---------------- sidebar ----------------
st.sidebar.title("Filters")


def multi(label, options, key):
    return st.sidebar.multiselect(label, options, default=options, key=key)


gender = multi("Gender", sorted(base.gender.unique()), "g")
ages = multi("Age group", AGE_ORDER, "a")
degs = multi("Degree level", [d for d in DEGREE_ORDER + ["Other"]], "d")
cities = st.sidebar.multiselect("City (empty = all)", sorted(c for c in base.city.unique() if c != "Unknown"), key="c")
sleep = multi("Sleep duration", SLEEP_ORDER, "s")
diet = multi("Dietary habits", DIET_ORDER, "diet")
fam = multi("Family history of mental illness", ["Yes", "No"], "f")
sui = multi("Ever had suicidal thoughts", ["Yes", "No"], "st")
page = st.sidebar.radio("Page", ["Overview", "Sleep & lifestyle", "Stress & academics", "Demographics & geography",
                                 "Relationships", "Data quality & notes"])

m = base.gender.isin(gender) & base.age_group.isin(ages) & base.degree_level.isin(degs) & base.family_history.isin(fam) & base.suicidal_thoughts.isin(sui)
if set(sleep) != set(SLEEP_ORDER): m &= base.sleep_duration.isin(sleep)
if set(diet) != set(DIET_ORDER): m &= base.diet.isin(diet)
if cities: m &= base.city.isin(cities)
df = base[m]
st.sidebar.caption(f"**{len(df):,}** of {len(base):,} students selected")
st.sidebar.info("The depression indicator is a survey label, not a clinical diagnosis. If you or someone you know is struggling, "
                "please talk to a counsellor or call Tele-MANAS (India): 14416.")

st.title("Student Lifestyle, Academic Stress and Well-being")
st.caption("Dataset: Student Depression Dataset (Indian students, survey-based, cross-sectional). Exploratory analysis – associations, not causes.")
if len(df) < MIN_N:
    st.warning(f"Only {len(df)} students match the filters (minimum {MIN_N}). Please widen the filters.")
    st.stop()


def show(fig, note=None):
    st.plotly_chart(fig, width="stretch")
    if note:
        st.caption("💡 " + note)


if page == "Overview":
    k = st.columns(5)
    k[0].metric("Students", f"{len(df):,}")
    k[1].metric("Depression indicator", f"{100 * df.depression.mean():.1f}%", f"{100 * (df.depression.mean() - base.depression.mean()):+.1f} pts vs all")
    k[2].metric("Sleep < 5 hours", f"{100 * (df.sleep_duration == 'Less than 5 hours').mean():.1f}%")
    k[3].metric("Mean academic pressure", f"{df.academic_pressure.mean():.2f} / 5")
    k[4].metric("Mean financial stress", f"{df.financial_stress.mean():.2f} / 5")
    a, b = st.columns([3, 2])
    with a:
        show(C.hours_pressure_histograms(df))
    with b:
        show(C.status_donut(df))
    st.subheader("Auto-insights")
    for s in insights.generate(df, base):
        st.markdown("- " + s)
    show(C.factor_gap(df), "A bigger gap means the depression-indicator rate differs more between groups of that factor. It does not prove the factor causes depression.")
    with st.expander("Data table & download"):
        st.dataframe(df.head(2000), width="stretch")
        st.download_button("Download filtered CSV", df.to_csv(index=False), "filtered_students.csv")

elif page == "Sleep & lifestyle":
    a, b = st.columns(2)
    with a: show(C.sleep_distribution(df))
    with b: show(C.stacked_status(df, "sleep_duration"))
    a, b = st.columns(2)
    with a: show(C.stacked_status(df, "diet"))
    with b: show(C.rate_heatmap(df, "diet", "sleep_duration"), "Cells with fewer than 30 students are hidden.")
    show(C.hours_line(df))

elif page == "Stress & academics":
    a, b = st.columns(2)
    with a: show(C.rate_bar(df, "academic_pressure"))
    with b: show(C.rate_bar(df, "financial_stress"))
    a, b = st.columns(2)
    with a: show(C.cgpa_box(df, st.selectbox("Group CGPA by", ["academic_pressure_level", "sleep_duration", "age_group", "degree_level"],
                                             format_func=lambda x: FACTORS[x])))
    with b: show(C.rate_heatmap(df, "financial_stress", "academic_pressure_level"))
    show(C.pressure_stress_bubble(df))
    a, b = st.columns(2)
    with a: show(C.rate_bar(df, "cgpa_band"))
    with b: show(C.rate_bar(df, "family_history"))

elif page == "Demographics & geography":
    a, b = st.columns(2)
    with a: show(C.age_histogram(df))
    with b: show(C.rate_bar(df, "age_group"))
    a, b = st.columns(2)
    with a: show(C.rate_bar(df[df.degree_level != "Other"], "degree_level"))
    with b: show(C.rate_bar(df, "gender"))
    show(C.city_rates(df), "City differences are small relative to their confidence intervals for many cities; avoid ranking cities.")

elif page == "Relationships":
    show(C.correlation_heatmap(df, st.radio("Method", ["spearman", "pearson"], horizontal=True)),
         "Spearman suits ordinal 1-5 ratings. Values near 0 mean little monotonic association.")
    show(C.sankey(df), "The flow shows how students are grouped across categories, not a causal sequence.")
    x = st.selectbox("Heatmap columns", ["financial_stress", "diet", "sleep_duration", "study_hours_band", "age_group"], format_func=lambda v: FACTORS[v])
    y = st.selectbox("Heatmap rows", ["academic_pressure_level", "sleep_duration", "diet", "age_group", "degree_level"], format_func=lambda v: FACTORS[v])
    show(C.rate_heatmap(df, x, y) if x != y else C.empty("Choose two different factors."))

else:
    rep, log = get_report()
    st.markdown(rep)
    st.subheader("Limitations")
    st.markdown("""- Cross-sectional survey: associations only; no causal or temporal claims.
- The depression indicator is a dataset label, not a clinical diagnosis; the exact collection method is not documented in the file.
- Sample of students from Indian cities; not representative of all students or of MIT-WPU.
- Sleep, diet and pressure are self-reported; the 'suicidal thoughts' item is sensitive and used only in aggregate.
- Small groups are hidden or flagged; always read confidence intervals.""")
