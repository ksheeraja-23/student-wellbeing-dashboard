"""Data-driven insight sentences. They describe associations only and never claim causation."""
from __future__ import annotations
import pandas as pd
from .stats import rate_table, MIN_N


def _fmt(v):
    return str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v)


def _top_bottom(df, col, name, min_n=100):
    t = rate_table(df, col, min_n)
    if len(t) < 2: return None
    hi, lo = t.loc[t.rate.idxmax()], t.loc[t.rate.idxmin()]
    return f"By **{name}**, the depression-indicator rate is highest for **{_fmt(hi[col])}** ({hi.rate:.0f}%, n={int(hi.n):,}) and lowest for **{_fmt(lo[col])}** ({lo.rate:.0f}%, n={int(lo.n):,})."


def generate(df: pd.DataFrame, baseline: pd.DataFrame) -> list[str]:
    if len(df) < MIN_N:
        return [f"Only {len(df)} students match the filters – too few for reliable insights (minimum {MIN_N})."]
    out = []
    r, rb = 100 * df.depression.mean(), 100 * baseline.depression.mean()
    cmp = "about the same as" if abs(r - rb) < 1 else ("higher than" if r > rb else "lower than")
    out.append(f"**{r:.1f}%** of the {len(df):,} selected students have the depression indicator, {cmp} the full-sample rate of {rb:.1f}%.")
    for col, name in (("sleep_duration", "sleep duration"), ("academic_pressure", "academic pressure (1-5)"), ("financial_stress", "financial stress (1-5)"),
                      ("diet", "dietary habits")):
        s = _top_bottom(df, col, name)
        if s: out.append(s)
    for col, label in (("suicidal_thoughts", "reported ever having suicidal thoughts"), ("family_history", "have a family history of mental illness")):
        t = rate_table(df, col, 30).set_index(col)
        if {"Yes", "No"} <= set(t.index):
            out.append(f"Students who {label}: **{t.loc['Yes','rate']:.0f}%** have the depression indicator vs **{t.loc['No','rate']:.0f}%** of those who do not.")
    c = df[["academic_pressure", "financial_stress", "study_satisfaction", "study_hours", "cgpa", "age"]].corrwith(df["depression"], method="spearman").dropna()
    if len(c):
        k = c.abs().idxmax()
        out.append(f"Among the numeric variables, **{k.replace('_', ' ')}** has the strongest rank association with the depression indicator (Spearman ρ = {c[k]:.2f}). Association does not imply causation.")
    return out
