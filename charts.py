"""Plotly figure builders shared by the notebook, the dashboard and the PNG exporter.

Colour roles: Depressed = orange, Not depressed = blue (fixed everywhere). Magnitude = single-hue ramps.
Diverging (correlation) = blue / grey / orange. Every chart has a title, labelled axes and hover values.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .preprocess import LABELS, ORDINAL, FACTORS, SLEEP_ORDER, PRESSURE_ORDER
from .stats import rate_table, MIN_N

NAMES = {**FACTORS, "academic_pressure": "Academic pressure (1-5)", "financial_stress": "Financial stress (1-5)",
         "study_hours": "Work/study hours per day"}
DEP, NOT = "#eb6834", "#2a78d6"
BLUE = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SEQ_ORANGE = [[0, "#fdf0ea"], [.25, "#f7c4ab"], [.5, "#f09670"], [.75, "#eb6834"], [1, "#a9400f"]]
DIV = [[0, "#eb6834"], [0.5, "#f1f0ed"], [1, "#2a78d6"]]


def _style(fig, title, height=430, legend=True):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16, color=INK)), template="plotly_white", height=height,
        font=dict(color=MUTED, size=12), paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        margin=dict(l=60, r=24, t=64, b=60), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, title_text=""),
        hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID)
    return fig


def empty(msg="Not enough records for this chart with the current filters."):
    fig = go.Figure(); fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _style(fig, "", 300, False)


# ---------------- A. distributions ----------------
def sleep_distribution(df):
    d = df.dropna(subset=["sleep_duration"]).groupby("sleep_duration", observed=True).size().reindex(SLEEP_ORDER, fill_value=0)
    pct = 100 * d / max(d.sum(), 1)
    fig = go.Figure(go.Bar(x=d.index, y=d.values, marker=dict(color=BLUE, cornerradius=4), text=[f"{p:.0f}%" for p in pct],
                           textposition="outside", customdata=pct, hovertemplate="%{x}<br>%{y:,} students (%{customdata:.1f}%)<extra></extra>"))
    fig.update_xaxes(title="Reported sleep duration"); fig.update_yaxes(title="Students", range=[0, d.max() * 1.15 if d.max() else 1])
    return _style(fig, "How long do students sleep?", 400, False)


def status_donut(df):
    d = df["depression_status"].value_counts().reindex(["Depressed", "Not depressed"]).fillna(0)
    fig = go.Figure(go.Pie(labels=d.index, values=d.values, hole=.58, sort=False, marker=dict(colors=[DEP, NOT], line=dict(color="#fcfcfb", width=2)),
                           textinfo="percent", hovertemplate="%{label}: %{value:,} students (%{percent})<extra></extra>"))
    return _style(fig, "Share of students with the depression indicator", 400)


def age_histogram(df):
    fig = go.Figure()
    for s, c in (("Not depressed", NOT), ("Depressed", DEP)):
        fig.add_histogram(x=df.loc[df.depression_status == s, "age"], name=s, marker=dict(color=c, line=dict(color="#fcfcfb", width=1)),
                          xbins=dict(start=17.5, end=60.5, size=1), hovertemplate="Age %{x}: %{y} students<extra>" + s + "</extra>")
    fig.update_layout(barmode="stack"); fig.update_xaxes(title="Age (years)"); fig.update_yaxes(title="Students")
    return _style(fig, "Age distribution of students", 400)


def hours_pressure_histograms(df):
    cols = ["academic_pressure", "study_satisfaction", "financial_stress", "study_hours"]
    fig = make_subplots(rows=1, cols=4, subplot_titles=[LABELS[c] for c in cols], horizontal_spacing=.06)
    for i, c in enumerate(cols, 1):
        d = df[c].value_counts().sort_index()
        fig.add_bar(x=d.index, y=d.values, marker=dict(color=BLUE, cornerradius=3), showlegend=False,
                    hovertemplate=f"{LABELS[c]}: %{{x}}<br>%{{y:,}} students<extra></extra>", row=1, col=i)
    fig.update_yaxes(title_text="Students", col=1)
    return _style(fig, "Distribution of pressure, satisfaction, stress and study hours", 380, False)


# ---------------- B. depression rate by factor ----------------
def rate_bar(df, col, title=None, xtitle=None, horizontal=False, min_n=MIN_N, height=430):
    t = rate_table(df, col, min_n)
    if t.empty:
        return empty()
    overall = 100 * df["depression"].mean()
    x, y = (t["rate"], t[col].astype(str)) if horizontal else (t[col].astype(str), t["rate"])
    err = dict(type="data", symmetric=False, array=t["hi"] - t["rate"], arrayminus=t["rate"] - t["lo"], color=MUTED, thickness=1.2, width=4)
    fig = go.Figure(go.Bar(x=x, y=y, orientation="h" if horizontal else "v", marker=dict(color=DEP, cornerradius=4),
                           **({"error_x": err} if horizontal else {"error_y": err}),
                           customdata=np.c_[t["n"], t["lo"], t["hi"]],
                           hovertemplate="%{" + ("y" if horizontal else "x") + "}<br>Depression rate: %{" + ("x" if horizontal else "y") + ":.1f}%"
                           "<br>95% CI %{customdata[1]:.1f}–%{customdata[2]:.1f}%<br>n = %{customdata[0]:,}<extra></extra>"))
    (fig.add_vline if horizontal else fig.add_hline)(**{("x" if horizontal else "y"): overall}, line=dict(color=MUTED, dash="dot"),
                                                     annotation_text=f"Overall {overall:.0f}%", annotation_position="top left" if not horizontal else "top")
    name = xtitle or NAMES.get(col, col)
    (fig.update_yaxes if not horizontal else fig.update_xaxes)(title="Students with depression indicator (%)", range=[0, 100])
    (fig.update_xaxes if not horizontal else fig.update_yaxes)(title=name, **({"autorange": "reversed"} if horizontal else {}))
    return _style(fig, title or f"Depression rate by {name.lower()} (bars: 95% CI)", height, False)


def stacked_status(df, col, title=None, xtitle=None):
    d = df.dropna(subset=[col])
    ct = pd.crosstab(d[col], d["depression_status"])
    ct = ct.reindex(columns=["Not depressed", "Depressed"], fill_value=0)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for s, c in (("Not depressed", NOT), ("Depressed", DEP)):
        fig.add_bar(x=pct.index.astype(str), y=pct[s], name=s, marker=dict(color=c, line=dict(color="#fcfcfb", width=2)), customdata=ct[s],
                    hovertemplate="%{x}<br>" + s + ": %{y:.1f}% (%{customdata:,} students)<extra></extra>")
    fig.update_layout(barmode="stack"); fig.update_xaxes(title=xtitle or NAMES.get(col, col)); fig.update_yaxes(title="% of students", range=[0, 100])
    return _style(fig, title or f"Depression indicator by {(xtitle or NAMES.get(col, col)).lower()}", 420)


def cgpa_box(df, col="academic_pressure_level"):
    d = df.dropna(subset=[col]); fig = go.Figure()
    for s, c in (("Not depressed", NOT), ("Depressed", DEP)):
        x = d[d.depression_status == s]
        fig.add_box(x=x[col].astype(str), y=x["cgpa"], name=s, marker=dict(color=c, size=3, opacity=.5), line=dict(color=c))
    fig.update_layout(boxmode="group"); fig.update_xaxes(title=NAMES.get(col, col)); fig.update_yaxes(title="CGPA (0-10)")
    return _style(fig, f"CGPA by {NAMES.get(col, col).lower()} and depression status", 430)


def hours_line(df):
    t = rate_table(df, "study_hours", MIN_N)
    if t.empty: return empty()
    fig = go.Figure()
    fig.add_scatter(x=t["study_hours"], y=t["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=t["study_hours"], y=t["lo"], line=dict(width=0), fill="tonexty", fillcolor="rgba(235,104,52,.18)", showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=t["study_hours"], y=t["rate"], mode="lines+markers", line=dict(color=DEP, width=2), marker=dict(size=8, line=dict(color="#fcfcfb", width=2)),
                    customdata=t["n"], hovertemplate="%{x} h/day<br>Depression rate %{y:.1f}%<br>n = %{customdata:,}<extra></extra>", showlegend=False)
    fig.update_xaxes(title="Work/study hours per day", dtick=1); fig.update_yaxes(title="Depression rate (%)", range=[0, 100])
    return _style(fig, "Depression rate by daily work/study hours (band: 95% CI)", 400, False)


def city_rates(df, top=30):
    d = df[df.city != "Unknown"]
    t = rate_table(d, "city", 100).sort_values("rate", ascending=False).head(top)
    if t.empty: return empty()
    fig = go.Figure(go.Bar(x=t["rate"], y=t["city"], orientation="h", marker=dict(color=DEP, cornerradius=3), customdata=np.c_[t["n"], t["lo"], t["hi"]],
                           error_x=dict(type="data", symmetric=False, array=t["hi"] - t["rate"], arrayminus=t["rate"] - t["lo"], color=MUTED, thickness=1, width=3),
                           hovertemplate="%{y}<br>Rate %{x:.1f}% (95% CI %{customdata[1]:.1f}–%{customdata[2]:.1f})<br>n = %{customdata[0]:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=""); fig.update_xaxes(title="Depression rate (%)", range=[0, 100])
    return _style(fig, f"Depression rate by city (cities with ≥100 students; n shown on hover)", max(420, 22 * len(t) + 120), False)


def factor_gap(df):
    rows = []
    for col, name in FACTORS.items():
        t = rate_table(df, col, 100)
        if len(t) >= 2:
            hi, lo = t.loc[t.rate.idxmax()], t.loc[t.rate.idxmin()]
            rows.append((name, hi.rate - lo.rate, f"{hi[col]} ({hi.rate:.0f}%)", f"{lo[col]} ({lo.rate:.0f}%)"))
    g = pd.DataFrame(rows, columns=["factor", "gap", "highest", "lowest"]).sort_values("gap")
    if g.empty: return empty()
    fig = go.Figure(go.Bar(x=g["gap"], y=g["factor"], orientation="h", marker=dict(color=BLUE, cornerradius=4),
                           customdata=np.c_[g["highest"], g["lowest"]],
                           hovertemplate="%{y}<br>Gap: %{x:.1f} percentage points<br>Highest: %{customdata[0]}<br>Lowest: %{customdata[1]}<extra></extra>"))
    fig.update_xaxes(title="Gap in depression rate between highest and lowest group (percentage points)")
    return _style(fig, "Which factors show the largest differences in depression rate?", 460, False)


# ---------------- C. relationships ----------------
def correlation_heatmap(df, method="spearman"):
    cols = [c for c in ORDINAL if c in df]
    corr = df[cols].corr(method=method).round(2); names = [LABELS[c] for c in cols]
    fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1, colorscale=DIV, text=corr.values, texttemplate="%{text}",
                               colorbar=dict(title=method.title() + " ρ"), hovertemplate="%{y} vs %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed"); fig.update_xaxes(tickangle=-35)
    return _style(fig, f"{method.title()} correlation between ordinal variables", 520, False)


def rate_heatmap(df, x, y, title=None, min_n=MIN_N):
    d = df.dropna(subset=[x, y])
    g = d.groupby([y, x], observed=True)["depression"].agg(n="size", rate="mean").reset_index()
    g["rate"] *= 100
    rate = g.pivot(index=y, columns=x, values="rate"); n = g.pivot(index=y, columns=x, values="n")
    rate = rate.where(n >= min_n)
    if rate.empty: return empty()
    text = rate.round(0).astype("Int64").astype(str).replace("<NA>", "")
    fig = go.Figure(go.Heatmap(z=rate.values, x=rate.columns.astype(str), y=rate.index.astype(str), colorscale=SEQ_ORANGE, zmin=0, zmax=100,
                               text=text.values, texttemplate="%{text}%", customdata=n.values, colorbar=dict(title="Rate (%)"),
                               hovertemplate=f"{NAMES.get(y, y)}: %{{y}}<br>{NAMES.get(x, x)}: %{{x}}<br>Depression rate: %{{z:.1f}}%<br>n = %{{customdata:,}}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=NAMES.get(y, y)); fig.update_xaxes(title=NAMES.get(x, x))
    return _style(fig, title or f"Depression rate (%) by {NAMES.get(y, y).lower()} and {NAMES.get(x, x).lower()}", 440, False)


def pressure_stress_bubble(df):
    g = df.groupby(["academic_pressure", "financial_stress"])["depression"].agg(n="size", rate="mean").reset_index(); g["rate"] *= 100
    g = g[g.n >= 1]
    fig = go.Figure(go.Scatter(x=g["academic_pressure"], y=g["financial_stress"], mode="markers", customdata=np.c_[g["n"], g["rate"]],
                               marker=dict(size=g["n"], sizemode="area", sizeref=2 * g["n"].max() / 55**2, color=g["rate"], colorscale=SEQ_ORANGE, cmin=0, cmax=100,
                                           colorbar=dict(title="Rate (%)"), line=dict(color="#fcfcfb", width=2), opacity=.9),
                               hovertemplate="Academic pressure %{x}, financial stress %{y}<br>n = %{customdata[0]:,}<br>Depression rate %{customdata[1]:.1f}%<extra></extra>"))
    fig.update_xaxes(title=LABELS["academic_pressure"], dtick=1, range=[.4, 5.6]); fig.update_yaxes(title=LABELS["financial_stress"], dtick=1, range=[.4, 5.6])
    return _style(fig, "Academic pressure vs financial stress (size = students, colour = depression rate)", 480, False)


def sankey(df):
    d = df.dropna(subset=["sleep_duration"]); cols = ["sleep_duration", "academic_pressure_level", "depression_status"]
    labels, idx, colors = [], {}, []
    pal = {"Depressed": DEP, "Not depressed": NOT}
    for c in cols:
        for v in (d[c].cat.categories if hasattr(d[c], "cat") else sorted(d[c].unique())):
            if (d[c] == v).any():
                idx[(c, v)] = len(labels); labels.append(str(v)); colors.append(pal.get(v, "#9b9a94"))
    s, t, v, lc = [], [], [], []
    for a, b in zip(cols[:-1], cols[1:]):
        g = d.groupby([a, b], observed=True).size().reset_index(name="n")
        for _, r in g.iterrows():
            s.append(idx[(a, r[a])]); t.append(idx[(b, r[b])]); v.append(int(r["n"]))
            lc.append({"Depressed": "rgba(235,104,52,.35)", "Not depressed": "rgba(42,120,214,.30)"}.get(r[b], "rgba(120,120,120,.22)"))
    fig = go.Figure(go.Sankey(arrangement="snap", node=dict(label=labels, color=colors, pad=16, thickness=16, line=dict(color="#fcfcfb", width=1)),
                              link=dict(source=s, target=t, value=v, color=lc)))
    return _style(fig, "Flow: sleep duration → academic pressure → depression indicator", 480, False)


# ---------------- registry ----------------
def all_report_figures(df) -> dict:
    return {
        "01_sleep_distribution": sleep_distribution(df),
        "02_depression_share_donut": status_donut(df),
        "03_age_histogram": age_histogram(df),
        "04_pressure_stress_distributions": hours_pressure_histograms(df),
        "05_rate_by_academic_pressure": rate_bar(df, "academic_pressure"),
        "06_rate_by_financial_stress": rate_bar(df, "financial_stress"),
        "07_sleep_depression_stacked": stacked_status(df, "sleep_duration"),
        "08_diet_depression_stacked": stacked_status(df, "diet"),
        "09_cgpa_box_by_pressure": cgpa_box(df),
        "10_spearman_heatmap": correlation_heatmap(df),
        "11_pressure_stress_rate_heatmap": rate_heatmap(df, "financial_stress", "academic_pressure_level"),
        "12_sleep_diet_rate_heatmap": rate_heatmap(df, "diet", "sleep_duration"),
        "13_pressure_stress_bubble": pressure_stress_bubble(df),
        "14_study_hours_line": hours_line(df),
        "15_rate_by_age_group": rate_bar(df, "age_group"),
        "16_rate_by_degree_level": rate_bar(df[df.degree_level != "Other"], "degree_level"),
        "17_rate_by_city": city_rates(df),
        "18_factor_gap_ranking": factor_gap(df),
        "19_sankey": sankey(df),
    }
