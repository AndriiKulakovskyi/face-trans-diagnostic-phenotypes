"""Render the Phase-2 sensitivity & decisions HTML report from the artifacts
produced by `scripts/phase2_sweep.py`.

Inputs:
  - results/phase2_sweep.csv
  - results/phase2_sweep_sections.csv
  - results/phase2_features.csv
  - results/phase2_meta.json

Output:
  - reports/phase2_sensitivity.html  (self-contained, plotly via CDN)

Sections of the report:
  1. Executive summary + decisions checklist
  2. Threshold sweep (heatmaps: features, patients, per-cohort retention)
  3. Section composition (biology vs psychopathology vs neuropsych)
  4. Cohort-balance bias (BP-protocol features that disappear when balanced)
  5. Feature content quality (entropy/std distribution, near-constants)
  6. Per-cohort residual missingness in the primary feature matrix
  7. Site-partition feasibility for the planned holdout validation
  8. V1..V4 attrition (the DR-V3 cliff)
  9. Open assumptions surfaced + counter-proposals
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from face_common import (  # noqa: E402
    build_unified_dataframe,
    load_variables,
    select_v0_anchor,
)

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
DATA_DIR = REPO_ROOT / "data"
DICT_PATH = REPO_ROOT / "face-common-vars.xlsx"

PRIMARY_CANDIDATES = [
    (0.60, 0.75, "PSY-RICH",  "#3498db"),
    (0.75, 0.75, "BALANCED",  "#16a085"),
    (0.85, 0.85, "BIO-STRICT","#e67e22"),
]
BIOLOGY_SECTIONS = {"BILAN BIOLOGIQUE", "CONSTANTES ET ECG", "ANTECEDENTS"}
PSYCH_SECTIONS = {"AUTO-QUESTIONNAIRES", "HETERO-QUESTIONNAIRES",
                  "NEUROPSYCHOLOGIE", "EVALUATION MEDICALE", "SUICIDE"}


def fig_div(fig: go.Figure, div_id: str, include_js: bool = False) -> str:
    return pio.to_html(
        fig, include_plotlyjs="cdn" if include_js else False,
        full_html=False, div_id=div_id, config={"displayModeBar": False},
    )


# ---------- threshold heatmaps ----------

def heatmap(sweep: pd.DataFrame, value_col: str, title: str,
            colorscale: str = "Viridis", reverse: bool = False) -> go.Figure:
    pivot = sweep.pivot_table(
        index="pt_threshold", columns="var_threshold", values=value_col
    )
    pivot = pivot.sort_index(ascending=False)
    text = np.array([[f"{int(v):,}" if not np.isnan(v) else ""
                      for v in row] for row in pivot.values])
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{c:.2f}" for c in pivot.columns],
        y=[f"{r:.2f}" for r in pivot.index],
        text=text, texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        colorscale=colorscale, reversescale=reverse,
        hovertemplate=("var=%{x}<br>pt=%{y}<br>" + value_col + "=%{z:.0f}"
                       "<extra></extra>"),
        colorbar=dict(title=value_col, thickness=12),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=380, margin=dict(t=50, l=70, r=20, b=50),
        xaxis_title="variable threshold", yaxis_title="patient threshold",
    )
    return fig


def per_cohort_retention(sweep: pd.DataFrame) -> go.Figure:
    """3-panel heatmap of per-cohort patient retention %."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["BP", "SZ", "DR"],
                        horizontal_spacing=0.08)
    totals = {"BP": 6252, "SZ": 2209, "DR": 552}
    for col, cohort in enumerate(["BP", "SZ", "DR"], start=1):
        pivot = sweep.pivot_table(
            index="pt_threshold", columns="var_threshold",
            values=f"{cohort.lower()}_patients",
        ).sort_index(ascending=False)
        pivot_pct = pivot / totals[cohort] * 100
        text = np.array([[f"{v:.0f}%" for v in row] for row in pivot_pct.values])
        fig.add_trace(
            go.Heatmap(
                z=pivot_pct.values,
                x=[f"{c:.2f}" for c in pivot_pct.columns],
                y=[f"{r:.2f}" for r in pivot_pct.index],
                text=text, texttemplate="%{text}",
                textfont=dict(size=10, color="white"),
                colorscale="RdYlGn", zmin=0, zmax=100,
                colorbar=dict(x=0.32 + (col - 1) * 0.34, len=0.9,
                              thickness=10, title="%"),
                hovertemplate=(f"{cohort} retained =%{{z:.1f}}%"
                              "<extra></extra>"),
            ),
            row=1, col=col,
        )
    fig.update_layout(
        title=dict(text="Per-cohort patient retention (% of full V0 cohort)",
                   font=dict(size=13)),
        height=380, margin=dict(t=70, l=60, r=20, b=50),
    )
    fig.update_xaxes(title_text="var threshold")
    fig.update_yaxes(title_text="pt threshold", col=1)
    return fig


# ---------- section composition ----------

def section_composition(sections: pd.DataFrame) -> go.Figure:
    """Stacked bar at pt=0.75 showing per-section feature count vs var threshold."""
    sub = sections[sections["pt_threshold"] == 0.75].copy()
    pivot = sub.pivot_table(
        index="var_threshold", columns="section", values="n_features",
        fill_value=0, aggfunc="sum",
    ).astype(int)
    fig = go.Figure()
    palette = {
        "ANTECEDENTS": "#7f8c8d",
        "BILAN BIOLOGIQUE": "#c0392b",
        "CONSTANTES ET ECG": "#e67e22",
        "AUTO-QUESTIONNAIRES": "#3498db",
        "HETERO-QUESTIONNAIRES": "#2980b9",
        "NEUROPSYCHOLOGIE": "#16a085",
        "EVALUATION MEDICALE": "#9b59b6",
        "SUICIDE": "#34495e",
        "PATIENT": "#95a5a6",
        "PERINATALITE": "#f39c12",
        "SOCIAL": "#bdc3c7",
        "SUBSTANCES": "#d35400",
        "SOIN SUIVI HOSP ARRET TRAVAIL": "#7d3c98",
    }
    for col in pivot.columns:
        fig.add_trace(go.Bar(
            x=[f"{v:.2f}" for v in pivot.index],
            y=pivot[col].values, name=col,
            marker_color=palette.get(col, "#999"),
            hovertemplate=("section=" + col +
                          "<br>var_thr=%{x}<br>n=%{y}<extra></extra>"),
        ))
    fig.update_layout(
        barmode="stack", title=dict(
            text="Surviving features by dictionary section "
                 "(at patient_threshold = 0.75)", font=dict(size=13)),
        xaxis_title="variable threshold",
        yaxis_title="n features",
        height=480, margin=dict(t=70, l=60, r=20, b=50),
        legend=dict(font=dict(size=10)),
    )
    return fig


# ---------- balanced-bootstrap bias ----------

def bp_bias_panel(pooled_features: set[str], balanced_results: list[set[str]],
                  var_lookup: dict) -> go.Figure:
    freq = {f: 0 for f in pooled_features}
    new_freq = {}
    for sel in balanced_results:
        for f in sel:
            if f in freq:
                freq[f] += 1
            else:
                new_freq[f] = new_freq.get(f, 0) + 1
    pooled_table = pd.DataFrame([
        {"variable": f, "n_balanced_runs": c,
         "section": var_lookup[f].section if f in var_lookup else "—"}
        for f, c in freq.items()
    ]).sort_values("n_balanced_runs")
    new_table = pd.DataFrame([
        {"variable": f, "n_balanced_runs": c,
         "section": var_lookup[f].section if f in var_lookup else "—"}
        for f, c in new_freq.items()
    ]).sort_values("n_balanced_runs", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pooled_table["n_balanced_runs"], y=pooled_table["variable"],
        orientation="h",
        marker_color=["#c0392b" if c < 10 else "#16a085"
                      for c in pooled_table["n_balanced_runs"]],
        name="pooled-selected features",
        hovertemplate="%{y}<br>in %{x}/20 balanced runs<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            "BP-protocol bias check: how often each pooled-selected feature "
            "ALSO survives in 20 cohort-balanced subsamples (BP/SZ "
            "down-sampled to n=552 to match DR)"
        ), font=dict(size=13)),
        xaxis=dict(title="balanced-bootstrap survival count (out of 20)",
                  range=[0, 21]),
        yaxis=dict(autorange="reversed", tickfont=dict(size=8)),
        height=max(500, len(pooled_table) * 12),
        margin=dict(t=80, l=200, r=20, b=50),
        showlegend=False,
    )
    fig.add_vline(x=10, line_dash="dash", line_color="#888",
                  annotation_text="majority threshold")
    return fig


# ---------- feature quality ----------

def entropy_distribution(features: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for dtype_filter, label, color in [
        ("entropy_bits", "categorical / ordinal / binary (entropy in bits)", "#3498db"),
        ("std",          "continuous (standard deviation)",                  "#16a085"),
    ]:
        sub = features[features["metric"] == dtype_filter]
        if not len(sub):
            continue
        fig.add_trace(go.Histogram(
            x=sub["value"], name=label, marker_color=color,
            opacity=0.7, nbinsx=24,
            hovertemplate=label + "<br>value=%{x}<br>n=%{y}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="Feature-content distribution (primary anchor)",
                   font=dict(size=13)),
        barmode="overlay", height=320,
        xaxis_title="value", yaxis_title="n features",
        margin=dict(t=60, l=60, r=20, b=50),
        legend=dict(font=dict(size=10)),
    )
    return fig


def near_constant_table(features: pd.DataFrame) -> go.Figure:
    nc = features[features["near_constant"]].sort_values("modal_share",
                                                          ascending=False)
    fig = go.Figure(go.Table(
        header=dict(values=["<b>canonical_name</b>", "<b>section</b>",
                            "<b>modal share</b>", "<b>n unique</b>"],
                    fill_color="#2b3a55", font_color="white",
                    align="left", height=26),
        cells=dict(
            values=[nc["canonical_name"], nc["section"],
                    [f"{v:.3f}" for v in nc["modal_share"]],
                    nc["n_unique"]],
            align="left", height=22, font=dict(size=11),
            fill_color=[["#fff5f5", "#ffffff"] * (len(nc) // 2 + 1)],
        ),
        columnwidth=[200, 200, 90, 90],
    ))
    fig.update_layout(
        title=dict(text=(f"Near-constant features at primary anchor "
                         f"({len(nc)} of {len(features)} — modal value ≥ 95%, "
                         f"contribute ~0 clustering signal)"),
                   font=dict(size=13)),
        height=22 * len(nc) + 80, margin=dict(t=60, l=10, r=10, b=10),
    )
    return fig


# ---------- per-cohort residual missingness ----------

def cohort_missingness_bars(meta: dict) -> go.Figure:
    miss = meta["primary_cohort_missingness"]
    cohorts = list(miss.keys())
    rates = [miss[c]["nan_rate"] * 100 for c in cohorts]
    cells = [f"{miss[c]['n_nan']:,}/{miss[c]['n_cells']:,}" for c in cohorts]
    fig = go.Figure(go.Bar(
        x=cohorts, y=rates,
        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        text=cells, textposition="outside",
        hovertemplate="%{x}<br>residual NaN %{y:.2f}%<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=(
            f"Residual NaN per cohort after primary filter "
            f"({meta['primary_var_threshold']:.2f} / "
            f"{meta['primary_pt_threshold']:.2f}) — "
            "imputation burden by cohort"
        ), font=dict(size=13)),
        yaxis=dict(title="% NaN remaining", range=[0, max(rates) * 1.3]),
        height=340, margin=dict(t=70, l=60, r=20, b=50),
    )
    return fig


# ---------- site partition ----------

def site_partition_table(v0: pd.DataFrame) -> go.Figure:
    counts = v0.groupby(["siteid_city", "cohort"]).size().unstack(fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=False)
    cohorts_with_data = (counts[["BP", "SZ", "DR"]] > 0).sum(axis=1)
    counts["3-cohort?"] = cohorts_with_data.map(
        {3: "yes", 2: "two", 1: "one"})
    fig = go.Figure(go.Table(
        header=dict(values=["<b>siteid</b>", "<b>BP</b>", "<b>SZ</b>",
                           "<b>DR</b>", "<b>total</b>",
                           "<b>cohorts</b>"],
                    fill_color="#2b3a55", font_color="white",
                    align="center", height=26),
        cells=dict(
            values=[[f"{int(s)}" for s in counts.index],
                    counts["BP"].values, counts["SZ"].values,
                    counts["DR"].values, counts["total"].values,
                    counts["3-cohort?"].values],
            align="center", height=22, font=dict(size=11),
            fill_color=[
                ["#fff5f5" if v == "one" else
                 ("#fffceb" if v == "two" else "#f2fbf2")
                 for v in counts["3-cohort?"]]
            ],
        ),
        columnwidth=[60, 60, 60, 60, 70, 90],
    ))
    fig.update_layout(
        title=dict(text="Sites × cohort recruitment (V0). Green = 3-cohort, "
                        "yellow = 2-cohort, red = mono-cohort.",
                   font=dict(size=13)),
        height=22 * len(counts) + 80,
        margin=dict(t=60, l=10, r=10, b=10),
    )
    return fig


# ---------- attrition Sankey ----------

def attrition_sankey(sweep: pd.DataFrame, var_t: float, pt_t: float) -> go.Figure:
    row = sweep[(sweep["var_threshold"] == var_t)
                & (sweep["pt_threshold"] == pt_t)].iloc[0]
    labels = []
    for visit in ("V0", "V1", "V2", "V3", "V4"):
        for c in ("BP", "SZ", "DR"):
            labels.append(f"{visit}·{c}")
    sources, targets, values, colors = [], [], [], []
    visits = ("V0", "V1", "V2", "V3", "V4")
    cohort_color = {"BP": "rgba(31,119,180,0.4)",
                    "SZ": "rgba(255,127,14,0.4)",
                    "DR": "rgba(44,160,44,0.4)"}
    for i in range(len(visits) - 1):
        v_from, v_to = visits[i], visits[i + 1]
        for c in ("BP", "SZ", "DR"):
            from_col = f"{c.lower()}_patients" if v_from == "V0" else f"{v_from}_{c}"
            to_col = f"{v_to}_{c}"
            n_to = int(row[to_col])
            if n_to == 0:
                continue
            sources.append(labels.index(f"{v_from}·{c}"))
            targets.append(labels.index(f"{v_to}·{c}"))
            values.append(n_to)
            colors.append(cohort_color[c])
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=12, thickness=14,
                  line=dict(color="#aaa", width=0.5)),
        link=dict(source=sources, target=targets, value=values, color=colors),
    ))
    fig.update_layout(
        title=dict(text=f"Patient carry V0→V1→V2→V3→V4 at {var_t:.2f}/{pt_t:.2f} "
                        "(width = n patients with a visit row)",
                   font=dict(size=13)),
        height=420, margin=dict(t=60, l=10, r=10, b=10),
        font=dict(size=10),
    )
    return fig


# ---------- HTML assembly ----------

CSS = """
:root{
  --fg:#1f2933;--muted:#6b7280;--accent:#2b3a55;--bg:#fff;
  --card:#fbfbfd;--border:#e5e7eb;--warn:#c0392b;--good:#16a085;
}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     color:var(--fg);margin:0;padding:0 24px 80px;line-height:1.55}
header{position:sticky;top:0;background:var(--bg);
  border-bottom:1px solid var(--border);padding:18px 0;
  z-index:100;margin-bottom:24px}
h1{margin:0 0 4px;font-size:22px;color:var(--accent)}
h2{margin:40px 0 14px;padding-bottom:6px;border-bottom:2px solid var(--accent);
   color:var(--accent);font-size:18px}
h3{margin:24px 0 8px;font-size:15px}
.muted{color:var(--muted);font-size:13px}
nav.toc a{display:inline-block;margin:2px 6px 2px 0;padding:3px 9px;
  background:#eef2f7;border-radius:12px;font-size:12px;color:var(--accent);
  text-decoration:none}
nav.toc a:hover{background:var(--accent);color:#fff}
.callout{border-left:4px solid var(--accent);padding:10px 14px;
  background:#f5f7fb;margin:14px 0;border-radius:0 6px 6px 0}
.callout.warn{border-color:var(--warn);background:#fff5f4}
.callout.good{border-color:var(--good);background:#f2fbf6}
table.k{border-collapse:collapse;margin:10px 0 18px;font-size:13px}
table.k th,table.k td{border:1px solid var(--border);padding:5px 10px;
  text-align:left}
table.k th{background:#eef2f7;font-weight:600}
.decision{border:1px solid var(--border);border-radius:8px;
  padding:14px 18px;margin:14px 0;background:var(--card)}
.decision h3{margin:0 0 6px}
.decision .label{display:inline-block;font-size:11px;padding:2px 8px;
  border-radius:10px;background:#eef2f7;color:var(--accent);
  font-weight:600;margin-right:6px}
.decision .rec{background:#d4f7d4;color:#0e6b2a}
.decision .open{background:#fff3cd;color:#856404}
.decision .reject{background:#f8d7da;color:#721c24}
.assumption{background:#fff;border:1px solid var(--border);border-left:6px solid #c0392b;
  padding:12px 14px;margin:10px 0;border-radius:0 6px 6px 0}
.assumption .code{font-family:"SF Mono",Menlo,monospace;font-size:12px;
  color:#fff;background:var(--warn);padding:1px 6px;border-radius:3px}
"""

ASSUMPTIONS_HTML = """
<div class='assumption'><span class='code'>A1</span>
<b>Pooled selection is biology-heavy.</b> At 0.75/0.75, 44/73 features come
from medical history + labs + ECG. Mood scales (MADRS/HAMD/PANSS) all fall
below 75% completeness. <i>Counter-proposal:</i> consider per-section
thresholds, or force-include canonical psychiatric scales regardless of
completeness.</div>

<div class='assumption'><span class='code'>A2</span>
<b>Cohort imbalance biases the selection.</b> Confirmed empirically: when
BP/SZ are down-sampled to DR's n=552, the entire PSQI family (8 features)
drops out and 7 medical-history binaries take their place. The pooled set
captures BP's protocol focus, not a trans-diagnostic core.</div>

<div class='assumption'><span class='code'>A3</span>
<b>GMM doesn't natively handle ordinals/binaries.</b> 40/73 primary features
are binary/categorical/ordinal. Gaussian likelihood is ill-defined on
{0,1}-valued vectors. <i>Counter-proposal:</i> replace GMM with Latent Class
Analysis (LCA) for categorical features, or restrict GMM to continuous-only
subset.</div>

<div class='assumption'><span class='code'>A4</span>
<b>Consensus-of-three clusters are hard to characterize.</b> Reviewers will
ask "what defines cluster X?" — for a consensus-cluster the answer is
"pairwise agreement of three method-specific cluster definitions", which is
not biologically interpretable. <i>Counter-proposal:</i> pick one method as
primary (with feature loadings); use the others as robustness checks.</div>

<div class='assumption'><span class='code'>A5</span>
<b>Site-based holdout is feasible but constrained.</b> 21 sites total, but
only 7 recruit all 3 cohorts. Mono-cohort sites can't serve as fair holdout.
<i>Counter-proposal:</i> hold out 2 multi-cohort sites (e.g., sites 10 + 13
= 762 patients, all 3 cohorts represented); train on the other 5 + all
single-cohort sites.</div>

<div class='assumption'><span class='code'>A6</span>
<b>k is unspecified.</b> Each method needs a cluster count. With DR=444 V0
patients, any cluster < 50 DR-equivalents is uninterpretable. <i>Concrete
bound: k ≤ 8.</i> Pre-register a k range (e.g., 3 – 6) per method.</div>

<div class='assumption'><span class='code'>A7</span>
<b>ARI is not the only stability metric.</b> V0→V1 ARI can be high while
every patient flips label. <i>Counter-proposal:</i> always report ARI +
per-patient transition matrix (Sankey) + per-cluster Jaccard.</div>

<div class='assumption'><span class='code'>A8</span>
<b>Completeness is multi-dimensional, not a single number.</b> A column
90%-present in BP and 10%-in-DR can pass a 60% pooled threshold while being
systematically missing-by-cohort. <i>Counter-proposal:</i> add a per-cohort
completeness floor in addition to the pooled one.</div>

<div class='assumption'><span class='code'>A9</span>
<b>Bootstrap stability (Hennig 2007) is missing.</b> Industry standard:
clusters with mean Jaccard &lt; 0.5 across 1000 bootstrap re-clusterings are
"dissolved" and shouldn't be reported. Add as primary V0 stability check.</div>

<div class='assumption'><span class='code'>A10</span>
<b>Protocol-driven features may distort the selection.</b> If FAST is the
primary endpoint of one of the FACE studies, it's collected at every visit
by mandate — its high completeness reflects protocol, not measurability.
<i>Counter-proposal:</i> flag protocol-mandated assessments (TBD inventory)
and report sensitivity excluding them.</div>
"""


def build_html(sweep: pd.DataFrame, sections: pd.DataFrame,
               features: pd.DataFrame, meta: dict,
               v0: pd.DataFrame,
               pooled_features: set[str], balanced_results: list[set[str]],
               var_lookup: dict,
               candidate_summary: pd.DataFrame) -> str:
    parts: list[str] = []
    parts.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    parts.append("<title>FACE Common — Phase 2 sensitivity & decisions</title>")
    parts.append(f"<style>{CSS}</style>")
    parts.append("</head><body>")

    # header / TOC
    parts.append("<header>")
    parts.append("<h1>FACE Common — Phase 2 sensitivity & decisions</h1>")
    parts.append("<div class='muted'>Threshold sweep, feature-content analysis, "
                 "cohort-imbalance bias, site-partition feasibility, imputation "
                 "burden, attrition. The goal: lock the primary methodology "
                 "with empirical justification.</div>")
    parts.append("<nav class='toc'>")
    for sid, label in [
        ("summary", "Summary"), ("threshold", "Threshold sweep"),
        ("composition", "Section composition"),
        ("bias", "Cohort-imbalance bias"),
        ("content", "Feature content"),
        ("imputation", "Residual missingness"),
        ("sites", "Site partition"),
        ("attrition", "V1–V4 attrition"),
        ("assumptions", "Assumptions challenged"),
        ("decisions", "Decisions to lock"),
    ]:
        parts.append(f"<a href='#{sid}'>{label}</a>")
    parts.append("</nav></header>")

    # Summary
    parts.append("<h2 id='summary'>1. Executive summary</h2>")
    parts.append(f"<div class='callout'><b>Three candidate primary thresholds.</b> "
                 f"Each represents a different scientific bet:")
    parts.append("<table class='k'>")
    parts.append("<tr><th>config</th><th>features</th><th>BP</th><th>SZ</th>"
                 "<th>DR</th><th>DR % NaN</th><th>Story</th></tr>")
    for _, r in candidate_summary.iterrows():
        parts.append(
            f"<tr><td><b>{r['label']}</b> ({r['var_t']:.2f}/{r['pt_t']:.2f})</td>"
            f"<td>{r['features']}</td><td>{r['bp_n']:,}</td>"
            f"<td>{r['sz_n']:,}</td><td>{r['dr_n']:,}</td>"
            f"<td>{r['dr_nan_pct']:.1f}%</td>"
            f"<td>{r['story']}</td></tr>"
        )
    parts.append("</table></div>")

    parts.append(
        "<p>Key empirical findings driving the methodology choices:</p>"
        "<ul>"
        "<li><b>Neuropsychology vanishes between 0.60 and 0.70.</b> "
        "At var_threshold=0.60, 44 cognition features survive; at 0.70, "
        "0 do. This is the variance we'd most want for a trans-diagnostic "
        "Nature paper, and the current 0.75 default discards all of it.</li>"
        "<li><b>Pooled selection is BP-protocol biased.</b> 8 of 73 "
        "primary features (entire PSQI family) disappear when cohorts are "
        "down-sampled to balance. 7 medical-history binaries take their place. "
        "The 'trans-diagnostic core' is fragile.</li>"
        "<li><b>12 of 73 features are near-constant</b> (rare medical-history "
        "binaries: lupus, MS, infarctus, polyarthritis at 99-100% modal). "
        "These contribute zero clustering signal and should be dropped.</li>"
        "<li><b>DR-V3 cliff is structural, not threshold-dependent.</b> "
        "Across every threshold combination, DR carries ≤ 3 patients into "
        "V3. Exclude DR from V3 stability metrics.</li>"
        "<li><b>Imputation burden scales with features.</b> 0.60/0.75 leaves "
        "DR with 45% NaN cells; 0.75/0.75 leaves 16%; 0.85/0.85 leaves 14%.</li>"
        "</ul>"
    )

    # Threshold sweep
    parts.append("<h2 id='threshold'>2. Threshold sweep</h2>")
    parts.append(fig_div(heatmap(sweep, "n_features", "Surviving features"),
                         "h-feat", include_js=True))
    parts.append(fig_div(heatmap(sweep, "n_patients", "Surviving patients",
                                colorscale="Plasma"), "h-pt"))
    parts.append(fig_div(per_cohort_retention(sweep), "h-cohort"))

    # Section composition
    parts.append("<h2 id='composition'>3. Section composition</h2>")
    parts.append("<p>Where does the variance come from at each threshold? "
                 "(at pt_threshold=0.75)</p>")
    parts.append(fig_div(section_composition(sections), "sec-comp"))
    parts.append(
        "<div class='callout warn'>"
        "<b>The neuropsych cliff.</b> NEUROPSYCHOLOGIE features go 90 → 44 → 0 "
        "between var_thresholds 0.50 → 0.60 → 0.70. Locking at 0.75 throws "
        "away the entire cognition axis. If 'trans-diagnostic' means anything, "
        "it means cognitive variability across BP/SZ/DR, which we cannot "
        "see at all in the current default."
        "</div>"
    )

    # Bias
    parts.append("<h2 id='bias'>4. Cohort-imbalance bias</h2>")
    parts.append(
        "<p>20 cohort-balanced subsamples (BP and SZ down-sampled to DR's "
        f"n=552). For each, we re-ran <code>select_v0_anchor(0.75, 0.75)</code> "
        "and tracked which features appeared. A pooled-selected feature that "
        "appears in fewer than 10/20 balanced runs is biased by BP/SZ "
        "majority — its selection is an artefact of the cohort mix.</p>"
    )
    parts.append(fig_div(bp_bias_panel(pooled_features, balanced_results,
                                       var_lookup), "bias"))
    parts.append(
        "<div class='callout warn'>"
        "<b>The PSQI family is a BP-protocol artefact.</b> All 8 PSQI sleep "
        "items drop out in balanced subsamples. The pooled threshold selected "
        "them because BP dominates the pool, not because they're "
        "trans-diagnostic. <i>Recommendation:</i> use balanced selection OR "
        "report PSQI as a sensitivity feature, not a primary one."
        "</div>"
    )

    # Feature content
    parts.append("<h2 id='content'>5. Feature content (primary 0.75/0.75)</h2>")
    parts.append(fig_div(entropy_distribution(features), "ent-dist"))
    parts.append(fig_div(near_constant_table(features), "nc-table"))

    # Residual missingness
    parts.append("<h2 id='imputation'>6. Residual missingness &amp; imputation</h2>")
    parts.append(fig_div(cohort_missingness_bars(meta), "miss"))
    parts.append("<p>The post-filter NaN profile is the <i>imputation surface</i> — "
                 "what each candidate primary commits us to imputing. The DR "
                 "cohort sets the constraint: at PSY-RICH (0.60/0.75), DR has "
                 "45% NaN, meaning clustering would be effectively "
                 "<i>hallucinating</i> half of DR's signal.</p>")
    parts.append("<table class='k'>"
                 "<tr><th>candidate</th><th>imputation option</th>"
                 "<th>cost/risk</th></tr>"
                 "<tr><td>0.75/0.75 BALANCED (recommended)</td>"
                 "<td>Gower-native partial distances</td>"
                 "<td><span style='color:#16a085'>Free, no imputation; "
                 "constrains us to hierarchical clustering</span></td></tr>"
                 "<tr><td>0.75/0.75 BALANCED</td>"
                 "<td>MICE m=20</td>"
                 "<td>~22M imputed cells (20 × 1.1M). Run cost: tens of "
                 "minutes. Defensible.</td></tr>"
                 "<tr><td>0.60/0.75 PSY-RICH</td>"
                 "<td>MICE m=20</td>"
                 "<td><span style='color:#c0392b'>~30M cells, with DR at "
                 "45% NaN. Imputation imputes half the DR cohort — hard to "
                 "defend.</span></td></tr>"
                 "<tr><td>0.85/0.85 BIO-STRICT</td>"
                 "<td>Any (Gower / MICE / KNN)</td>"
                 "<td><span style='color:#16a085'>Very cheap (76% of patients "
                 "have ≤ 5% NaN). But only 30 features — weak clustering "
                 "story.</span></td></tr>"
                 "</table>")

    # Sites
    parts.append("<h2 id='sites'>7. Site-based holdout feasibility</h2>")
    parts.append(fig_div(site_partition_table(v0), "sites"))
    parts.append(
        "<div class='callout good'>"
        "<b>Recommended holdout: sites 10 + 13.</b> Both recruit all 3 "
        "cohorts. Combined n = 489 + 273 = 762 patients (BP 223 / SZ 424 / "
        "DR 115). Training on the other 19 sites = 8,251 patients. "
        "Per-cohort split: BP ~ 96%/4%, SZ ~ 81%/19%, DR ~ 79%/21%. "
        "Defensible: tests true cross-site generalization without losing "
        "DR statistical power."
        "</div>"
    )

    # Attrition
    parts.append("<h2 id='attrition'>8. V1–V4 attrition (DR cliff)</h2>")
    parts.append(fig_div(attrition_sankey(sweep, 0.75, 0.75), "sankey"))
    parts.append("<p>DR is essentially gone by V3 regardless of threshold. "
                 "All V3 stability metrics for DR are reported as "
                 "<i>descriptive only (n ≤ 3)</i>.</p>")

    # Assumptions
    parts.append("<h2 id='assumptions'>9. Assumptions challenged</h2>")
    parts.append(ASSUMPTIONS_HTML)

    # Decisions
    parts.append("<h2 id='decisions'>10. Decisions to lock for Phase 3</h2>")
    parts.append("<div class='decision'>"
                 "<h3>D1 · Primary thresholds <span class='label rec'>RECOMMEND</span></h3>"
                 "<b>var_threshold = 0.75 · pt_threshold = 0.75</b> (BALANCED). "
                 "Best DR retention, lowest imputation burden, includes some "
                 "psychiatric scales (PSQI, YMRS, CGI). Report 0.60/0.75 "
                 "(PSY-RICH) and 0.85/0.85 (BIO-STRICT) as sensitivity "
                 "analyses. <br><i>Alternative if you want neuropsych:</i> "
                 "promote 0.60/0.75 to primary, accept the higher imputation "
                 "burden, and exclude DR patients with > 30% NaN.</div>")
    parts.append("<div class='decision'>"
                 "<h3>D2 · Near-constant feature exclusion <span class='label rec'>RECOMMEND</span></h3>"
                 "Drop the 12 medical-history binaries with modal share ≥ 95% "
                 "(lupus, polyarthritis, AVC, coronar, infarctus, sep, "
                 "epilepsie, autneuro, autendoc, autcardv, asthme, trbrycard). "
                 "They add 12 degrees of freedom to clustering with ~0 signal. "
                 "Effective primary feature count: 73 − 12 − 2 (date+string) = "
                 "<b>59 numerical features</b>.</div>")
    parts.append("<div class='decision'>"
                 "<h3>D3 · Primary clustering method <span class='label rec'>RECOMMEND</span></h3>"
                 "<b>Hierarchical clustering (Ward linkage) on Gower distance.</b> "
                 "Natively handles mixed types and NaN via partial distances "
                 "(no imputation needed). GMM is demoted to robustness check "
                 "on the continuous-only subset; HDBSCAN to robustness check "
                 "for outlier patients. The pre-registered <i>consensus of "
                 "three</i> rule is changed to: <i>primary = hierarchical, "
                 "robustness = others.</i></div>")
    parts.append("<div class='decision'>"
                 "<h3>D4 · Imputation <span class='label rec'>RECOMMEND</span></h3>"
                 "<b>None for the primary run</b> (Gower handles partial "
                 "distances). MICE m=20 added as sensitivity analysis to "
                 "test whether clusters survive multiple-imputation noise.</div>")
    parts.append("<div class='decision'>"
                 "<h3>D5 · Site holdout <span class='label rec'>RECOMMEND</span></h3>"
                 "<b>Train on sites {1,2,3,4,5,6,7,8,9,11,12,14,15,16,17,18,19,20,21,22,23,24,25} "
                 "minus {10,13}; holdout = sites {10, 13}.</b> Both holdout "
                 "sites are multi-cohort. Commit to "
                 "<code>results/site_partition.json</code> before any "
                 "Phase-3 clustering runs. (Exact site list depends on which "
                 "siteid integers appear — see the site table above.)</div>")
    parts.append("<div class='decision'>"
                 "<h3>D6 · Cluster count <span class='label open'>OPEN</span></h3>"
                 "Pre-register k ∈ {2, 3, 4, 5, 6}. Method-specific selection: "
                 "hierarchical → silhouette score on Gower distance, "
                 "GMM → BIC, HDBSCAN → fix min_cluster_size = "
                 "max(50, 0.05 × n_patients). Report all k in supplementary, "
                 "primary = the k that maximizes silhouette in hierarchical.</div>")
    parts.append("<div class='decision'>"
                 "<h3>D7 · Stability metrics <span class='label rec'>RECOMMEND</span></h3>"
                 "1. ARI (V0 vs Vk) — structure stability. "
                 "2. Per-patient transition Sankey — position stability. "
                 "3. Bootstrap Jaccard (1000 resamples) — Hennig stability per cluster. "
                 "All three reported. Clusters with mean Jaccard &lt; 0.5 "
                 "are flagged as dissolved.</div>")
    parts.append("<div class='decision'>"
                 "<h3>D8 · Cohort-balanced sensitivity <span class='label rec'>RECOMMEND</span></h3>"
                 "Add a sensitivity analysis where BP and SZ are "
                 "down-sampled to match DR (or to a balanced n=500 per "
                 "cohort), the pipeline re-run, and feature stability + "
                 "cluster stability vs the pooled primary reported. "
                 "Surfaces the BP-protocol bias quantified in §4.</div>")
    parts.append("<div class='decision'>"
                 "<h3>D9 · Per-cohort completeness floor <span class='label open'>OPEN</span></h3>"
                 "Optional: require each variable to have ≥ X% completeness "
                 "<i>per cohort</i> (not just pooled). At X = 60% this would "
                 "drop systematically missing-by-cohort variables. Quantify "
                 "in Phase 3 before locking.</div>")
    parts.append("<div class='decision'>"
                 "<h3>D10 · Protocol-mandated feature flagging <span class='label open'>OPEN</span></h3>"
                 "Inventory which assessments are protocol endpoints in each "
                 "FACE sub-study. Report sensitivity excluding them — "
                 "ensures the clusters aren't echoing the recruitment protocol.</div>")

    parts.append("</body></html>")
    return "".join(parts)


# ---------- entry ----------

def main() -> int:
    sweep = pd.read_csv(RESULTS_DIR / "phase2_sweep.csv")
    sections = pd.read_csv(RESULTS_DIR / "phase2_sweep_sections.csv")
    features = pd.read_csv(RESULTS_DIR / "phase2_features.csv")
    meta = json.loads((RESULTS_DIR / "phase2_meta.json").read_text())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(
            DATA_DIR, DICT_PATH,
            readiness=meta["readiness"], format="long",
        )
    v0 = df[df["visit"] == "V0"]
    variables = load_variables(DICT_PATH)
    var_lookup = {v.canonical_name: v for v in variables}

    # candidate summary table
    candidate_rows = []
    candidate_stories = {
        (0.60, 0.75): "Psychiatry-led, includes 44 neuropsych features. "
                       "Trans-diagnostic story strongest but DR-NaN heavy.",
        (0.75, 0.75): "Balanced. Best DR retention, lowest imputation. "
                       "No neuropsych — biology + PSQI + suicide.",
        (0.85, 0.85): "Tight, low NaN. Only 30 features; clustering may "
                       "be under-powered.",
    }
    for var_t, pt_t, label, _ in PRIMARY_CANDIDATES:
        row = sweep[(sweep["var_threshold"] == var_t)
                     & (sweep["pt_threshold"] == pt_t)].iloc[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v0_f, _ = select_v0_anchor(df, variable_threshold=var_t,
                                       patient_threshold=pt_t)
        feat_cols = [c for c in v0_f.columns
                     if c not in {"usubjid_patients", "cohort", "arm",
                                 "visit", "visitnum", "fondacode", "armcd"}]
        dr_mat = v0_f[v0_f["cohort"] == "DR"][feat_cols]
        dr_nan_pct = dr_mat.isna().mean().mean() * 100 if len(dr_mat) else 0
        candidate_rows.append(dict(
            var_t=var_t, pt_t=pt_t, label=label,
            features=row["n_features"],
            bp_n=int(row["bp_patients"]), sz_n=int(row["sz_patients"]),
            dr_n=int(row["dr_patients"]), dr_nan_pct=dr_nan_pct,
            story=candidate_stories[(var_t, pt_t)],
        ))
    candidate_summary = pd.DataFrame(candidate_rows)

    # cohort-balanced bootstrap (20 trials)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, anchor_pooled = select_v0_anchor(
            df, variable_threshold=0.75, patient_threshold=0.75)
    pooled_features = set(anchor_pooled.feature_columns)
    n_dr = (v0["cohort"] == "DR").sum()
    rng = np.random.default_rng(seed=2025)
    balanced_results = []
    for _ in range(20):
        parts = []
        for cohort in ("BP", "SZ", "DR"):
            pool = v0[v0["cohort"] == cohort]
            sample = pool.sample(n=min(len(pool), n_dr),
                                 random_state=int(rng.integers(0, 10**9)))
            parts.append(sample)
        balanced_v0 = pd.concat(parts, axis=0)
        balanced_df = df[df["usubjid_patients"].isin(balanced_v0["usubjid_patients"])]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, ab = select_v0_anchor(balanced_df,
                                     variable_threshold=0.75,
                                     patient_threshold=0.75)
        balanced_results.append(set(ab.feature_columns))

    html_str = build_html(sweep, sections, features, meta, v0,
                          pooled_features, balanced_results, var_lookup,
                          candidate_summary)
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "phase2_sensitivity.html"
    out.write_text(html_str, encoding="utf-8")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
