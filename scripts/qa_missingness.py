"""Interactive HTML QA report on per-variable missingness across V0..V4.

Generates a single self-contained HTML file:

  - Overview block (Plotly heatmaps per cohort: section × visit, colour = mean
    missingness; plus a patient×visit count table and a top-30 most-missing
    variables bar chart).
  - Table of contents linking to each clinical section.
  - Per-section blocks. For every variable in the section: a Plotly metadata
    table (canonical_name, dtype, unit, readiness, source columns, rule,
    rationale) AND a Plotly grouped-bar chart of missingness % at V0..V4 for
    each cohort (BP/SZ/DR). Cohorts without a source CSV column show a
    distinct grey 'no source' bar.

Outputs: reports/qa_missingness.html (single file, plotly.js loaded via CDN)
         results/qa_missingness.csv  (raw numbers — useful for downstream tooling)

Run: python3 scripts/qa_missingness.py
"""
from __future__ import annotations

import html
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "archive"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from face_common import build_unified_dataframe, load_variables  # noqa: E402


DATA_DIR = REPO_ROOT / "data"
DICT_PATH = REPO_ROOT / "face-common-vars.xlsx"
RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"

VISITS = ["V0", "V1", "V2", "V3", "V4"]
COHORTS = ["BP", "SZ", "DR"]
COHORT_COLORS = {"BP": "#1f77b4", "SZ": "#ff7f0e", "DR": "#2ca02c"}
NO_SRC_COLOR = "#cccccc"

IDENTIFIERS = {"usubjid_patients", "fondacode", "arm", "armcd",
               "visitnum", "visit", "cohort"}


# ---------------------------------------------------------------------------
# Data computation
# ---------------------------------------------------------------------------

def compute_missingness(df: pd.DataFrame, feature_vars: list) -> pd.DataFrame:
    df_v = df[df["visit"].isin(VISITS)].copy()
    subs = {(c, v): df_v[(df_v["cohort"] == c) & (df_v["visit"] == v)]
            for c in COHORTS for v in VISITS}
    rows = []
    for var in feature_vars:
        canonical = var.canonical_name
        for cohort in COHORTS:
            has_src = bool(var.source_col(cohort))
            for visit in VISITS:
                sub = subs[(cohort, visit)]
                n_rows = len(sub)
                if not has_src:
                    rows.append(dict(canonical=canonical, section=var.section,
                                     dtype=var.dtype,
                                     readiness=var.cluster_readiness.split(" ")[0],
                                     cohort=cohort, visit=visit,
                                     n_rows=n_rows, n_missing=n_rows,
                                     miss_pct=np.nan, no_source=True))
                    continue
                n_missing = int(sub[canonical].isna().sum()) if n_rows else 0
                miss_pct = (n_missing / n_rows * 100.0) if n_rows else np.nan
                rows.append(dict(canonical=canonical, section=var.section,
                                 dtype=var.dtype,
                                 readiness=var.cluster_readiness.split(" ")[0],
                                 cohort=cohort, visit=visit,
                                 n_rows=n_rows, n_missing=n_missing,
                                 miss_pct=miss_pct, no_source=False))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def fig_to_div(fig: go.Figure, div_id: str, include_js: bool = False) -> str:
    return pio.to_html(
        fig, include_plotlyjs="cdn" if include_js else False,
        full_html=False, div_id=div_id, config={"displayModeBar": False},
    )


def overview_heatmaps(miss_df: pd.DataFrame) -> go.Figure:
    sections = sorted(miss_df["section"].unique())
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=[f"Cohort: {c}" for c in COHORTS],
        horizontal_spacing=0.08,
    )
    for col_idx, cohort in enumerate(COHORTS, start=1):
        sub = miss_df[(miss_df["cohort"] == cohort) & (~miss_df["no_source"])]
        pivot = (sub.groupby(["section", "visit"])["miss_pct"].mean()
                    .unstack("visit").reindex(index=sections, columns=VISITS))
        text_grid = np.where(
            np.isnan(pivot.values), "",
            np.vectorize(lambda v: f"{v:.0f}")(np.where(np.isnan(pivot.values), 0, pivot.values))
        )
        fig.add_trace(
            go.Heatmap(
                z=pivot.values, x=VISITS, y=sections,
                colorscale="RdYlGn_r", zmin=0, zmax=100,
                text=text_grid, texttemplate="%{text}",
                textfont=dict(size=10),
                colorbar=dict(title="miss %", x=0.32 + (col_idx - 1) * 0.34,
                              len=0.92, thickness=12),
                hovertemplate=("section=%{y}<br>visit=%{x}"
                              "<br>mean miss%=%{z:.1f}<extra></extra>"),
            ),
            row=1, col=col_idx,
        )
    fig.update_layout(
        title=dict(text="Mean missingness % per section × visit × cohort",
                   font=dict(size=15)),
        height=520, margin=dict(t=80, l=140, r=20, b=40),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def patient_count_table(df: pd.DataFrame) -> go.Figure:
    df_v = df[df["visit"].isin(VISITS)]
    counts = (df_v.groupby(["cohort", "visit"]).size()
                  .unstack("visit").reindex(index=COHORTS, columns=VISITS, fill_value=0))
    n_pts = (df_v.groupby("cohort")["usubjid_patients"].nunique()
                  .reindex(COHORTS, fill_value=0))
    header_vals = ["cohort", "unique patients"] + VISITS
    row_vals = [
        COHORTS,
        n_pts.values.tolist(),
    ] + [counts[v].values.tolist() for v in VISITS]
    fig = go.Figure(data=[go.Table(
        header=dict(values=[f"<b>{h}</b>" for h in header_vals],
                    fill_color="#2b3a55", font_color="white",
                    align="center", height=28),
        cells=dict(values=row_vals, align="center",
                  fill_color=[["#f5f5f5", "#ffffff", "#f5f5f5"]],
                  font=dict(size=12), height=24),
    )])
    fig.update_layout(
        title=dict(text="Patient×visit row counts (V0–V4)",
                   font=dict(size=14)),
        height=190, margin=dict(t=50, l=10, r=10, b=10),
    )
    return fig


def top_missing_bar(miss_df: pd.DataFrame, n: int = 30) -> go.Figure:
    has_src = miss_df[~miss_df["no_source"]]
    rank = (has_src.groupby("canonical")["miss_pct"].mean()
                  .sort_values(ascending=False).head(n))
    fig = go.Figure(go.Bar(
        x=rank.values, y=rank.index, orientation="h",
        marker_color="#c0392b",
        hovertemplate="%{y}<br>mean miss%=%{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Top {n} highest-missingness variables (mean across V0–V4)",
                   font=dict(size=14)),
        height=max(420, 22 * n),
        xaxis=dict(title="mean missingness %", range=[0, 105]),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        margin=dict(t=60, l=180, r=20, b=40),
    )
    return fig


def variable_metadata_table(var) -> go.Figure:
    """Compact 2-column Plotly table with the variable's metadata."""
    def trunc(s: str, n: int = 240) -> str:
        s = s or ""
        return s if len(s) <= n else s[: n - 1] + "…"

    fields = [
        ("canonical_name", var.canonical_name),
        ("section", var.section),
        ("dtype", var.dtype),
        ("unit / value set", var.unit_or_value_set),
        ("cluster readiness", var.cluster_readiness),
        ("BP source col", var.bp_csv_col or "— (not in BP CSV)"),
        ("SZ source col", var.sz_csv_col or "— (not in SZ CSV)"),
        ("DR source col", var.dr_csv_col or "— (not in DR CSV)"),
        ("label", trunc(var.label, 200)),
        ("clinical rationale", trunc(var.clinical_rationale, 600)),
        ("findings", trunc(var.findings, 600)),
        ("harmonization rule", trunc(var.rule, 800)),
    ]
    fig = go.Figure(data=[go.Table(
        header=dict(values=[f"<b>field</b>", f"<b>value</b>"],
                    fill_color="#2b3a55", font_color="white",
                    align="left", height=26),
        cells=dict(
            values=[[k for k, _ in fields], [v for _, v in fields]],
            align="left",
            fill_color=[["#f5f7fb", "#ffffff"] * (len(fields) // 2 + 1)],
            font=dict(size=11),
            height=22,
        ),
        columnwidth=[140, 700],
    )])
    fig.update_layout(
        height=22 * len(fields) + 50,
        margin=dict(t=10, l=0, r=0, b=0),
    )
    return fig


def variable_missingness_bar(var, miss_df: pd.DataFrame) -> go.Figure:
    """Grouped bar: 5 visit clusters × 3 cohort bars. miss % on y-axis."""
    fig = go.Figure()
    for cohort in COHORTS:
        sub = miss_df[(miss_df["canonical"] == var.canonical_name)
                      & (miss_df["cohort"] == cohort)]
        sub = sub.set_index("visit").reindex(VISITS)
        no_src = sub["no_source"].fillna(True).values
        heights = np.where(no_src, 100.0, sub["miss_pct"].fillna(0).values)
        n_rows = sub["n_rows"].fillna(0).astype(int).values
        n_missing = sub["n_missing"].fillna(0).astype(int).values
        bar_colors = [NO_SRC_COLOR if n else COHORT_COLORS[cohort] for n in no_src]
        pattern = ["/" if n else "" for n in no_src]
        custom = np.stack([n_rows, n_missing, no_src.astype(int)], axis=-1)
        fig.add_trace(go.Bar(
            name=cohort, x=VISITS, y=heights,
            marker=dict(color=bar_colors,
                        pattern_shape=pattern,
                        pattern_size=4,
                        pattern_fillmode="overlay"),
            customdata=custom,
            hovertemplate=(
                f"<b>{var.canonical_name}</b> · cohort=<b>{cohort}</b><br>"
                "visit=%{x}<br>"
                "missing=%{customdata[1]}/%{customdata[0]} rows<br>"
                "miss %=%{y:.1f}"
                "<extra></extra>"
            ),
        ))
    fig.update_layout(
        barmode="group",
        height=300,
        yaxis=dict(title="missingness %", range=[0, 105],
                   tickvals=[0, 25, 50, 75, 100]),
        xaxis=dict(title="visit"),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(t=20, l=60, r=20, b=60),
        bargap=0.18, bargroupgap=0.06,
    )
    fig.add_annotation(
        x=0.99, y=1.02, xref="paper", yref="paper",
        text="grey = no source CSV column", showarrow=False,
        font=dict(size=10, color="#777"), align="right",
    )
    return fig


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def section_slug(section: str) -> str:
    return "section-" + (section or "unknown").lower().replace(" ", "-").replace("/", "-")


CSS = """
:root {
  --fg: #1f2933;
  --muted: #6b7280;
  --accent: #2b3a55;
  --bg: #ffffff;
  --card-bg: #fbfbfd;
  --border: #e5e7eb;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg); color: var(--fg);
  margin: 0; padding: 0 24px 80px;
  line-height: 1.5;
}
header {
  position: sticky; top: 0; background: var(--bg);
  border-bottom: 1px solid var(--border);
  padding: 18px 0 12px; z-index: 100;
  margin-bottom: 24px;
}
h1 { margin: 0 0 4px; font-size: 22px; }
h2 {
  margin: 36px 0 16px; padding-bottom: 6px;
  border-bottom: 2px solid var(--accent);
  color: var(--accent); font-size: 18px;
}
h3 { margin: 26px 0 10px; font-size: 15px; color: var(--accent); }
.muted { color: var(--muted); font-size: 13px; }
nav.toc { margin: 8px 0 18px; }
nav.toc a {
  display: inline-block; margin: 2px 6px 2px 0;
  padding: 3px 9px; background: #eef2f7;
  border-radius: 12px; font-size: 12px;
  color: var(--accent); text-decoration: none;
}
nav.toc a:hover { background: var(--accent); color: white; }
.var-card {
  border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px 4px;
  margin: 18px 0; background: var(--card-bg);
  display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
}
.var-card .meta-box { min-width: 0; }
.var-card .plot-box { min-width: 0; }
.var-card h3 {
  grid-column: 1 / -1; margin: 4px 0 8px;
  font-family: "SF Mono", Menlo, monospace; font-size: 14px;
  color: #111;
}
@media (max-width: 1100px) {
  .var-card { grid-template-columns: 1fr; }
}
.summary-row { display: flex; flex-wrap: wrap; gap: 18px; margin: 12px 0; }
.summary-row > div { flex: 1; min-width: 350px; }
.section-summary {
  font-size: 13px; color: var(--muted); margin: 4px 0 12px;
}
.pill {
  display: inline-block; padding: 2px 8px;
  border-radius: 10px; font-size: 11px; margin-right: 4px;
  background: #e5edf7; color: var(--accent);
}
.pill.ready { background: #d4f7d4; color: #137533; }
.pill.partial { background: #fff3cd; color: #856404; }
.pill.notusable { background: #f8d7da; color: #721c24; }
.back-to-top {
  position: fixed; bottom: 20px; right: 24px;
  background: var(--accent); color: white;
  border-radius: 20px; padding: 8px 14px; font-size: 12px;
  text-decoration: none; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
"""


def readiness_pill(readiness: str) -> str:
    head = readiness.split(" ")[0].lower()
    cls = {"ready": "ready", "partial": "partial"}.get(head, "notusable")
    return f'<span class="pill {cls}">{html.escape(head.upper())}</span>'


def build_html(df: pd.DataFrame, miss_df: pd.DataFrame,
               feature_vars: list) -> str:
    sections = sorted({v.section or "—" for v in feature_vars})
    vars_by_section: dict[str, list] = {s: [] for s in sections}
    for v in feature_vars:
        vars_by_section[v.section or "—"].append(v)
    for s in vars_by_section:
        vars_by_section[s].sort(key=lambda v: v.canonical_name)

    chunks: list[str] = []
    chunks.append("<!DOCTYPE html><html lang='en'><head>")
    chunks.append("<meta charset='utf-8'>")
    chunks.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    chunks.append("<title>FACE Common QA – Missingness Report</title>")
    chunks.append(f"<style>{CSS}</style>")
    chunks.append("</head><body>")

    # ----- Header / TOC -----------------------------------------------
    chunks.append("<header>")
    chunks.append("<h1>FACE Common QA — Missingness Report (V0–V4)</h1>")
    chunks.append("<div class='muted'>One block per dictionary section. "
                  "Each variable shows its metadata table and a missingness "
                  "bar chart per cohort × visit. Bars are grey when the "
                  "cohort has no source CSV column.</div>")
    chunks.append("<nav class='toc'>")
    chunks.append("<a href='#overview'>Overview</a>")
    for s in sections:
        n = len(vars_by_section[s])
        chunks.append(
            f"<a href='#{section_slug(s)}'>{html.escape(s)} ({n})</a>"
        )
    chunks.append("</nav>")
    chunks.append("</header>")

    # ----- Overview block ---------------------------------------------
    chunks.append("<h2 id='overview'>Overview</h2>")
    chunks.append("<div class='summary-row'>")
    chunks.append("<div>")
    chunks.append(fig_to_div(patient_count_table(df), "patient-counts",
                             include_js=True))
    chunks.append("</div>")
    chunks.append("</div>")
    chunks.append(fig_to_div(overview_heatmaps(miss_df), "overview-heatmap"))
    chunks.append(fig_to_div(top_missing_bar(miss_df, n=30), "top-missing"))

    # ----- Per-section variable blocks --------------------------------
    for sec in sections:
        sec_vars = vars_by_section[sec]
        sec_id = section_slug(sec)
        n_ready = sum(1 for v in sec_vars
                      if v.cluster_readiness.startswith("READY"))
        n_partial = len(sec_vars) - n_ready
        chunks.append(f"<h2 id='{sec_id}'>{html.escape(sec)}</h2>")
        chunks.append(
            f"<div class='section-summary'>{len(sec_vars)} variables "
            f"({n_ready} READY · {n_partial} PARTIAL)</div>"
        )
        for v in sec_vars:
            meta_id = f"meta-{v.canonical_name}"
            plot_id = f"plot-{v.canonical_name}"
            chunks.append("<div class='var-card'>")
            chunks.append(
                f"<h3>{html.escape(v.canonical_name)} "
                f"{readiness_pill(v.cluster_readiness)} "
                f"<span class='muted' style='font-weight: normal; "
                f"font-family: -apple-system, sans-serif;'>"
                f"· {html.escape(v.dtype)}</span></h3>"
            )
            chunks.append("<div class='meta-box'>")
            chunks.append(fig_to_div(variable_metadata_table(v), meta_id))
            chunks.append("</div>")
            chunks.append("<div class='plot-box'>")
            chunks.append(
                fig_to_div(variable_missingness_bar(v, miss_df), plot_id)
            )
            chunks.append("</div>")
            chunks.append("</div>")

    chunks.append("<a class='back-to-top' href='#overview'>↑ top</a>")
    chunks.append("</body></html>")
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print("Loading unified dataframe (READY + PARTIAL, long)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(
            DATA_DIR, DICT_PATH,
            readiness=["READY", "PARTIAL"], format="long",
        )

    variables = load_variables(DICT_PATH)
    feature_vars = [
        v for v in variables
        if v.cluster_readiness.startswith(("READY", "PARTIAL"))
        and v.canonical_name not in IDENTIFIERS
    ]
    print(f"  variables: {len(feature_vars)}")
    print(f"  sections : {sorted({v.section or '—' for v in feature_vars})}")

    print("Computing missingness table...")
    miss_df = compute_missingness(df, feature_vars)
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "qa_missingness.csv"
    miss_df.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path} ({len(miss_df):,} rows)")

    print("Building HTML report (this generates ~700 Plotly figures)...")
    html_str = build_html(df, miss_df, feature_vars)
    REPORTS_DIR.mkdir(exist_ok=True)
    html_path = REPORTS_DIR / "qa_missingness.html"
    html_path.write_text(html_str, encoding="utf-8")
    size_mb = html_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {html_path} ({size_mb:.1f} MB)")
    print("\nOpen in a browser:")
    print(f"  open {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
