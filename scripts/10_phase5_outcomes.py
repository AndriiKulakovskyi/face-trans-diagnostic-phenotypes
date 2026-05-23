"""Phase 5 — do the trans-diagnostic axes predict outcomes BEYOND DSM?

The value test. For each V1 follow-up outcome we fit three nested, cross-validated
models and ask whether adding the 6 V0 dimensional axes improves out-of-sample
prediction over DSM diagnosis:

    M0 (DSM):      Y_V1 ~ baseline(Y_V0) + age + sex + cohort + arm
    M1 (axes):     Y_V1 ~ baseline(Y_V0) + age + sex + axis1..6
    M2 (combined): Y_V1 ~ baseline + age + sex + cohort + arm + axis1..6

Leakage-safe: predictors are V0, outcome is V1, and we always adjust for the V0
BASELINE of the outcome (so EGF/hospitalization — which also feed the axes — are
predicted as a *trajectory*, not circularly). Continuous outcomes → 5-fold CV R²
(Ridge); binary → 5-fold CV AUC (logistic). We also report per-axis standardized
effects (statsmodels) and a likelihood-ratio / F test for the added axes.

Outcomes (feasible follow-up coverage): EGF/GAF functioning, any-hospitalization,
EQ-5D quality of life. (Work disability dropped — not measured at follow-up.)

Artifacts: results/phase5_headtohead.csv, results/phase5_axis_effects.csv,
reports/phase5.html.
Run:  python3 scripts/10_phase5_outcomes.py [--visit V1]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402

from trans_diag import build_unified_dataframe  # noqa: E402
from trans_diag.outcomes import (  # noqa: E402  shared head-to-head helpers
    OUTCOMES,
    added_axes_test,
    axis_betas,
    cv_metric,
)

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
AXES_PATH = RESULTS_DIR / "dimensional_final_scores.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--visit", default="V1", help="follow-up visit for the outcome")
    args = ap.parse_args()
    REPORTS_DIR.mkdir(exist_ok=True)

    axes = pd.read_parquet(AXES_PATH)
    axis_cols = list(axes.columns)
    axes.index = pd.MultiIndex.from_arrays(
        [axes.index.get_level_values("cohort").astype(str),
         axes.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    axes = axes.reset_index()
    axes["pid"] = axes["cohort"] + "::" + axes["patient_id"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
    df["pid"] = df["cohort"].str.lower() + "::" + df["usubjid_patients"].astype(str)
    v0 = df[df["visit"] == "V0"].set_index("pid")
    vk = df[df["visit"] == args.visit].set_index("pid")

    base = pd.DataFrame(index=v0.index)
    base["age"], base["sex"] = pd.to_numeric(v0["age"], errors="coerce"), pd.to_numeric(v0["sex"], errors="coerce")
    # DSM diagnosis = arm (7 subtypes); arm already implies cohort, so adding cohort
    # would make the design collinear/singular.
    dsm = pd.get_dummies(v0["arm"].astype(str), drop_first=True).astype(float)
    dsm_cols = list(dsm.columns)
    base = base.join(dsm)
    A = axes.set_index("pid")[axis_cols]

    head_rows, eff_rows = [], []
    for name, kind, col, tf in OUTCOMES:
        if col not in df.columns:
            continue
        y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
        yk = pd.to_numeric(vk[col], errors="coerce")
        if tf is not None:
            yk = tf(yk)
        d = base.join(y0).join(A).join(yk.rename("y")).dropna(
            subset=["y", "baseline", "age", "sex"] + axis_cols)
        if kind == "binary" and (d["y"].nunique() < 2 or d["y"].mean() < 0.02):
            continue
        n = len(d)
        if n < 200:
            continue
        bc = ["baseline", "age", "sex"]
        m0 = cv_metric(d[bc + dsm_cols].to_numpy(float), d["y"].to_numpy(float), kind)
        m1 = cv_metric(d[bc + axis_cols].to_numpy(float), d["y"].to_numpy(float), kind)
        m2 = cv_metric(d[bc + dsm_cols + axis_cols].to_numpy(float), d["y"].to_numpy(float), kind)
        yv = d["y"].to_numpy(float)
        p_added = added_axes_test(d, bc, dsm_cols, axis_cols, yv, kind)
        betas = axis_betas(d, bc + dsm_cols + axis_cols, axis_cols, yv, kind)
        metric = "R2" if kind == "continuous" else "AUC"
        head_rows.append({"outcome": name, "n": n, "metric": metric,
                          "DSM": round(m0, 3), "axes": round(m1, 3), "combined": round(m2, 3),
                          "axes_minus_DSM": round(m1 - m0, 3), "added_axes_p": p_added})
        print(f"{name}: n={n} {metric}  DSM={m0:.3f}  axes={m1:.3f}  combined={m2:.3f}  "
              f"(axes−DSM {m1-m0:+.3f}; added-axes p={p_added:.1e})")
        for a in axis_cols:
            eff_rows.append({"outcome": name, "axis": a, "beta": betas[a]})

    suf = args.visit
    head = pd.DataFrame(head_rows); head.to_csv(RESULTS_DIR / f"phase5_headtohead_{suf}.csv", index=False)
    eff = pd.DataFrame(eff_rows); eff.to_csv(RESULTS_DIR / f"phase5_axis_effects_{suf}.csv", index=False)
    _report(head, eff, axis_cols, args.visit)
    print(f"\nWrote results/phase5_* + reports/phase5.html. Done.")
    return 0


def _report(head, eff, axis_cols, visit):
    rows = "".join(
        f"<tr><td>{r.outcome}</td><td>{r.n}</td><td>{r.metric}</td><td>{r.DSM}</td>"
        f"<td>{r.axes}</td><td>{r.combined}</td><td><b>{r.axes_minus_DSM:+}</b></td>"
        f"<td>{r.added_axes_p:.1e}</td></tr>" for r in head.itertuples())
    # forest plot of standardized axis effects per outcome
    f = go.Figure()
    for out in eff["outcome"].unique():
        sub = eff[eff["outcome"] == out]
        f.add_scatter(x=sub["beta"], y=sub["axis"], mode="markers", name=out,
                      marker=dict(size=9))
    f.add_vline(x=0, line_dash="dash")
    f.update_layout(title="Standardized axis effects on each V1 outcome (combined model)",
                    height=420, xaxis_title="β (per SD of axis)", margin=dict(t=46, l=140))
    css = "body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}h1{color:#2b3a55}table{border-collapse:collapse;font-size:13px;margin:12px 0}th,td{border:1px solid #e5e7eb;padding:5px 10px}th{background:#eef2f7}.c{background:#f5f7fb;border-left:4px solid #2b3a55;padding:10px 14px;margin:12px 0}"
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            f"<h1>Phase 5 — do dimensional axes beat DSM on {visit} outcomes?</h1>",
            "<div class='c'>Nested 5-fold CV. M0=DSM(+baseline,age,sex), M1=axes(+same), "
            "M2=both. 'axes−DSM' &gt; 0 ⇒ axes carry information DSM lacks; 'added-axes p' "
            "tests adding axes to the DSM model.</div>",
            "<table><tr><th>outcome</th><th>n</th><th>metric</th><th>DSM (M0)</th>"
            "<th>axes (M1)</th><th>combined (M2)</th><th>axes−DSM</th><th>added-axes p</th></tr>",
            rows, "</table>",
            pio.to_html(f, include_plotlyjs="cdn", full_html=False), "</body></html>"]
    (REPORTS_DIR / f"phase5_{visit}.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
