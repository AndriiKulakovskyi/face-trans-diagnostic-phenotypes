"""Phase 5 — do the trans-diagnostic axes predict outcomes BEYOND DSM?

The value test. For each V1 follow-up outcome we fit nested, cross-validated
models and ask whether adding the 6 V0 dimensional axes improves out-of-sample
prediction over DSM diagnosis. Post-audit (2026-05), TWO head-to-head specs
are reported:

    Original (pre-audit, retained for back-compat):
        M0_arm  : Y_V1 ~ baseline + age + sex + arm                     (7 subtype dummies)
        M1_axes : Y_V1 ~ baseline + age + sex + axes                    (6 axes; NO cohort)
        M2_full : Y_V1 ~ baseline + age + sex + arm + axes

    Fair (post-audit):
        M0_arm     : same as above
        M1_fair    : Y_V1 ~ baseline + age + sex + cohort_dum + axes    (cohort parity)
        M2_fair    : Y_V1 ~ baseline + age + sex + arm + axes
        M0_scales  : Y_V1 ~ baseline + age + sex + arm + raw_QIDS + raw_MADRS + raw_STAI
                     (the depression-scale comparator: do dimensional axes add value
                      beyond standard depression scales the clinic already collects?)

Why the fair spec: the original M1 omits cohort, but M0_arm carries 7 subtype
dummies that encode cohort + within-cohort subtype. The axes thus quietly act as
a cohort surrogate that M1 needs to compete with arm. Adding cohort dummies (2
levels) to M1 restores comparator parity. The scales comparator tests whether
the dimensional model is doing anything beyond what raw clinical scales already do.

Hospitalization outcome (post-audit): now **incident** between V0 and V1
(V1 lifetime-count > V0 lifetime-count), not (V1 lifetime > 0) — the latter was
near-deterministically driven by V0 lifetime alone (lifetime counts are
non-decreasing), making the comparison uninterpretable.

Leakage-safe: predictors are V0, outcome is V1, and we always adjust for the V0
BASELINE of the outcome (so EGF/hospitalization — which also feed the axes — are
predicted as a *trajectory*, not circularly). Continuous outcomes → 5-fold CV R²
(Ridge); binary → 5-fold CV AUC (logistic). We also report per-axis standardized
effects (statsmodels) and a likelihood-ratio / F test for the added axes.

Outcomes (feasible follow-up coverage): EGF/GAF functioning, incident
hospitalization, EQ-5D quality of life. (Work disability dropped — not measured
at follow-up.)

Artifacts: results/phase5_headtohead.csv, results/phase5_axis_effects.csv,
results/reports/phase5.html.
Run:  python3 scripts/10_phase5_outcomes.py [--visit V1]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402

from trans_diag import build_unified_dataframe  # noqa: E402
from trans_diag.outcomes import (  # noqa: E402  shared head-to-head helpers
    OUTCOMES,
    added_axes_test,
    apply_outcome_tf,
    axis_betas,
    cohort_dummies,
    cv_metric,
)

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "results" / "reports"
AXES_PATH = RESULTS_DIR / "dimensional_final_scores.parquet"
DOMAINS_PATH = RESULTS_DIR / "cluster_domains_scores.parquet"
# raw clinical-scale columns used as a comparator ("do the axes add over standard
# depression scales the clinic already collects?"). These are the residualized
# domain scores (age/sex partialled), same source as the FA. NaN-tolerant: a
# patient missing any scale is dropped from the M0_scales comparator only.
SCALE_COMPARATOR = ["qidsr", "madrs", "stai"]


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
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "data" / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
    df["pid"] = df["cohort"].str.lower() + "::" + df["usubjid_patients"].astype(str)
    v0 = df[df["visit"] == "V0"].set_index("pid")
    vk = df[df["visit"] == args.visit].set_index("pid")

    # raw clinical-scale comparator (residualized domain scores, joined by pid)
    dom = pd.read_parquet(DOMAINS_PATH)
    dom.index = pd.MultiIndex.from_arrays(
        [dom.index.get_level_values("cohort").astype(str),
         dom.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    dom = dom.reset_index()
    dom["pid"] = dom["cohort"] + "::" + dom["patient_id"]
    scale_cols_present = [c for c in SCALE_COMPARATOR if c in dom.columns]
    scale_cols = [f"scale_{c}" for c in scale_cols_present]
    scales_df = dom.set_index("pid")[scale_cols_present].rename(
        columns={c: f"scale_{c}" for c in scale_cols_present})

    base = pd.DataFrame(index=v0.index)
    base["age"], base["sex"] = pd.to_numeric(v0["age"], errors="coerce"), pd.to_numeric(v0["sex"], errors="coerce")
    # DSM diagnosis = arm (7 subtypes); arm already implies cohort.
    dsm = pd.get_dummies(v0["arm"].astype(str), drop_first=True).astype(float)
    dsm_cols = list(dsm.columns)
    base = base.join(dsm)
    # Cohort dummies (2 of 3 levels) — added to M1 for fair head-to-head parity
    # with M0_arm (which carries cohort via the arm dummies).
    cohort_dum, cohort_cols = cohort_dummies(v0["cohort"])
    base = base.join(cohort_dum)
    A = axes.set_index("pid")[axis_cols]

    head_rows, eff_rows = [], []
    for name, kind, col, tf in OUTCOMES:
        if col not in df.columns:
            continue
        y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
        # vk and v0 have different patient sets (V1 is a subset of V0); align
        # yk to v0's index so the (y0, yk) transform compares the same patient.
        yk = pd.to_numeric(vk[col], errors="coerce").reindex(y0.index)
        if tf is not None:
            yk = apply_outcome_tf(y0, yk, tf)
        d = base.join(y0).join(A).join(yk.rename("y")).dropna(
            subset=["y", "baseline", "age", "sex"] + axis_cols)
        if kind == "binary" and (d["y"].nunique() < 2 or d["y"].mean() < 0.02):
            continue
        n = len(d)
        if n < 200:
            continue
        # Raw-scale comparator: join on the subset of patients with all comparator
        # scales observed; if too few patients have them, skip the comparator only.
        d_scales = d.join(scales_df, how="left").dropna(subset=scale_cols)
        bc = ["baseline", "age", "sex"]
        m0_arm = cv_metric(d[bc + dsm_cols].to_numpy(float), d["y"].to_numpy(float), kind)
        m1_axes = cv_metric(d[bc + axis_cols].to_numpy(float), d["y"].to_numpy(float), kind)
        m1_fair = cv_metric(d[bc + cohort_cols + axis_cols].to_numpy(float),
                            d["y"].to_numpy(float), kind)
        m2_full = cv_metric(d[bc + dsm_cols + axis_cols].to_numpy(float),
                            d["y"].to_numpy(float), kind)
        # M0_scales evaluated on the (slightly smaller) common support so the
        # DSM/scales/axes comparison is apples-to-apples within the same patients.
        if len(d_scales) >= 200 and scale_cols:
            m0_arm_sub = cv_metric(d_scales[bc + dsm_cols].to_numpy(float),
                                   d_scales["y"].to_numpy(float), kind)
            m0_scales = cv_metric(d_scales[bc + dsm_cols + scale_cols].to_numpy(float),
                                  d_scales["y"].to_numpy(float), kind)
            m1_fair_sub = cv_metric(d_scales[bc + cohort_cols + axis_cols].to_numpy(float),
                                    d_scales["y"].to_numpy(float), kind)
            n_scales = len(d_scales)
            scales_minus_dsm = m0_scales - m0_arm_sub
            axes_minus_scales = m1_fair_sub - m0_scales
        else:
            m0_arm_sub = m0_scales = m1_fair_sub = float("nan")
            n_scales = 0
            scales_minus_dsm = axes_minus_scales = float("nan")
        yv = d["y"].to_numpy(float)
        p_added = added_axes_test(d, bc, dsm_cols, axis_cols, yv, kind)
        betas = axis_betas(d, bc + dsm_cols + axis_cols, axis_cols, yv, kind)
        metric = "R2" if kind == "continuous" else "AUC"
        head_rows.append({
            "outcome": name, "n": n, "metric": metric,
            "DSM": round(m0_arm, 3),
            "axes_orig": round(m1_axes, 3),
            "axes_fair": round(m1_fair, 3),
            "combined": round(m2_full, 3),
            "axes_orig_minus_DSM": round(m1_axes - m0_arm, 3),
            "axes_fair_minus_DSM": round(m1_fair - m0_arm, 3),
            "added_axes_p": p_added,
            # Raw-scales comparator (computed on a possibly-smaller subset where
            # QIDS/MADRS/STAI are jointly observed; n_scales reports that subset).
            "n_scales": n_scales,
            "DSM_plus_scales": round(m0_scales, 3) if n_scales else None,
            "scales_minus_DSM": round(scales_minus_dsm, 3) if n_scales else None,
            "axes_fair_minus_scales": round(axes_minus_scales, 3) if n_scales else None,
        })
        print(f"{name}: n={n} {metric}  DSM={m0_arm:.3f}  axes(orig)={m1_axes:.3f}  "
              f"axes(fair)={m1_fair:.3f}  combined={m2_full:.3f}\n"
              f"   axes(orig)−DSM={m1_axes-m0_arm:+.3f}   axes(fair)−DSM={m1_fair-m0_arm:+.3f}  "
              f"(p_added={p_added:.1e})")
        if n_scales:
            print(f"   [scales subset n={n_scales}] DSM+scales={m0_scales:.3f}  "
                  f"scales−DSM={scales_minus_dsm:+.3f}  axes(fair)−scales={axes_minus_scales:+.3f}")
        for a in axis_cols:
            eff_rows.append({"outcome": name, "axis": a, "beta": betas[a]})

    suf = args.visit
    head = pd.DataFrame(head_rows); head.to_csv(RESULTS_DIR / f"phase5_headtohead_{suf}.csv", index=False)
    eff = pd.DataFrame(eff_rows); eff.to_csv(RESULTS_DIR / f"phase5_axis_effects_{suf}.csv", index=False)
    _report(head, eff, axis_cols, args.visit)
    print("\nWrote results/phase5_* + results/reports/phase5.html. Done.")
    return 0


def _report(head, eff, axis_cols, visit):
    def _fmt(v):
        return "" if v is None or (isinstance(v, float) and (v != v)) else f"{v:+.3f}" if isinstance(v, (int, float)) else str(v)
    rows = "".join(
        f"<tr><td>{r.outcome}</td><td>{r.n}</td><td>{r.metric}</td><td>{r.DSM}</td>"
        f"<td>{r.axes_orig}</td><td>{r.axes_fair}</td><td>{r.combined}</td>"
        f"<td>{_fmt(r.axes_orig_minus_DSM)}</td>"
        f"<td><b>{_fmt(r.axes_fair_minus_DSM)}</b></td>"
        f"<td>{_fmt(r.scales_minus_DSM)}</td>"
        f"<td>{_fmt(r.axes_fair_minus_scales)}</td>"
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
            "<div class='c'>Nested 5-fold CV. <b>axes(orig)</b> = the pre-audit "
            "comparator (baseline+age+sex+axes, <i>no cohort</i>). <b>axes(fair)</b> = the "
            "post-audit fair comparator (baseline+age+sex+cohort+axes), which restores cohort "
            "parity with M0 (whose arm dummies already encode cohort). The 'scales−DSM' and "
            "'axes(fair)−scales' columns test whether the dimensional axes add value beyond raw "
            "QIDS+MADRS+STAI scores (a near-zero value would mean the axes' advantage is the "
            "depression scales themselves). Hospitalization is now <b>incident</b> "
            "(V1_lt&gt;V0_lt) rather than (V1_lt&gt;0).</div>",
            "<table><tr><th>outcome</th><th>n</th><th>metric</th><th>DSM</th>"
            "<th>axes(orig)</th><th>axes(fair)</th><th>combined</th>"
            "<th>orig−DSM</th><th>fair−DSM</th><th>scales−DSM</th>"
            "<th>fair−scales</th><th>added-axes p</th></tr>",
            rows, "</table>",
            pio.to_html(f, include_plotlyjs="cdn", full_html=False), "</body></html>"]
    (REPORTS_DIR / f"phase5_{visit}.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
