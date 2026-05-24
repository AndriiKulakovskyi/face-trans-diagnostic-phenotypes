"""Phase 4 (on axes) — temporal stability of the 6 dimensional axes, V0→V4.

The cluster version (09_longitudinal_coherence.py) needed a classifier and discretized
a continuum. Axes are continuous, so we ask the cleaner question directly: does a
patient keep their score on each axis across annual visits?

Pipeline (imputation-free, consistent with the dimensional model; LABBOOK E19):
  per visit → to_harmonized_dataset(DOMAIN_SECTIONS) → pool → build_domain_scores
  (common scale) → restrict to the 54 V0 domains → residualize on per-visit age+sex
  (spline + cross-fit) → standardize per domain (NaN kept, NO imputation) → PROJECT the
  locked imputation-free V0 loadings (dimensional_final_loadings.csv) onto every
  (patient, visit) via masked posterior-mean scoring → axis scores.

Then per axis, per visit: V0↔Vk test-retest correlation (Pearson/Spearman) + ICC(2,1)
on patients present at both → a trait↔state gradient.

Artifacts: results/longitudinal_axes_{stability.csv,scores.parquet}, reports/longitudinal_axes.html.
Run:  python3 scripts/08_longitudinal_axes.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from trans_diag import (  # noqa: E402
    ADMINISTRATIVE_FEATURES,
    COHORT_TO_CODE,
    DOMAIN_SECTIONS,
    build_domain_scores,
    build_unified_dataframe,
    load_variables,
    residualize_features,
    to_harmonized_dataset,
)
from trans_diag.masked_fa import masked_scores  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
VISITS = ["V0", "V1", "V2", "V3", "V4"]
K = 6
SPLINE_DF, CROSS_FIT, RANDOM = 4, 5, 0
# axis order = SS order of the imputation-free model (07_dimensional_refine; LABBOOK E19):
# the former ADHD/impulsivity/trauma axis is gone — impulsivity (WURS/BIS) merges into
# mania/activation, and the 6th axis is now socio-occupational/work-disability.
AXIS_NAMES = ["depression_severity", "later_onset", "mania_activation",
              "illness_burden", "metabolic", "work_disability"]


def icc21(a, b):
    """ICC(2,1) two-way random, single rater (test-retest absolute agreement)."""
    Y = np.column_stack([a, b]); n, k = Y.shape
    gm = Y.mean()
    ms_r = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    ms_e = (((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum()
            / ((n - 1) * (k - 1)))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom > 0 else float("nan")


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    variables = load_variables(REPO_ROOT / "face-common-vars.xlsx")
    exclude = set(ADMINISTRATIVE_FEATURES) | {v.canonical_name for v in variables
                                              if v.canonical_name.endswith("_mhoccur")}
    v0_domains = list(pd.read_parquet(RESULTS_DIR / "cluster_domains_scores.parquet").columns)

    print("Building per-visit domain scores...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
        items, covs = [], []
        for v in VISITS:
            fv = df[df["visit"] == v]
            if fv.empty:
                continue
            ds = to_harmonized_dataset(df, variables, visit=v, exclude=exclude,
                                       sections=DOMAIN_SECTIONS)
            X = ds.X.copy()
            X.index = pd.MultiIndex.from_arrays(
                [X.index.get_level_values("cohort"), X.index.get_level_values("patient_id"),
                 [v] * len(X)], names=("cohort", "patient_id", "visit"))
            items.append(X)
            cov = fv[["cohort", "usubjid_patients", "age", "sex"]].copy()
            cov.index = pd.MultiIndex.from_arrays(
                [fv["cohort"].map(COHORT_TO_CODE).to_numpy(),
                 fv["usubjid_patients"].astype(str).to_numpy(), [v] * len(fv)],
                names=("cohort", "patient_id", "visit"))
            covs.append(cov[["age", "sex"]])
    items = pd.concat(items)
    covars = pd.concat(covs)
    covars = covars[~covars.index.duplicated(keep="first")].reindex(items.index)

    scores, _ = build_domain_scores(items, variables)
    scores = scores.reindex(columns=v0_domains)
    scores_r = residualize_features(scores, covars, spline_df=SPLINE_DF,
                                    cross_fit=CROSS_FIT, random_state=RANDOM)
    # standardize per domain, KEEPING NaN (no imputation), then project the LOCKED
    # imputation-free V0 loadings onto every (patient, visit) via masked posterior-mean scoring
    # — the same loadings applied to every visit (cf. Methods §2.10), not a per-visit refit.
    Z = (scores_r - scores_r.mean()) / scores_r.std(ddof=0)
    final_load = (pd.read_csv(RESULTS_DIR / "dimensional_final_loadings.csv")
                  .pivot(index="domain", columns="axis", values="loading").reindex(v0_domains))
    load = final_load[sorted(final_load.columns, key=lambda c: int(str(c).replace("axis", "")))].to_numpy()
    axis = pd.DataFrame(masked_scores(Z.to_numpy(), load), index=Z.index, columns=AXIS_NAMES)
    axis.to_parquet(RESULTS_DIR / "longitudinal_axes_scores.parquet")
    n_scored = int(axis["depression_severity"].notna().sum())
    print(f"projected the {K} locked imputation-free V0 axes onto {n_scored:,} (patient,visit) rows "
          f"via masked scoring (no imputation).")

    # test-retest stability per axis per visit
    print("\ntest-retest stability (V0↔Vk Pearson r):")
    rows = []
    for a in AXIS_NAMES:
        wide = axis[a].unstack("visit")
        for v in VISITS[1:]:
            if v not in wide.columns:
                continue
            pair = wide[["V0", v]].dropna()
            if len(pair) < 50:
                continue
            r = float(pearsonr(pair["V0"], pair[v])[0])
            rho = float(spearmanr(pair["V0"], pair[v]).statistic)
            icc = icc21(pair["V0"].to_numpy(), pair[v].to_numpy())
            rows.append({"axis": a, "visit": v, "n": len(pair), "pearson": r,
                         "spearman": rho, "icc": icc})
        m = [x["pearson"] for x in rows if x["axis"] == a]
        print(f"  {a:24s} V0↔Vk r = {[round(x,2) for x in m]}")
    stab = pd.DataFrame(rows)
    stab.to_csv(RESULTS_DIR / "longitudinal_axes_stability.csv", index=False)

    # trait↔state ranking (mean V0↔V1/V2 r)
    early = stab[stab["visit"].isin(["V1", "V2"])].groupby("axis")["pearson"].mean().sort_values(ascending=False)
    print("\ntrait↔state gradient (mean V0↔V1/V2 r):")
    for a, r in early.items():
        tag = "TRAIT-like" if r >= 0.5 else "intermediate" if r >= 0.35 else "STATE-like"
        print(f"  {a:24s} {r:.2f}  {tag}")

    meta = {"K": K, "axis_names": AXIS_NAMES,
            "method": "locked imputation-free V0 loadings projected via masked scoring (no imputation)",
            "trait_state_meanr_V1V2": {a: round(float(r), 3) for a, r in early.items()}}
    (RESULTS_DIR / "longitudinal_axes_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    _report(stab, early)
    print("\nWrote results/longitudinal_axes_* + reports/longitudinal_axes.html. Done.")
    return 0


def _report(stab, early):
    piv = stab.pivot_table(index="axis", columns="visit", values="pearson")
    piv = piv.reindex(early.index)
    f = go.Figure(go.Heatmap(z=piv.to_numpy(), x=list(piv.columns), y=list(piv.index),
                             text=piv.round(2).to_numpy(), texttemplate="%{text}",
                             colorscale="Viridis", zmin=0, zmax=1,
                             colorbar=dict(title="V0↔Vk r", thickness=12)))
    f.update_layout(title="Axis test-retest stability (V0↔Vk Pearson r) — trait↔state gradient",
                    height=360, xaxis_title="follow-up visit", yaxis_title="axis (trait→state)",
                    margin=dict(t=46, l=170))
    css = "body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}h1{color:#2b3a55}.c{background:#f2fbf6;border-left:4px solid #16a085;padding:10px 14px;margin:12px 0}"
    trait = "; ".join(f"{a} {r:.2f}" for a, r in early.items())
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Phase 4 (axes) — temporal stability of the dimensional axes</h1>",
            f"<div class='c'>Mean V0↔V1/V2 test-retest r by axis: {trait}. Higher = "
            "trait-like (persists); lower = state-like (fluctuates).</div>",
            pio.to_html(f, include_plotlyjs="cdn", full_html=False), "</body></html>"]
    (REPORTS_DIR / "longitudinal_axes.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
