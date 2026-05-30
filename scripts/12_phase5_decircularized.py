"""De-circularized head-to-head (review issue #1).

The locked dimensional axes contain the V0 values of the outcomes we predict:
the depression axis loads on EQ-5D, EQ-VAS, EGF and FAST; the illness-burden axis
loads on the lifetime-hospitalization counts. Predicting those same outcomes from
axes that contain them — even with baseline adjustment — is partly circular and
unfair to DSM (a bare label with no such content).

This script re-runs the head-to-head with axes that EXCLUDE each outcome's own
measure(s) before the factor analysis is refit, then compares:
  - DSM (M0)            baseline + age + sex + arm
  - axes_full (M1)      refit on all 54 domains (≈ the locked, circular axes)
  - axes_clean (M1*)    refit WITHOUT the outcome's domains (de-circularized)
  - combined_clean (M2) DSM + axes_clean
All nested 5-fold CV, leakage-safe (predictors V0, outcome V1, baseline-adjusted).

Per-outcome exclusions (own measure + same-instrument synonym):
  EQ-5D quality of life  → drop {eq5d, eq}
  EGF functioning        → drop {egf, fast}
  any hospitalization    → drop {nboccur_hospitalisation_lt, hodur_hospitalisation_lt}

Writes results/phase5_decircularized.csv. Run: python3 scripts/12_phase5_decircularized.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trans_diag import build_unified_dataframe  # noqa: E402
from trans_diag.masked_fa import (  # noqa: E402  (same estimator as 07)
    masked_loadings,
    masked_scores,
)
from trans_diag.outcomes import (  # noqa: E402  (reuse the exact CV metric)
    apply_outcome_tf,
    cohort_dummies,
    cv_metric,
)

RES = REPO / "results"
SCORES = RES / "cluster_domains_scores.parquet"     # residualized domain scores
K = json.loads((RES / "dimensional_final_meta.json").read_text())["K"]  # locked by 07
# (name, kind, outcome_col, transform, exclude_from_axes)
OUTCOMES = [
    ("EQ-5D quality of life", "continuous", "eq5d", None, ["eq5d", "eq"]),
    ("EGF functioning", "continuous", "egf", None, ["egf", "fast"]),
    ("any hospitalization", "binary", "nboccur_hospitalisation_lt",
     lambda y0, yk: (yk > 0).astype(float),
     ["nboccur_hospitalisation_lt", "hodur_hospitalisation_lt"]),
]


def fit_axes(scores_df: pd.DataFrame, exclude) -> pd.DataFrame:
    """Refit the K=7 varimax axes IMPUTATION-FREE on the residualized domains minus `exclude`.

    Uses the SAME estimator as the locked model (07): masked pairwise-complete correlation ->
    principal-axis factoring + varimax -> masked posterior-mean scores on each patient's observed
    support (no cell ever filled). The previous version mean-filled (z.fillna(0) + sklearn FA),
    which §3.8 shows reweights correlations by co-observation and biases the weakest factor — so
    the de-circularization now probes the published masked model, not a superseded mean-fill one.
    """
    cols = [c for c in scores_df.columns if c not in set(exclude)]
    sub = scores_df[cols]
    load = masked_loadings(sub, K)
    for a in range(K):                                       # orient: defining domain positive
        j = int(np.argmax(np.abs(load[:, a])))
        if load[j, a] < 0:
            load[:, a] = -load[:, a]
    load = load[:, np.argsort(-(load ** 2).sum(0))]          # order by sum-of-squares (as in 07)
    z = (sub - sub.mean()) / sub.std(ddof=0)
    scores = masked_scores(z, load)                          # NaN for <K observed (no imputation)
    out = pd.DataFrame(scores, index=scores_df.index, columns=[f"axis{i+1}" for i in range(K)])
    out.index = [f"{c}::{p}" for c, p in scores_df.index]
    return out


def main() -> int:
    sc = pd.read_parquet(SCORES)
    sc.index = pd.MultiIndex.from_arrays(
        [sc.index.get_level_values("cohort").astype(str),
         sc.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO / "data", REPO / "data" / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
    df["pid"] = df["cohort"].str.lower() + "::" + df["usubjid_patients"].astype(str)
    v0 = df[df["visit"] == "V0"].set_index("pid")
    v1 = df[df["visit"] == "V1"].set_index("pid")

    base = pd.DataFrame(index=v0.index)
    base["age"] = pd.to_numeric(v0["age"], errors="coerce")
    base["sex"] = pd.to_numeric(v0["sex"], errors="coerce")
    dsm = pd.get_dummies(v0["arm"].astype(str), drop_first=True).astype(float)
    dsm_cols = list(dsm.columns)
    base = base.join(dsm)
    # Post-audit: add cohort dummies for fair head-to-head parity (M0_arm
    # encodes cohort via the 7-level arm; the axes model without cohort was
    # forced to act as a cohort surrogate). See 10_phase5_outcomes.py for context.
    cohort_dum, cohort_cols = cohort_dummies(v0["cohort"])
    base = base.join(cohort_dum)

    axes_full = fit_axes(sc, exclude=[])               # ≈ locked (circular) axes
    axis_cols = list(axes_full.columns)

    rows = []
    for name, kind, col, tf, excl in OUTCOMES:
        if col not in df.columns:
            continue
        axes_clean = fit_axes(sc, exclude=excl)
        y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
        # align V1 to V0's patient index (different patient sets)
        yk = pd.to_numeric(v1[col], errors="coerce").reindex(y0.index)
        if tf is not None:
            yk = apply_outcome_tf(y0, yk, tf)
        bc = ["baseline", "age", "sex"]
        d = (base.join(y0).join(yk.rename("y"))
             .join(axes_full.add_suffix("_f")).join(axes_clean.add_suffix("_c"))
             .dropna(subset=["y", "baseline", "age", "sex"]
                     + [f"{a}_f" for a in axis_cols] + [f"{a}_c" for a in axis_cols]))
        if kind == "binary" and (d["y"].nunique() < 2 or d["y"].mean() < 0.02):
            continue
        if len(d) < 200:
            continue
        n = len(d); y = d["y"].to_numpy(float)
        m0 = cv_metric(d[bc + dsm_cols].to_numpy(float), y, kind)
        # axes_full / axes_clean with cohort parity (post-audit fair comparator)
        m1_full = cv_metric(d[bc + cohort_cols + [f"{a}_f" for a in axis_cols]].to_numpy(float),
                            y, kind)
        m1_clean = cv_metric(d[bc + cohort_cols + [f"{a}_c" for a in axis_cols]].to_numpy(float),
                             y, kind)
        # Also keep an axes-alone (no cohort) row for backward comparison
        m1_clean_orig = cv_metric(d[bc + [f"{a}_c" for a in axis_cols]].to_numpy(float), y, kind)
        m2_clean = cv_metric(d[bc + dsm_cols + [f"{a}_c" for a in axis_cols]].to_numpy(float), y, kind)
        metric = "R2" if kind == "continuous" else "AUC"
        rows.append({"outcome": name, "n": n, "metric": metric, "excluded": "+".join(excl),
                     "DSM": round(m0, 3),
                     "axes_full_fair": round(m1_full, 3),
                     "axes_clean_fair": round(m1_clean, 3),
                     "axes_clean_orig": round(m1_clean_orig, 3),
                     "combined_clean": round(m2_clean, 3),
                     "full_fair_minus_DSM": round(m1_full - m0, 3),
                     "clean_fair_minus_DSM": round(m1_clean - m0, 3),
                     "clean_orig_minus_DSM": round(m1_clean_orig - m0, 3),
                     "combined_minus_DSM": round(m2_clean - m0, 3)})
        print(f"{name}: n={n} {metric}  DSM={m0:.3f} | axes_full(fair)={m1_full:.3f} "
              f"(Δ{m1_full-m0:+.3f}) → axes_clean(fair)={m1_clean:.3f} "
              f"(Δ{m1_clean-m0:+.3f}; orig {m1_clean_orig:.3f} Δ{m1_clean_orig-m0:+.3f}) | "
              f"combined_clean={m2_clean:.3f} (Δ{m2_clean-m0:+.3f})   [drop {excl}]")
    out = pd.DataFrame(rows)
    out.to_csv(RES / "phase5_decircularized.csv", index=False)
    print("\nWrote results/phase5_decircularized.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
