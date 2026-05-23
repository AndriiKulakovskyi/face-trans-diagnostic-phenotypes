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

Writes results/phase5_decircularized.csv. Run: python3 scripts/phase5_decircularized.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from sklearn.decomposition import FactorAnalysis  # noqa: E402

from face_common import build_unified_dataframe  # noqa: E402
from phase5_outcomes import cv_metric  # noqa: E402  (reuse the exact CV metric)

RES = REPO / "results"
SCORES = RES / "cluster_domains_scores.parquet"     # residualized 54 domains
K = 6
# (name, kind, outcome_col, transform, exclude_from_axes)
OUTCOMES = [
    ("EQ-5D quality of life", "continuous", "eq5d", None, ["eq5d", "eq"]),
    ("EGF functioning", "continuous", "egf", None, ["egf", "fast"]),
    ("any hospitalization", "binary", "nboccur_hospitalisation_lt",
     lambda s: (s > 0).astype(float), ["nboccur_hospitalisation_lt", "hodur_hospitalisation_lt"]),
]


def fit_axes(scores_df: pd.DataFrame, exclude) -> pd.DataFrame:
    """Refit the K=6 varimax axes on the residualized domains minus `exclude`."""
    cols = [c for c in scores_df.columns if c not in set(exclude)]
    sub = scores_df[cols]
    z = (sub - sub.mean()) / sub.std(ddof=0)
    X = z.fillna(0.0).to_numpy(np.float64)
    fa = FactorAnalysis(n_components=K, rotation="varimax", random_state=0).fit(X)
    sc = fa.transform(X)
    order = np.argsort(-(fa.components_.T ** 2).sum(0))      # by SS loading
    sc = sc[:, order]
    out = pd.DataFrame(sc, index=scores_df.index, columns=[f"axis{i+1}" for i in range(K)])
    out.index = [f"{c}::{p}" for c, p in scores_df.index]
    return out


def main() -> int:
    sc = pd.read_parquet(SCORES)
    sc.index = pd.MultiIndex.from_arrays(
        [sc.index.get_level_values("cohort").astype(str),
         sc.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO / "data", REPO / "face-common-vars.xlsx",
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

    axes_full = fit_axes(sc, exclude=[])               # ≈ locked (circular) axes
    axis_cols = list(axes_full.columns)

    rows = []
    for name, kind, col, tf, excl in OUTCOMES:
        if col not in df.columns:
            continue
        axes_clean = fit_axes(sc, exclude=excl)
        y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
        yk = pd.to_numeric(v1[col], errors="coerce")
        yk = tf(yk) if tf is not None else yk
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
        m1_full = cv_metric(d[bc + [f"{a}_f" for a in axis_cols]].to_numpy(float), y, kind)
        m1_clean = cv_metric(d[bc + [f"{a}_c" for a in axis_cols]].to_numpy(float), y, kind)
        m2_clean = cv_metric(d[bc + dsm_cols + [f"{a}_c" for a in axis_cols]].to_numpy(float), y, kind)
        metric = "R2" if kind == "continuous" else "AUC"
        rows.append({"outcome": name, "n": n, "metric": metric, "excluded": "+".join(excl),
                     "DSM": round(m0, 3), "axes_full": round(m1_full, 3),
                     "axes_clean": round(m1_clean, 3), "combined_clean": round(m2_clean, 3),
                     "full_minus_DSM": round(m1_full - m0, 3),
                     "clean_minus_DSM": round(m1_clean - m0, 3),
                     "combined_minus_DSM": round(m2_clean - m0, 3)})
        print(f"{name}: n={n} {metric}  DSM={m0:.3f} | axes_full={m1_full:.3f} "
              f"(Δ{m1_full-m0:+.3f}) → axes_clean={m1_clean:.3f} (Δ{m1_clean-m0:+.3f}) | "
              f"combined_clean={m2_clean:.3f} (Δ{m2_clean-m0:+.3f})   [drop {excl}]")
    out = pd.DataFrame(rows)
    out.to_csv(RES / "phase5_decircularized.csv", index=False)
    print(f"\nWrote results/phase5_decircularized.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
