"""Repeated-CV confidence intervals for the V1 head-to-head (review issue #10).

The published metrics come from a single 5-fold split. Here we repeat the 5-fold CV
R times (different fold partitions) and report, for each outcome, the mean and 95%
percentile interval of the CV metric and — paired across the SAME fold partitions —
of the key difference Dimensions − DSM (and combined − DSM).

Caveat: a repeated-CV interval captures fold-partition variance, not cohort
sampling variance, so it is a lower bound on true uncertainty.

Writes results/phase5_ci.csv. Run: python3 scripts/11_phase5_ci.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from trans_diag import build_unified_dataframe  # noqa: E402

RES = REPO / "results"
AXES_PATH = RES / "dimensional_final_scores.parquet"
R = 200
OUTCOMES = [
    ("EQ-5D quality of life", "continuous", "eq5d", None),
    ("EGF functioning", "continuous", "egf", None),
    ("any hospitalization", "binary", "nboccur_hospitalisation_lt",
     lambda s: (s > 0).astype(float)),
]


def repeated_cv(X, y, kind, R=R):
    """R repeats of 5-fold CV; returns the per-repeat mean metric (paired by seed)."""
    Xs = StandardScaler().fit_transform(X)
    out = np.empty(R)
    for r in range(R):
        if kind == "continuous":
            cv = KFold(5, shuffle=True, random_state=r)
            out[r] = np.mean(cross_val_score(Ridge(alpha=1.0), Xs, y, cv=cv, scoring="r2"))
        else:
            cv = StratifiedKFold(5, shuffle=True, random_state=r)
            out[r] = np.mean(cross_val_score(LogisticRegression(max_iter=2000), Xs, y,
                                             cv=cv, scoring="roc_auc"))
    return out


def ci(a):
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main() -> int:
    axes = pd.read_parquet(AXES_PATH)
    axis_cols = list(axes.columns)
    axes.index = pd.MultiIndex.from_arrays(
        [axes.index.get_level_values("cohort").astype(str),
         axes.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    A = axes.copy()
    A.index = [f"{c}::{p}" for c, p in A.index]

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

    rows = []
    for name, kind, col, tf in OUTCOMES:
        if col not in df.columns:
            continue
        y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
        yk = pd.to_numeric(v1[col], errors="coerce")
        yk = tf(yk) if tf is not None else yk
        bc = ["baseline", "age", "sex"]
        d = (base.join(y0).join(A).join(yk.rename("y"))
             .dropna(subset=["y", "baseline", "age", "sex"] + axis_cols))
        if kind == "binary" and (d["y"].nunique() < 2 or d["y"].mean() < 0.02):
            continue
        if len(d) < 200:
            continue
        y = d["y"].to_numpy(float)
        m0 = repeated_cv(d[bc + dsm_cols].to_numpy(float), y, kind)
        m1 = repeated_cv(d[bc + axis_cols].to_numpy(float), y, kind)
        m2 = repeated_cv(d[bc + dsm_cols + axis_cols].to_numpy(float), y, kind)
        metric = "R2" if kind == "continuous" else "AUC"
        d0, d1, d2 = m1 - m0, m2 - m0, m2 - m1
        (mm0, l0, h0), (mm1, l1, h1), (mm2, l2, h2) = ci(m0), ci(m1), ci(m2)
        (dd, dl, dh) = ci(d0)            # Dimensions − DSM
        (cd, cl, ch) = ci(d1)            # combined − DSM
        rows.append({"outcome": name, "n": len(d), "metric": metric,
                     "DSM": f"{mm0:.3f} [{l0:.3f},{h0:.3f}]",
                     "axes": f"{mm1:.3f} [{l1:.3f},{h1:.3f}]",
                     "combined": f"{mm2:.3f} [{l2:.3f},{h2:.3f}]",
                     "dim_minus_DSM": f"{dd:+.3f} [{dl:+.3f},{dh:+.3f}]",
                     "combined_minus_DSM": f"{cd:+.3f} [{cl:+.3f},{ch:+.3f}]",
                     "dim>DSM_excludes_0": bool(dl > 0 or dh < 0)})
        print(f"{name}: n={len(d)} {metric}  DSM={mm0:.3f}[{l0:.3f},{h0:.3f}]  "
              f"axes={mm1:.3f}[{l1:.3f},{h1:.3f}]  Δdim−DSM={dd:+.3f}[{dl:+.3f},{dh:+.3f}]  "
              f"Δcomb−DSM={cd:+.3f}[{cl:+.3f},{ch:+.3f}]")
    pd.DataFrame(rows).to_csv(RES / "phase5_ci.csv", index=False)
    print(f"\nWrote results/phase5_ci.csv (R={R} repeats)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
