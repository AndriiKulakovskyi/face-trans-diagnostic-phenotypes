"""Repeated-CV confidence intervals for the V1 head-to-head (review issue #10).

The published metrics come from a single 5-fold split. Here we repeat the 5-fold CV
R times (different fold partitions) and report, for each outcome, the mean and 95%
percentile interval of the CV metric and — paired across the SAME fold partitions —
of the key difference Dimensions − DSM (and combined − DSM).

Caveat: a repeated-CV interval captures fold-partition variance, not patient-
sampling variance, so it is a lower bound on true uncertainty. We therefore ALSO
compute a patient-cluster bootstrap CI (B=400, resample patients with replacement,
fit all 3 nested models on each bootstrap draw, record paired Δ) — this is the
proper sampling-variance interval. Report both.

Post-audit (2026-05):
  - M1 was misspecified by omitting cohort dummies while M0 carried arm (which
    encodes cohort + within-cohort subtype). Now we report BOTH the original spec
    (M1_orig) AND the fair spec (M1_fair = baseline + age + sex + cohort + axes)
    so the per-outcome Δs are directly comparable.
  - Hospitalization column ``nboccur_hospitalisation_lt`` is empirically an
    interval count at follow-up (V1 mean 0.18; V0 mean 2.73), so the outcome
    (V1 > 0) is the interval incidence; V0 lifetime count enters as prior-history
    baseline. The original transform is retained.

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
from trans_diag.outcomes import (  # noqa: E402
    OUTCOMES,
    apply_outcome_tf,
    cohort_dummies,
)

RES = REPO / "results"
AXES_PATH = RES / "dimensional_final_scores.parquet"
R = 200         # fold-partition CV repeats (paired-tight, same as before audit)
BOOT = 200      # patient-cluster bootstrap repeats for sampling-variance CI (post-audit)


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


def patient_bootstrap(X_list, y, kind, B=BOOT, seed=0):
    """Patient-cluster bootstrap of CV metric for a list of nested designs.

    For B bootstrap replicates, resample patients with replacement, then run a
    single 5-fold CV for each design (single fold partition for speed; the fold-
    partition variance is separately handled by ``repeated_cv``). Returns an
    (n_designs, B) array of mean metrics, paired across designs at each replicate.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty((len(X_list), B))
    for b in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        # skip degenerate binary draws (all 0 or all 1)
        if kind == "binary" and (np.unique(yb).size < 2):
            out[:, b] = np.nan
            continue
        for j, X in enumerate(X_list):
            Xb = StandardScaler().fit_transform(X[idx])
            if kind == "continuous":
                cv = KFold(5, shuffle=True, random_state=0)
                out[j, b] = np.mean(cross_val_score(Ridge(alpha=1.0), Xb, yb, cv=cv, scoring="r2"))
            else:
                cv = StratifiedKFold(5, shuffle=True, random_state=0)
                out[j, b] = np.mean(cross_val_score(LogisticRegression(max_iter=2000), Xb, yb,
                                                    cv=cv, scoring="roc_auc"))
    return out


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
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
    cohort_dum, cohort_cols = cohort_dummies(v0["cohort"])
    base = base.join(cohort_dum)

    rows = []
    for name, kind, col, tf in OUTCOMES:
        if col not in df.columns:
            continue
        y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
        yk = pd.to_numeric(v1[col], errors="coerce").reindex(y0.index)
        if tf is not None:
            yk = apply_outcome_tf(y0, yk, tf)
        bc = ["baseline", "age", "sex"]
        d = (base.join(y0).join(A).join(yk.rename("y"))
             .dropna(subset=["y", "baseline", "age", "sex"] + axis_cols))
        if kind == "binary" and (d["y"].nunique() < 2 or d["y"].mean() < 0.02):
            continue
        if len(d) < 200:
            continue
        y = d["y"].to_numpy(float)
        X0 = d[bc + dsm_cols].to_numpy(float)
        X1_orig = d[bc + axis_cols].to_numpy(float)
        X1_fair = d[bc + cohort_cols + axis_cols].to_numpy(float)
        X2 = d[bc + dsm_cols + axis_cols].to_numpy(float)
        m0 = repeated_cv(X0, y, kind)
        m1_orig = repeated_cv(X1_orig, y, kind)
        m1_fair = repeated_cv(X1_fair, y, kind)
        m2 = repeated_cv(X2, y, kind)
        metric = "R2" if kind == "continuous" else "AUC"
        (mm0, l0, h0), (mm1o, l1o, h1o), (mm1f, l1f, h1f), (mm2, l2, h2) = (
            ci(m0), ci(m1_orig), ci(m1_fair), ci(m2))
        (dd_o, dl_o, dh_o) = ci(m1_orig - m0)
        (dd_f, dl_f, dh_f) = ci(m1_fair - m0)
        (cd, cl, ch) = ci(m2 - m0)
        # patient-cluster bootstrap for sampling-variance CI (M1_fair, M2)
        boot = patient_bootstrap([X0, X1_fair, X2], y, kind, B=BOOT)
        bm0, bm1f, bm2 = boot[0], boot[1], boot[2]
        (b_dd_f, b_dl_f, b_dh_f) = ci(bm1f - bm0)
        (b_cd,   b_cl,   b_ch)   = ci(bm2 - bm0)
        rows.append({"outcome": name, "n": len(d), "metric": metric,
                     "DSM": f"{mm0:.3f} [{l0:.3f},{h0:.3f}]",
                     "axes_orig": f"{mm1o:.3f} [{l1o:.3f},{h1o:.3f}]",
                     "axes_fair": f"{mm1f:.3f} [{l1f:.3f},{h1f:.3f}]",
                     "combined": f"{mm2:.3f} [{l2:.3f},{h2:.3f}]",
                     # CV-only (fold-partition variance — original headline format)
                     "dim_minus_DSM_orig": f"{dd_o:+.3f} [{dl_o:+.3f},{dh_o:+.3f}]",
                     "dim_minus_DSM": f"{dd_f:+.3f} [{dl_f:+.3f},{dh_f:+.3f}]",
                     "combined_minus_DSM": f"{cd:+.3f} [{cl:+.3f},{ch:+.3f}]",
                     # Patient-cluster bootstrap (sampling-variance interval)
                     "dim_minus_DSM_boot": f"{b_dd_f:+.3f} [{b_dl_f:+.3f},{b_dh_f:+.3f}]",
                     "combined_minus_DSM_boot": f"{b_cd:+.3f} [{b_cl:+.3f},{b_ch:+.3f}]",
                     "dim>DSM_excludes_0": bool(dl_f > 0 or dh_f < 0),
                     "dim>DSM_excludes_0_boot": bool(b_dl_f > 0 or b_dh_f < 0)})
        print(f"{name}: n={len(d)} {metric}\n"
              f"  DSM={mm0:.3f}[{l0:.3f},{h0:.3f}]  axes(orig)={mm1o:.3f}[{l1o:.3f},{h1o:.3f}]  "
              f"axes(fair)={mm1f:.3f}[{l1f:.3f},{h1f:.3f}]\n"
              f"  Δdim(orig)−DSM={dd_o:+.3f}[{dl_o:+.3f},{dh_o:+.3f}]  "
              f"Δdim(fair)−DSM={dd_f:+.3f}[{dl_f:+.3f},{dh_f:+.3f}]\n"
              f"  Δdim(fair)−DSM (patient bootstrap, B={BOOT}) = {b_dd_f:+.3f}[{b_dl_f:+.3f},{b_dh_f:+.3f}]")
    pd.DataFrame(rows).to_csv(RES / "phase5_ci.csv", index=False)
    print(f"\nWrote results/phase5_ci.csv (R={R} CV repeats; B={BOOT} patient-bootstrap reps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
