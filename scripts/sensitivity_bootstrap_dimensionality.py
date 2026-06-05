"""Sensitivity — bootstrap robustness of the second-order dimensionality (LABBOOK V2-23).

Why: adding a single orthogonal construct moved the split-half "first-collapse-minus-1" lock from
K=4 to K=3, which looked alarming. This quantifies what is and isn't robust, by **cohort-stratified
nonparametric bootstrap** of the Stage-2 construct scores (resample patients within cohort, recompute
the masked Φ₁ and everything downstream of it). It reports three things:

  1. eigenvalue / eigengap 95% CIs of Φ₁  — *which* directions are well-separated;
  2. the distribution of the locked K (the actual split-half rule)  — is the scalar K stable?
  3. per-factor stability  — fixing K=6, how often does each factor recover (Tucker congruence ≥ 0.85)?

Result (see docs/PHENOTYPE_ATLAS.md, manuscript §3.1): the first 2-3 eigengaps are bounded off 0 but
gap λ4-λ5 ≈ 0 (a degenerate eigenpair); the locked K is a *noisy* estimator (mode 3); yet every factor
recovers in 98-100% of resamples. The data is "3 weakly-correlated axes + reproducible ORTHOGONAL
standalones," and the scalar K conflates "#reproducible factors (>=6)" with "#correlated axes (3)".

Deterministic (fixed per-replicate seeds). Writes results/hfa/bootstrap_dimensionality.json.
Run:  python3 scripts/sensitivity_bootstrap_dimensionality.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from trans_diag.masked_fa import masked_correlation, paf_loadings, varimax

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
MP, COV, FLOOR = 100, 0.30, 0.85
B_EIG, B_K, N_SPLITS, KMAX, KREF = 400, 50, 10, 6, 6


def main() -> None:
    S = pd.read_pickle(OUT / "stage2_scores.pkl").set_index(["cohort", "patient_id"])
    cov, var = S.notna().mean(), S.var()
    keep = [c for c in S.columns if not ((var[c] < 1e-9) or pd.isna(var[c])) and cov[c] >= COV]
    coh = np.asarray(S.index.get_level_values("cohort"))
    idxc = {c: np.where(coh == c)[0] for c in np.unique(coh)}
    Sk = S[keep].to_numpy(float)

    def stdz(rows):
        X = Sk[rows]; mu = np.nanmean(X, 0); sd = np.nanstd(X, 0); sd[sd == 0] = 1
        return pd.DataFrame((X - mu) / sd)

    def resample(rng):
        return np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idxc.values()])

    def loadings(Z, K):
        return varimax(paf_loadings(masked_correlation(Z, MP), K))

    def matched(A, B):
        M = np.abs(A.T @ B) / (np.sqrt(np.outer((A * A).sum(0), (B * B).sum(0))) + 1e-12)
        r, c = linear_sum_assignment(-M)
        return M[r, c]

    def kselect(Z, rng):
        K2, collapsed = 1, False
        for K in range(2, KMAX + 1):
            m = len(Z); mins = []
            for _ in range(N_SPLITS):
                p = rng.permutation(m); a, b = p[: m // 2], p[m // 2:]
                mins.append(matched(loadings(Z.iloc[a], K), loadings(Z.iloc[b], K)).min())
            c = float(np.mean(mins))
            if not collapsed:
                if c >= FLOOR:
                    K2 = K
                else:
                    collapsed = True
        return K2

    # reference K=KREF factors (full sample), named by top construct
    Lref = loadings(stdz(np.arange(len(Sk))), KREF)
    refnames = [keep[int(np.argmax(np.abs(Lref[:, k])))] for k in range(KREF)]
    print(f"constructs={len(keep)} | reference K={KREF} factors: {refnames}", flush=True)

    # (1) eigenvalue / eigengap bootstrap
    EV = np.zeros((B_EIG, 8))
    for i in range(B_EIG):
        rng = np.random.default_rng(1000 + i)
        EV[i] = np.sort(np.linalg.eigvalsh(masked_correlation(stdz(resample(rng)), MP)))[::-1][:8]
        if (i + 1) % 100 == 0:
            print(f"  eig boot {i + 1}/{B_EIG}", flush=True)
    gaps = -np.diff(EV, axis=1)

    # (2)+(3) K distribution + per-factor stability (shared resamples)
    Ks, STAB = [], np.zeros((B_K, KREF))
    for i in range(B_K):
        rng = np.random.default_rng(7000 + i); Z = stdz(resample(rng))
        Ks.append(kselect(Z, rng))
        STAB[i] = matched(Lref, loadings(Z, KREF))
        if (i + 1) % 10 == 0:
            print(f"  K+stability boot {i + 1}/{B_K}  K so far={dict(pd.Series(Ks).value_counts())}", flush=True)

    res = {
        "eig_mean": EV.mean(0).round(2).tolist(),
        "eig_ci": [np.percentile(EV, 2.5, 0).round(2).tolist(), np.percentile(EV, 97.5, 0).round(2).tolist()],
        "gap_mean": gaps.mean(0).round(2).tolist(),
        "gap_ci": [np.percentile(gaps, 2.5, 0).round(2).tolist(), np.percentile(gaps, 97.5, 0).round(2).tolist()],
        "K_dist": {int(k): int(v) for k, v in pd.Series(Ks).value_counts().sort_index().items()},
        "ref_factors": refnames,
        "stability_pct": {refnames[k]: round(float((STAB[:, k] >= FLOOR).mean()) * 100, 1) for k in range(KREF)},
        "stability_meancong": {refnames[k]: round(float(STAB[:, k].mean()), 2) for k in range(KREF)},
        "n_boot_eig": B_EIG, "n_boot_K": B_K,
    }
    json.dump(res, open(OUT / "bootstrap_dimensionality.json", "w"), indent=2)
    print("\n=== eigengaps (mean [95% CI]) ===")
    for i in range(7):
        print(f"  gap{i+1}: {res['gap_mean'][i]:.2f}  [{res['gap_ci'][0][i]:.2f}, {res['gap_ci'][1][i]:.2f}]")
    print(f"K distribution: {res['K_dist']}")
    print(f"per-factor stability %: {res['stability_pct']}")
    print(f"\nsaved -> {OUT / 'bootstrap_dimensionality.json'}")


if __name__ == "__main__":
    main()
