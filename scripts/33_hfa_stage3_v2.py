"""Stage 3 (v2) — second-order / bifactor general dimensions (the named trans-diagnostic axes).

Implements Stage 3 of docs/HIERARCHICAL_FA_PLAN.md. Factors the construct-correlation matrix Phi_1
(Stage 2 construct scores) into general dimensions, locks K by masked split-half Tucker congruence
(NOT Kaiser), and TESTS whether a single general factor ("p-factor") is warranted via ECV/Schmid-
Leiman rather than assuming one. Masked / no-imputation throughout.

=== STATISTICAL-CORRECTNESS AUDIT (printed first) — guards the usual scale/FA mistakes ===
  * construct scores STANDARDIZED (z) before correlation/scoring -> single-item and multi-item
    constructs comparable (no scale artefact);
  * correlation is PEARSON (note: binary-derived constructs are Pearson-attenuated; polychoric is the
    D9 phase-2 sensitivity) -> reported, not silently ignored;
  * sparse constructs (coverage < floor) EXCLUDED from extraction (a 6%-observed construct cannot be
    reliably placed) and reported, not left to distort the matrix;
  * Phi_1 forced positive-definite (nearest_pd) before factoring; pre-repair neg-eigen mass reported;
  * factor signs deterministically oriented; K by split-half reproducibility, not eigenvalue>1.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from factor_analyzer import Rotator
from scipy.optimize import linear_sum_assignment

from trans_diag.masked_fa import masked_correlation, masked_scores, paf_loadings, varimax

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
MIN_PAIR = 100
COVERAGE_FLOOR = 0.30        # a construct observed in <30% of patients cannot be reliably placed
K_FLOOR = 0.85               # lock K = max K whose split-half mean-min Tucker congruence >= this
SEED = 0


def tucker(A, B):
    return np.abs(A.T @ B) / (np.sqrt(np.outer((A * A).sum(0), (B * B).sum(0))) + 1e-12)


def splithalf_congruence(Z, K, n_splits=15, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(Z)
    mins = []
    for _ in range(n_splits):
        p = rng.permutation(n)
        a, b = p[: n // 2], p[n // 2:]
        La = varimax(paf_loadings(masked_correlation(Z.iloc[a], MIN_PAIR), K))
        Lb = varimax(paf_loadings(masked_correlation(Z.iloc[b], MIN_PAIR), K))
        M = tucker(La, Lb)
        r, c = linear_sum_assignment(-M)
        mins.append(M[r, c].min())
    return float(np.mean(mins))


def main() -> None:
    S = pd.read_pickle(OUT / "stage2_scores_v2.pkl").set_index(["cohort", "patient_id"])
    fit = pd.read_csv(OUT / "stage2_construct_fit_v2.csv").set_index("construct")

    # ---------------- AUDIT ----------------
    print("=== STATISTICAL-CORRECTNESS AUDIT ===")
    cov = S.notna().mean()
    var = S.var()
    degenerate = sorted(var[(var < 1e-9) | var.isna()].index)
    print(f"  constructs: {S.shape[1]} | degenerate (zero-variance): {len(degenerate)} {degenerate}")
    keep = [c for c in S.columns if c not in degenerate and cov[c] >= COVERAGE_FLOOR]
    excluded = sorted(set(S.columns) - set(keep))
    print(f"  excluded (coverage<{COVERAGE_FLOOR:.0%} -> cannot place): {len(excluded)}")
    print(f"    {[f'{c}({cov[c]:.2f})' for c in sorted(excluded, key=lambda c: cov[c])][:12]}")
    Z = ((S[keep] - S[keep].mean()) / S[keep].std())            # STANDARDIZE before anything
    print(f"  standardized {len(keep)} constructs (mean~0, sd~1): "
          f"mean|mu|={Z.mean().abs().mean():.0e}, mean sd={Z.std().mean():.2f}")
    # Phi_1 conditioning
    Rraw = Z.corr(min_periods=MIN_PAIR).to_numpy()
    zeroed = (~np.isfinite(Rraw)).sum() / 2 / (len(keep) * (len(keep) - 1) / 2)
    R0 = np.nan_to_num(Rraw, nan=0.0); np.fill_diagonal(R0, 1.0)
    w = np.linalg.eigvalsh(R0)
    negmass = np.abs(w[w < 0]).sum() / np.abs(w).sum()
    Phi1 = masked_correlation(Z, MIN_PAIR)
    print(f"  Phi_1: {zeroed*100:.1f}% cells <min_pair (set 0); pre-repair neg-eigen mass {negmass*100:.1f}% "
          f"-> nearest_pd. binary-derived constructs are Pearson-attenuated (polychoric = D9 phase-2).")

    # ---------------- K by split-half reproducibility ----------------
    # K-lock = first collapse minus 1 (NOT max-K>=floor: split-half congruence is non-monotonic and
    # over-extraction shows spurious recovery + Heywood cases at high K — script 07's convention).
    print("\n=== K (second-order dimensions) by masked split-half congruence ===")
    K2, collapsed = 1, False
    for K in range(2, 11):
        c = splithalf_congruence(Z, K)
        note = "" if collapsed else (" reproducible" if c >= K_FLOOR else "  <-- FIRST COLLAPSE")
        print(f"  K={K}: split-half mean-min Tucker congruence = {c:.2f}{note}")
        if not collapsed:
            if c >= K_FLOOR:
                K2 = K
            else:
                collapsed = True     # ignore any spurious recovery at higher K
    print(f"  locked K2 = {K2} (first collapse minus 1; spurious high-K recovery ignored)")

    # ---------------- extract oblique second-order dimensions ----------------
    A = paf_loadings(Phi1, K2)
    rot = Rotator(method="promax")
    L2 = rot.fit_transform(A)                 # constructs x K2 oblique pattern
    Phi2 = rot.phi_                           # dimension correlations
    # deterministic sign: orient each dimension so its largest |loading| is positive
    for k in range(K2):
        if L2[np.argmax(np.abs(L2[:, k])), k] < 0:
            L2[:, k] *= -1; Phi2[k, :] *= -1; Phi2[:, k] *= -1
    L2df = pd.DataFrame(L2, index=keep, columns=[f"dim{k+1}" for k in range(K2)])
    heywood = int((np.abs(L2) > 1.0).sum())
    print(f"  Heywood check (|loading|>1, improper): {heywood}  "
          f"({'OK — proper solution' if heywood == 0 else 'IMPROPER — over-extracted'})")

    # ---------------- general-factor test (Schmid-Leiman ECV) ----------------
    wq, Vq = np.linalg.eigh(Phi2)
    gamma = Vq[:, -1] * np.sqrt(max(wq[-1], 0))      # 2nd-order general loadings of the K2 dims
    gamma = np.clip(gamma, -0.99, 0.99)
    g = L2 @ gamma                                   # SL general loading per construct
    spec = L2 * np.sqrt(1 - gamma ** 2)              # SL specific loadings
    ecv = float((g ** 2).sum() / ((g ** 2).sum() + (spec ** 2).sum()))
    print(f"\n=== general-factor test (Schmid-Leiman) ===")
    print(f"  ECV (explained common variance by a general factor) = {ecv:.2f}  "
          f"({'general p-factor warranted' if ecv >= 0.5 else 'multidimensional — no dominant p-factor'})")
    print(f"  mean |dimension correlation| (Phi_2) = {np.abs(Phi2[np.triu_indices(K2,1)]).mean():.2f}")

    # ---------------- name the axes ----------------
    print(f"\n=== the {K2} trans-diagnostic dimensions (top constructs, |loading|>0.3) ===")
    for k in range(K2):
        top = L2df.iloc[:, k].reindex(L2df.iloc[:, k].abs().sort_values(ascending=False).index)
        top = top[top.abs() > 0.30].head(8)
        print(f"  dim{k+1}: " + ", ".join(f"{c}({v:+.2f})" for c, v in top.items()))

    # ---------------- score patients on the dimensions (no imputation) ----------------
    F = masked_scores(Z.to_numpy(float), L2)
    Fdf = pd.DataFrame(F, index=Z.index, columns=L2df.columns)
    gscore = masked_scores(Z.to_numpy(float), g.reshape(-1, 1))[:, 0]
    Fdf["general"] = gscore
    print(f"\n  patient scores: {Fdf.shape}; dimension-score coverage "
          f"{[round(Fdf[c].notna().mean(),2) for c in L2df.columns]}")

    L2df.assign(general_SL=g).round(3).to_csv(OUT / "stage3_loadings_v2.csv")
    Fdf.reset_index().to_pickle(OUT / "stage3_scores_v2.pkl")
    pd.DataFrame(Phi2, index=L2df.columns, columns=L2df.columns).round(3).to_csv(OUT / "stage3_phi2_v2.csv")
    print(f"\nsaved -> {OUT}/stage3_loadings_v2.csv, stage3_scores_v2.pkl, stage3_phi2_v2.csv")


if __name__ == "__main__":
    main()
