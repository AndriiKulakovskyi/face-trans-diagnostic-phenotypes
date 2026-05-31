"""K-selection deep dive (v2) — how many second-order trans-diagnostic dimensions?

Stage 3 locked K=4 by "first-collapse-minus-1" on the MINIMUM split-half congruence. But the min
collapses if ONE factor is unstable, so it can UNDER-extract: it ignores that several OTHER factors
at higher K may be perfectly reproducible. This script reconciles multiple criteria and, crucially,
reports the PER-FACTOR congruence profile (how many of the K factors actually reproduce), not just
the min. It also tracks where the clinically-important constructs (mania, suicidality, metabolic vs
inflammatory) land as K grows — to see if K=4 is merging distinct dimensions.

Criteria: (1) Φ₁ eigenvalue scree; (2) masked Horn parallel analysis; (3) per-K: Heywood count +
variance explained + the descending per-factor split-half congruence vector -> #reproducible factors;
(4) factor genealogy K=4..7 (what splits / what is added).
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

from trans_diag.masked_fa import masked_correlation, paf_loadings, varimax

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
MIN_PAIR = 100
COVERAGE_FLOOR = 0.30
SEED = 0


def tucker(A, B):
    return np.abs(A.T @ B) / (np.sqrt(np.outer((A * A).sum(0), (B * B).sum(0))) + 1e-12)


def perfactor_congruence(Z, K, n_splits=25, seed=SEED):
    """Descending per-rank split-half Tucker congruence (varimax, Hungarian-matched), averaged."""
    rng = np.random.default_rng(seed)
    n = len(Z)
    profs = []
    for _ in range(n_splits):
        p = rng.permutation(n)
        a, b = p[: n // 2], p[n // 2:]
        La = varimax(paf_loadings(masked_correlation(Z.iloc[a], MIN_PAIR), K))
        Lb = varimax(paf_loadings(masked_correlation(Z.iloc[b], MIN_PAIR), K))
        M = tucker(La, Lb)
        r, c = linear_sum_assignment(-M)
        profs.append(np.sort(M[r, c])[::-1])
    return np.mean(profs, axis=0)


def masked_parallel(Z, n_iter=25, seed=SEED, q=95):
    real = np.sort(np.linalg.eigvalsh(masked_correlation(Z, MIN_PAIR)))[::-1]
    arr = Z.to_numpy(float)
    rng = np.random.default_rng(seed)
    null = np.empty((n_iter, arr.shape[1]))
    for it in range(n_iter):
        Zp = arr.copy()
        for j in range(arr.shape[1]):
            o = np.where(np.isfinite(Zp[:, j]))[0]
            Zp[o, j] = Zp[o[rng.permutation(o.size)], j]
        null[it] = np.sort(np.linalg.eigvalsh(masked_correlation(pd.DataFrame(Zp, columns=Z.columns), MIN_PAIR)))[::-1]
    return real, np.percentile(null, q, axis=0)


def top_constructs(L2df, k, n=4):
    s = L2df.iloc[:, k]
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    s = s[s.abs() > 0.30].head(n)
    return ",".join(f"{c}{'+' if v > 0 else '-'}" for c, v in s.items())


def main() -> None:
    S = pd.read_pickle(OUT / "stage2_scores_v2.pkl").set_index(["cohort", "patient_id"])
    cov = S.notna().mean()
    keep = [c for c in S.columns if cov[c] >= COVERAGE_FLOOR and S[c].var() > 1e-9]
    Z = (S[keep] - S[keep].mean()) / S[keep].std()
    Phi1 = masked_correlation(Z, MIN_PAIR)

    # (1) scree + (2) parallel analysis
    real, thr = masked_parallel(Z)
    k_pa = int((real > thr).sum())
    print(f"75 constructs | Φ₁ eigenvalues>1: {int((real>1).sum())}  top8: {np.round(real[:8],2)}")
    print(f"parallel analysis (Horn): K_PA = {k_pa}\n")

    # (3) per-K diagnostics
    track = ["mania_activation", "suicidal_ideation", "adiposity", "inflammation",
             "cardiac_history", "atopic_inflammatory"]
    print(f"{'K':>2s} {'Heywood':>7s} {'var%':>5s} {'#repro(cong>=.85)':>17s}  per-factor congruence (descending)")
    sols = {}
    for K in range(2, 10):
        A = paf_loadings(Phi1, K)
        rot = Rotator(method="promax")
        L2 = rot.fit_transform(A)
        L2df = pd.DataFrame(L2, index=keep, columns=[f"d{k+1}" for k in range(K)])
        sols[K] = L2df
        heywood = int((np.abs(L2) > 1.0).sum())
        var = float((L2 ** 2).sum() / len(keep))
        prof = perfactor_congruence(Z, K)
        nrepro = int((prof >= 0.85).sum())
        print(f"{K:2d} {heywood:7d} {var*100:4.0f}% {nrepro:17d}  [{' '.join(f'{x:.2f}' for x in prof)}]")

    # (4) genealogy: where do the tracked constructs load, K=4..7
    print("\n=== where key constructs load (|loading|, '-' if <0.30), K=4..7 ===")
    print(f"{'construct':22s} " + " ".join(f"K{K:<5d}" for K in (4, 5, 6, 7)))
    for c in track:
        row = []
        for K in (4, 5, 6, 7):
            if c in sols[K].index:
                v = sols[K].loc[c]
                best = v.abs().max()
                row.append(f"{best:.2f}@{v.abs().idxmax()}" if best > 0.30 else "  -   ")
            else:
                row.append("  -   ")
        print(f"{c:22s} " + " ".join(f"{r:11s}" for r in row))

    # (4b) factor compositions at K=4,5,6,7 (what splits / is added)
    for K in (4, 5, 6, 7):
        print(f"\n--- K={K} factors ---")
        for k in range(K):
            print(f"  d{k+1}: {top_constructs(sols[K], k)}")


if __name__ == "__main__":
    main()
