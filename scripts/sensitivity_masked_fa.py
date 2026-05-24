"""Sensitivity: are the six varimax axes an artifact of the factor-analysis mean-fill?

The locked dimensional model (`07_dimensional_refine.py`) is the one step that imputes:
it standardizes the 54 residual domain scores and mean-fills the ~35% missing cells with
0 before sklearn's `FactorAnalysis` (which needs a complete matrix). This script re-derives
the loadings WITHOUT any imputation, by factoring the **pairwise-complete (masked)
correlation matrix** of the same residual domains — each correlation is estimated only from
the patients who have *both* domains, so no cell is ever filled — and compares the resulting
varimax loadings to the locked ones by Tucker's congruence.

To separate the *imputation* effect from the *extraction-method* effect (principal-axis
factoring here vs sklearn maximum-likelihood FA in the locked model), we report a 2x2:

  locked        = sklearn ML-FA + varimax on the MEAN-FILLED matrix      (the published axes)
  PAF(meanfill) = principal-axis factoring + varimax on corr(MEAN-FILLED) (extraction control)
  PAF(masked)   = principal-axis factoring + varimax on the MASKED corr   (no imputation)

  cong(locked, PAF-masked)      -> headline: do the published axes survive imputation-free?
  cong(PAF-meanfill, PAF-masked)-> isolates the imputation effect (extraction held fixed)
  cong(locked, PAF-meanfill)    -> sanity: PAF ~ ML-FA (extraction effect alone)

Writes results/sensitivity_masked_fa.json. Reads cluster_domains_scores.parquet (per-patient,
gitignored) so it runs only where the data are present. No downstream artifact is modified.
Run:  python3 scripts/sensitivity_masked_fa.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_DIR = REPO_ROOT / "results"
SCORES_PATH = RESULTS_DIR / "cluster_domains_scores.parquet"
LOADINGS_PATH = RESULTS_DIR / "dimensional_final_loadings.csv"
MIN_PAIR = 100  # min co-observed patients to trust a pairwise correlation; else treat as 0


def nearest_pd(A: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive-definite correlation matrix."""
    A = (A + A.T) / 2.0
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 1e-8, None)
    P = (V * w) @ V.T
    d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
    P = P / np.outer(d, d)              # renormalize to unit diagonal (a correlation matrix)
    P = (P + P.T) / 2.0
    np.fill_diagonal(P, 1.0)
    return P


def paf_loadings(R: np.ndarray, k: int, n_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Principal-axis factoring of a correlation matrix R -> p x k unrotated loadings.

    Classical iterated communality estimation: replace the diagonal with communalities,
    take the top-k eigenpairs, recompute communalities, repeat. Initial communalities are
    the squared multiple correlations (SMC).
    """
    p = R.shape[0]
    try:
        h2 = np.clip(1.0 - 1.0 / np.clip(np.diag(np.linalg.pinv(R)), 1e-6, None), 0.0, 1.0)
    except np.linalg.LinAlgError:
        h2 = np.full(p, 0.5)
    L = np.zeros((p, k))
    prev = h2.copy()
    for _ in range(n_iter):
        Rr = R.copy()
        np.fill_diagonal(Rr, h2)
        w, V = np.linalg.eigh(Rr)                 # ascending
        idx = np.argsort(w)[::-1][:k]
        L = V[:, idx] * np.sqrt(np.clip(w[idx], 0.0, None))
        h2 = np.clip(np.sum(L ** 2, axis=1), 0.0, 1.0)
        if np.max(np.abs(h2 - prev)) < tol:
            break
        prev = h2.copy()
    return L


def varimax(L: np.ndarray, gamma: float = 1.0, n_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    """Kaiser varimax rotation of a p x k loading matrix (orthogonal, simple structure)."""
    p, k = L.shape
    Rot = np.eye(k)
    d = 0.0
    for _ in range(n_iter):
        Lam = L @ Rot
        G = L.T @ (Lam ** 3 - (gamma / p) * Lam @ np.diag(np.diag(Lam.T @ Lam)))
        u, s, vt = np.linalg.svd(G)
        Rot = u @ vt
        dn = float(np.sum(s))
        if d != 0.0 and dn / d < 1 + tol:
            break
        d = dn
    return L @ Rot


def match(La: np.ndarray, Lb: np.ndarray):
    """Greedily match each column of La to the best unused column of Lb by |Tucker congruence|.

    Returns list of (a_index, b_index, congruence, sign)."""
    used: set[int] = set()
    out = []
    for a in range(La.shape[1]):
        best, bj, bsign = -1.0, -1, 1
        for b in range(Lb.shape[1]):
            if b in used:
                continue
            den = np.linalg.norm(La[:, a]) * np.linalg.norm(Lb[:, b])
            dot = float(La[:, a] @ Lb[:, b])
            phi = abs(dot) / den if den > 0 else 0.0
            if phi > best:
                best, bj, bsign = phi, b, int(np.sign(dot)) or 1
        out.append((a, bj, best, bsign))
        used.add(bj)
    return out


def main() -> int:
    sc = pd.read_parquet(SCORES_PATH)
    domains = list(sc.columns)
    p = len(domains)

    # locked (published) loadings: sklearn ML-FA + varimax on the mean-filled matrix
    lock = pd.read_csv(LOADINGS_PATH)
    Llock_df = lock.pivot(index="domain", columns="axis", values="loading").reindex(domains)
    axis_cols = sorted(Llock_df.columns, key=lambda c: int(str(c).replace("axis", "")))
    Llock = Llock_df[axis_cols].to_numpy(float)
    K = len(axis_cols)

    # --- the matrix actually fed to factor analysis, and its observed fraction ---
    z = (sc - sc.mean()) / sc.std(ddof=0)
    obs_frac = float(sc.notna().to_numpy().mean())          # the "65% observed" figure
    Xf = z.fillna(0.0).to_numpy(np.float64)                 # mean-filled (the imputation)

    # --- pairwise co-observation diagnostics ---
    M = sc.notna().to_numpy(float)
    coobs = M.T @ M                                          # p x p co-observed patient counts
    off = ~np.eye(p, dtype=bool)
    thin = int(((coobs < MIN_PAIR) & off).sum() // 2)
    co_off = coobs[off]

    # --- correlation matrices ---
    Rf = np.corrcoef(Xf, rowvar=False)                      # corr of the mean-filled matrix
    Rf = nearest_pd(Rf)
    Rm = sc.corr(min_periods=MIN_PAIR).to_numpy(float)      # MASKED pairwise-complete corr
    n_nan = int(np.isnan(Rm[off]).sum() // 2)
    Rm[~np.isfinite(Rm)] = 0.0                              # thin/undefined pairs -> 0 (no fabrication of a value)
    np.fill_diagonal(Rm, 1.0)
    min_eig_masked = float(np.linalg.eigvalsh((Rm + Rm.T) / 2).min())   # how non-PD the raw masked corr is
    Rm = nearest_pd(Rm)

    # --- factor each, varimax-rotate ---
    L_paf_meanfill = varimax(paf_loadings(Rf, K))
    L_paf_masked = varimax(paf_loadings(Rm, K))

    # --- congruences ---
    head = match(Llock, L_paf_masked)            # headline: published vs imputation-free
    imp = match(L_paf_meanfill, L_paf_masked)    # isolates imputation (PAF held fixed)
    extr = match(Llock, L_paf_meanfill)          # isolates extraction (mean-fill held fixed)

    def summ(m):
        phis = [phi for (_, _, phi, _) in m]
        return {"per_axis": [round(x, 3) for x in phis],
                "min": round(float(np.min(phis)), 3), "mean": round(float(np.mean(phis)), 3)}

    axis_names = lock.sort_values("axis").drop_duplicates("axis")  # for readable labels
    print(f"matrix: {len(sc):,} patients x {p} domains | observed fraction = {obs_frac:.3f} "
          f"(mean-fills {1 - obs_frac:.0%})")
    print(f"pairwise co-observation: min={int(co_off.min())}, median={int(np.median(co_off))}, "
          f"max={int(co_off.max())}; {thin} of {p*(p-1)//2} pairs < {MIN_PAIR} co-obs "
          f"({n_nan} undefined -> 0); masked-corr min eigenvalue {min_eig_masked:+.3f} (pre-PD).")
    print()
    print(f"HEADLINE  cong(locked sklearn-FA[mean-fill], PAF[MASKED, no imputation]):")
    print(f"          per-axis {summ(head)['per_axis']}  min={summ(head)['min']} mean={summ(head)['mean']}")
    print(f"isolate imputation  cong(PAF[mean-fill], PAF[masked]): "
          f"min={summ(imp)['min']} mean={summ(imp)['mean']}  {summ(imp)['per_axis']}")
    print(f"isolate extraction  cong(locked, PAF[mean-fill]):      "
          f"min={summ(extr)['min']} mean={summ(extr)['mean']}  {summ(extr)['per_axis']}")
    print()
    verdict = ("PRESERVED — the published axes reproduce imputation-free"
               if summ(head)["min"] >= 0.85 else
               "PARTIAL — some axes shift; inspect per-axis congruence")
    print(f"VERDICT: {verdict} (Lorenzo-Seva & ten Berge: >=0.85 fair, >=0.95 equal).")

    # --- per-axis diagnostic: name each locked axis, its masked-congruence, and its sparsity ---
    obs_by_domain = sc.notna().mean()
    print("\nper locked axis  |  top domains  |  masked-congruence  |  mean obs-frac of top-5 domains")
    for (a, b, phi, sign) in head:
        s = pd.Series(Llock[:, a], index=domains).sort_values(key=abs, ascending=False)
        of = float(obs_by_domain[s.head(5).index].mean())
        flag = "  <-- DIVERGES" if phi < 0.85 else ""
        print(f"  {axis_cols[a]}: " + ", ".join(f"{d}({v:+.2f})" for d, v in s.head(3).items())
              + f"  | cong={phi:.3f} | obs={of:.2f}{flag}")
    # for the divergent axis, show what the no-imputation solution puts in its place
    aw, bw, phiw, _ = min(head, key=lambda t: t[2])
    sl = pd.Series(Llock[:, aw], index=domains).sort_values(key=abs, ascending=False).head(6)
    sm = pd.Series(L_paf_masked[:, bw], index=domains).sort_values(key=abs, ascending=False).head(6)
    print(f"\nDIVERGENT locked {axis_cols[aw]} (best masked congruence {phiw:.3f}):")
    print("  locked (mean-fill) top: " + ", ".join(f"{d}({v:+.2f})" for d, v in sl.items()))
    print("  masked (no-imput.) top: " + ", ".join(f"{d}({v:+.2f})" for d, v in sm.items()))

    # --- STEP 1: split-half reproducibility WITHOUT imputation (masked) vs WITH (mean-fill) ---
    # The imputation-free analogue of 07_dimensional_refine's K-selection: how many factors
    # reproduce across random halves when the covariance is estimated pairwise-complete?
    rng = np.random.default_rng(0)
    order = rng.permutation(len(sc))
    hlf = len(sc) // 2
    Ah, Bh = sc.iloc[order[:hlf]], sc.iloc[order[hlf:]]
    zA, zB = z.iloc[order[:hlf]], z.iloc[order[hlf:]]

    def masked_load(sub, k):
        Rk = sub.corr(min_periods=MIN_PAIR).to_numpy(float)
        Rk[~np.isfinite(Rk)] = 0.0
        np.fill_diagonal(Rk, 1.0)
        return varimax(paf_loadings(nearest_pd(Rk), k))

    def fill_load(zsub, k):
        return varimax(paf_loadings(nearest_pd(np.corrcoef(
            zsub.fillna(0.0).to_numpy(np.float64), rowvar=False)), k))

    print("\nSTEP 1 — split-half reproducibility (min/mean Tucker congruence): mean-fill vs imputation-free")
    print("   K    mean-fill (min/mean)    masked/no-imput. (min/mean)")
    curve = []
    for k in range(3, 11):
        mf = [phi for (_, _, phi, _) in match(fill_load(zA, k), fill_load(zB, k))]
        mk = [phi for (_, _, phi, _) in match(masked_load(Ah, k), masked_load(Bh, k))]
        curve.append({"k": k,
                      "meanfill_min": round(float(np.min(mf)), 3), "meanfill_mean": round(float(np.mean(mf)), 3),
                      "masked_min": round(float(np.min(mk)), 3), "masked_mean": round(float(np.mean(mk)), 3)})
        print(f"  {k:>2}      {np.min(mf):.2f} / {np.mean(mf):.2f}              {np.min(mk):.2f} / {np.mean(mk):.2f}")
    cdf = pd.DataFrame(curve)
    okm = cdf[(cdf["masked_min"] >= 0.85) & (cdf["k"] <= 8)]
    okf = cdf[(cdf["meanfill_min"] >= 0.85) & (cdf["k"] <= 8)]
    K_masked = int(okm["k"].max()) if len(okm) else 0
    K_fill = int(okf["k"].max()) if len(okf) else 0
    print(f"\n  reproducible K (largest K<=8, min-congruence>=0.85): mean-fill={K_fill}  |  imputation-free={K_masked}")

    out = {
        "n_patients": int(len(sc)), "n_domains": p, "K": K,
        "observed_fraction": round(obs_frac, 4), "mean_filled_fraction": round(1 - obs_frac, 4),
        "min_pair_threshold": MIN_PAIR,
        "pair_coobs_min": int(co_off.min()), "pair_coobs_median": int(np.median(co_off)),
        "pairs_below_threshold": thin, "pairs_undefined": n_nan,
        "masked_corr_min_eigenvalue_prePD": round(min_eig_masked, 4),
        "congruence_locked_vs_masked": summ(head),
        "congruence_meanfill_vs_masked_isolates_imputation": summ(imp),
        "congruence_locked_vs_meanfill_isolates_extraction": summ(extr),
        "verdict": verdict,
        "split_half_reproducibility": curve,
        "reproducible_K_meanfill": K_fill,
        "reproducible_K_imputation_free": K_masked,
    }
    (RESULTS_DIR / "sensitivity_masked_fa.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote results/sensitivity_masked_fa.json. Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
