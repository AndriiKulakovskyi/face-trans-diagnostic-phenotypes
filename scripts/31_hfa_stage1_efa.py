"""Stage 1 (v2) — data-driven first-order EFA on the masked correlation.

Implements Stage 1 of docs/HIERARCHICAL_FA_PLAN.md. Purpose: SEE the empirical construct
structure before committing the hybrid anchors — where the clinical grouping is confirmed,
where it splits/merges, which items are orphans. Masked / no-imputation throughout.

Method: masked Horn parallel analysis (column-permutation, respects the missingness pattern)
-> number of first-order factors K1; principal-axis factoring (paf_loadings) + varimax
(orthogonal, tested); item -> dominant factor; agreement with the clinical constructs; factor
inter-correlations from masked posterior scores; leave-BP-out robustness (Tucker congruence).

Oblique rotation + Phi_1 proper are deferred to Stage 2/3 (the exploratory item->factor mapping
is robust to the orthogonal-vs-oblique choice).
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from trans_diag.domains import BIOLOGY_COMPOSITES, COGNITIVE_COMPOSITES, instrument_stem
from trans_diag.masked_fa import masked_correlation, masked_scores, paf_loadings, varimax
from trans_diag.variable import load_variables

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
MIN_PAIR = 100
SEED = 0


def construct_of(item: str, by: dict) -> str:
    """Clinical construct label for an item (biology composite / cognition / instrument stem)."""
    for dom, mem in BIOLOGY_COMPOSITES.items():
        if item in [c for c, _ in mem]:
            return f"bio:{dom}"
    for dom, mem in COGNITIVE_COMPOSITES.items():
        if item in [c for c, _ in mem]:
            return f"cog:{dom}"
    sec = by[item].section
    if sec in ("BILAN BIOLOGIQUE", "CONSTANTES ET ECG"):
        return f"bio:{item}"          # uncomposited lab/vital -> its own construct
    if sec == "NEUROPSYCHOLOGIE":
        return f"cog:{instrument_stem(item)}"
    return instrument_stem(item)


def masked_parallel_analysis(Z: pd.DataFrame, k_max: int, n_iter: int = 20, q: float = 95.0):
    """Horn parallel analysis on the masked correlation, preserving the missingness pattern.

    Each null replicate independently permutes every column's OBSERVED values (NaN positions
    fixed), so cross-column correlation is destroyed while marginals + missingness are kept.
    K1 = number of real eigenvalues exceeding the q-th percentile of the null eigenvalues.
    """
    real = np.sort(np.linalg.eigvalsh(masked_correlation(Z, MIN_PAIR)))[::-1]
    arr = Z.to_numpy(float)
    rng = np.random.default_rng(SEED)
    null = np.empty((n_iter, arr.shape[1]))
    for it in range(n_iter):
        Zp = arr.copy()
        for j in range(arr.shape[1]):
            obs = np.where(np.isfinite(Zp[:, j]))[0]
            Zp[obs, j] = Zp[obs[rng.permutation(obs.size)], j]
        Rp = masked_correlation(pd.DataFrame(Zp, columns=Z.columns), MIN_PAIR)
        null[it] = np.sort(np.linalg.eigvalsh(Rp))[::-1]
    thresh = np.percentile(null, q, axis=0)
    k1 = int(np.sum(real > thresh))
    return min(k1, k_max), real, thresh


def topk_congruence(A: np.ndarray, B: np.ndarray, k: int):
    """For the top-k factors of A (by sum-of-squares), the best |Tucker congruence| with ANY
    factor of B. Reports (mean, min) over those k. Not forcing a 1-1 match avoids the greedy-
    matcher collapse that makes a full-K min meaningless when most factors are tail noise."""
    M = np.abs(A.T @ B) / (np.sqrt(np.outer((A * A).sum(0), (B * B).sum(0))) + 1e-12)
    top = np.argsort(-(A ** 2).sum(0))[:k]
    best = M[top].max(1)
    return float(best.mean()), float(best.min())


def main() -> None:
    Z = pd.read_pickle(OUT / "stage0_Z_resid.pkl")
    meta = pd.read_pickle(OUT / "stage0_meta.pkl")
    by = {v.canonical_name: v for v in load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))}
    cols = list(Z.columns)

    # 1) number of first-order factors
    k1, real_ev, thresh = masked_parallel_analysis(Z, k_max=45)
    print(f"masked parallel analysis: K1 = {k1} first-order factors "
          f"(eigs>1 = {int((real_ev > 1).sum())}; PA keeps eigenvalue > null 95th pct)")

    # 2) PAF + varimax at K1
    R = masked_correlation(Z, MIN_PAIR)
    L = varimax(paf_loadings(R, k1))
    Ld = pd.DataFrame(L, index=cols)
    dom = np.abs(L).argmax(1)                       # each item's dominant factor
    constructs = {c: construct_of(c, by) for c in cols}

    # order factors by sum-of-squares
    order = np.argsort(-(L ** 2).sum(0))
    print("\n=== first-order factors (top loading items + clinical construct) ===")
    fac_label = {}
    for rank, f in enumerate(order, 1):
        idx = np.argsort(-np.abs(L[:, f]))[:7]
        top = [(cols[i], L[i, f], constructs[cols[i]]) for i in idx if abs(L[i, f]) > 0.30]
        if not top:
            continue
        cons = Counter(c for _, _, c in top)
        name = cons.most_common(1)[0][0]
        fac_label[f] = name
        members = ", ".join(f"{n}({l:+.2f})" for n, l, _ in top)
        tag = "PURE" if len(cons) == 1 else f"MIX({len(cons)})"
        print(f"  F{rank:02d} [{name:22s} {tag:7s}] {members}")

    # 3) clinical-construct purity: do a construct's items land on ONE factor?
    print("\n=== clinical constructs: confirmed vs split (multi-item constructs only) ===")
    rows = []
    by_con: dict[str, list[int]] = {}
    for i, c in enumerate(cols):
        by_con.setdefault(constructs[c], []).append(i)
    for con, idxs in sorted(by_con.items()):
        if len(idxs) < 2:
            continue
        facs = [dom[i] for i in idxs]
        modal, cnt = Counter(facs).most_common(1)[0]
        purity = cnt / len(idxs)
        verdict = "confirmed" if purity >= 0.8 else ("split" if purity < 0.5 else "partial")
        rows.append((con, len(idxs), round(purity, 2), len(set(facs)), verdict))
    pur = pd.DataFrame(rows, columns=["construct", "n_items", "purity", "n_factors", "verdict"])
    print(pur.sort_values(["verdict", "n_items"], ascending=[True, False]).to_string(index=False))

    # 4) factor inter-correlations from masked posterior scores: how many factor PAIRS correlate?
    #    (most of 42 construct-factors are genuinely independent, e.g. a lab vs a sleep factor; the
    #     second-order layer is formally warranted/tested by ECV/omega-H in Stage 3, not here.)
    F = masked_scores(Z.to_numpy(float), L)
    Phi = pd.DataFrame(F).corr(min_periods=MIN_PAIR).to_numpy()
    off = np.abs(Phi[np.triu_indices(k1, 1)])
    print(f"\nfactor inter-correlations: {int((off > 0.3).sum())} of {len(off)} pairs |r|>0.3 "
          f"(max={np.nanmax(off):.2f}) -> correlated symptom blocks exist; second-order tested in Stage 3")

    # 5) leave-BP-out robustness of the SUBSTANTIVE (top-12) factors (preliminary; full split-half
    #    stability + K-lock is Stage 3). Best congruence with any non-BP factor, not forced 1-1.
    nonbp = (meta["cohort"] != "bp").values
    L_nonbp = varimax(paf_loadings(masked_correlation(Z[nonbp], MIN_PAIR), k1))
    cmean, cmin = topk_congruence(L, L_nonbp, k=12)
    print(f"leave-BP-out congruence of top-12 factors: mean={cmean:.2f}, min={cmin:.2f}  "
          f"(>=0.85 mean = substantive structure reproduces in SZ+DR)")

    # save
    Ld.columns = [f"F{i+1}" for i in range(k1)]
    Ld.insert(0, "construct", [constructs[c] for c in cols])
    Ld.insert(1, "dom_factor", [f"F{d+1}" for d in dom])
    Ld.round(3).to_csv(OUT / "stage1_loadings.csv")
    pur.to_csv(OUT / "stage1_construct_purity.csv", index=False)
    print(f"\nsaved -> {OUT}/stage1_loadings.csv, stage1_construct_purity.csv")


if __name__ == "__main__":
    main()
