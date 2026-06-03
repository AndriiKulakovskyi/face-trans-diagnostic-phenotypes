"""Sensitivity (v2) — reproduces the empirical evidence in docs/AGGREGATION_RATIONALE.md.

Four analyses, all on the V0 matrix, masked / no-imputation:
  A. conditioning of the masked correlation, item-level (177) vs domain-level (69);
  B. unidimensionality of each multi-item construct (does averaging drop a dimension?);
  C. granularity invariance — canonical correlation between item-FA and domain-FA scores;
  D. item-count / redundancy bias (share of item-axes per instrument).

These are the numbers cited in the rationale; this script makes them auditable. It uses the
ORIGINAL flat-domain pipeline (DOMAIN_SECTIONS -> 177 items / 69 domains) on purpose, since the
rationale is *why we move away from flat domains*. Run: python3 scripts/sensitivity_aggregation.py
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
from trans_diag.domains import (
    BIOLOGY_COMPOSITES,
    COGNITIVE_COMPOSITES,
    DOMAIN_SECTIONS,
    _robust_z,
    build_domain_scores,
    instrument_stem,
)
from trans_diag.masked_fa import masked_loadings, masked_scores, nearest_pd

warnings.simplefilter("ignore")
MIN_PAIR = 100
SEED = 0


def load_matrices():
    df = build_unified_dataframe(str(ROOT / "data"), str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL"], format="long")
    vs = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    ds = to_harmonized_dataset(df, vs, visit="V0", sections=list(DOMAIN_SECTIONS), normalize=False)
    Xi_raw = ds.X
    Xi = Xi_raw.apply(_robust_z)
    Xd, _ = build_domain_scores(Xi_raw, vs, biology=BIOLOGY_COMPOSITES, cognition=COGNITIVE_COMPOSITES)
    return Xi_raw, Xi, Xd, {v.canonical_name: v for v in vs}, ds.metadata["cohort"]


def A_conditioning(Xi, Xd):
    print("=== A. conditioning: item (177) vs domain (69) masked correlation ===")
    for lab, M in [("item", Xi), ("domain", Xd)]:
        O = M.notna().astype(int).to_numpy()
        N = O.T @ O
        off = N[np.triu_indices(N.shape[0], 1)]
        R = M.corr(min_periods=MIN_PAIR).to_numpy(float)
        zeroed = (~np.isfinite(R)).sum() / 2 / len(off)
        R0 = np.nan_to_num(R, nan=0.0)
        np.fill_diagonal(R0, 1.0)
        w = np.linalg.eigvalsh(R0)
        neg = np.abs(w[w < 0]).sum() / np.abs(w).sum()
        cond = np.linalg.cond(nearest_pd(R0))
        print(f"  {lab:6s}: median pairN={int(np.median(off))}, pairs<{MIN_PAIR}={zeroed*100:.1f}%, "
              f"neg-eig mass={neg*100:.1f}%, cond={cond:.1e}")


def B_unidimensionality(Xi, by):
    print("\n=== B. unidimensionality per multi-item construct ===")
    bio = {c: d for d, m in BIOLOGY_COMPOSITES.items() for c, _ in m}
    cog = {c: d for d, m in COGNITIVE_COMPOSITES.items() for c, _ in m}
    groups: dict[str, list[str]] = {}
    for c in Xi.columns:
        k = f"bio:{bio[c]}" if c in bio else (f"cog:{cog[c]}" if c in cog else instrument_stem(c))
        groups.setdefault(k, []).append(c)
    rng = np.random.default_rng(SEED)
    print(f"  {'construct':22s} {'p':>2s} {'n':>5s} {'VAF1':>5s} {'PA_k':>4s} {'r(mean,PC1)':>11s}")
    for k in sorted(g for g in groups if len(groups[g]) >= 2):
        cc = Xi[groups[k]].apply(_robust_z).dropna()
        if len(cc) < 40:
            print(f"  {k:22s} {len(groups[k]):2d} {len(cc):5d}   (too few complete cases)")
            continue
        Zk = cc.values[:, cc.values.std(0) > 1e-9]
        C = np.corrcoef(Zk, rowvar=False)
        C = np.nan_to_num((C + C.T) / 2, nan=0.0)
        np.fill_diagonal(C, 1.0)
        w, V = np.linalg.eigh(C)
        o = np.argsort(w)[::-1]
        w, V = w[o], V[:, o]
        vaf1 = w[0] / w.sum()
        null = [np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal(Zk.shape), rowvar=False)))[::-1]
                for _ in range(40)]
        pak = int((w > np.percentile(null, 95, axis=0)).sum())
        r = abs(np.corrcoef(Zk @ V[:, 0], Zk.mean(1))[0, 1])
        print(f"  {k:22s} {Zk.shape[1]:2d} {len(cc):5d} {vaf1*100:4.0f}% {pak:4d} {r:11.2f}")


def _cca(A, B):
    ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    A, B = A[ok], B[ok]
    A = (A - A.mean(0)) / A.std(0)
    B = (B - B.mean(0)) / B.std(0)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)


def C_convergence(Xi, Xd):
    print("\n=== C. granularity invariance: canonical corr (item-FA vs domain-FA scores) ===")
    Zi = (Xi - Xi.mean()) / Xi.std()
    Zd = (Xd - Xd.mean()) / Xd.std()
    rng = np.random.default_rng(SEED)
    for K in (5, 6, 7):
        Fi = masked_scores(Zi.values, masked_loadings(Xi, K, MIN_PAIR))
        Fd = masked_scores(Zd.values, masked_loadings(Xd, K, MIN_PAIR))
        cc = _cca(Fi, Fd)
        null = _cca(Fi, Fd[rng.permutation(len(Fd))]).max()
        print(f"  K={K}: {np.round(cc, 2)}  (perm-null max={null:.2f})")


def D_itemcount(Xi):
    print("\n=== D. item-count / redundancy bias (share of item-axes per instrument) ===")
    bio = {c: d for d, m in BIOLOGY_COMPOSITES.items() for c, _ in m}
    cog = {c: d for d, m in COGNITIVE_COMPOSITES.items() for c, _ in m}
    cnt: Counter = Counter()
    for c in Xi.columns:
        cnt[f"bio:{bio[c]}" if c in bio else (f"cog:{cog[c]}" if c in cog else instrument_stem(c))] += 1
    tot = Xi.shape[1]
    for k, n in cnt.most_common(6):
        print(f"  {k:22s} {n:3d} items = {n/tot*100:4.1f}%")
    sui = sum(n for k, n in cnt.items() if k in ("cssrs", "isf", "ltsv", "ltsg"))
    print(f"  suicide block (cssrs+isf+ltsv+ltsg): {sui} = {sui/tot*100:.1f}% of item-axes")


def main():
    Xi_raw, Xi, Xd, by, coh = load_matrices()
    print(f"item matrix {Xi.shape} | domain matrix {Xd.shape} | "
          f"cohorts { {k: int(v) for k, v in coh.value_counts().items()} }\n")
    A_conditioning(Xi, Xd)
    B_unidimensionality(Xi, by)
    C_convergence(Xi, Xd)
    D_itemcount(Xi)


if __name__ == "__main__":
    main()
