"""Export the phenotype-atlas factor scores as a tidy patient × factor feature table.

Reads the Stage-2 construct scores (``results/hfa/stage2_scores.pkl``), scores the named phenotype
factors (``trans_diag.phenotype``; see ``docs/PHENOTYPE_ATLAS.md``) on observed support only, and
writes ``results/hfa/phenotype_features.csv`` with one ``<factor>`` score column and one
``<factor>__cov`` coverage column (fraction of member constructs observed) per factor.

These are the **predictive features** view: deliberately orthogonal, non-redundant directions. Gate
each feature with its ``__cov`` column (e.g. require ≥ 0.5) — internalizing is SZ-proxy-only,
illness-course is ~half-covered in DR, substance-use is BP/SZ only.

Run:  python3 scripts/export_phenotype_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from trans_diag.phenotype import AXES, FACTOR_META, STANDALONES, build_phenotype_factors

OUT = ROOT / "results" / "hfa"


def main() -> None:
    S = pd.read_pickle(OUT / "stage2_scores.pkl").set_index(["cohort", "patient_id"])
    scores, coverage = build_phenotype_factors(S)
    coh = np.asarray(scores.index.get_level_values("cohort"))

    # ── tidy wide table: <factor> score + <factor>__cov coverage ──
    out = scores.copy()
    out = out.join(coverage.add_suffix("__cov")).reset_index()
    out.to_csv(OUT / "phenotype_features.csv", index=False)

    # ── summary: per-cohort coverage (fraction of patients with usable ≥50% coverage) ──
    print(f"phenotype features: {scores.shape[1]} factors × {len(scores)} patients "
          f"-> {OUT / 'phenotype_features.csv'}\n")
    print(f"{'factor':20} {'kind':10} {'cov BP/SZ/DR (≥50% usable)':30} direction")
    for fac in list(AXES) + list(STANDALONES):
        if fac not in coverage:
            continue
        usable = (coverage[fac] >= 0.5).to_numpy()
        pc = {c: f"{usable[coh == c].mean():.2f}" for c in ("bp", "sz", "dr")}
        m = FACTOR_META[fac]
        print(f"{fac:20} {m['kind']:10} {pc['bp']+'/'+pc['sz']+'/'+pc['dr']:30} {m['direction']}")

    # ── sanity: the 3 axis features should track the pipeline's Stage-3 dims ──
    f3 = OUT / "stage3_scores.pkl"
    if f3.exists():
        D = pd.read_pickle(f3).set_index(["cohort", "patient_id"])
        print("\nsanity — atlas axis vs pipeline Stage-3 dim (|Pearson r|, should be high):")
        for fac, dim in [("internalizing", "dim1"), ("cognition", "dim2"), ("cardiometabolic", "dim3")]:
            a, b = scores[fac].align(D[dim], join="inner")
            m = a.notna() & b.notna()
            r = np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 10 else np.nan
            print(f"  {fac:16} ~ {dim}: |r| = {abs(r):.2f}  (n={int(m.sum())})")

    print("\ninter-feature correlation (orthogonality — off-diagonal should be ~0):")
    C = scores.corr(min_periods=100)
    off = C.to_numpy()[np.triu_indices(scores.shape[1], 1)]
    print(f"  mean |r| = {np.nanmean(np.abs(off)):.3f}  | max |r| = {np.nanmax(np.abs(off)):.3f}")


if __name__ == "__main__":
    main()
