"""Sensitivity (v2) — data-anchored decomposition of the medical-comorbidity flags.

Audits the Stage 2 decision (LABBOOK V2-8): the 24 `*_mhoccur` flags do NOT form one construct
(pooled VAF1 ~0.06-0.38). This script shows, step by step:
  1. prevalence per flag per cohort -> only ~8 flags are common enough to subgroup (13 are <2%);
  2. cohort-cleaned (within-BP) association + hierarchical clustering -> 2 weak but interpretable
     co-occurrence clusters (cardiac, atopic/inflammatory) + standalones;
  3. validation: VAF1, any-positive prevalence, cross-cohort + bootstrap stability.
Conclusion encoded in scripts/32_hfa_stage2_v2.py: cardiac_history + atopic_inflammatory +
standalone {migraine, head_trauma, peptic_ulcer, other_neuro}; the 12 flags <2% -> Stage-4 validators.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset

warnings.simplefilter("ignore")


def vaf1(M: pd.DataFrame) -> float:
    Mc = M.dropna()
    Mc = Mc.loc[:, Mc.std() > 0]
    if Mc.shape[1] < 2 or len(Mc) < 30:
        return np.nan
    w = np.linalg.eigvalsh(np.corrcoef(Mc.values, rowvar=False))[::-1]
    return float(w[0] / w.sum())


def main() -> None:
    df = build_unified_dataframe(str(ROOT / "data"), str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL", "NOT USABLE", "ID"], format="long")
    vs = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    ds = to_harmonized_dataset(df, vs, visit="V0", sections=None, residualize_on=None, normalize=False)
    X, coh = ds.X, ds.metadata["cohort"]
    flags = sorted(c for c in X.columns if c.endswith("_mhoccur"))
    F = X[flags].apply(lambda s: pd.to_numeric(s, errors="coerce"))

    # 1. prevalence
    prev = (F == 1).mean() * 100
    print("=== 1. prevalence (% positive) ===")
    print(f"  >=5% (subgroup-able): {sorted(c.replace('_mhoccur','') for c in flags if prev[c] >= 5)}")
    print(f"  2-5%: {sorted(c.replace('_mhoccur','') for c in flags if 2 <= prev[c] < 5)}")
    print(f"  <2% (un-clusterable noise -> validators): {sorted(c.replace('_mhoccur','') for c in flags if prev[c] < 2)}")

    # 2. cohort-cleaned association + clustering on the >=2% flags
    prevalent = [c for c in flags if prev[c] >= 2]
    B = F.loc[(coh == "bp").values, prevalent]
    R = B.corr().fillna(0).to_numpy()
    D = 1 - R
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    Zl = linkage(squareform(D, checks=False), method="average")
    print("\n=== 2. within-BP clustering of the >=2% flags ===")
    for k in (3, 4):
        cl = fcluster(Zl, k, criterion="maxclust")
        groups: dict[int, list[str]] = {}
        for c, lab in zip(prevalent, cl, strict=False):
            groups.setdefault(lab, []).append(c.replace("_mhoccur", ""))
        print(f"  k={k}: " + " | ".join("{" + ",".join(g) + "}" for g in groups.values()))

    # 3. validate the encoded subgroups
    GROUPS = {
        "cardiac_history": ["hta_mhoccur", "autcardv_mhoccur", "trbrycard_mhoccur"],
        "atopic_inflammatory": ["acne_mhoccur", "eczema_mhoccur", "cheveux_mhoccur",
                                 "toxidermi_mhoccur", "psoriasis_mhoccur"],
    }
    rng = np.random.default_rng(0)
    print("\n=== 3. validation of encoded subgroups ===")
    for name, items in GROUPS.items():
        sub = F[items]
        anypos = (sub == 1).any(axis=1).mean() * 100
        vs_ = {c: vaf1(sub[(coh == c).values]) for c in ("bp", "sz", "dr")}
        bp = sub[(coh == "bp").values].dropna()
        bp = bp.loc[:, bp.std() > 0]
        boots = [np.linalg.eigvalsh(np.corrcoef(bp.values[rng.integers(0, len(bp), len(bp))], rowvar=False))[::-1][0]
                 / sub.shape[1] for _ in range(200)]
        print(f"  {name:22s} any+={anypos:4.1f}%  VAF1 BP/SZ/DR={vs_['bp']:.2f}/{vs_['sz']:.2f}/{vs_['dr']:.2f}  "
              f"bootstrap CI [{np.percentile(boots,2.5):.2f}-{np.percentile(boots,97.5):.2f}]")
    print(f"  reference: pooled 24-flag bin VAF1={vaf1(F):.2f} ; clean construct (adiposity) "
          f"VAF1={vaf1(X[['bmi','wstcir','weight']].apply(lambda s: pd.to_numeric(s, errors='coerce'))):.2f}")


if __name__ == "__main__":
    main()
