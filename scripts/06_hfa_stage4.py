"""Stage 4 (v2) — validation of the trans-diagnostic dimensions (K=3 primary, K=6 sensitivity).

Implements Stage 4 of docs/legacy_v2/planning/HIERARCHICAL_FA_PLAN.md. Checks the recovered axes are real phenotype,
not artifacts:
  1. CONFOUND BATTERY — variance in each dim explained by cohort / DSM-arm / sex / age / education /
     site / per-patient missingness (eta^2 categorical, R^2 continuous). An axis that is mostly a
     confound is flagged.
  2. TRANS-DIAGNOSTIC test — do the dims cut ACROSS cohorts/DSM (low-moderate eta^2) or just mark a
     cohort? Per-cohort means reported.
  3. LEAVE-COHORT-OUT reproducibility — re-extract dropping BP, then DR; Tucker congruence vs full.
  4. MISSING-MANIA investigation — where does mania_activation's variance go?
  5. GRANULARITY INVARIANCE — canonical corr of the hierarchical dims vs a flat-domain masked FA
     (anti-circularity: is the structure an artifact of construct aggregation?).
  6. K=6 SENSITIVITY — confound + DSM for the 5 reproducible dims.

Masked / no-imputation. Scores are standardized; eta^2/R^2 on observed support only.
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

from trans_diag import AXIS_INDEX_TO_NAME, build_unified_dataframe, load_variables, to_harmonized_dataset
from trans_diag.domains import (
    BIOLOGY_COMPOSITES,
    COGNITIVE_COMPOSITES,
    DOMAIN_SECTIONS,
    build_domain_scores,
)
from trans_diag.masked_fa import masked_correlation, masked_loadings, masked_scores, paf_loadings

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
MIN_PAIR = 100
COVERAGE_FLOOR = 0.30


def eta2(y: np.ndarray, g: np.ndarray) -> float:
    m = np.isfinite(y) & pd.notna(g)
    y, g = y[m], np.asarray(g)[m]
    if len(y) < 10 or len(set(g)) < 2:
        return np.nan
    gm = y.mean()
    ssb = sum(len(y[g == lv]) * (y[g == lv].mean() - gm) ** 2 for lv in set(g))
    sst = ((y - gm) ** 2).sum()
    return float(ssb / sst) if sst > 0 else np.nan


def r2(y: np.ndarray, x: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 10 or np.std(x[m]) == 0:
        return np.nan
    return float(np.corrcoef(y[m], x[m])[0, 1] ** 2)


def extract(S: pd.DataFrame, K: int) -> pd.DataFrame:
    cov = S.notna().mean()
    keep = [c for c in S.columns if cov[c] >= COVERAGE_FLOOR and S[c].var() > 1e-9]
    Z = (S[keep] - S[keep].mean()) / S[keep].std()
    L = Rotator(method="promax").fit_transform(paf_loadings(masked_correlation(Z, MIN_PAIR), K))
    for k in range(K):
        if L[np.argmax(np.abs(L[:, k])), k] < 0:
            L[:, k] *= -1
    return pd.DataFrame(L, index=keep, columns=[f"d{k+1}" for k in range(K)])


def congruence(LA: pd.DataFrame, LB: pd.DataFrame):
    common = [c for c in LA.index if c in LB.index]
    A, B = LA.loc[common].to_numpy(), LB.loc[common].to_numpy()
    M = np.abs(A.T @ B) / (np.sqrt(np.outer((A * A).sum(0), (B * B).sum(0))) + 1e-12)
    r, c = linear_sum_assignment(-M)
    return M[r, c]


def cca(A, B):
    ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    A, B = A[ok], B[ok]
    A = (A - A.mean(0)) / A.std(0)
    B = (B - B.mean(0)) / B.std(0)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)


def main() -> None:
    F = pd.read_pickle(OUT / "stage3_scores.pkl").set_index(["cohort", "patient_id"])
    S = pd.read_pickle(OUT / "stage2_scores.pkl").set_index(["cohort", "patient_id"])
    Lfull = pd.read_csv(OUT / "stage3_loadings.csv", index_col=0)
    DIMS = [c for c in Lfull.columns if c.startswith("dim")]   # K locked by Stage 3 (data-driven)
    Lfull = Lfull[DIMS]
    K = len(DIMS)
    NAMES = {d: AXIS_INDEX_TO_NAME.get(d, d) for d in DIMS}

    # reload covariates / labels at V0, aligned to F's patients
    df = build_unified_dataframe(str(ROOT / "data"), str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL", "NOT USABLE", "ID"], format="long")
    vs = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    ds = to_harmonized_dataset(df, vs, visit="V0", sections=None, residualize_on=None, normalize=False)
    cov = ds.X.reindex(F.index)
    conf = pd.DataFrame(index=F.index)
    conf["cohort"] = F.index.get_level_values("cohort")
    conf["dsm_arm"] = ds.metadata.reindex(F.index)["dsm_diagnosis"]
    conf["sex"] = pd.to_numeric(cov.get("sex"), errors="coerce")
    conf["age"] = pd.to_numeric(cov.get("age"), errors="coerce")
    conf["education"] = pd.to_numeric(cov.get("education_years"), errors="coerce")
    conf["site"] = pd.to_numeric(cov.get("siteid_city"), errors="coerce")
    conf["missingness"] = S.reindex(F.index).isna().mean(axis=1).values

    # 1+2. confound battery + trans-diagnostic
    print("=== 1-2. CONFOUND BATTERY  (eta^2 categorical / R^2 continuous; flag > 0.25) ===")
    print(f"{'dim':18s} {'cohort':>7s} {'DSMarm':>7s} {'sex':>6s} {'age':>6s} {'educ':>6s} {'site':>6s} {'missg':>6s}")
    for d in DIMS:
        y = F[d].to_numpy()
        vals = {"cohort": eta2(y, conf.cohort), "dsm_arm": eta2(y, conf.dsm_arm), "sex": eta2(y, conf.sex.values),
                "age": r2(y, conf.age.values), "educ": r2(y, conf.education.values),
                "site": eta2(y, conf.site.values), "missg": r2(y, conf.missingness.values)}
        flag = " <- ".join([""] + [k for k, v in vals.items() if pd.notna(v) and v > 0.25])
        print(f"{NAMES[d]:18s} " + " ".join(f"{vals[k]:6.2f}" for k in
              ["cohort", "dsm_arm", "sex", "age", "educ", "site", "missg"]) + flag)
    print("  per-cohort means (trans-diagnostic = dim varies WITHIN, not just BETWEEN cohorts):")
    print("   ", F.groupby(F.index.get_level_values("cohort"))[DIMS].mean().round(2).to_dict("index"))

    # 3. leave-cohort-out reproducibility
    print(f"\n=== 3. LEAVE-COHORT-OUT reproducibility (Tucker congruence vs full K={K}) ===")
    full = Lfull.rename(columns={f"dim{k+1}": f"d{k+1}" for k in range(K)})
    for drop in ("bp", "dr", "sz"):
        sub = S[S.index.get_level_values("cohort") != drop]
        cong = congruence(full, extract(sub, K))
        print(f"  drop {drop.upper()} (n={len(sub)}): per-dim congruence {np.round(np.sort(cong)[::-1],2)} "
              f"(min {cong.min():.2f})")

    # 4. missing-mania investigation
    print("\n=== 4. MISSING-MANIA investigation ===")
    def _corr(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan
    if "mania_activation" in S.columns:
        mania = S["mania_activation"].reindex(F.index)
        cors = {d: _corr(mania.values, F[d].values) for d in DIMS}
        print(f"  mania_activation corr with dims: {{{', '.join(f'{NAMES[d]}:{cors[d]:+.2f}' for d in DIMS)}}}")
        print(f"  mania eta^2 ~ cohort = {eta2(mania.to_numpy(), conf.cohort):.2f} "
              f"(high -> mania is a cohort marker, can't be a trans-diag axis); "
              f"coverage {mania.notna().mean():.2f}")
        print(f"  mania per-cohort mean: "
              f"{mania.groupby(F.index.get_level_values('cohort')).mean().round(2).to_dict()}")

    # 5. granularity invariance vs flat-domain masked FA
    print(f"\n=== 5. GRANULARITY INVARIANCE — hierarchical K={K} vs flat-domain masked FA ===")
    dsf = to_harmonized_dataset(df, vs, visit="V0", sections=list(DOMAIN_SECTIONS), normalize=False)
    Xd, _ = build_domain_scores(dsf.X, vs, biology=BIOLOGY_COMPOSITES, cognition=COGNITIVE_COMPOSITES)
    Zd = (Xd - Xd.mean()) / Xd.std()
    Fd = masked_scores(Zd.to_numpy(float), masked_loadings(Xd, K, MIN_PAIR))
    Fd = pd.DataFrame(Fd, index=Xd.index).reindex(F.index).to_numpy()
    ccs = cca(F[DIMS].to_numpy(), Fd)
    print(f"  canonical correlations (hier-K{K} vs flat-domain-K{K}): {np.round(ccs,2)}  "
          f"(>=0.8 on the top dims => structure not an aggregation artifact)")

    # 6. K=6 sensitivity
    print("\n=== 6. K=6 SENSITIVITY (5 reproducible dims) — confound + DSM ===")
    L6 = extract(S, 6)
    F6 = pd.DataFrame(masked_scores(((S[L6.index] - S[L6.index].mean()) / S[L6.index].std()).to_numpy(float),
                                    L6.to_numpy()), index=S.index, columns=L6.columns).reindex(F.index)
    for k in range(6):
        d = f"d{k+1}"
        top = L6[d].reindex(L6[d].abs().sort_values(ascending=False).index)
        top = ",".join(top[top.abs() > 0.3].head(3).index)
        y = F6[d].to_numpy()
        print(f"  {d}: eta2(cohort)={eta2(y, conf.cohort):.2f} eta2(DSM)={eta2(y, conf.dsm_arm):.2f}  [{top}]")

    print(f"\nVALIDATION SUMMARY: see flags above. axes.py to be locked only if the K={K} dims are "
          "confound-clean, leave-cohort-out reproducible, and granularity-invariant.")


if __name__ == "__main__":
    main()
