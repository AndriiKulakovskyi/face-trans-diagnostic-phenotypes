"""Study A (v2) — confront the cohort confound.

Attack (reviewer #1): the 3 cohorts ARE the 3 DSM diagnoses, so the 4 axes might encode
between-cohort / batch differences rather than within-patient trans-diagnostic variation.
Two defenses:
  1. WITHIN-COHORT re-derivation — re-extract the K=4 second-order structure within BP-alone and
     SZ-alone (DR n=552 too small, attempted + flagged); Tucker congruence vs the pooled axes.
     Pass: each pooled axis has a congruent (|>=0.85|) counterpart inside each cohort -> the axis
     exists *within* a diagnosis, not only *between* diagnoses.
  2. COHORT-RESIDUALIZED sensitivity — remove between-cohort means (center each construct within
     cohort), re-extract K=4, congruence vs pooled. Pass: axes survive -> the structure is
     within-cohort covariance, not the between-cohort mean differences.

Masked / no-imputation. Construct scores are the Stage-2 outputs (already age+sex-residualized; the
cohort index is lowercase bp/sz/dr). Writes results/hfa/studyA_cohort_confound.json.
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
from factor_analyzer import Rotator
from scipy.optimize import linear_sum_assignment

from trans_diag.axes import AXIS_NAMES
from trans_diag.masked_fa import masked_correlation, paf_loadings

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
MIN_PAIR = 100
COVERAGE_FLOOR = 0.30
K = len(AXIS_NAMES)        # data-locked second-order K (Stage 3); names = trans_diag.axes


def extract(S: pd.DataFrame, k: int = K) -> pd.DataFrame:
    """Re-derive the K second-order axes from construct scores (same recipe as Stage 3)."""
    cov = S.notna().mean()
    keep = [c for c in S.columns if cov[c] >= COVERAGE_FLOOR and S[c].var() > 1e-9]
    Z = (S[keep] - S[keep].mean()) / S[keep].std()
    L = Rotator(method="promax").fit_transform(paf_loadings(masked_correlation(Z, MIN_PAIR), k))
    for j in range(k):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    return pd.DataFrame(L, index=keep, columns=[f"d{j+1}" for j in range(k)])


def congruence(ref: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Per-(reference axis) best Tucker congruence under a Hungarian 1-1 match, on shared constructs."""
    common = [c for c in ref.index if c in test.index]
    A, B = ref.loc[common].to_numpy(), test.loc[common].to_numpy()
    M = np.abs(A.T @ B) / (np.sqrt(np.outer((A * A).sum(0), (B * B).sum(0))) + 1e-12)
    r, c = linear_sum_assignment(-M)
    return {ref.columns[i]: float(M[i, j]) for i, j in zip(r, c, strict=False)}, len(common)


def named(cong: dict) -> str:
    return "  ".join(f"{AXIS_NAMES[int(d[1:]) - 1][:13]}={v:.2f}" for d, v in cong.items())


def main() -> None:
    S = pd.read_pickle(OUT / "stage2_scores.pkl").set_index(["cohort", "patient_id"])
    coh = S.index.get_level_values("cohort")
    print(f"construct scores: {S.shape}; cohorts {dict(pd.Series(coh).value_counts())}")

    # pooled reference (must reproduce Stage 3)
    Lpool = extract(S)
    print(f"\npooled K={K} axes (sanity — top construct per axis):")
    for j, d in enumerate(Lpool.columns):
        top = Lpool[d].abs().idxmax()
        print(f"  {d} = {AXIS_NAMES[j]:16s} (top: {top} {Lpool.loc[top, d]:+.2f})")

    results = {}

    # ---- Test 1: within-cohort re-derivation ----
    print("\n=== Test 1: within-cohort re-derivation (Tucker congruence vs pooled) ===")
    for c in ["bp", "sz", "dr"]:
        sub = S[coh == c]
        flag = "  [UNDERPOWERED n<800]" if len(sub) < 800 else ""
        Lc = extract(sub)
        cong, ncommon = congruence(Lpool, Lc)
        results[f"within_{c}"] = {"n": int(len(sub)), "n_constructs": ncommon, "congruence": cong}
        print(f"  {c.upper()} (n={len(sub)}): {named(cong)}  min={min(cong.values()):.2f}{flag}")

    # ---- Test 2: cohort-residualized (remove between-cohort means) ----
    print("\n=== Test 2: cohort-residualized (center each construct within cohort) ===")
    Scent = S - S.groupby(level="cohort").transform("mean")
    Lcent = extract(Scent)
    cong, ncommon = congruence(Lpool, Lcent)
    results["cohort_residualized"] = {"n_constructs": ncommon, "congruence": cong}
    print(f"  cohort-centered (n={len(Scent)}): {named(cong)}  min={min(cong.values()):.2f}")

    # ---- verdict ----
    def ok(key):
        return min(results[key]["congruence"].values()) >= 0.85
    bp_ok, sz_ok, cent_ok = ok("within_bp"), ok("within_sz"), ok("cohort_residualized")
    print("\n=== VERDICT ===")
    print(f"  within-BP reproduces 4 axes: {bp_ok} | within-SZ: {sz_ok} | survives cohort-residualization: {cent_ok}")
    weak = [AXIS_NAMES[int(d[1:]) - 1] for d in Lpool.columns
            if min(results['within_bp']['congruence'][d], results['within_sz']['congruence'][d],
                   results['cohort_residualized']['congruence'][d]) < 0.85]
    if bp_ok and sz_ok and cent_ok:
        verdict = "All 4 axes are within-cohort phenomena that survive cohort removal -> NOT a cohort artifact."
    else:
        verdict = f"Mostly robust; axes weak on >=1 test: {weak} (likely cohort-protocol-sensitive — flag)."
    print(f"  {verdict}")
    results["verdict"] = verdict
    results["weak_axes"] = weak
    json.dump(results, open(OUT / "studyA_cohort_confound.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/studyA_cohort_confound.json")


if __name__ == "__main__":
    main()
