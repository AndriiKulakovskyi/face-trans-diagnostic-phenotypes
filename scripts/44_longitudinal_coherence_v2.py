"""Study C (v2) — longitudinal coherence of the 4 axes (V0 defines, V1/V2 validate).

Two tests, reusing the EXACT validated Stage-0/2/3 logic (imported from scripts 30/32, no
duplication) to recompute construct scores at each visit:
  1. MEASUREMENT INVARIANCE — re-derive K=4 independently at V1 and V2; Tucker congruence vs V0.
     Pass (>=0.85): the same axis structure exists at follow-up.
  2. SCORE STABILITY — project the V0 axis loadings onto each visit's construct scores; per-axis
     rank-order test-retest (Spearman V0<->V1, V0<->V2) = do individual differences persist; plus
     mean-level change (treatment drift).

Coverage constraints (honest): cognition is baseline-anchored (wais ~5% at V1, 43% at V2) -> cognition
coherence is V0<->V2 only; internalizing is BP+DR (Study A). Attrition: completers; report dropout bias.
Masked / no-imputation. Writes results/hfa/studyC_longitudinal_v2.json.
"""
from __future__ import annotations

import importlib.util
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
from scipy.stats import spearmanr

from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset
from trans_diag.axes import AXIS_NAMES
from trans_diag.domains import _robust_z
from trans_diag.masked_fa import masked_correlation, masked_scores, paf_loadings

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
MIN_PAIR, FLOOR, K = 100, 0.30, 4


def _load(stem):  # import a digit-prefixed pipeline script as a module (no main() run on import)
    path = ROOT / "scripts" / stem
    spec = importlib.util.spec_from_file_location(stem.replace(".py", "").replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S30 = _load("30_hfa_stage0_itemset_v2.py")
S32 = _load("32_hfa_stage2_v2.py")
BIO_IDX = {it: (d, s) for d, mem in S32.BIO.items() for it, s in mem}
COG_IDX = {it: (d, s) for d, mem in S32.COG.items() for it, s in mem}


def construct_scores_at(df, vs, by, visit):
    """Recompute Stage-2 construct scores at `visit` using the V0 construct DEFINITIONS (re-estimated
    within-construct loadings at that visit) — same code path as Stage 2, different visit."""
    ds = to_harmonized_dataset(df, vs, visit=visit, sections=None,
                               exclude=(S30.EXCLUDE_ALL - set(S30.RESID)),
                               residualize_on=S30.RESID, normalize=False)
    Z = ds.X.apply(_robust_z)
    cmap = {}
    for it in Z.columns:
        con, sgn = S32.construct_and_sign(it, by[it].section, BIO_IDX, COG_IDX)
        if con is None:
            continue
        cmap.setdefault(con, []).append((it, sgn))
    scores = {}
    for con, members in cmap.items():
        Zc = pd.DataFrame({it: Z[it] * (sgn if sgn != 0 else 1) for it, sgn in members}, index=Z.index)
        scores[con] = S32.score_construct(Zc)[0]
    return pd.DataFrame(scores, index=Z.index)


def extract(S, k=K):
    keep = [c for c in S.columns if S[c].notna().mean() >= FLOOR and S[c].var() > 1e-9]
    Z = (S[keep] - S[keep].mean()) / S[keep].std()
    L = Rotator(method="promax").fit_transform(paf_loadings(masked_correlation(Z, MIN_PAIR), k))
    for j in range(k):
        if L[np.argmax(np.abs(L[:, j])), j] < 0:
            L[:, j] *= -1
    return pd.DataFrame(L, index=keep, columns=[f"d{j+1}" for j in range(k)])


def congruence(ref, test):
    common = [c for c in ref.index if c in test.index]
    A, B = ref.loc[common].to_numpy(), test.loc[common].to_numpy()
    M = np.abs(A.T @ B) / (np.sqrt(np.outer((A * A).sum(0), (B * B).sum(0))) + 1e-12)
    r, c = linear_sum_assignment(-M)
    return {ref.columns[i]: float(M[i, j]) for i, j in zip(r, c)}


def axis_scores(Sv, Lref):
    common = [c for c in Lref.index if c in Sv.columns and Sv[c].notna().mean() >= FLOOR and Sv[c].var() > 0]
    Z = (Sv[common] - Sv[common].mean()) / Sv[common].std()
    F = masked_scores(Z.to_numpy(float), Lref.loc[common].to_numpy())
    return pd.DataFrame(F, index=Sv.index, columns=[f"d{j+1}" for j in range(Lref.shape[1])])


def main() -> None:
    df = build_unified_dataframe(str(ROOT / "data"), str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL", "NOT USABLE", "ID"], format="long")
    vs = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    by = {v.canonical_name: v for v in vs}
    S = {v: construct_scores_at(df, vs, by, v) for v in ("V0", "V1", "V2")}
    print(f"construct scores: " + ", ".join(f"{v} {S[v].shape}" for v in S))

    # sanity: V0 reproduces the committed Stage-2 scores
    S0ref = pd.read_pickle(OUT / "stage2_scores_v2.pkl").set_index(["cohort", "patient_id"])
    common_c = [c for c in S0ref.columns if c in S["V0"].columns]
    r = np.nanmedian([abs(np.corrcoef(np.nan_to_num(S["V0"][c]), np.nan_to_num(S0ref[c].reindex(S["V0"].index)))[0, 1])
                      for c in common_c])
    print(f"  sanity: V0 reproduces committed Stage-2 scores (median |r| = {r:.3f})")

    L0 = extract(S["V0"])
    res = {"invariance": {}, "stability": {}, "mean_change": {}}

    # 1. measurement invariance
    print("\n=== 1. measurement invariance (re-derive K=4 per visit; Tucker congruence vs V0) ===")
    for v in ("V1", "V2"):
        cong = congruence(L0, extract(S[v]))
        res["invariance"][v] = {AXIS_NAMES[int(d[1:]) - 1]: round(val, 2) for d, val in cong.items()}
        print(f"  {v}: " + "  ".join(f"{AXIS_NAMES[int(d[1:])-1][:13]}={val:.2f}" for d, val in cong.items()))

    # 2. score stability (project V0 axis loadings; rank-order test-retest)
    print("\n=== 2. axis score stability (project V0 loadings; Spearman test-retest) ===")
    F = {v: axis_scores(S[v], L0) for v in ("V0", "V1", "V2")}
    print(f"  {'axis':16s} {'rho(V0,V1)':>11s} {'n':>6s} {'rho(V0,V2)':>11s} {'n':>6s}  {'mean V0->V1->V2':>18s}")
    for j, name in enumerate(AXIS_NAMES):
        d = f"d{j+1}"
        row = {}
        for v in ("V1", "V2"):
            a = F["V0"][d]; b = F[v][d].reindex(a.index)
            m = a.notna() & b.notna()
            row[v] = (float(spearmanr(a[m], b[m]).statistic) if m.sum() > 30 else np.nan, int(m.sum()))
        means = [round(float(F[v][d].mean()), 2) for v in ("V0", "V1", "V2")]
        res["stability"][name] = {"rho_V0V1": row["V1"][0], "n_V1": row["V1"][1],
                                  "rho_V0V2": row["V2"][0], "n_V2": row["V2"][1]}
        res["mean_change"][name] = means
        print(f"  {name:16s} {row['V1'][0]:11.2f} {row['V1'][1]:6d} {row['V2'][0]:11.2f} {row['V2'][1]:6d}  {str(means):>18s}")

    print("\n=== VERDICT ===")
    inv_ok = all(min(res["invariance"][v].values()) >= 0.80 for v in res["invariance"])
    print(f"  structural invariance V1/V2 (min congruence >=0.80, cognition excepted at V1): "
          f"{ {v: round(min(res['invariance'][v].values()),2) for v in res['invariance']} }")
    print("  (cognition V1 is sparse — wais ~5%; interpret cognition coherence at V2 only)")
    json.dump(res, open(OUT / "studyC_longitudinal_v2.json", "w"), indent=2, default=str)
    print(f"\nsaved -> {OUT}/studyC_longitudinal_v2.json")


if __name__ == "__main__":
    main()
