"""Study B (v2) — symptom <-> biology orthogonality + the integrated-vs-symptom-only p-factor.

The one potentially non-derivative message. Two linked tests:
  1. ORTHOGONALITY — between-block (symptom<->biology / <->cognition) construct correlations vs
     within-block. Claim: symptoms and biology are nearly orthogonal (between |r| ~ 0).
  2. p-FACTOR IS A SYMPTOM-ONLY ARTIFACT — general-factor strength for nested construct sets:
     symptom-only -> +cognition -> +biology -> full. Metrics: (a) first-factor share lambda1/sum(lambda)
     (K-free, comparable); (b) ECV via Schmid-Leiman at the data-locked K. Prediction: a general factor is strong in
     symptom-only and dissolves as biology/cognition are admitted.

Per Study A: the mood (internalizing) scales are BP+DR-only, so the CLEAN test is **within BP+DR**
(symptoms + biology both measured); pooled is reported as a (confounded) sensitivity. Masked / no-
imputation. Writes results/hfa/studyB_orthogonality.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from factor_analyzer import Rotator

from trans_diag.axes import AXIS_NAMES
from trans_diag.masked_fa import masked_correlation, paf_loadings
from trans_diag.variable import load_variables

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
MIN_PAIR = 100
SEED = 0
K_LOCK = len(AXIS_NAMES)        # data-locked second-order K (Stage 3); ECV computed at this K
SEC2BLOCK = {
    "AUTO-QUESTIONNAIRES": "symptom", "HETERO-QUESTIONNAIRES": "symptom",
    "SUICIDE": "symptom", "EVALUATION MEDICALE": "symptom",
    "BILAN BIOLOGIQUE": "biology", "CONSTANTES ET ECG": "biology",
    "NEUROPSYCHOLOGIE": "cognition",
}  # everything else -> "other" (course / comorbidity / substance / social / perinatal)


def block_map(fit, item_sec):
    out = {}
    for con in fit.index:
        secs = [item_sec.get(it) for it in str(fit.loc[con, "items"]).split(",") if it in item_sec]
        blocks = [SEC2BLOCK.get(s, "other") for s in secs]
        out[con] = Counter(blocks).most_common(1)[0][0] if blocks else "other"
    return out


def first_factor_share(Z):
    w = np.sort(np.linalg.eigvalsh(masked_correlation(Z, MIN_PAIR)))[::-1]
    return float(w[0] / w.sum()), float(w[0] / w[1])


def ecv(Z, K=K_LOCK):
    """Schmid-Leiman explained-common-variance of a single general factor (general-factor strength)."""
    if Z.shape[1] <= K:
        return np.nan
    A = paf_loadings(masked_correlation(Z, MIN_PAIR), K)
    rot = Rotator(method="promax")
    L2 = rot.fit_transform(A)
    wq, Vq = np.linalg.eigh(rot.phi_)
    gamma = np.clip(Vq[:, -1] * np.sqrt(max(wq[-1], 0)), -0.99, 0.99)
    g = L2 @ gamma
    spec = L2 * np.sqrt(1 - gamma ** 2)
    return float((g ** 2).sum() / ((g ** 2).sum() + (spec ** 2).sum()))


def analyze(S, blk, label):
    print(f"\n########## {label}  (n={len(S)}) ##########")
    Z = (S - S.mean()) / S.std()
    cons = {b: [c for c in Z.columns if blk[c] == b and Z[c].notna().mean() >= 0.30 and Z[c].std() > 0]
            for b in ("symptom", "biology", "cognition", "other")}
    print("  block sizes (coverage>=30%): " + ", ".join(f"{b}={len(v)}" for b, v in cons.items()))

    # 1. orthogonality
    R = pd.DataFrame(masked_correlation(Z, MIN_PAIR), index=Z.columns, columns=Z.columns)
    def block_meanabs(b1, b2):
        a, b = cons[b1], cons[b2]
        sub = R.loc[a, b].to_numpy()
        if b1 == b2:
            sub = sub[np.triu_indices(len(a), 1)]
        return float(np.nanmean(np.abs(sub))) if sub.size else np.nan
    print("  mean |construct r|:")
    print(f"    within  symptom={block_meanabs('symptom','symptom'):.2f}  biology={block_meanabs('biology','biology'):.2f}  cognition={block_meanabs('cognition','cognition'):.2f}")
    print(f"    between symptom<->biology={block_meanabs('symptom','biology'):.2f}  "
          f"symptom<->cognition={block_meanabs('symptom','cognition'):.2f}  "
          f"biology<->cognition={block_meanabs('biology','cognition'):.2f}")

    # 2. general-factor strength for nested sets
    sets = {"symptom_only": cons["symptom"],
            "symptom+cognition": cons["symptom"] + cons["cognition"],
            "symptom+biology": cons["symptom"] + cons["biology"],
            "full(all blocks)": cons["symptom"] + cons["biology"] + cons["cognition"] + cons["other"]}
    print("  general-factor strength (lower = more multidimensional):")
    print(f"    {'set':20s} {'p':>3s} {'1st-factor share':>16s} {'l1/l2':>6s} {f'ECV(K={K_LOCK})':>9s}")
    out = {"block_sizes": {b: len(v) for b, v in cons.items()},
           "orthogonality": {f"{b1}_{b2}": block_meanabs(b1, b2)
                             for b1 in cons for b2 in cons if b1 <= b2}, "sets": {}}
    for name, members in sets.items():
        Zs = Z[members]
        ff, ratio = first_factor_share(Zs)
        e = ecv(Zs)
        print(f"    {name:20s} {len(members):3d} {ff:15.2f} {ratio:6.1f} {e:9.2f}")
        out["sets"][name] = {"p": len(members), "first_factor_share": ff, "l1_l2": ratio, "ecv": e}
    return out


def main() -> None:
    S = pd.read_pickle(OUT / "stage2_scores.pkl").set_index(["cohort", "patient_id"])
    fit = pd.read_csv(OUT / "stage2_construct_fit.csv").set_index("construct")
    item_sec = {v.canonical_name: v.section for v in load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))}
    blk = block_map(fit, item_sec)
    coh = S.index.get_level_values("cohort")

    res = {}
    res["BP+DR"] = analyze(S[coh.isin(["bp", "dr"])], blk, "PRIMARY: BP+DR (mood + biology both measured)")
    res["pooled"] = analyze(S, blk, "SENSITIVITY: pooled (confounded by SZ mood-gap)")

    # verdict from the clean BP+DR analysis
    o = res["BP+DR"]["orthogonality"]; s = res["BP+DR"]["sets"]
    sb = o.get("biology_symptom", o.get("symptom_biology"))
    ff_sym, ff_full = s["symptom_only"]["first_factor_share"], s["full(all blocks)"]["first_factor_share"]
    orth_v = "~orthogonal" if sb < 0.12 else "correlated"
    pf_v = ("general factor is symptom-bound; dissolves when biology/cognition admitted"
            if ff_sym > 1.4 * ff_full else "no clear symptom-only advantage")
    print("\n=== VERDICT (BP+DR) ===")
    print(f"  symptom<->biology mean|r| = {sb:.2f}  ({orth_v})")
    print(f"  1st-factor share: symptom-only {ff_sym:.2f} -> full {ff_full:.2f}  ({pf_v})")
    json.dump(res, open(OUT / "studyB_orthogonality.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/studyB_orthogonality.json")


if __name__ == "__main__":
    main()
