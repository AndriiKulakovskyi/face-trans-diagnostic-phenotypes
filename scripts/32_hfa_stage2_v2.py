"""Stage 2 (v2) — hybrid first-order measurement model (clinical anchors, data-revised).

Implements Stage 2 of docs/HIERARCHICAL_FA_PLAN.md. Builds the construct layer that REPLACES the
flat masked-mean domain scores. "Hybrid" = clinical anchors (the construct boundaries) revised by
the Stage 1 data (split multidimensional constructs, drop misfitting items, recover dropped labs).

Each construct's score = the WITHIN-construct masked 1-factor posterior (paf_loadings -> masked_scores)
of its sign-oriented items, on observed support only (no imputation). This fixes, vs the flat mean:
  - item weights are estimated, not flat (B1);  - totals + sub-scores no longer double-count (B4);
  - misfitting items are dropped, not averaged in (B3: CTQ denial; B5: CGI non-severity);
  - signs are explicit -> uniform "higher = more severe" where a severity pole exists (B7);
  - multidimensional constructs are split, not collapsed (metabolic -> 4; C-SSRS/ISF -> 2).

Outputs the construct scores (Stage 3 input), Phi_1 (construct correlations), per-construct fit,
and a target-rotation check of where the data disagrees with the clinical map. Reviewer-facing
caveats are printed and written to the report.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from trans_diag.domains import instrument_stem
from trans_diag.masked_fa import masked_correlation, masked_scores, paf_loadings, varimax
from trans_diag.variable import load_variables

warnings.simplefilter("ignore")
OUT = ROOT / "results" / "hfa"
MIN_PAIR = 100

# ---- biology panels (explicit clinical grouping; data-driven metabolic split from Stage 1) ----
# sign: +1 higher=worse, -1 lower=worse, 0 = no severity pole (polarity data-defined, documented).
BIO = {
    "adiposity":     [("bmi", 1), ("wstcir", 1), ("weight", 1)],
    "blood_pressure":[("sysbpsupine", 1), ("diabpsupine", 1), ("sysbpstanding", 1), ("diabpstanding", 1)],
    "lipids_hdl":    [("hdl", -1), ("trig", 1), ("cholhdl_lbstresc", 1)],
    "cholesterol":   [("chol", 1), ("ldl", 1)],
    "glycemia":      [("gluc", 1), ("hba1c", 1)],
    "inflammation":  [("crp", 1), ("wbc", 1), ("neut", 1), ("mono_lbstresc", 1)],
    "red_cell":      [("hgb", 0), ("hct_lbstresc", 0), ("rbc", 0), ("mcv", 0), ("mch_lbstresc", 0), ("mchc_lbstresc", 0)],
    "hepatic":       [("alt_lbstresc", 1), ("ast_lbstresc", 1), ("ggt_lbstresc", 1), ("alp_lbstresc", 1), ("bili_lbstresc", 1)],
    "renal":         [("creat_lbstresc", 1), ("urea_lbstresc", 1), ("urate", 1), ("creatclr_lbstresc", -1)],
    "thyroid":       [("tsh_lbstresc", 1), ("t3fr_lbstresc", -1), ("t4fr_lbstresc", -1)],  # hypothyroid pole
    "electrolytes":  [("sodium_lbstresc", 0), ("k_lbstresc", 0), ("cl_lbstresc", 0), ("bicarb_lbstresc", 0)],
    "autonomic_hr":  [("eghrmn", 1), ("hrsupine", 1), ("hrstanding", 1)],
    "serum_protein": [("alb_lbstresc", -1), ("prot_lbstresc", -1), ("ca_lbstresc", 0)],
}
# ---- cognition (from neuropsy_features.yaml; +2-cohort sensitivity items recovered in Stage 0) ----
COG = {
    "verbal_reasoning":  [("wais_similitudes_std", -1)],
    "working_memory":    [("wais_digitspan_std", -1), ("arithstd_w", -1)],
    "processing_speed":  [("wais_code_std", -1), ("wais_ivt_index", -1), ("symbol04_w", -1)],
    "perceptual_reason": [("mat_std_w", -1)],
    "psychomotor_speed": [("tmt_a_time_sec", 1)],
    "executive":         [("tmt_b_time_sec", 1)],
}
# ---- symptom splits / drops / special groupings (override the instrument-stem default) ----
SPLIT = {
    **{f"cssrs{i:02d}": ("cssrs_severity", 1) for i in (1, 2, 3, 4, 5, 6)},
    **{f"cssrs{i:02d}": ("cssrs_intensity", 1) for i in (8, 9, 10, 11, 12)},
    **{c: ("suicidal_ideation", 1) for c in ("isf01", "isf02", "isf03", "isf04", "isf05")},
    **{c: ("attempt_history", 1) for c in ("isf07", "isf08", "isf08a", "isf09", "isf09a")},
    "altman": ("mania_activation", 1), "ymrs": ("mania_activation", 1),
    "cgi01": ("cgi_severity", 1),
}
DROP = {"ctq40", "ctq41", "cgi02", "cgi03", "cgi03a", "cgi03b"}   # denial (B3); CGI non-severity (B5)

# ---- medical-history flags: data-anchored decomposition of the old VAF1=0.38 24-flag bin ----
# (scripts/_comorbidity step analysis, LABBOOK V2-8): the 24 flags split into 2 weakly-coherent but
# STABLE co-occurrence clusters (cardiac 2.7-2.9x lift, VAF1 0.38; atopic 1.6-2.5x, VAF1 0.24) +
# prevalent standalones; the 12 flags <2% prevalence are un-clusterable noise -> dropped from the
# dimensional inputs, RETAINED as Stage-4 validators (does the metabolic/inflammation axis predict them?).
COMORBID = {
    "hta_mhoccur": ("cardiac_history", 1), "autcardv_mhoccur": ("cardiac_history", 1),
    "trbrycard_mhoccur": ("cardiac_history", 1),
    "acne_mhoccur": ("atopic_inflammatory", 1), "eczema_mhoccur": ("atopic_inflammatory", 1),
    "cheveux_mhoccur": ("atopic_inflammatory", 1), "toxidermi_mhoccur": ("atopic_inflammatory", 1),
    "psoriasis_mhoccur": ("atopic_inflammatory", 1),
    "migraine_mhoccur": ("migraine", 1), "traumacra_mhoccur": ("head_trauma", 1),
    "ulcgasduo_mhoccur": ("peptic_ulcer", 1),     # 2.3% -> standalone (>=2% kept)
}
RARE_DROP = {"autendoc_mhoccur", "autneuro_mhoccur", "avc_mhoccur", "cirrhose_mhoccur",
             "coronar_mhoccur", "epilepsie_mhoccur", "genetique_mhoccur", "hepmedic_mhoccur",
             "hvc_mhoccur", "infarctus_mhoccur", "inflachro_mhoccur", "sep_mhoccur",
             "vih_mhoccur"}                         # all <2% prevalence -> Stage-4 validators


def construct_and_sign(item: str, sec: str, bio_idx: dict, cog_idx: dict) -> tuple:
    if item in DROP:
        return None, 0
    if item in bio_idx:
        return bio_idx[item]
    if item in cog_idx:
        return cog_idx[item]
    if item in SPLIT:
        return SPLIT[item]
    if item in RARE_DROP:
        return None, 0                            # <2% comorbidity flag -> Stage-4 validator, not input
    if item in COMORBID:
        return COMORBID[item]
    if item.endswith("_mhoccur"):
        return f"{item.replace('_mhoccur','')}_hx", 1   # any unlisted flag -> standalone
    if item in ("sudose_cigarettes_lt", "suncf_cigarettes_lt", "fagers"):
        return "nicotine", 1          # dependence (age-start/stop are timing -> own constructs)
    if sec in ("BILAN BIOLOGIQUE", "CONSTANTES ET ECG"):
        return f"bio_{item}", 0                        # uncomposited lab -> own construct (data-defined pole)
    return instrument_stem(item), 1                    # symptom instrument: higher = worse


def score_construct(Zc: pd.DataFrame):
    """Within-construct masked 1-factor posterior score + VAF1; deterministic 'higher=worse' sign."""
    if Zc.shape[1] == 1:
        s = Zc.iloc[:, 0]
        return s, 1.0, np.array([1.0])
    R = masked_correlation(Zc, MIN_PAIR)
    L = paf_loadings(R, 1)
    s = pd.Series(masked_scores(Zc.to_numpy(float), L)[:, 0], index=Zc.index)
    if np.corrcoef(np.nan_to_num(s), np.nan_to_num(Zc.mean(1)))[0, 1] < 0:   # orient to item mean
        s, L = -s, -L
    w = np.linalg.eigvalsh(R)[::-1]
    return s, float(w[0] / w.sum()), L[:, 0]


def main() -> None:
    Z = pd.read_pickle(OUT / "stage0_Z_resid_v2.pkl")
    meta = pd.read_pickle(OUT / "stage0_meta_v2.pkl")
    by = {v.canonical_name: v for v in load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))}
    bio_idx = {it: (d, s) for d, mem in BIO.items() for it, s in mem}
    cog_idx = {it: (d, s) for d, mem in COG.items() for it, s in mem}

    # build construct -> [(item, sign)]
    cmap: dict[str, list[tuple[str, int]]] = {}
    dropped = []
    for it in Z.columns:
        con, sgn = construct_and_sign(it, by[it].section, bio_idx, cog_idx)
        if con is None:
            dropped.append(it)
            continue
        cmap.setdefault(con, []).append((it, sgn))

    # score every construct (sign-orient items first)
    scores, fit = {}, []
    for con, members in cmap.items():
        Zc = pd.DataFrame({it: Z[it] * (sgn if sgn != 0 else 1) for it, sgn in members}, index=Z.index)
        s, vaf1, load = score_construct(Zc)
        scores[con] = s
        fit.append({"construct": con, "n_items": len(members), "vaf1": round(vaf1, 2),
                    "coverage": round(float(s.notna().mean()), 2),
                    "polarity": "severity" if any(sg != 0 for _, sg in members) else "data-defined",
                    "items": ",".join(it for it, _ in members)})
    S = pd.DataFrame(scores, index=Z.index)
    fitdf = pd.DataFrame(fit).set_index("construct").sort_values("n_items", ascending=False)

    n_multi = int((fitdf.n_items > 1).sum())
    contaminants = [d for d in dropped if d in DROP]
    validators = [d for d in dropped if d in RARE_DROP]
    print(f"first-order constructs: {S.shape[1]}  ({n_multi} multi-item, {S.shape[1]-n_multi} single-item)")
    print(f"items used: {int(fitdf.n_items.sum())} / {Z.shape[1]}")
    print(f"  dropped as contaminants ({len(contaminants)}): {contaminants}")
    print(f"  dropped as <2% comorbidity -> Stage-4 validators ({len(validators)}): "
          f"{[v.replace('_mhoccur','') for v in validators]}")
    pd.Series(sorted(validators)).to_csv(OUT / "stage2_comorbidity_validators_v2.csv", index=False, header=["flag"])
    print(f"\n=== multi-item constructs (within-construct unidimensionality VAF1) ===")
    print(fitdf[fitdf.n_items > 1][["n_items", "vaf1", "coverage", "polarity", "items"]]
          .to_string(max_colwidth=52))

    # Phi_1: construct-score correlations (for Stage 3 / second-order)
    Phi = S.corr(min_periods=MIN_PAIR)
    off = np.abs(Phi.to_numpy()[np.triu_indices(S.shape[1], 1)])
    print(f"\nPhi_1 (construct correlations): {int(np.nansum(off > 0.3))} of {len(off)} pairs |r|>0.3, "
          f"max={np.nanmax(off):.2f}  -> second-order layer (Stage 3)")
    # strongest correlated construct pairs (the second-order seeds)
    P = Phi.to_numpy().copy(); np.fill_diagonal(P, 0)
    cn = list(Phi.columns)
    seen = set()
    print("  top correlated construct pairs:")
    for a, b in sorted(zip(*np.unravel_index(np.argsort(-np.abs(P), axis=None), P.shape)),
                       key=lambda ab: -abs(P[ab]))[:200]:
        if a >= b or (a, b) in seen or abs(P[a, b]) < 0.3:
            continue
        seen.add((a, b))
        if len(seen) <= 8:
            print(f"    {cn[a]:20s} ~ {cn[b]:20s} r={P[a,b]:+.2f}")

    # reviewer check: where does the data DISAGREE with the clinical map? (global target rotation)
    cons_order = list(cmap.keys())
    cidx = {c: i for i, c in enumerate(cons_order)}
    A = varimax(paf_loadings(masked_correlation(Z, MIN_PAIR), len(cons_order)))
    # crude alignment: each construct-factor = factor whose top items are that construct's items
    item_con = {it: con for con, mem in cmap.items() for it, _ in mem}
    weak = []
    for it in Z.columns:
        if it in DROP or it not in item_con:
            continue
        i = list(Z.columns).index(it)
        if np.abs(A[i]).max() < 0.30:
            weak.append((it, item_con[it], round(float(np.abs(A[i]).max()), 2)))
    print(f"\nreviewer check — items with NO strong loading anywhere (max|loading|<0.30): {len(weak)}")
    for it, con, m in sorted(weak, key=lambda x: x[2])[:15]:
        print(f"    {it:24s} (assigned {con:18s}) max|load|={m}")

    # save
    fitdf.to_csv(OUT / "stage2_construct_fit_v2.csv")
    S.reset_index().to_pickle(OUT / "stage2_scores_v2.pkl")
    Phi.round(3).to_csv(OUT / "stage2_phi1_v2.csv")
    print(f"\nsaved -> {OUT}/stage2_construct_fit_v2.csv, stage2_scores_v2.pkl, stage2_phi1_v2.csv")


if __name__ == "__main__":
    main()
