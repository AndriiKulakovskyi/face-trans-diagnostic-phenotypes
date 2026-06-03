"""Golden-numbers regression (v2): the manuscript's headline values must match the result files.

Each assertion pins a number the manuscript (``results/manuscript/manuscript.md``) states — with its
§/Table/Figure location — to the corresponding committed *aggregate* artifact under ``results/hfa/``
(loadings, correlation matrices, and the Study A–D JSONs hold no per-patient data). If a pipeline
re-run (scripts 30–48_v2) changes a result, the artifact changes and the matching assertion fails,
forcing a synchronized update of BOTH this test and the manuscript. Tolerances absorb the manuscript's
rounding (3 decimals) + BLAS round-off.

**v2 model (LABBOOK V2-9..V2-19).** K=4 second-order trans-diagnostic axes — internalizing, cognition,
illness_course, cardiometabolic — with NO dominant general factor (ECV 0.36) and NO discrete subtypes
beyond the DSM cohorts. Mania & suicidality are valid but orthogonal standalone constructs (not axes).
The v1 K=6 golden numbers are archived at git tag ``v1-archive-2026-05-30`` — do not carry them over.

**2026-06-03 dictionary-review re-baseline.** Six variables were added (CVLT total/short/long-delay recall,
verbal fluency phonemic+semantic, QIDS-13 anhedonia) and the suicide skip-logic was decoded. The K=4
backbone, dimensional verdict, and no-p-factor result all held; the changes were: item set 188→194; the
**cognition axis is now memory-anchored** (CVLT leads dim2; its sign flipped, magnitudes preserved);
anhedonia joined internalizing; and the weak-axes caveat gained ``cardiometabolic`` (DR n=552 underpowered).
Suicidality stayed orthogonal even after its coverage was recovered.

If the artifacts are absent (a fresh clone that hasn't run the pipeline), the tests skip — results/hfa/
is gitignored, so run the v2 pipeline (scripts ``30_*_v2`` … ``48_*_v2``) to regenerate them.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HFA = Path(__file__).resolve().parents[1] / "results" / "hfa"


def _need(name: str) -> Path:
    p = HFA / name
    if not p.exists():
        pytest.skip(f"{name} absent — run scripts/30–48_*_v2 to regenerate")
    return p


def _json(name: str) -> dict:
    return json.loads(_need(name).read_text())


def _csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(_need(name), **kw)


# ── §2.1 / Table 1 + §2.7 Stage 0 — cohorts, item set, near-singular item correlation ──────
def test_cohorts_and_itemset():
    d = _json("stage0_diagnostics_v2.json")
    assert d["n_patients"] == 9013                                   # Table 1 total
    assert d["cohort_n"] == {"bp": 6252, "sz": 2209, "dr": 552}      # Table 1 per-cohort
    assert d["n_items"] == 194                                       # §2.7 Stage 0 item set (188 +6: CVLT×3, fluency×2, QIDS-13 anhedonia)
    assert d["eig_gt1"] >= 50                                        # factorable (54 eigenvalues > 1)
    assert d["eig_top20"][0] > 12 and d["eig_top20"][1] > 9          # scree 12.6, 10.1, …
    # the item correlation is near-singular (κ ≈ 1.3e9) — this MOTIVATES aggregation (§2.6)
    assert d["cond_R"] > 1e8


# ── §2.6 / §3.1 — aggregation concentrates construct signal (VAF1 wins vs collapsed mean) ───
def test_construct_unidimensionality():
    vaf = _csv("stage2_construct_fit_v2.csv").set_index("construct")["vaf1"]
    assert vaf["adiposity"] >= 0.90            # 0.93 (collapsed metabolic mean was 0.40)
    assert vaf["cholesterol"] >= 0.85          # 0.90
    assert vaf["autonomic_hr"] >= 0.80         # 0.86 (a recovered vitals construct)
    assert vaf["processing_speed"] >= 0.80     # 0.87
    assert vaf["blood_pressure"] >= 0.65       # 0.72
    assert vaf["mania_activation"] >= 0.65     # 0.71 — a valid construct (just orthogonal to the axes)


# ── Fig 2 / §3.1 — the four trans-diagnostic axes (top-loading constructs) ──────────────────
def test_four_axes_loadings():
    L = _csv("stage3_loadings_v2.csv", index_col=0)
    assert {"dim1", "dim2", "dim3", "dim4"}.issubset(L.columns)
    # dim1 internalizing — depression/anxiety/functioning
    assert L.loc["qidsr", "dim1"] >= 0.85 and L.loc["madrs", "dim1"] >= 0.80     # 0.93 / 0.89
    assert L.loc["staya", "dim1"] >= 0.70                                         # 0.83 anxiety
    # dim2 cognition — now MEMORY-anchored (CVLT verbal memory leads after the 2026-06-03 additions);
    # executive / processing speed / fluency co-load. dim2's sign is arbitrary (factor orientation) → abs().
    assert L.loc["cvlt_total_recall", "dim2"] >= 0.70                              # 0.79 verbal memory (new lead)
    assert L.loc["cvlt_long_delay_free_recall", "dim2"] >= 0.60                    # 0.77 delayed recall
    assert abs(L.loc["executive", "dim2"]) >= 0.60 and abs(L.loc["processing_speed", "dim2"]) >= 0.60   # 0.70 / 0.67
    assert abs(L.loc["verbal_fluency_semantic", "dim2"]) >= 0.40                   # 0.65 semantic fluency
    # dim3 illness course — age-of-onset top, inverse hospitalization burden
    assert L.loc["agedebut_hospitalisation", "dim3"] >= 0.80                      # 0.87
    assert L.loc["agetrt", "dim3"] >= 0.70                                        # 0.79
    assert L.loc["nboccur_hospitalisation_lt", "dim3"] < 0                        # inverse burden (-0.53)
    # dim4 cardiometabolic — lipids/inflammation/adiposity
    for c in ("lipids_hdl", "inflammation", "adiposity", "autonomic_hr"):
        assert L.loc[c, "dim4"] >= 0.40
    # mania is ORTHOGONAL to all four axes (|loading| < 0.30 everywhere) — not an axis
    assert L.loc["mania_activation", ["dim1", "dim2", "dim3", "dim4"]].abs().max() < 0.30   # ~0.08


# ── §3.1 / §3.3 — no dominant general ('p') factor ──────────────────────────────────────────
def test_no_pfactor():
    phi2 = _csv("stage3_phi2_v2.csv", index_col=0).to_numpy()
    offdiag = np.abs(phi2[np.triu_indices(4, 1)])
    assert offdiag.mean() <= 0.20                                  # weakly correlated axes (mean |Φ₂| 0.17)
    # the integrated-set Schmid–Leiman general factor is weak (pooled = the Stage-3 75-construct set)
    full = _json("studyB_orthogonality_v2.json")["pooled"]["sets"]["full(all blocks)"]
    assert abs(full["ecv_k4"] - 0.36) <= 0.05                      # ECV 0.36 → no dominant p-factor
    assert full["first_factor_share"] <= 0.12                      # 0.094


# ── §3.2 / Fig 4 — dimensional, NOT categorical (both arms) ─────────────────────────────────
def test_dimensional_not_categorical():
    p = _json("phase5_structure_v2.json")
    # arm A (the 4 dims + mania + suicidality): a continuum — HDBSCAN finds at most tiny micro-pockets
    # (<=2 dense, >=85% noise), and those pockets don't track cohort => no real discrete structure.
    assert p["A"]["hdbscan"]["n"] <= 2 and p["A"]["hdbscan"]["noise"] >= 0.85
    assert p["A"]["hdbscan"]["cohort_ari"] <= 0.10                 # micro-pockets are not cohort (0.04)
    assert max(p["A"]["dsm_ari"].values()) <= 0.06                 # k-means vs DSM ARI ≈ 0.03
    assert max(p["A"]["bimodality"]) <= 0.555                      # every axis unimodal (Sarle)
    # arm B (75 constructs): the ONLY dense clusters are the 3 cohorts themselves
    assert p["B"]["hdbscan"]["n"] == 3 and p["B"]["hdbscan"]["cohort_ari"] >= 0.98   # ARI 1.00


# ── §3.3 / Fig 3 — THE HEADLINE: symptoms ⊥ biology; p-factor is symptom-only ───────────────
def test_orthogonality_headline():
    b = _json("studyB_orthogonality_v2.json")["BP+DR"]
    o = b["orthogonality"]
    assert o["biology_symptom"] <= 0.06          # symptom ↔ biology ≈ 0.03 (near-orthogonal)
    assert o["cognition_symptom"] <= 0.10         # symptom ↔ cognition ≈ 0.07
    assert o["symptom_symptom"] >= 0.18           # within-symptom structure is real (0.24)
    # the general factor dissolves MONOTONICALLY as structured biology/cognition are admitted
    ff = {k: b["sets"][k]["first_factor_share"] for k in
          ("symptom_only", "symptom+cognition", "symptom+biology", "full(all blocks)")}
    assert ff["symptom_only"] >= 0.30 and ff["full(all blocks)"] <= 0.12      # 0.33 → 0.09
    assert ff["symptom_only"] > ff["symptom+cognition"] > ff["symptom+biology"] > ff["full(all blocks)"]


# ── §3.4 — the axes are not a cohort artifact (+ the internalizing measurement caveat) ──────
def test_studyA_cohort_confound():
    a = _json("studyA_cohort_confound_v2.json")
    assert min(a["cohort_residualized"]["congruence"].values()) >= 0.96   # cohort-residualized ≥0.96
    assert min(a["within_bp"]["congruence"].values()) >= 0.95             # within-BP all four ≥0.95
    assert a["weak_axes"] == ["internalizing", "cardiometabolic"]   # internalizing (SZ proxy) + cardiometabolic (DR n=552 underpowered, congruence 0.35)


# ── §3.5 / Fig 6 — longitudinal coherence (trait vs state vs fixed-historical) ──────────────
def test_studyC_longitudinal():
    c = _json("studyC_longitudinal_v2.json")
    inv = c["invariance"]["V2"]
    assert inv["internalizing"] >= 0.95 and inv["cardiometabolic"] >= 0.95   # structure persists (0.98/0.97)
    st = c["stability"]
    assert st["cardiometabolic"]["rho_V0V1"] >= 0.60                         # most trait-stable (0.66)
    assert st["cardiometabolic"]["rho_V0V1"] > st["internalizing"]["rho_V0V1"]   # > episodic mood (0.59)
    assert st["illness_course"]["rho_V0V1"] <= 0.25                          # fixed-historical (0.16)


# ── §3.6 / Fig 5 / Table 3 — predictive validity vs DSM (functioning robust, honest) ───────
def test_studyD_functioning():
    d = _json("studyD_predictive_v2.json")
    gaf_est, gaf_ci = d["GAF@V2"]["delta_axes_add_dsm"]
    assert abs(gaf_est - 0.046) <= 0.012 and gaf_ci[0] > 0          # GAF: dims add over DSM, CI excludes 0
    fast_est, fast_ci = d["FAST@V2"]["delta_axes_vs_dsm"]
    assert fast_est >= 0.02 and fast_ci[0] > 0                       # FAST: dims beat DSM, CI excludes 0
    assert d["dropout_auc_from_axes"] <= 0.56                        # attrition check: axes→dropout ≈ chance


# ── §3.6 / Fig 5 — relapse: the regression-to-the-mean confound, removed ────────────────────
def test_studyD_relapse_deconfounded():
    d2 = _json("studyD2_survival_v2.json")
    assert abs(d2["logistic"]["AUC"]["M0_base"] - 0.578) <= 0.02     # de-confounded baseline (was 0.765)
    add_est, add_ci = d2["logistic"]["d_axes_add_dsm"]
    assert add_est >= 0.02 and add_ci[0] > 0                         # dims add over DSM (+0.036), CI excludes 0
    # AUC ≈ 0.70 is reachable only via early-course (V0+V1) trajectory information
    d4 = _json("studyD4_trajectory_v2.json")
    assert d4["logistic"]["AUC"]["+traj"] >= 0.69                    # 0.70
    assert d4["gboost"]["AUC"]["+traj"] >= 0.68                      # 0.696
