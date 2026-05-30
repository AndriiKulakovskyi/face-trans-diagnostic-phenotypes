"""Golden-numbers regression: the manuscript's headline values must match the result files.

This is the safeguard the verification round wished for. Each assertion pins a number the
manuscript states (with its §/Table location) to the corresponding committed *aggregate*
artifact in results/ (no per-patient data — these CSV/JSON files are tracked, so the test runs
in CI without the confidential cohort). If a pipeline re-run changes a result, the artifact
changes and the matching assertion fails — forcing a synchronized update of BOTH this test and
the manuscript. Tolerances absorb the manuscript's rounding (3 decimals) + BLAS round-off.

Locked at K=6 after the DR neuropsychology extraction gap was closed (2026-05): cognition is now
one of the six trans-diagnostic dimensions (cognition_verbal), so the cognition checks below test
that integrated axis rather than the former BP/SZ-only sub-analysis.

If the artifacts are absent (e.g. a fresh clone that hasn't run the pipeline), the tests skip.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

RES = Path(__file__).resolve().parents[1] / "results"


def _need(name: str) -> Path:
    p = RES / name
    if not p.exists():
        pytest.skip(f"{name} absent — run scripts/00_run_all.py to regenerate")
    return p


def _lead(x) -> float:
    """Leading float of a cell like '0.302 [0.295,0.306]' or '+0.039 [+0.036,+0.042]'."""
    return float(str(x).split("[")[0].strip())


def _json(name: str) -> dict:
    return json.loads(_need(name).read_text())


# ── §3.4 / §3k / Table 3 — head-to-head outcomes (repeated-CV) ─────────────────────────
def test_outcomes_headtohead():
    """Post-audit (2026-05): the head-to-head reports BOTH ``axes_orig`` (pre-audit, no
    cohort in M1) and ``axes_fair`` (cohort dummies added to M1 for parity with M0's
    arm dummies). The headline ``dim_minus_DSM`` is now the fair version; the original
    survives as ``dim_minus_DSM_orig`` for back-compat. See FINDINGS §3k."""
    df = pd.read_csv(_need("phase5_ci.csv")).set_index("outcome")
    q = df.loc["EQ-5D quality of life"]
    assert abs(_lead(q["DSM"]) - 0.302) <= 0.005          # QoL: DSM R²
    assert abs(_lead(q["axes_fair"]) - 0.343) <= 0.006    # QoL: axes(fair) R²
    assert abs(_lead(q["dim_minus_DSM"]) - 0.039) <= 0.004   # axes(fair) beat DSM (+0.039)
    assert abs(_lead(q["dim_minus_DSM_orig"]) - 0.037) <= 0.004   # axes(orig) +0.037
    e = df.loc["EGF functioning"]
    assert abs(_lead(e["dim_minus_DSM"]) - 0.035) <= 0.005   # axes(fair) BEAT DSM by +0.035
    assert abs(_lead(e["dim_minus_DSM_orig"]) - 0.0) <= 0.005   # axes(orig) tied DSM (pre-audit)
    assert abs(_lead(e["combined_minus_DSM"]) - 0.035) <= 0.005
    h = df.loc["any hospitalization"]
    assert abs(_lead(h["DSM"]) - 0.747) <= 0.006          # hosp: DSM AUC
    # post-audit: with cohort parity, axes ≈ DSM on hospitalization (was -0.141 pre-audit)
    assert abs(_lead(h["dim_minus_DSM"])) <= 0.04         # axes(fair) ≈ DSM
    assert _lead(h["dim_minus_DSM_orig"]) < -0.10         # axes(orig) was DSM-dominated


# ── Table 2 — top loadings of the six imputation-free axes (incl. cognition) ───────────
def test_axis_loadings():
    L = pd.read_csv(_need("dimensional_final_loadings.csv"))
    top = {a: g.set_index("domain")["loading"].abs().idxmax()
           for a, g in L.groupby("axis")}
    val = {a: g.set_index("domain")["loading"].abs().max()
           for a, g in L.groupby("axis")}
    assert top["axis1"] == "qidsr" and abs(val["axis1"] - 0.89) <= 0.04        # depression
    assert top["axis2"] == "agetrt" and abs(val["axis2"] - 0.78) <= 0.04        # later onset
    assert top["axis3"] == "altman"                                             # mania / externalizing
    assert top["axis4"] == "nboccur_hospitalisation_lt"                         # illness burden
    assert top["axis5"] == "verbal_reasoning" and abs(val["axis5"] - 0.81) <= 0.05   # cognition (verbal)
    assert top["axis6"] == "metabolic_syndrome"                                 # metabolic


# ── §3.3 — confound independence + AE↔FA agreement + structure ────────────────────────
def test_confound_and_agreement():
    meta = _json("dimensional_final_meta.json")
    assert meta["K"] == 6
    assert max(meta["confound_max_corr"].values()) <= 0.02      # age/sex ≤0.017
    rc = _json("review_checks.json")
    assert rc["eta_cohort_max"] <= 0.11                          # cohort ≤0.106
    assert rc["eta_site_max"] <= 0.055                           # site ≤0.049
    assert rc["cca_observed"][0] >= 0.92                         # AE↔FA leading CCA 0.94
    assert rc["cca_observed"][0] > rc["cca_null_leading_p95"] + 0.5   # far above null
    assert abs(rc["continuum_rho"] - 0.50) <= 0.06             # mood↔psychosis ρ 0.50 (cognition shares PC1)
    assert rc["cohort_from_mask_bacc"] >= 0.95                   # ~98% from mask


# ── §3.3 / Fig 6c — DSM-subtype variance per axis ─────────────────────────────────────
def test_dsm_eta_squared():
    e = pd.read_csv(_need("dimensional_dsm_eta_squared.csv")).set_index("axis")["eta_sq"]
    # cognition, depression and illness-burden carry the most DSM-diagnosis variance (all ≤ ~0.13)
    assert abs(e["cognition_verbal"] - 0.132) <= 0.02          # most diagnosis-linked (SZ↓ verbal)
    assert abs(e["depression_severity"] - 0.119) <= 0.02
    assert abs(e["illness_burden"] - 0.111) <= 0.02
    # the remaining axes are minimally diagnosis-bound
    assert e[["later_onset", "mania_activation", "metabolic"]].max() <= 0.06


# ── §3.6 / Table 2 — trait–state gradient (V0↔V1) ─────────────────────────────────────
def test_trait_state_gradient():
    s = pd.read_csv(_need("longitudinal_axes_stability.csv"))
    v1 = s[s["visit"] == "V1"].set_index("axis")["pearson"]
    assert abs(v1["metabolic"] - 0.63) <= 0.04                 # most trait-like
    assert abs(v1["depression_severity"] - 0.58) <= 0.04
    assert v1["metabolic"] > v1["mania_activation"]            # metabolic more trait-like than mania
    assert v1["later_onset"] <= 0.15                            # static by construction


# ── §3.2 — discrete-vs-dimensional structure test ─────────────────────────────────────
def test_structure_test():
    st = _json("structure_test.json")
    assert abs(st["hdbscan"]["cohort_ari"] - 0.64) <= 0.04     # HDBSCAN ≈ cohort
    eig = st["eigengap"]["eigenvalues"]
    assert eig[1] < 0.01 and eig[2] < eig[5]                    # smooth low-end rise (no gap)


# ── §3.1 — confound ladder ────────────────────────────────────────────────────────────
def test_confound_ladder():
    cl = pd.read_csv(_need("confound_ladder.csv")).set_index("rung")
    assert cl.loc[1, "bootstrap_ari"] >= 0.90                   # rung 1 spurious-but-stable
    assert cl.loc[1, "ari_sister"] <= 0.40
    assert cl.loc[4, "ari_sex"] <= 0.02 and cl.loc[4, "ari_age"] <= 0.02   # collapses


# ── §2.12 / §3.7 — cognition enters the main model as ONE confound-clean axis ──────────
def test_cognition_axis_confound_clean():
    rc = _json("review_checks.json")
    assert rc["cognition_axes"] == ["cognition_verbal"]                       # one genuine cognitive axis
    assert rc["cognition_axis_eta_cohort"]["cognition_verbal"] <= 0.11        # not a cohort proxy (0.072)
    assert rc["cognition_axis_r2_from_availability"]["cognition_verbal"] <= 0.05  # not an availability proxy


# ── §4.6 — no general ('p') factor: confound-free axes are near-orthogonal ────────────
def test_pfactor_no_general_factor():
    m = _json("pfactor.json")
    assert m["oblique_phi_mean_offdiag"] <= 0.10                # axes near-orthogonal → weak/no general factor
    # the single dominant dimension collapses onto depression, not a broad general factor
    assert m["corr_with_axes"]["depression_severity"] >= 0.90
    others = [v for k, v in m["corr_with_axes"].items() if k != "depression_severity"]
    assert max(abs(v) for v in others) <= 0.30                 # ≈0 with the other five axes


# ── §3.5 / §3k — fold-honest re-fit removes only negligible optimism (Limitation 10) ──
def test_cvrefit_robustness():
    """Post-audit: 20_robustness_cvrefit.py now reports both ``axes_refit_orig_minus_DSM``
    (pre-audit, no cohort in M1) and ``axes_refit_fair_minus_DSM`` (cohort dummies added)."""
    rows = {r["outcome"]: r for r in _json("robustness_cvrefit.json")["headtohead_cvrefit"]}
    q = rows["EQ-5D quality of life"]
    assert abs(q["axes_refit_fair_minus_DSM"] - 0.040) <= 0.006    # QoL: fair advantage survives refit
    assert abs(q["axes_refit_orig_minus_DSM"] - 0.038) <= 0.006    # QoL: orig advantage too
    assert abs(q["optimism_alldata_minus_refit"]) <= 0.01           # ≈0 optimism for QoL
    e = rows["EGF functioning"]
    assert e["axes_refit_fair_minus_DSM"] >= 0.02                   # EGF: fair axes beat DSM (post-audit)
    assert abs(e["axes_refit_orig_minus_DSM"]) <= 0.02              # EGF: orig axes ≈ DSM
    assert e["combined_refit_minus_DSM"] >= 0.02                    # functioning complemented
    h = rows["any hospitalization"]
    assert abs(h["axes_refit_fair_minus_DSM"]) <= 0.04               # hosp: fair ≈ DSM (post-audit)
    assert h["axes_refit_orig_minus_DSM"] < -0.10                    # hosp: orig was DSM-dominated
    # the manuscript's "optimism ≤0.007" headline, across all outcomes
    assert max(abs(r["optimism_alldata_minus_refit"]) for r in rows.values()) <= 0.01


# ── §3.5 — within-FACE held-out replication (transportability; Limitation 9) ──────────
def test_replication_holdout():
    m = _json("replication_holdout.json")
    struct = {r["held_out_cohort"]: r for r in m["loco_structure"]}
    assert struct["dr"]["min_congruence"] >= 0.95          # structure near-identical without DR (0.97)
    assert struct["sz"]["mean_congruence"] >= 0.85          # mostly preserved without SZ (0.98)
    loso = {r["outcome"]: r for r in m["loso_outcomes"]}
    # post-audit (§3k): the headline LOSO numbers are the fair (cohort-controlled) deltas
    assert loso["EQ-5D quality of life"]["axes_fair_minus_DSM"] >= 0.03   # QoL transports
    assert abs(loso["any hospitalization"]["axes_fair_minus_DSM"]) <= 0.05   # hosp ≈ DSM (was -0.147)
    assert loso["any hospitalization"]["axes_orig_minus_DSM"] < -0.10        # orig: DSM-dominated


# ── §4.5 / Table 5 — parsimonious screening panel (sparse distillation) ────────────────
def test_screening_panel():
    m = _json("screening_panel_meta.json")
    assert m["n_questionnaire_items"] <= 15                          # parsimonious (≤15 features)
    rq = m["recon_r2_questionnaire"]
    assert rq["depression_severity"] >= 0.78 and rq["mania_activation"] >= 0.78   # symptom axes recover
    assert rq["cognition_verbal"] >= 0.40                            # cognition partly recoverable (education+functioning)
    assert rq["metabolic"] <= 0.20                                   # not questionnaire-recoverable (needs labs)
    # the flagged routine metabolic-panel add-on lifts the metabolic axis
    assert m["recon_r2_questionnaire_plus_labs"]["metabolic"] > rq["metabolic"] + 0.10
    # repeated-CV (R=200): the QoL advantage is robust — 95% CI excludes 0
    assert float(m["qol_panel_axes_minus_DSM_ci"].split("[")[1].split(",")[0]) > 0
    # functioning: the panel COMPLEMENTS DSM (combined − DSM CI lower bound > 0)
    assert float(m["egf_panel_combined_minus_DSM_ci"].split("[")[1].split(",")[0]) > 0
    # the group-aware per-axis panel still recovers the cognitive axis
    assert m["recon_r2_peraxis_plus_labs"]["cognition_verbal"] > 0.3
