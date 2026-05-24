"""Golden-numbers regression: the manuscript's headline values must match the result files.

This is the safeguard the verification round wished for. Each assertion pins a number the
manuscript states (with its §/Table location) to the corresponding committed *aggregate*
artifact in results/ (no per-patient data — these CSV/JSON files are tracked, so the test runs
in CI without the confidential cohort). If a pipeline re-run changes a result, the artifact
changes and the matching assertion fails — forcing a synchronized update of BOTH this test and
the manuscript. Tolerances absorb the manuscript's rounding (3 decimals) + BLAS round-off.

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


# ── §3.4 / Table 3 — head-to-head outcomes (repeated-CV) ──────────────────────────────
def test_outcomes_headtohead():
    df = pd.read_csv(_need("phase5_ci.csv")).set_index("outcome")
    q = df.loc["EQ-5D quality of life"]
    assert abs(_lead(q["DSM"]) - 0.302) <= 0.005          # QoL: DSM R²
    assert abs(_lead(q["axes"]) - 0.342) <= 0.005         # QoL: axes R²
    assert abs(_lead(q["dim_minus_DSM"]) - 0.039) <= 0.004   # axes beat DSM (+0.039)
    e = df.loc["EGF functioning"]
    assert abs(_lead(e["combined_minus_DSM"]) - 0.034) <= 0.004   # functioning complemented
    assert abs(_lead(e["dim_minus_DSM"]) - 0.000) <= 0.004        # dims alone ≈ DSM
    h = df.loc["any hospitalization"]
    assert abs(_lead(h["DSM"]) - 0.747) <= 0.005          # hosp: DSM AUC
    assert _lead(h["dim_minus_DSM"]) < -0.10              # DSM dominates hospitalization


# ── Table 2 — top loadings of the six imputation-free axes ────────────────────────────
def test_axis_loadings():
    L = pd.read_csv(_need("dimensional_final_loadings.csv"))
    top = {a: g.set_index("domain")["loading"].abs().idxmax()
           for a, g in L.groupby("axis")}
    val = {a: g.set_index("domain")["loading"].abs().max()
           for a, g in L.groupby("axis")}
    assert top["axis1"] == "qidsr" and abs(val["axis1"] - 0.89) <= 0.04        # depression
    assert top["axis2"] == "agetrt" and abs(val["axis2"] - 0.79) <= 0.04        # later onset
    assert top["axis4"] == "nboccur_hospitalisation_lt"                          # illness burden
    assert top["axis5"] == "metabolic_syndrome"                                  # metabolic
    assert top["axis6"] == "hooccur_arret_travail_actuel"                        # work-disability


# ── §3.3 — confound independence + AE↔FA agreement + structure ────────────────────────
def test_confound_and_agreement():
    meta = _json("dimensional_final_meta.json")
    assert meta["K"] == 6
    assert max(meta["confound_max_corr"].values()) <= 0.02      # age/sex ≤0.017
    rc = _json("review_checks.json")
    assert rc["eta_cohort_max"] <= 0.12                          # cohort ≤0.112
    assert rc["eta_site_max"] <= 0.055                           # site ≤0.051
    assert rc["cca_observed"][0] >= 0.95                         # AE↔FA leading CCA 0.98
    assert rc["cca_observed"][0] > rc["cca_null_leading_p95"] + 0.5   # far above null
    assert abs(rc["continuum_rho"] - 0.786) <= 0.02             # mood↔psychosis ρ 0.79
    assert rc["cohort_from_mask_bacc"] >= 0.95                   # 98% from mask


# ── §3.3 / Fig 6c — DSM-subtype variance per axis ─────────────────────────────────────
def test_dsm_eta_squared():
    e = pd.read_csv(_need("dimensional_dsm_eta_squared.csv")).set_index("axis")["eta_sq"]
    assert abs(e["illness_burden"] - 0.140) <= 0.01            # most diagnosis-linked
    assert abs(e["depression_severity"] - 0.123) <= 0.01
    assert e.drop(["illness_burden", "depression_severity"]).max() <= 0.08   # others ≤0.07


# ── §3.6 / Table 2 — trait–state gradient (V0↔V1) ─────────────────────────────────────
def test_trait_state_gradient():
    s = pd.read_csv(_need("longitudinal_axes_stability.csv"))
    v1 = s[s["visit"] == "V1"].set_index("axis")["pearson"]
    assert abs(v1["metabolic"] - 0.64) <= 0.04                 # most trait-like
    assert abs(v1["depression_severity"] - 0.58) <= 0.04
    assert v1["metabolic"] > v1["mania_activation"]            # metabolic more trait-like than mania
    assert v1["later_onset"] <= 0.15                            # static by construction


# ── §3.2 — discrete-vs-dimensional structure test ─────────────────────────────────────
def test_structure_test():
    st = _json("structure_test.json")
    assert abs(st["hdbscan"]["cohort_ari"] - 0.70) <= 0.03     # HDBSCAN ≈ cohort
    eig = st["eigengap"]["eigenvalues"]
    assert eig[1] < 0.01 and eig[2] < eig[5]                    # smooth low-end rise (no gap)


# ── §3.1 — confound ladder ────────────────────────────────────────────────────────────
def test_confound_ladder():
    cl = pd.read_csv(_need("confound_ladder.csv")).set_index("rung")
    assert cl.loc[1, "bootstrap_ari"] >= 0.90                   # rung 1 spurious-but-stable
    assert cl.loc[1, "ari_sister"] <= 0.40
    assert cl.loc[4, "ari_sex"] <= 0.02 and cl.loc[4, "ari_age"] <= 0.02   # collapses


# ── §3.7 — cognition semi-independent of symptom axes ─────────────────────────────────
def test_cognition_semi_independent():
    c = pd.read_csv(_need("cognition_bpsz_corr.csv"), index_col=0)
    assert c.abs().to_numpy().max() <= 0.30                     # max |r| ≈ 0.26


# ── §4.6 — no general ('p') factor: confound-free axes are near-orthogonal ────────────
def test_pfactor_no_general_factor():
    m = _json("pfactor.json")
    assert m["oblique_phi_mean_offdiag"] <= 0.10                # axes near-orthogonal → weak/no general factor
    # the single dominant dimension collapses onto depression, not a broad general factor
    assert m["corr_with_axes"]["depression_severity"] >= 0.90
    others = [v for k, v in m["corr_with_axes"].items() if k != "depression_severity"]
    assert max(abs(v) for v in others) <= 0.30                 # ≈0 with the other five axes


# ── §3.5 — fold-honest re-fit removes only negligible optimism (Limitation 10) ────────
def test_cvrefit_robustness():
    rows = {r["outcome"]: r for r in _json("robustness_cvrefit.json")["headtohead_cvrefit"]}
    q = rows["EQ-5D quality of life"]
    assert abs(q["axes_refit_minus_DSM"] - 0.039) <= 0.006      # QoL advantage survives refit (+0.040)
    assert abs(q["optimism_alldata_minus_refit"]) <= 0.01       # ≈0 optimism for QoL
    assert rows["EGF functioning"]["combined_refit_minus_DSM"] >= 0.02   # functioning still complemented
    assert rows["any hospitalization"]["axes_refit_minus_DSM"] < -0.10   # DSM still dominates hosp
    # the manuscript's "optimism ≤0.007" headline, across all outcomes
    assert max(abs(r["optimism_alldata_minus_refit"]) for r in rows.values()) <= 0.01


# ── §3.5 — within-FACE held-out replication (transportability; Limitation 9) ──────────
def test_replication_holdout():
    m = _json("replication_holdout.json")
    struct = {r["held_out_cohort"]: r for r in m["loco_structure"]}
    assert struct["dr"]["min_congruence"] >= 0.95          # structure near-identical without DR (0.98)
    assert struct["sz"]["mean_congruence"] >= 0.85          # mostly preserved without SZ (0.93)
    loso = {r["outcome"]: r for r in m["loso_outcomes"]}
    assert loso["EQ-5D quality of life"]["axes_minus_DSM"] >= 0.03   # QoL advantage transports to unseen sites (+0.042)
    assert loso["any hospitalization"]["axes_minus_DSM"] < -0.10     # hospitalization stays DSM-dominated
