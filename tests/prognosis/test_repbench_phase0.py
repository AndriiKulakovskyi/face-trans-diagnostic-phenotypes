"""Phase 0 foundation tests for the M4 representation benchmark — data wiring (copula objects), proper-score
metrics (CRPS, calibration), and identical-across-arms CV folds. No model fits here."""
from __future__ import annotations

import numpy as np
import pytest

from face.prognosis.repbench import ARCH, CANON, cv, data, metrics

# ------------------------------------------------------------------ data: RAW (143 indicators + mask)


def test_load_raw_shape_missingness_no_leakage():
    raw = data.load_raw()
    assert raw.shape == (9013, 143)
    assert list(raw.index.names) == ["cohort", "patient_id"]
    # missingness is preserved (never imputed) — a substantial fraction of cells is NaN
    frac_missing = float(raw.isna().to_numpy().mean())
    assert 0.2 < frac_missing < 0.7
    mask = data.raw_mask(raw)
    assert mask.shape == raw.shape and set(np.unique(mask.to_numpy())) <= {0, 1}
    # no follow-up / endpoint columns leak into the raw V0 block
    assert not any(("__V2" in c) or c.startswith("ep_") for c in raw.columns)


# ------------------------------------------------------------------ data: LATENT (copula coords + arch)


def test_load_latent_copula_schema():
    lat = data.load_latent()
    assert lat.shape[0] == 9013
    for ax in CANON:
        assert f"{ax}__mean" in lat.columns and f"{ax}__sd" in lat.columns
    for a in ARCH:
        assert a in lat.columns
    # copula coordinates are full-N: the posterior means are never missing
    mean_cols = [f"{ax}__mean" for ax in CANON]
    assert not lat[mean_cols].isna().any().any()
    assert list(lat.index.names) == ["cohort", "patient_id"]


def test_latent_blocks_nested_and_clean():
    blocks = data.latent_blocks()
    assert blocks["LAT-mu"] == [f"{ax}__mean" for ax in CANON]
    # nested: each richer arm contains the previous
    assert set(blocks["LAT-mu"]) < set(blocks["LAT-sigma"]) < set(blocks["LAT-A"])
    # no outcome leakage in any latent block
    flat = {c for cols in blocks.values() for c in cols}
    assert not any(("__V2" in c) or c.startswith("ep_") or c.startswith("egf__") for c in flat)


# ------------------------------------------------------------------ outcomes


def test_outcome_endpoints_definitions():
    panel = data.load_outcomes()
    rec = panel["ep_egf_recovery"]
    det = panel["ep_egf_deterioration"]
    n_rec, n_det = int(rec.notna().sum()), int(det.notna().sum())
    # plausible eligible Ns (recovery = impaired w/ V0&V2 GAF; deterioration = all w/ V0&V2 GAF)
    assert 300 < n_rec < 2200
    assert 1500 < n_det < 2600
    # recovery denominator is a subset of deterioration's (impaired ⊆ have-both-visits)
    assert n_rec <= n_det
    # recovery is only defined among the baseline-impaired (GAF < 61)
    impaired = panel.loc[rec.notna(), "egf__V0"]
    assert (impaired < 61).all()
    # binary
    assert set(np.unique(rec.dropna())) <= {0.0, 1.0}


# ------------------------------------------------------------------ assemble + cohort scope


def test_assemble_alignment_and_cohort_filter():
    full = data.assemble()
    assert len(full) == 9013
    for col in ("ep_egf_recovery", "ep_egf_deterioration", "egf__V0", "w_retained_V2"):
        assert col in full.columns
    # latent joined and aligned (no row blow-up, means present)
    assert not full["metabolic__mean"].isna().any()
    # episodic headline scope (cohort lives on the index, exposed via cohort_of)
    full_cohort = data.cohort_of(full)
    bd = data.assemble(cohorts=("bp", "dr"))
    assert set(np.unique(data.cohort_of(bd))) <= {"bp", "dr"}
    assert len(bd) == int(np.isin(full_cohort, ["bp", "dr"]).sum())
    assert 0 < len(bd) < len(full)


def test_eligible_mask_matches_notna():
    full = data.assemble()
    m = data.eligible(full, "egf_recovery")
    assert m.sum() == int(full["ep_egf_recovery"].notna().sum())


# ------------------------------------------------------------------ metrics: CRPS


def test_crps_gaussian_closed_form():
    # CRPS of N(0,1) at y=0 = 2*phi(0) - 1/sqrt(pi) = 0.7979 - 0.5642 = 0.23369...
    assert metrics.crps_gaussian([0.0], [0.0], [1.0]) == pytest.approx(0.233692, abs=1e-5)
    # a forecast centred on the truth beats one that is off
    good = metrics.crps_gaussian([5.0], [5.0], [2.0])
    bad = metrics.crps_gaussian([5.0], [0.0], [2.0])
    assert good < bad


def test_crps_ensemble_matches_gaussian():
    rng = np.random.default_rng(0)
    mu, sigma, n = 3.0, 2.0, 400
    y = rng.normal(mu, sigma, n)
    samples = rng.normal(mu, sigma, size=(2000, n))     # well-specified ensemble
    ens = metrics.crps_ensemble(y, samples)
    closed = metrics.crps_gaussian(y, np.full(n, mu), np.full(n, sigma))
    assert ens == pytest.approx(closed, rel=0.05)


# ------------------------------------------------------------------ metrics: calibration + decision


def test_calibration_slope_perfect():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 8000)
    y = (rng.uniform(size=p.size) < p).astype(int)      # perfectly calibrated by construction
    slope = metrics.calibration_slope(y, p)
    assert slope == pytest.approx(1.0, abs=0.15)


def test_net_benefit_band_keys():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < p).astype(int)
    nb = metrics.net_benefit_band(y, p)
    assert set(nb) == {"thresholds", "model", "treat_all", "treat_none"}
    assert len(nb["model"]) == len(nb["thresholds"])


# ------------------------------------------------------------------ CV: identical folds across arms


def test_make_folds_identical_and_cover():
    rng = np.random.default_rng(3)
    n = 1000
    cohort = rng.choice(["bp", "sz", "dr"], n)
    y = (rng.uniform(size=n) < 0.2).astype(int)
    f1 = cv.make_folds(y, cohort, n_splits=5, seed=123)
    f2 = cv.make_folds(y, cohort, n_splits=5, seed=123)     # feature-independent ⇒ reproducible
    for (a_tr, a_te), (b_tr, b_te) in zip(f1, f2, strict=False):
        assert np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)
    # one repeat's test folds cover every row exactly once
    assert np.array_equal(cv.oof_indices(f1), np.arange(n))
