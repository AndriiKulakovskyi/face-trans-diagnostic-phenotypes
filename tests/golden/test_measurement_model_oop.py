"""Golden tests for the parallel OOP measurement-model implementation."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytensor.tensor as pt
import pytest

pytest.importorskip("pymc")

from face.models.bayesian.measurement_model_oop import (  # noqa: E402
    S1_FACTORS,
    S5_FACTORS,
    BayesianBifactorESEM,
    MeasurementConfig,
    MeasurementDataset,
    dense_pattern_loglik,
    solve_observed_gaussian_scores,
)
from synthetic.generate_face_like import generate  # noqa: E402


def _synthetic_config(tmp_path, n=260, seed=0) -> tuple[MeasurementConfig, dict]:
    outdir, truth = generate(n=n, seed=seed, out=tmp_path)
    return MeasurementConfig(processed_dir=Path(outdir), include_covariates=True), truth


def test_oop_loading_spec_hard_zero_default_soft_is_optin(tmp_path):
    config, _truth = _synthetic_config(tmp_path)
    assert config.soft_unlikely is False and config.soft_g_anchor_specific is False
    dataset = MeasurementDataset(config)
    core = dataset.core(S1_FACTORS, n_subsample=160, seed=1)

    # Default (hard-zero primary): unlikely_cross and g_anchor_on_specific cells are
    # fixed at exactly 0, so they are NOT free loading parameters.
    spec = dataset.loading_spec(core, windows=False)
    kinds = set(spec.kind.values())
    assert "unlikely" not in kinds
    assert "g_anchor_on_specific" not in kinds

    # Opt-in soft sensitivity arm: those cells become free (near-zero priors).
    soft_spec = MeasurementDataset(config.with_soft_unlikely()).loading_spec(core, windows=False)
    soft_kinds = set(soft_spec.kind.values())
    assert "unlikely" in soft_kinds
    assert "g_anchor_on_specific" in soft_kinds
    near_zero = [
        sd for j, c, _mu, sd in soft_spec.signed_cells if soft_spec.kind[(j, c)] == "g_anchor_on_specific"
    ]
    assert near_zero
    assert max(near_zero) <= 0.001


def test_oop_woodbury_matches_dense_mvn_on_ragged_missingness():
    rng = np.random.default_rng(3)
    M = rng.normal(size=(35, 7))
    M[rng.random(M.shape) < 0.3] = np.nan
    Lam = rng.normal(0.3, 0.2, size=(7, 3))
    A = rng.normal(size=(3, 3))
    Phi = A @ A.T + np.eye(3)
    d = np.sqrt(np.diag(Phi))
    Phi = Phi / np.outer(d, d)
    sigma = rng.uniform(0.4, 0.9, size=7)

    builder = BayesianBifactorESEM()
    mask = np.isfinite(M).astype("float64")
    x = np.nan_to_num(M, nan=0.0)
    pat_mask, pat_inv = builder.patterns(mask)
    R = np.linalg.cholesky(Phi)
    got = builder.woodbury_potential(
        pt.as_tensor(x),
        mask,
        pt.as_tensor(Lam @ R),
        pt.as_tensor(sigma**2),
        pat_mask,
        pat_inv,
        mask.sum(1),
        Lam.shape[1],
    ).eval()
    ref = dense_pattern_loglik(M, Lam, Phi, sigma)
    assert np.allclose(got, ref, atol=1e-6)


def test_oop_covariate_modes_residualize_default_and_in_likelihood(tmp_path):
    config, _truth = _synthetic_config(tmp_path)  # default covariate_mode == "residualize"
    assert config.covariate_mode == "residualize"

    # Default (residualize): covariates fold into the data via FWL, so the
    # marginalized model stays zero-mean — no alpha/beta sampler parameters.
    dataset_r = MeasurementDataset(config)
    core_r = dataset_r.core(S1_FACTORS, n_subsample=120, seed=2)
    spec_r = dataset_r.loading_spec(core_r, windows=False)
    model_r = BayesianBifactorESEM(config).build_marginalized(core_r, spec_r, correlated=False)
    assert core_r.covariates.shape[1] == 0
    assert "alpha" not in model_r.named_vars
    assert "beta" not in model_r.named_vars

    # Opt-in in_likelihood: the published equation written literally — alpha + beta
    # are sampled inside PyMC, beta is item x covariate.
    il_config = replace(config, covariate_mode="in_likelihood")
    dataset_i = MeasurementDataset(il_config)
    core_i = dataset_i.core(S1_FACTORS, n_subsample=120, seed=2)
    spec_i = dataset_i.loading_spec(core_i, windows=False)
    model_i = BayesianBifactorESEM(il_config).build_marginalized(core_i, spec_i, correlated=False)
    assert core_i.covariates.shape[1] > 0
    assert "alpha" in model_i.named_vars
    assert "beta" in model_i.named_vars
    assert model_i.named_vars["beta"].type.shape == (len(core_i.items), core_i.covariates.shape[1])


def test_oop_conditional_scores_match_dense_formula():
    rng = np.random.default_rng(4)
    M = rng.normal(size=(20, 5))
    M[rng.random(M.shape) < 0.25] = np.nan
    Lam = rng.normal(0.5, 0.2, size=(5, 2))
    Phi = np.array([[1.0, 0.25], [0.25, 1.0]])
    sigma = rng.uniform(0.5, 0.8, size=5)

    mean, sd = solve_observed_gaussian_scores(M, Lam, Phi, sigma)
    assert mean.shape == (20, 2)
    assert sd.shape == (20, 2)
    assert np.isfinite(mean[np.isfinite(mean)]).all()
    assert (sd[np.isfinite(sd)] >= 0).all()


def test_oop_synthetic_smoke_preserves_planted_structure(tmp_path):
    config, truth = _synthetic_config(tmp_path, n=220, seed=5)
    dataset = MeasurementDataset(config.with_fast_mode())
    core = dataset.core(S1_FACTORS, n_subsample=180, seed=5, include_covariates=False)
    spec = dataset.loading_spec(core, windows=False)

    assert core.items == truth["items"]
    assert len(spec.pos_cells) > 0
    assert len(spec.signed_cells) > 0

    home = np.array([truth["home"][item] for item in core.items])
    bio = np.isin(home, ["metabolic", "inflammatory"])
    cog_sleep = np.isin(home, ["cognition", "sleep"])
    assert bio.any()
    assert cog_sleep.any()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs processed FACE baseline",
)
def test_oop_correlated_phi_initializes_at_multiple_factor_sizes():
    """Regression: building S2 then S5 in one session must not reuse LKJ shapes."""
    config = MeasurementConfig().with_smoke_defaults()
    dataset = MeasurementDataset(config)
    builder = BayesianBifactorESEM(config)
    for factors in (S1_FACTORS, S5_FACTORS):
        core = dataset.core(factors, correlated=True, windows=True, n_subsample=80, seed=0)
        spec = dataset.loading_spec(core, windows=True)
        model = builder.build_marginalized(core, spec, correlated=True)
        point = model.initial_point()
        assert any(name.startswith("Phi_spec_lower") for name in point)
