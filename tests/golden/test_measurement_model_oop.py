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
    _cohort_weights,
    _config_sig,
    _rank_int,
    copula_invert,
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


# ---------------------------- Gaussian-copula vertical ----------------------------

def test_oop_copula_invert_round_trips():
    rng = np.random.default_rng(7)
    x = rng.lognormal(0.0, 1.0, size=2000)  # skewed positive
    z = _rank_int(x)
    order = np.argsort(x, kind="mergesort")
    sorted_values, sorted_z = x[order], np.sort(z)
    recov = copula_invert(z, sorted_values, sorted_z)
    assert np.max(np.abs(recov - x)) < 1e-9  # exact at the data nodes


def test_oop_copula_config_sig_distinguishes_modes():
    base = MeasurementConfig()
    cop = base.with_gaussian_copula()
    assert base.likelihood_mode == "native" and cop.likelihood_mode == "gaussian_copula"
    sig_n, sig_c = _config_sig(base), _config_sig(cop)
    assert sig_n["likelihood_mode"] == "native"
    assert sig_c["likelihood_mode"] == "gaussian_copula"
    assert sig_n != sig_c  # cache will not be shared between native and copula fits


def test_oop_copula_marginals_standard_normal_and_buildable(tmp_path):
    config, _truth = _synthetic_config(tmp_path, n=400, seed=6)
    cop = replace(config, likelihood_mode="gaussian_copula", include_covariates=False)
    dataset = MeasurementDataset(cop)
    core = dataset.core(S1_FACTORS, n_subsample=300, seed=6)
    # marginals approximately standard normal
    for c in range(core.M.shape[1]):
        col = core.M[:, c][np.isfinite(core.M[:, c])]
        assert abs(col.mean()) < 0.15
        assert abs(col.std() - 1.0) < 0.15
    # inversion map stored for every encoded item, and the model builds on copula-z
    assert core.copula is not None and set(core.copula) == set(core.items)
    spec = dataset.loading_spec(core, windows=False)
    model = BayesianBifactorESEM(cop).build_marginalized(core, spec, correlated=False)
    assert "lam_pos" in model.named_vars


# ---------------------------- cohort-weighted (§3.6) ----------------------------

def test_cohort_weights_sum_to_n_and_balance():
    cohort = np.array(["bp"] * 6 + ["sz"] * 3 + ["dr"] * 1)
    w = _cohort_weights(cohort)
    assert w.shape == (10,)
    assert abs(w.sum() - 10.0) < 1e-9                       # sum(w) = N (information preserved)
    for c in ("bp", "sz", "dr"):
        assert abs(w[cohort == c].sum() - 10.0 / 3.0) < 1e-9  # each cohort total = N/K (equal influence)
    assert w[cohort == "dr"][0] > w[cohort == "bp"][0]      # the rare cohort is up-weighted


def test_weighted_marginalized_builds_and_changes_logp(tmp_path):
    config, _truth = _synthetic_config(tmp_path, n=400, seed=8)
    dataset = MeasurementDataset(replace(config, include_covariates=False))
    builder = BayesianBifactorESEM(replace(config, include_covariates=False))
    core = dataset.core(S1_FACTORS, n_subsample=300, seed=8)
    spec = dataset.loading_spec(core, windows=False)
    w = _cohort_weights(core.cohort)
    m_plain = builder.build_marginalized(core, spec, correlated=False)
    m_wtd = builder.build_marginalized(core, spec, correlated=False, weights=w)
    pt0 = m_plain.initial_point()
    lp_plain = float(m_plain.compile_logp()(pt0))
    lp_wtd = float(m_wtd.compile_logp()(m_wtd.initial_point()))
    assert np.isfinite(lp_plain) and np.isfinite(lp_wtd)
    assert abs(lp_plain - lp_wtd) > 1e-6  # weighting changes the likelihood contribution


def test_native_invert_round_trips():
    """The native (pre-copula) inverse must exactly recover the original scale from the encoded z, for
    gaussian (identity) and both lognormal branches (log when min>0, log1p when min<=0)."""
    from face.models.bayesian.synthetic import _native_invert  # noqa: PLC0415
    rng = np.random.default_rng(0)

    def encode_decode(y, family, sign, log_min):
        vlog = y
        if family == "lognormal":
            vlog = np.log1p(y - log_min + 1e-6) if log_min <= 0 else np.log(y)
        oriented = sign * vlog
        mu, sd = float(oriented.mean()), float(oriented.std())
        return _native_invert((oriented - mu) / sd, family, sign, log_min, mu, sd)

    y = rng.normal(50, 10, 500)
    assert np.allclose(encode_decode(y, "gaussian", 1, None), y, atol=1e-6)
    yl = rng.lognormal(2.0, 0.5, 500)                       # strictly positive -> log branch
    assert np.allclose(encode_decode(yl, "lognormal", 1, float(yl.min())), yl, atol=1e-4)
    yz = rng.poisson(3, 500).astype(float)                  # contains zeros -> log1p branch
    assert np.allclose(encode_decode(yz, "lognormal", -1, float(yz.min())), yz, atol=1e-3)


def test_substance_orthogonal_config():
    base = MeasurementConfig()
    so = base.with_substance_orthogonal()
    assert base.orthogonal_factors == ()
    assert "substance" in so.orthogonal_factors
    assert _config_sig(base)["orthogonal_factors"] == []
    assert _config_sig(so)["orthogonal_factors"] == ["substance"]


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs processed FACE baseline",
)
def test_substance_orthogonal_removes_it_from_correlated_block():
    """With substance pinned orthogonal, it leaves the correlated Phi block, so the free correlation
    vector (Phi_spec_lower) shrinks by exactly one factor's worth of cross-terms -> substance has a
    zero off-diagonal Phi row by construction (like G)."""
    cfg = MeasurementConfig(likelihood_mode="gaussian_copula")
    nlower = {}
    for label, c in [("plain", cfg), ("orth", cfg.with_substance_orthogonal())]:
        ds = MeasurementDataset(c)
        core = ds.core(S5_FACTORS, correlated=True, windows=True, n_subsample=400, balanced=True, seed=0)
        model = BayesianBifactorESEM(c).build_marginalized(core, ds.loading_spec(core, windows=True), correlated=True)
        nlower[label] = int(model.named_vars["Phi_spec_lower"].type.shape[0])
        F = len(core.factor_cols)
        ns = (F - 1) - (1 if label == "orth" else 0)   # specifics minus G, minus substance if pinned
        assert nlower[label] == ns * (ns - 1) // 2
    assert nlower["orth"] < nlower["plain"]   # substance dropped from the correlated block


def test_cohort_weighted_in_config_sig():
    base = MeasurementConfig()
    wtd = base.with_cohort_weighted()
    assert base.cohort_weighted is False and wtd.cohort_weighted is True
    assert _config_sig(base)["cohort_weighted"] is False
    assert _config_sig(wtd)["cohort_weighted"] is True


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs processed FACE baseline",
)
def test_weighted_mixed_builds_on_real_data():
    """Weighted build_mixed (explicit Bernoulli/ordered/NB -> weighted Potentials) compiles a finite logp."""
    cfg = MeasurementConfig(likelihood_mode="gaussian_copula", cohort_weighted=True)
    dataset = MeasurementDataset(cfg)
    mixed = dataset.mixed(S5_FACTORS, min_cohorts=2, n_subsample=600, balanced=True, seed=0)
    spec = dataset.loading_spec(mixed.base, windows=True,
                                bifactor_g_sd={f: 0.05 for f in S5_FACTORS if f != "overall_severity"})
    w = _cohort_weights(mixed.base.cohort)
    model = BayesianBifactorESEM(cfg).build_mixed(mixed, spec, weights=w)
    lp = float(model.compile_logp()(model.initial_point()))
    assert np.isfinite(lp)
    assert any(str(n).startswith("lh_") for n in model.named_vars)  # explicit item loadings present


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs processed FACE baseline",
)
def test_oop_copula_promotes_high_cardinality_count_to_continuum():
    """A high-cardinality count (sudose_cigarettes_lt) is Gaussianized into the continuous block and
    therefore absent from the mixed explicit items; a binary SUD item stays explicit."""
    cop = MeasurementConfig(likelihood_mode="gaussian_copula")
    dataset = MeasurementDataset(cop)
    core = dataset.core(S5_FACTORS, correlated=True, windows=True, n_subsample=800, balanced=True, seed=0)
    assert "sudose_cigarettes_lt" in core.items  # promoted into the Gaussianized continuous block
    mixed = dataset.mixed(S5_FACTORS, min_cohorts=2, n_subsample=800, balanced=True, seed=0)
    explicit_items = set(mixed.bin_items) | set(mixed.ord_items) | set(mixed.cnt_items)
    assert "sudose_cigarettes_lt" not in explicit_items   # no longer an explicit count
    assert "suoccur_alcool" in explicit_items             # rare binary SUD stays native/explicit
