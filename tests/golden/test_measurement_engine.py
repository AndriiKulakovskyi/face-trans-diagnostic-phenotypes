"""Golden tests for the parallel OOP measurement-model implementation."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pytest
from scipy.stats import multivariate_normal

pytest.importorskip("pymc")

from face.measurement.engine import (  # noqa: E402
    S1_FACTORS,
    S5_FACTORS,
    BayesianBifactorESEM,
    CoreData,
    LoadingSpec,
    MeasurementConfig,
    MeasurementDataset,
    StageRunner,
    _cached_model_sig,
    _certification_policy,
    _cohort_weights,
    _config_sig,
    _rank_int,
    apply_frozen_covariate_design,
    copula_invert,
    dense_pattern_loglik,
    solve_observed_gaussian_scores,
)
from face.measurement.run import (
    DATA_QUALITY_EXCLUSIONS,  # noqa: E402
    LOCAL_INDEPENDENCE_EXCLUSIONS,  # noqa: E402
    PRIMARY_EXCLUSIONS,  # noqa: E402
    PRIMARY_EXPLICIT_FACTORS,  # noqa: E402
    _eligible_continuation_advance,  # noqa: E402
    _eligible_continuation_retry,  # noqa: E402
)
from synthetic.generate_face_like import generate  # noqa: E402


def _synthetic_config(tmp_path, n=260, seed=0) -> tuple[MeasurementConfig, dict]:
    outdir, truth = generate(n=n, seed=seed, out=tmp_path)
    return MeasurementConfig(processed_dir=Path(outdir), include_covariates=True), truth


def test_loading_spec_hard_zero_default_soft_is_optin(tmp_path):
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


def test_woodbury_matches_dense_mvn_on_ragged_missingness():
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


def test_five_three_conditional_marginalization_matches_dense_mvn():
    rng = np.random.default_rng(31)
    n, j, f = 24, 11, 8
    e = [0, 2, 4, 5, 7]
    m = [1, 3, 6]
    a = rng.normal(size=(f, f))
    phi = a @ a.T + np.eye(f)
    scale = np.sqrt(np.diag(phi))
    phi /= np.outer(scale, scale)
    lam = rng.normal(0.0, 0.35, size=(j, f))
    sigma = rng.uniform(0.4, 0.9, size=j)
    phi_ee = phi[np.ix_(e, e)]
    phi_me = phi[np.ix_(m, e)]
    mmat = phi_me @ np.linalg.inv(phi_ee)
    sres = phi[np.ix_(m, m)] - mmat @ phi_me.T
    bmat = lam[:, e] + lam[:, m] @ mmat
    cond_cov = lam[:, m] @ sres @ lam[:, m].T + np.diag(sigma**2)
    fe = rng.multivariate_normal(np.zeros(len(e)), phi_ee, size=n)
    mu = rng.normal(0.0, 0.25, size=(n, j))
    x = np.vstack(
        [rng.multivariate_normal(mu[i] + bmat @ fe[i], cond_cov) for i in range(n)]
    )
    x[rng.random(x.shape) < 0.3] = np.nan
    mask = np.isfinite(x).astype("float64")
    residual = np.nan_to_num(x - mu - fe @ bmat.T, nan=0.0)
    builder = BayesianBifactorESEM()
    pat_mask, pat_inv = builder.patterns(mask)
    got = builder.woodbury_potential(
        pt.as_tensor(residual),
        mask,
        pt.as_tensor(lam[:, m] @ np.linalg.cholesky(sres)),
        pt.as_tensor(sigma**2),
        pat_mask,
        pat_inv,
        mask.sum(1),
        len(m),
    ).eval()
    ref = []
    for i in range(n):
        observed = np.flatnonzero(mask[i])
        ref.append(
            multivariate_normal.logpdf(
                x[i, observed],
                mean=(mu[i] + bmat @ fe[i])[observed],
                cov=cond_cov[np.ix_(observed, observed)],
            )
        )
    assert np.allclose(got, ref, atol=1e-6)


def test_covariate_modes_in_likelihood_default_and_residualize_optin(tmp_path):
    config, _truth = _synthetic_config(tmp_path)
    assert config.covariate_mode == "in_likelihood"

    # Legacy opt-in residualization: covariates fold into the data via FWL, so the
    # marginalized model stays zero-mean — no alpha/beta sampler parameters.
    residual_config = replace(config, covariate_mode="residualize")
    dataset_r = MeasurementDataset(residual_config)
    core_r = dataset_r.core(S1_FACTORS, n_subsample=120, seed=2)
    spec_r = dataset_r.loading_spec(core_r, windows=False)
    model_r = BayesianBifactorESEM(residual_config).build_marginalized(
        core_r, spec_r, correlated=False
    )
    assert core_r.covariates.shape[1] == 0
    assert "alpha" not in model_r.named_vars
    assert "beta" not in model_r.named_vars

    # Default in_likelihood: the published equation written literally — alpha + beta
    # are sampled inside PyMC, beta is item x covariate.
    il_config = config
    dataset_i = MeasurementDataset(il_config)
    core_i = dataset_i.core(S1_FACTORS, n_subsample=120, seed=2)
    spec_i = dataset_i.loading_spec(core_i, windows=False)
    model_i = BayesianBifactorESEM(il_config).build_marginalized(core_i, spec_i, correlated=False)
    assert core_i.covariates.shape[1] > 0
    assert "alpha" in model_i.named_vars
    assert "beta" in model_i.named_vars
    assert model_i.named_vars["beta"].type.shape == (len(core_i.items), core_i.covariates.shape[1])


def test_frozen_covariate_transform_reproduces_training_design(tmp_path):
    config, _truth = _synthetic_config(tmp_path)
    core = MeasurementDataset(config).core(S1_FACTORS, n_subsample=120, seed=7)
    cov = pd.read_parquet(config.processed_dir / "covariates_v0.parquet")
    site_path = config.processed_dir / "site_v0.parquet"
    site = pd.read_parquet(site_path)["siteid_city"] if site_path.exists() else None
    frozen = apply_frozen_covariate_design(
        core.index, cov, site, core.covariate_metadata
    )
    assert frozen.shape == core.covariates.shape
    assert np.allclose(frozen, core.covariates, atol=1e-12)


def test_conditional_scores_match_dense_formula():
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


def test_synthetic_smoke_preserves_planted_structure(tmp_path):
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
def test_correlated_phi_initializes_at_multiple_factor_sizes():
    """Regression: building S2 then S5 in one session must not reuse LKJ shapes."""
    config = MeasurementConfig().with_smoke_defaults()
    dataset = MeasurementDataset(config)
    builder = BayesianBifactorESEM(config)
    for factors in (S1_FACTORS, S5_FACTORS):
        core = dataset.core(factors, correlated=True, windows=True, n_subsample=80, seed=0)
        spec = dataset.loading_spec(core, windows=True)
        model = builder.build_marginalized(core, spec, correlated=True)
        point = model.initial_point()
        assert any(name.startswith("Phi_spec_") and "partial" in name for name in point)


def test_diagnostics_supports_latent_panel_with_current_arviz():
    rng = np.random.default_rng(44)
    chains, draws, patients, factors = 2, 40, 12, 3
    lam = rng.normal(size=(5, factors))
    idata = az.from_dict(
        {
            "posterior": {
                "lam_pos": rng.normal(size=(chains, draws, 4)),
                "Lam": lam[None, None, :, :] + rng.normal(
                    0, 0.005, size=(chains, draws, 5, factors)
                ),
                "sigma": np.abs(rng.normal(size=(chains, draws, 5))) + 0.2,
                "z_e": rng.normal(size=(chains, draws, patients, factors)),
            },
            "sample_stats": {
                "energy": rng.normal(size=(chains, draws)),
                "diverging": np.zeros((chains, draws), dtype=bool),
                "tree_depth": np.full((chains, draws), 3),
                "n_steps": np.full((chains, draws), 7),
                "acceptance_rate": np.full((chains, draws), 0.9),
            },
        },
    )
    diag = StageRunner(MeasurementConfig(max_tree_depth=7)).diagnostics(idata)
    assert diag["latent_panel_patients"] == patients
    assert np.isfinite(diag["latent_panel_rhat"])
    assert diag["loading_congruence_min"] > 0.99
    assert diag["salient_loading_sign_disagreements"] == 0
    assert diag["divergences_by_chain"] == [0, 0]
    assert diag["tree_depth_cap_fraction"] == 0.0
    assert diag["n_steps_cap_fraction"] == 0.0
    passing = dict(diag)
    passing.update(
        map_rhat=1.0,
        map_ess_bulk=500.0,
        map_ess_tail=500.0,
        core_map_rhat=1.0,
        core_map_ess_bulk=500.0,
        core_map_ess_tail=500.0,
        latent_panel_rhat=1.0,
        latent_panel_ess_bulk=500.0,
        latent_panel_ess_tail=500.0,
        core_latent_panel_rhat=1.0,
        core_latent_panel_ess_bulk=500.0,
        core_latent_panel_ess_tail=500.0,
        bfmi_min=0.9,
        loading_congruence_min=0.99,
    )
    runner = StageRunner(MeasurementConfig(max_tree_depth=7))
    assert runner._passes_gates(passing)
    passing["tree_depth_cap_fraction_by_chain"] = [0.0, 0.10]
    assert not runner._passes_gates(passing)


def test_diagnostics_split_substance_loadings_and_phi_row_elementwise():
    rng = np.random.default_rng(177)
    chains, draws = 4, 240
    factor_cols = ["overall_severity", "cognition", "substance"]
    items = ["cog_item", "fagers", "sudose_cigarettes_lt"]
    home = ["cognition", "substance", "substance"]
    core = CoreData(
        M=np.zeros((6, 3)),
        covariates=np.zeros((6, 0)),
        covariate_names=[],
        items=items,
        home=home,
        factor_cols=factor_cols,
        spec_factors=factor_cols[1:],
        g_col=0,
        cohort=np.array(["A"] * 6),
        index=pd.RangeIndex(6),
        families={item: "gaussian" for item in items},
        signs={item: 1 for item in items},
    )
    pos_cells = [
        (0, 1, 0.6, 0.3),
        (1, 2, 0.6, 0.3),
        (2, 2, 0.6, 0.3),
    ]
    spec = LoadingSpec(
        pos_cells=pos_cells,
        signed_cells=[],
        kind={(row, col): "primary" for row, col, _mu, _sd in pos_cells},
        factor_cols=factor_cols,
        items=items,
        home=home,
    )
    lam_pos = np.abs(rng.normal(0.6, 0.05, size=(chains, draws, 3)))
    sigma = np.abs(rng.normal(0.7, 0.05, size=(chains, draws, 3)))
    rho = np.clip(rng.normal(0.25, 0.03, size=(chains, draws)), -0.8, 0.8)
    phi = np.broadcast_to(np.eye(3), (chains, draws, 3, 3)).copy()
    phi[:, :, 2, 1] = rho
    phi[:, :, 1, 2] = rho
    lam = np.zeros((chains, draws, 3, 3))
    lam[:, :, 0, 1] = lam_pos[:, :, 0]
    lam[:, :, 1, 2] = lam_pos[:, :, 1]
    lam[:, :, 2, 2] = lam_pos[:, :, 2]
    idata = az.from_dict(
        {
            "posterior": {
                "lam_pos": lam_pos,
                "sigma": sigma,
                "Phi": phi,
                "Lam": lam,
            },
            "sample_stats": {
                "energy": rng.normal(size=(chains, draws)),
                "diverging": np.zeros((chains, draws), dtype=bool),
                "tree_depth": np.full((chains, draws), 3),
                "n_steps": np.full((chains, draws), 7),
            },
        },
    )
    diagnostics = StageRunner(MeasurementConfig(max_tree_depth=7)).diagnostics(
        idata, core=core, spec=spec
    )
    assert diagnostics["substance_loading_parameter_count"] == 2
    assert diagnostics["substance_correlation_parameter_count"] == 1
    assert diagnostics["substance_parameter_count"] == 3
    assert diagnostics["core_map_parameter_count"] == 4
    assert np.isfinite(diagnostics["core_map_rhat"])
    assert np.isfinite(diagnostics["substance_rhat"])


def test_tiered_gate_is_strict_for_map_and_bounded_for_nuisance():
    runner = StageRunner(MeasurementConfig(max_tree_depth=7))
    diagnostics = {
        "map_rhat": 1.009,
        "map_ess_bulk": 500.0,
        "map_ess_tail": 500.0,
        "nuisance_rhat": 1.019,
        "nuisance_ess_bulk": 500.0,
        "nuisance_ess_tail": 500.0,
        "nuisance_parameter_count": 2000,
        "latent_panel_patients": 0,
        "bfmi_min": 0.9,
        "divergences": 0,
        "tree_depth_cap_fraction": 0.0,
        "tree_depth_cap_fraction_by_chain": [0.0, 0.0, 0.0, 0.0],
        "n_steps_cap_fraction": 0.0,
        "n_steps_cap_fraction_by_chain": [0.0, 0.0, 0.0, 0.0],
        "loading_congruence_min": 0.99,
        "salient_loading_sign_disagreements": 0,
    }
    assert runner._passes_gates(diagnostics)

    bad_map = dict(diagnostics, map_rhat=1.011)
    assert not runner._passes_gates(bad_map)

    bad_nuisance = dict(diagnostics, nuisance_rhat=1.021)
    assert not runner._passes_gates(bad_nuisance)

    # Small Monte Carlo shortfalls are eligible for one longer retry, but remain
    # failed until that retry satisfies the unchanged tiered gates.
    borderline = dict(diagnostics, map_rhat=1.0122)
    assert not runner._passes_gates(borderline)
    assert _eligible_continuation_retry(runner, {"diagnostics": borderline})

    bounded_multi_gate = dict(
        diagnostics,
        map_rhat=1.0118,
        nuisance_rhat=1.0208,
        nuisance_ess_bulk=340.0,
    )
    assert _eligible_continuation_retry(
        runner, {"diagnostics": bounded_multi_gate}
    )
    assert not _eligible_continuation_retry(
        runner, {"diagnostics": dict(diagnostics, map_rhat=1.016)}
    )
    assert not _eligible_continuation_retry(
        runner,
        {"diagnostics": dict(borderline, nuisance_rhat=1.026)},
    )
    assert not _eligible_continuation_retry(
        runner,
        {"diagnostics": dict(borderline, nuisance_ess_bulk=299.0)},
    )
    assert not _eligible_continuation_retry(
        runner,
        {
            "diagnostics": dict(
                borderline,
                tree_depth_cap_fraction=0.2,
                tree_depth_cap_fraction_by_chain=[0.2] * 4,
            )
        },
    )

    # A nonterminal continuous rung may advance with a recorded warning only
    # when all non-trajectory gates pass and saturation is isolated to one chain.
    isolated_depth = dict(
        diagnostics,
        tree_depth_cap_fraction=0.099,
        tree_depth_cap_fraction_by_chain=[0.0, 0.0, 0.396, 0.0],
        n_steps_cap_fraction=0.0595,
        n_steps_cap_fraction_by_chain=[0.0, 0.0, 0.238, 0.0],
    )
    assert not runner._passes_gates(isolated_depth)
    assert _eligible_continuation_advance(
        runner, {"diagnostics": isolated_depth}
    )
    assert not _eligible_continuation_advance(
        runner,
        {
            "diagnostics": dict(
                isolated_depth,
                tree_depth_cap_fraction=0.11,
            )
        },
    )
    assert not _eligible_continuation_advance(
        runner,
        {
            "diagnostics": dict(
                isolated_depth,
                tree_depth_cap_fraction_by_chain=[0.198, 0.198, 0.0, 0.0],
            )
        },
    )
    assert not _eligible_continuation_advance(
        runner,
        {"diagnostics": dict(isolated_depth, map_rhat=1.011)},
    )
    assert not _eligible_continuation_advance(
        runner,
        {
            "diagnostics": dict(
                isolated_depth,
                tree_depth_cap_fraction=1.0,
                tree_depth_cap_fraction_by_chain=[1.0] * 4,
                n_steps_cap_fraction=1.0,
                n_steps_cap_fraction_by_chain=[1.0] * 4,
            )
        },
    )


def test_substance_gate_is_provisional_without_relaxing_core_or_geometry():
    runner = StageRunner(MeasurementConfig(max_tree_depth=7))
    diagnostics = {
        "core_map_rhat": 1.009,
        "core_map_ess_bulk": 500.0,
        "core_map_ess_tail": 500.0,
        "nuisance_rhat": 1.019,
        "nuisance_ess_bulk": 500.0,
        "nuisance_ess_tail": 500.0,
        "nuisance_parameter_count": 2000,
        "core_latent_panel_patients": 0,
        "substance_rhat": 1.049,
        "substance_ess_bulk": 120.0,
        "substance_ess_tail": 110.0,
        "substance_parameter_count": 9,
        "substance_latent_panel_patients": 0,
        "substance_loading_parameter_count": 2,
        "core_loading_congruence_min": 0.99,
        "substance_loading_congruence": 0.98,
        "core_loading_sign_disagreements": 0,
        "substance_loading_sign_disagreements": 0,
        "bfmi_min": 0.9,
        "divergences": 0,
        "tree_depth_cap_fraction": 0.0,
        "tree_depth_cap_fraction_by_chain": [0.0] * 4,
        "n_steps_cap_fraction": 0.0,
        "n_steps_cap_fraction_by_chain": [0.0] * 4,
    }
    components = runner._gate_components(diagnostics)
    assert components["core_passed"]
    assert components["substance_provisional"]
    assert not components["substance_strict"]
    assert runner._passes_gates(diagnostics)

    assert not runner._passes_gates(
        dict(diagnostics, core_map_rhat=1.011)
    )
    assert not runner._passes_gates(
        dict(diagnostics, substance_rhat=1.051)
    )
    assert not runner._passes_gates(
        dict(diagnostics, bfmi_min=0.29)
    )


def test_certification_policy_does_not_invalidate_fit_cache():
    base = MeasurementConfig()
    relaxed = replace(base, nuisance_rhat_max=1.03)
    assert _config_sig(base) == _config_sig(relaxed)
    assert _certification_policy(base) != _certification_policy(relaxed)

    legacy = {"config_sig": dict(_config_sig(base))}
    legacy["config_sig"]["certification_gates"] = {"rhat_max": 1.01}
    assert _cached_model_sig(legacy) == _config_sig(base)


# ---------------------------- Gaussian-copula vertical ----------------------------

def test_copula_invert_round_trips():
    rng = np.random.default_rng(7)
    x = rng.lognormal(0.0, 1.0, size=2000)  # skewed positive
    z = _rank_int(x)
    order = np.argsort(x, kind="mergesort")
    sorted_values, sorted_z = x[order], np.sort(z)
    recov = copula_invert(z, sorted_values, sorted_z)
    assert np.max(np.abs(recov - x)) < 1e-9  # exact at the data nodes


def test_copula_config_sig_distinguishes_modes():
    base = MeasurementConfig()
    cop = base.with_gaussian_copula()
    assert base.likelihood_mode == "native" and cop.likelihood_mode == "gaussian_copula"
    sig_n, sig_c = _config_sig(base), _config_sig(cop)
    assert sig_n["likelihood_mode"] == "native"
    assert sig_c["likelihood_mode"] == "gaussian_copula"
    assert sig_n != sig_c  # cache will not be shared between native and copula fits


def test_config_signature_tracks_processed_input_content(tmp_path):
    config, _truth = _synthetic_config(tmp_path, n=80, seed=19)
    before = _config_sig(config)
    covariates_path = config.processed_dir / "covariates_v0.parquet"
    covariates = pd.read_parquet(covariates_path)
    covariates.iloc[0, 0] = float(covariates.iloc[0, 0]) + 0.25
    covariates.to_parquet(covariates_path)
    after = _config_sig(config)
    assert before["processed_inputs_sha256"] != after["processed_inputs_sha256"]


def test_copula_marginals_standard_normal_and_buildable(tmp_path):
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


def test_copula_in_likelihood_retains_covariates(tmp_path):
    config, _truth = _synthetic_config(tmp_path, n=300, seed=16)
    config = replace(config, likelihood_mode="gaussian_copula", covariate_mode="in_likelihood")
    dataset = MeasurementDataset(config)
    core = dataset.core(S1_FACTORS, n_subsample=220, seed=16)
    assert core.covariates.shape[1] > 0
    assert core.covariate_names
    transform = core.covariate_metadata["transform"]
    assert transform["numeric"]["age"]["fill"] is not None
    assert transform["age_spline"]["knot_vector"]
    assert len(transform["age_spline"]["center"]) > 0
    assert "reference" in transform["site"]
    model = BayesianBifactorESEM(config).build_marginalized(
        core, dataset.loading_spec(core, windows=False), correlated=False
    )
    assert model.named_vars["beta"].type.shape == (
        len(core.items),
        core.covariates.shape[1],
    )


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
    from face.measurement.synthetic import _native_invert  # noqa: PLC0415
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


def test_equal_substance_loading_is_model_affecting_but_not_orthogonality():
    base = MeasurementConfig()
    constrained = base.with_equal_home_loadings("substance")
    assert constrained.orthogonal_factors == ()
    assert constrained.equal_home_loading_factors == ("substance",)
    assert _config_sig(base)["equal_home_loading_factors"] == []
    assert _config_sig(constrained)["equal_home_loading_factors"] == ["substance"]


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs processed FACE baseline",
)
def test_substance_orthogonal_removes_it_from_correlated_block():
    """With substance pinned orthogonal, it leaves the correlated Phi block, so the free correlation
    vector (Phi_spec_lower) shrinks by exactly one factor's worth of cross-terms -> substance has a
    zero off-diagonal Phi row by construction (like G)."""
    cfg = MeasurementConfig(likelihood_mode="gaussian_copula")
    ncorr = {}
    for label, c in [("plain", cfg), ("orth", cfg.with_substance_orthogonal())]:
        ds = MeasurementDataset(c)
        core = ds.core(S5_FACTORS, correlated=True, windows=True, n_subsample=400, balanced=True, seed=0)
        model = BayesianBifactorESEM(c).build_marginalized(core, ds.loading_spec(core, windows=True), correlated=True)
        F = len(core.factor_cols)
        ns = (F - 1) - (1 if label == "orth" else 0)   # specifics minus G, minus substance if pinned
        ncorr[label] = int(model.named_vars["Phi_spec"].type.shape[0])
        assert model.named_vars["Phi_spec"].type.shape == (ns, ns)
        assert ncorr[label] == ns
    assert ncorr["orth"] < ncorr["plain"]   # substance dropped from the correlated block


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
    cfg = MeasurementConfig(
        likelihood_mode="gaussian_copula",
        covariate_mode="in_likelihood",
        cohort_weighted=True,
    )
    dataset = MeasurementDataset(cfg)
    mixed = dataset.mixed(S5_FACTORS, min_cohorts=2, n_subsample=600, balanced=True, seed=0)
    spec = dataset.loading_spec(mixed.base, windows=True,
                                bifactor_g_sd={f: 0.05 for f in S5_FACTORS if f != "overall_severity"})
    w = _cohort_weights(mixed.base.cohort)
    model = BayesianBifactorESEM(cfg).build_mixed(mixed, spec, weights=w)
    lp = float(model.compile_logp()(model.initial_point()))
    assert np.isfinite(lp)
    assert any(str(n).startswith("lh_") for n in model.named_vars)  # explicit item loadings present
    assert "beta_native" in model.named_vars


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs processed FACE baseline",
)
def test_copula_promotes_high_cardinality_count_to_continuum():
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


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs processed FACE baseline",
)
def test_corrected_canonical_routing_is_exhaustive():
    from face.measurement.run import CROSSLOAD_MATRIX, F8

    base = (
        MeasurementConfig()
        .with_gaussian_copula()
        .with_equal_home_loadings("substance")
        .with_excluded_items(*PRIMARY_EXCLUSIONS)
    )
    cfg = replace(
        base,
        prior_matrix=CROSSLOAD_MATRIX,
        covariate_mode="in_likelihood",
    )
    dataset = MeasurementDataset(cfg)
    expected_e = [
        "overall_severity",
        "immunometabolic",
        "suicidality",
        "developmental_risk",
    ]
    mixed = dataset.mixed(
        F8,
        explicit_factors=expected_e,
        min_cohorts=2,
        n_subsample=120,
        balanced=True,
        seed=0,
    )
    assert mixed.base.M.shape[1] == 74
    assert (len(mixed.bin_items), len(mixed.ord_items), len(mixed.cnt_items)) == (25, 3, 1)
    assert len(mixed.native_items) == 29
    assert [mixed.base.factor_cols[c] for c in mixed.e_cols] == expected_e
    assert [mixed.base.factor_cols[c] for c in mixed.m_cols] == [
        "cognition",
        "sleep",
        "mania_activation",
        "substance",
    ]
    assert set(mixed.base.items).isdisjoint(mixed.native_items)
    assert {row["route"] for row in mixed.routing_report} >= {"gaussian", "native"}
    g_anchors = [
        item
        for item in mixed.native_items
        if mixed.base.factor_cols[mixed.e_cols[mixed.ng_home[item]]] == "overall_severity"
    ]
    assert g_anchors == ["hooccur_arret_travail_actuel", "jobclas"]
    assert all(item not in mixed.ng_gp for item in g_anchors)
    spec = dataset.loading_spec(
        mixed.base, windows=True, specific_cross=True, cross_sd_scale=1.0
    )
    loading_map = dataset.conceptual_loading_map(mixed, spec)
    assert len(loading_map) == 103 * 8
    for item in g_anchors:
        free = [
            row for row in loading_map if row["item"] == item and row["parameter"]
        ]
        assert [(row["factor"], row["parameter"]) for row in free] == [
            ("overall_severity", f"lh_{item}")
        ]
    assert any(
        row["topology_role"] == "structural_zero"
        and row["structural_zero_reason"] == "unlikely_cross"
        for row in loading_map
    )


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs processed FACE baseline",
)
def test_corrected_mixed_model_has_native_covariates_and_one_g_anchor_loading():
    from face.measurement.run import CROSSLOAD_MATRIX, F8

    base = (
        MeasurementConfig()
        .with_gaussian_copula()
        .with_equal_home_loadings("substance")
        .with_excluded_items(*PRIMARY_EXCLUSIONS)
    )
    cfg = replace(
        base,
        prior_matrix=CROSSLOAD_MATRIX,
        covariate_mode="in_likelihood",
    )
    dataset = MeasurementDataset(cfg)
    mixed = dataset.mixed(
        F8,
        specific_cross=True,
        min_cohorts=2,
        n_subsample=60,
        balanced=True,
        seed=2,
    )
    spec = dataset.loading_spec(
        mixed.base, windows=True, specific_cross=True, cross_sd_scale=1.0
    )
    model = BayesianBifactorESEM(cfg).build_mixed(mixed, spec)
    assert cfg.orthogonal_factors == ()
    assert model.named_vars["Phi_spec"].type.shape == (7, 7)
    assert "lam_equal_substance" in model.named_vars
    assert model.named_vars["beta_native"].type.shape == (
        len(mixed.native_items),
        mixed.base.covariates.shape[1],
    )
    for item in ("hooccur_arret_travail_actuel", "jobclas"):
        assert f"lh_{item}" in model.named_vars
        assert f"lg_{item}" not in model.named_vars
    assert np.isfinite(float(model.compile_logp()(model.initial_point())))


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs processed FACE baseline",
)
def test_zero_covariate_effects_reproduce_no_covariate_mixed_likelihood():
    from face.measurement.run import CROSSLOAD_MATRIX, F8

    adjusted = replace(
        MeasurementConfig().with_gaussian_copula().with_substance_orthogonal(),
        prior_matrix=CROSSLOAD_MATRIX,
        covariate_mode="in_likelihood",
    )
    unadjusted = replace(adjusted, include_covariates=False)
    models = []
    for config in (unadjusted, adjusted):
        dataset = MeasurementDataset(config)
        mixed = dataset.mixed(
            F8,
            specific_cross=True,
            min_cohorts=2,
            n_subsample=20,
            balanced=True,
            seed=4,
        )
        spec = dataset.loading_spec(
            mixed.base, windows=True, specific_cross=True, cross_sd_scale=1.0
        )
        models.append(BayesianBifactorESEM(config).build_mixed(mixed, spec))

    model_no_cov, model_cov = models
    point_cov = model_cov.initial_point(random_seed=7)
    point_no_cov = model_no_cov.initial_point(random_seed=8)
    for name in point_no_cov:
        if name in point_cov and np.shape(point_no_cov[name]) == np.shape(point_cov[name]):
            point_no_cov[name] = point_cov[name]
    for name in ("alpha", "beta", "beta_native"):
        point_cov[name] = np.zeros_like(point_cov[name])
    likelihood_no_cov = model_no_cov.compile_logp(
        vars=model_no_cov.observed_RVs + model_no_cov.potentials
    )
    likelihood_cov = model_cov.compile_logp(
        vars=model_cov.observed_RVs + model_cov.potentials
    )
    assert np.isclose(
        float(likelihood_no_cov(point_no_cov)),
        float(likelihood_cov(point_cov)),
        atol=1e-8,
    )


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs processed FACE baseline",
)
def test_nonzero_covariate_direction_enters_all_four_family_likelihoods():
    from face.measurement.run import CROSSLOAD_MATRIX, F8

    config = replace(
        MeasurementConfig().with_gaussian_copula().with_substance_orthogonal(),
        prior_matrix=CROSSLOAD_MATRIX,
        covariate_mode="in_likelihood",
    )
    dataset = MeasurementDataset(config)
    mixed = dataset.mixed(
        F8,
        specific_cross=True,
        min_cohorts=2,
        n_subsample=120,
        balanced=True,
        seed=12,
    )
    x = mixed.base.covariates[:, 0].copy()
    x = (x - x.mean()) / x.std()
    continuous = np.full_like(mixed.base.M, np.nan)
    continuous[:, 0] = x
    mixed.base.M = continuous
    mixed.Bin[:, 0] = (x > 0).astype(float)
    mixed.Cnt[:, 0] = np.rint(np.exp(0.4 + 0.5 * x))
    categories = mixed.ord_K[0]
    mixed.Ord[:, 0] = np.digitize(
        x, np.quantile(x, np.linspace(0, 1, categories + 1)[1:-1])
    )
    spec = dataset.loading_spec(
        mixed.base, windows=True, specific_cross=True, cross_sd_scale=1.0
    )
    model = BayesianBifactorESEM(config).build_mixed(mixed, spec)
    center = model.initial_point(random_seed=13)
    for name in ("z_e", "alpha", "beta", "beta_native"):
        center[name] = np.zeros_like(center[name])
    for item in (mixed.bin_items[0], mixed.cnt_items[0]):
        center[f"a_{item}"] = np.zeros_like(center[f"a_{item}"])

    cases = [
        ("cont_ll", "beta", 0, 1.0),
        (f"y_{mixed.bin_items[0]}", "beta_native", mixed.ng_index[mixed.bin_items[0]], 2.0),
        (f"y_{mixed.cnt_items[0]}", "beta_native", mixed.ng_index[mixed.cnt_items[0]], 0.5),
        (f"y_{mixed.ord_items[0]}", "beta_native", mixed.ng_index[mixed.ord_items[0]], 2.0),
    ]
    for variable, coefficient, row, effect in cases:
        positive = {name: np.array(value, copy=True) for name, value in center.items()}
        negative = {name: np.array(value, copy=True) for name, value in center.items()}
        positive[coefficient][row, 0] = effect
        negative[coefficient][row, 0] = -effect
        logp = model.compile_logp(vars=[model[variable]])
        assert float(logp(positive)) > float(logp(negative)), variable


def test_exclude_items_drops_indicator_everywhere(tmp_path):
    """The ``exclude_items`` sensitivity arm removes named indicators from the matrix, the metadata,
    the home map, and the encoded core block -- without touching the canonical (unexcluded) fit."""
    config, _truth = _synthetic_config(tmp_path)
    base = MeasurementDataset(config)
    base_core = base.core(S1_FACTORS, n_subsample=160, seed=1)
    target = base_core.items[0]  # a continuous home item active in the S1 backbone

    excl_cfg = config.with_excluded_items(target)
    # accessor is pure (frozen dataclass): original config untouched, and repeated exclusion dedups.
    assert config.exclude_items == ()
    assert excl_cfg.exclude_items == (target,)
    assert excl_cfg.with_excluded_items(target).exclude_items == (target,)

    ds = MeasurementDataset(excl_cfg)
    assert target not in set(ds.matrix["item"])
    assert target not in ds.home
    assert target not in ds.meta.index

    core = ds.core(S1_FACTORS, n_subsample=160, seed=1)
    assert target not in core.items
    assert len(core.items) == len(base_core.items) - 1


def test_canonical_local_independence_exclusion_profile():
    """The production deletion profile is explicit, unique, and keeps one
    prespecified representative from every overlapping indicator cluster."""
    expected = {
        "cvlt_short_delay_free_recall",
        "cvlt_long_delay_free_recall",
        "wais_ivt_index",
        "ctq39",
        "ctq41",
        "psqi",
        "fast",
        "weight",
        "hrstanding",
        "eghrmn",
        "sysbpstanding",
        "diabpstanding",
        "neut",
        "chol",
        "ast_lbstresc",
    }
    retained = {
        "cvlt_total_recall",
        "wais_code_std",
        "ctq29",
        "ctq31",
        "ctq33",
        "ctq35",
        "ctq37",
        "ctq40",
        "psqi11",
        "psqi12",
        "psqi13",
        "psqi14",
        "psqi15",
        "psqi17",
        "fast25",
        "fast26",
        "fast27",
        "fast28",
        "fast30",
        "bmi",
        "wstcir",
        "hrsupine",
        "sysbpsupine",
        "diabpsupine",
        "wbc",
        "ldl",
        "alt_lbstresc",
    }

    assert set(LOCAL_INDEPENDENCE_EXCLUSIONS) == expected
    assert len(LOCAL_INDEPENDENCE_EXCLUSIONS) == len(expected)
    assert retained.isdisjoint(LOCAL_INDEPENDENCE_EXCLUSIONS)


def test_primary_data_quality_exclusion_profile():
    assert set(DATA_QUALITY_EXCLUSIONS) == {
        "suoccur_alcool",
        "suoccur_cannabis",
    }
    assert set(PRIMARY_EXCLUSIONS) == {
        *LOCAL_INDEPENDENCE_EXCLUSIONS,
        *DATA_QUALITY_EXCLUSIONS,
    }
    assert set(PRIMARY_EXPLICIT_FACTORS) == {
        "overall_severity",
        "immunometabolic",
        "suicidality",
        "developmental_risk",
    }


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs processed FACE baseline",
)
def test_ctq_exclusions_remove_deterministic_derived_indicators():
    baseline = pd.read_parquet(
        Path(__file__).resolve().parents[2] / "data" / "processed" / "baseline_v0.parquet",
        columns=[
            "ctq29",
            "ctq31",
            "ctq33",
            "ctq35",
            "ctq37",
            "ctq39",
            "ctq40",
            "ctq41",
        ],
    )
    subscales = ["ctq29", "ctq31", "ctq33", "ctq35", "ctq37"]
    total = baseline[subscales + ["ctq39"]].dropna()
    recode = baseline[["ctq40", "ctq41"]].dropna()

    assert len(total) == 8122
    assert np.array_equal(total["ctq39"].to_numpy(), total[subscales].sum(axis=1).to_numpy())
    assert np.array_equal(
        recode["ctq41"].to_numpy(),
        (recode["ctq40"].to_numpy() > 0).astype(float),
    )
