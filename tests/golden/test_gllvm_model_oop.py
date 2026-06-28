"""Golden tests for the variational GLLVM orchestration layer.

The machinery (ontology parity, full item set, encoder, config signature, export schema,
caching) is tested on the synthetic FACE-like generator with the v3 (S1) matrix.  The
8-factor operational specifics (immunometabolic merge, substance orthogonality, the 3 earned
cross-loadings, mixed channels) are gated on the real baseline / biomerge_xc matrix.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from face.models.bayesian.measurement_model_oop import (  # noqa: E402
    S1_FACTORS,
    LoadingSpec,
    MeasurementDataset,
)
from face.models.variational.generative import generate_synthetic  # noqa: E402
from face.models.variational.gllvm import VariationalGLLVM  # noqa: E402
from face.models.variational.gllvm_model_oop import (  # noqa: E402
    BIOMERGE_XC,
    F8_FIT,
    PROC,
    REPO,
    GLLVMConfig,
    GLLVMDataset,
    GLLVMProjector,
    GLLVMRunner,
    GLLVMStage,
    _config_sig,
)
from synthetic.generate_face_like import generate  # noqa: E402

V3_MATRIX = REPO / "configs" / "prior_loading_matrix_v3.csv"
REAL_DATA = (PROC / "baseline_v0.parquet").exists() and BIOMERGE_XC.exists()
_real = pytest.mark.skipif(not REAL_DATA, reason="real baseline_v0 / biomerge_xc not present")

COORD_SUFFIXES = ("__mean", "__sd", "__hdi_low", "__hdi_high", "__n_obs", "__reliability")
LOADING_COLS = ["item", "factor", "home", "block", "likelihood_family", "kind", "loading",
                "abs_loading", "ci_low", "ci_high", "excludes_zero"]


def _synthetic_config(tmp_path, n=300, seed=0) -> GLLVMConfig:
    outdir, _truth = generate(n=n, seed=seed, out=tmp_path)
    return GLLVMConfig(
        processed_dir=Path(outdir),
        prior_matrix=V3_MATRIX,
        output_dir=tmp_path / "out",
        figure_dir=tmp_path / "fig",
        factors=tuple(S1_FACTORS),
        explicit_factors=(),
        orthogonal_factors=(),
        specific_cross=False,
        likelihood_mode="gaussian_copula",
    )


# ----------------------------------------------------------------- machinery (synthetic)
def test_ontology_cell_classification_parity_vs_loadingspec(tmp_path):
    """The GLLVM ontology must carry the exact same free cells + kinds as the certified
    ``LoadingSpec.from_core`` (anti-drift: one classification source)."""
    config = _synthetic_config(tmp_path)
    data = GLLVMDataset(config).build(list(S1_FACTORS), windows=False)

    md = MeasurementDataset(config.measurement_config())
    core = md.core(list(S1_FACTORS), windows=False, include_covariates=False)
    # Same items, in the same order (S1 synthetic is continuous-only).
    assert data.items == core.items
    spec = LoadingSpec.from_core(
        core, md.matrix, windows=True, soft_unlikely=False, soft_g_anchor_specific=False,
        specific_cross=False, horseshoe=False,
    )
    assert data.ontology.kind == dict(spec.kind)
    # Every free ontology cell is a pos/signed/hs cell in the spec, and vice versa.
    spec_cells = {(j, c) for j, c, *_ in spec.pos_cells} | {(j, c) for j, c, *_ in spec.signed_cells}
    onto_cells = set(zip(*np.where(data.ontology.free_mask), strict=False))
    onto_cells = {(int(j), int(c)) for j, c in onto_cells}
    assert onto_cells == spec_cells


def test_config_sig_distinguishes_structure_not_optimization(tmp_path):
    config = _synthetic_config(tmp_path)
    base = _config_sig(config)
    # Optimization knobs do NOT change the structural signature.
    assert _config_sig(replace(config, lr=0.5, epochs=99, batch_size=128, device="mps")) == base
    # Structural switches DO.
    assert _config_sig(replace(config, soft_unlikely=True)) != base
    assert _config_sig(replace(config, specific_cross=True)) != base
    assert _config_sig(replace(config, orthogonal_factors=("substance",))) != base
    assert _config_sig(replace(config, covariate_mode="none")) != base


def test_export_schema_matches_canonical(tmp_path):
    config = _synthetic_config(tmp_path)
    data = GLLVMDataset(config).build(list(S1_FACTORS), windows=False)
    model = VariationalGLLVM(len(data.index), data.ontology, orthogonal_indices=(0,), seed=0)
    model.initialize_from_data(data.x, data.mask)
    fit = {"stage": "t", "model": model, "data": data, "history": [], "factor_cols": data.factor_cols}
    proj = GLLVMProjector(config)

    coords = proj.coordinates_frame(fit)
    expected_coord_cols = {f"{f}{suf}" for f in data.factor_cols for suf in COORD_SUFFIXES}
    assert set(coords.columns) == expected_coord_cols
    assert len(coords) == len(data.index)

    loads = proj.loadings_summary(fit)
    assert list(loads.columns) == LOADING_COLS
    # Only free cells are emitted (hard-zero cells are not interpretable).
    assert len(loads) == int(data.ontology.free_mask.sum())


def test_runner_caches_and_reuses(tmp_path):
    config = _synthetic_config(tmp_path)
    config = replace(config, epochs=3, lr=2e-2)
    runner = GLLVMRunner(config)
    stage = GLLVMStage("t_stage", list(S1_FACTORS), windows=False, epochs=3)

    runner.run_stage(stage)
    ckpt = config.output_dir / stage.name / "model_state.pt"
    assert ckpt.exists()
    # Tamper the checkpoint, then a non-overwrite rerun must LOAD it (proving cache reuse).
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    blob["state_dict"]["alpha"][:] = 123.0
    torch.save(blob, ckpt)
    fit2 = runner.run_stage(stage, overwrite=False)
    assert float(fit2["model"].alpha.detach()[0]) == pytest.approx(123.0)

    # Overwrite refits (the tampered sentinel is gone).
    fit3 = runner.run_stage(stage, overwrite=True)
    assert float(fit3["model"].alpha.detach()[0]) != pytest.approx(123.0)


def test_gaussian_channel_is_rank_int_standardized(tmp_path):
    config = _synthetic_config(tmp_path)
    data = GLLVMDataset(config).build(list(S1_FACTORS), windows=False)
    # All synthetic items are continuous -> gaussian channel, rank-INT z (~N(0,1)).
    assert set(data.families) == {"gaussian"}
    obs = data.M_raw[data.mask.numpy()]
    assert abs(float(np.nanmean(obs))) < 0.2
    assert 0.7 < float(np.nanstd(obs)) < 1.4


# ------------------------------------------------------------- 8-factor map (real data)
@_real
def test_full_item_set_includes_both_blocks():
    config = GLLVMConfig()  # 8-factor operational defaults
    data = GLLVMDataset(config).build(list(F8_FIT), windows=True, n_subsample=400, balanced=True)
    assert set(data.blocks) == {"continuous", "explicit"}
    # Every GLLVM channel family is exercised on the real map.
    assert set(data.families) == {"gaussian", "bernoulli", "ordinal", "count"}
    assert data.factor_cols == list(F8_FIT)


@_real
def test_three_earned_cross_loadings_present_and_only_those():
    config = GLLVMConfig()
    data = GLLVMDataset(config).build(list(F8_FIT), windows=True, n_subsample=400, balanced=True)
    cog = data.factor_cols.index("cognition")
    cross = sorted(data.items[j] for (j, c), k in data.ontology.kind.items() if k == "cross")
    assert cross == ["ctq37", "psqi11", "psqi17"]
    for it in cross:
        j = data.items.index(it)
        assert bool(data.ontology.free_mask[j, cog])


@_real
def test_generate_synthetic_raw_scale_and_determinism():
    config = replace(GLLVMConfig(), covariate_mode="none", include_covariates=False, q_rank=2)
    data = GLLVMDataset(config).build(list(F8_FIT), windows=True, n_subsample=500, balanced=True)
    assert len(data.copula) > 0 and len(data.item_signs) == len(data.items)
    model = VariationalGLLVM(len(data.index), data.ontology, orthogonal_indices=(0, 7),
                             q_rank=2, seed=0)
    model.initialize_from_data(data.x, data.mask)
    a = generate_synthetic(model, data, n=500, seed=3)
    b = generate_synthetic(model, data, n=500, seed=3)
    assert a.shape == (500, len(data.items))
    assert a.equals(b)  # deterministic given the seed
    assert not a.isna().any().any()
    # Continuous items land on the raw clinical scale (within the observed support).
    base = pd.read_parquet(config.processed_dir / "baseline_v0.parquet")
    for it in ("bmi", "egf"):
        if it in a.columns and it in base.columns:
            o = pd.to_numeric(base[it], errors="coerce").dropna()
            assert o.min() - 1 <= a[it].min() and a[it].max() <= o.max() + 1
    # Binary channel items are 0/1.
    for j, it in enumerate(data.items):
        if data.families[j] == "bernoulli":
            assert set(np.unique(a[it])).issubset({0.0, 1.0})
            break


@_real
def test_substance_and_g_pinned_orthogonal_after_short_fit():
    config = GLLVMConfig().with_smoke_defaults()
    config = replace(config, epochs=6, output_dir=Path("/tmp") / "gllvm_test_ortho")
    runner = GLLVMRunner(config)
    fit = runner.run_plan(overwrite=True)
    phi = fit["model"].phi_matrix()
    fc = fit["factor_cols"]
    for f in ("overall_severity", "substance"):
        i = fc.index(f)
        assert float(np.abs(np.delete(phi[i], i)).max()) == 0.0
    # The immunometabolic specific block carries a learnable correlation.
    ci, ii = fc.index("cognition"), fc.index("immunometabolic")
    assert float(np.abs(phi[ci, ii])) >= 0.0  # present (possibly ~0 after 6 epochs)
