"""Golden tests for the OOP soft-region strata engine (synthetic arrays — no heavy map needed)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from face.strata.engine import (
    CANON,
    ArchetypeModel,
    CoordinateSet,
    SoftRegionModel,
    StrataConfig,
    StrataProjector,
    StrataRunner,
    StructureGate,
    _confidence_tier,
    _config_sig,
    _normalized_entropy,
)
from face.strata.validation import assignment_usefulness, choose_K_operational


def _coords(N=600, K=3, *, seed=0, spread=5.0, X=None):
    """A synthetic CoordinateSet: K Gaussian blobs in 9-D with small known measurement error."""
    rng = np.random.default_rng(seed)
    D = len(CANON)
    if X is None:
        centers = rng.normal(0, spread, (K, D))
        lab = rng.integers(0, K, N)
        X = centers[lab] + rng.normal(0, 0.5, (N, D))
    else:
        centers, lab = None, None
    S = np.full((N, D), 0.25)
    draws = (X[None] + rng.normal(0, 0.3, (20, N, D))).astype("float32")
    cohort = rng.choice(["bp", "sz", "dr"], N)
    idx = pd.MultiIndex.from_arrays([cohort, [f"p{i}" for i in range(N)]], names=["cohort", "patient_id"])
    val = pd.DataFrame({"arm": rng.choice(["a", "b", "c"], N), "age": rng.normal(40, 10, N)}, index=idx)
    cs = CoordinateSet(X=X, S=S, draws=draws, dims=list(CANON), index=idx,
                       n_obs=np.full((N, D), 3), reliability=np.full((N, D), "well"), validation=val)
    return cs, lab, centers


def test_config_sig_and_factories():
    base = StrataConfig()
    assert _config_sig(base) == _config_sig(StrataConfig())          # stable
    assert base.with_full_si().use_full_Si and not base.use_full_Si
    assert _config_sig(base.with_full_si()) != _config_sig(base)     # full-Si changes the key
    assert _config_sig(base.with_smoke_defaults())["smoke"] is True


def test_entropy_and_confidence_tier():
    onehot = np.eye(3)[[0, 1, 2]]
    assert np.allclose(_normalized_entropy(onehot), 0.0)             # a single dominant region
    assert np.allclose(_normalized_entropy(np.full((2, 3), 1 / 3)), 1.0)  # maximally on-the-fence
    resp = np.array([[0.9, 0.05, 0.05], [0.6, 0.2, 0.2], [0.34, 0.33, 0.33]])
    assert list(_confidence_tier(resp, (0.5, 0.8))) == ["core", "soft", "boundary"]


def test_soft_region_recovers_centroids_and_tiers():
    coords, _, centers = _coords(N=600, K=3, spread=6.0, seed=1)
    rf = SoftRegionModel(StrataConfig()).fit(coords, 3, arm="A")
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(((centers[:, None] - rf.mu[None]) ** 2).sum(-1))
    assert np.allclose(centers[r], rf.mu[c], atol=1.2)               # deconvolved centroids ~ truth
    assert rf.resp.shape == (600, 3) and rf.tier.shape == (600,)
    assert set(np.unique(rf.tier)) <= {"core", "soft", "boundary"}


def test_archetype_weights_simplex_and_uncertainty():
    rng = np.random.default_rng(2)
    D = len(CANON)
    Z = rng.normal(0, 3, (3, D))
    X = rng.dirichlet([0.4] * 3, 400) @ Z                            # genuine simplex blends
    coords, _, _ = _coords(N=400, seed=2, X=X)
    af = ArchetypeModel(StrataConfig()).fit(coords, 3, arm="A", n_draw=6)
    assert af.W.shape == (400, 3)
    assert np.allclose(af.W.sum(1), 1.0, atol=1e-3)                  # simplex membership
    assert np.isfinite(af.W_sd).all() and af.explained_variance > 0.8


def test_structure_gate_distinguishes_clusters_from_continuum():
    blobs, _, _ = _coords(N=400, K=3, spread=8.0, seed=3)
    uniform, _, _ = _coords(N=400, X=np.random.default_rng(3).uniform(-3, 3, (400, len(CANON))))
    g = StructureGate(StrataConfig())
    sb = g.battery(blobs, arm="A")["verdict"]["clustered_score"]
    su = g.battery(uniform, arm="A")["verdict"]["clustered_score"]
    assert sb > su                                                  # blobs carry more clustered signal


def test_null_comparison_separation():
    """The single-Gaussian null test flags genuine separation (well-separated blobs) and clears a
    structureless Gaussian (apparent structure = continuum)."""
    g = StructureGate(StrataConfig())
    blobs, _, _ = _coords(N=400, K=3, spread=12.0, seed=8)
    assert g.null_comparison(blobs, n_null=4, Ks=range(2, 5))["z"]["best_silhouette"] > 2.0
    gauss, _, _ = _coords(N=400, X=np.random.default_rng(8).multivariate_normal(
        np.zeros(len(CANON)), np.eye(len(CANON)), 400))
    assert g.null_comparison(gauss, n_null=4, Ks=range(2, 5))["z"]["best_silhouette"] < 2.0


def test_assignment_usefulness_gate():
    rng = np.random.default_rng(0)
    conf = np.eye(3)[rng.integers(0, 3, 500)] * 0.9 + 0.05
    conf /= conf.sum(1, keepdims=True)
    assert assignment_usefulness(conf)["gate"] == "PASS"
    mushy = np.clip(np.full((500, 3), 1 / 3) + rng.normal(0, 0.01, (500, 3)), 1e-6, None)
    mushy /= mushy.sum(1, keepdims=True)
    assert assignment_usefulness(mushy)["gate"] == "FAIL"           # the mushy-middle is not useful


def test_choose_K_operational_picks_confident_stable_K():
    coords, _, _ = _coords(N=500, K=3, spread=6.0, seed=4)
    X, S, _ = coords.arm("A")
    ck = choose_K_operational(X, S, Ks=range(2, 5), seeds=(1, 2))
    assert ck["chosen_K"] in (2, 3, 4)
    assert all({"K", "bic", "confident_dominant_frac", "seed_ari"} <= set(r) for r in ck["sweep"])


def test_membership_frame_m3_contract():
    coords, _, _ = _coords(N=300, K=3, seed=5)
    rf = SoftRegionModel().fit(coords, 3, arm="A")
    af = ArchetypeModel().fit(coords, 3, arm="A", n_draw=5)
    frame = StrataProjector().membership_frame(coords, af, rf)
    assert {"cohort", "patient_id", "arm"} <= set(frame.columns)
    assert {"arch_dominant", "arch_dominant_name", "arch_entropy", "tess_MAP", "tess_entropy"} <= set(frame.columns)
    glob = [c for c in frame.columns if c.startswith(("arch_", "tess_"))]   # prognosis.frame's selector
    assert any(c.startswith("arch_w") for c in glob) and any(c.startswith("tess_r") for c in glob)
    assert len(frame) == 300


def test_membership_frame_exports_k_family():
    """The hand-off carries a nested K-family overlay (conventions) without polluting the operational
    ``tess_`` contract, and ``k_family_menu`` reports the per-K decision metrics M4/M5 consume."""
    coords, _, _ = _coords(N=300, K=3, seed=5)
    proj = StrataProjector()
    rf2 = SoftRegionModel().fit(coords, 2, arm="A")                  # operational K
    af = ArchetypeModel().fit(coords, 3, arm="A", n_draw=5)
    family = [SoftRegionModel().fit(coords, k, arm="A") for k in (2, 3, 4)]
    frame = proj.membership_frame(coords, af, rf2, region_family=family)
    op = [c for c in frame.columns if c.startswith("tess_")]         # prognosis.frame's selector
    assert any(c.startswith("tess_r") for c in op)
    assert not any(c.startswith("tess_k") for c in op)               # family uses tessfam_, never tess_
    for k in (2, 3, 4):
        rcols = [f"tessfam_k{k}_r{j}" for j in range(k)]
        assert set(rcols) <= set(frame.columns)
        assert np.allclose(frame[rcols].to_numpy().sum(1), 1.0, atol=1e-3)   # valid soft simplex
        assert {f"tessfam_k{k}_MAP", f"tessfam_k{k}_entropy", f"tessfam_k{k}_tier"} <= set(frame.columns)
    menu = proj.k_family_menu(coords, family)
    assert list(menu["K"]) == [2, 3, 4]
    assert {"mean_eta_specifics", "eta_overall_severity", "confident_dominant_frac"} <= set(menu.columns)


def test_select_A_operational_prefers_stable_A():
    """The operational A choice favours a reproducible archetype set: on clean simplex data it returns a
    valid A in the sweep whose reported stability is high (the EV scree alone would over-reach)."""
    rng = np.random.default_rng(7)
    Z = rng.normal(0, 3, (3, len(CANON)))
    X = rng.dirichlet([0.4] * 3, 400) @ Z
    coords, _, _ = _coords(N=400, seed=7, X=X)
    sel = ArchetypeModel(StrataConfig()).select_A_operational(coords, As=(2, 3, 4), stability_min=0.8)
    assert sel["chosen_A"] in (2, 3, 4)
    assert all({"A", "explained_variance", "stability"} <= set(r) for r in sel["sweep"])
    chosen = next(r for r in sel["sweep"] if r["A"] == sel["chosen_A"])
    assert chosen["stability"] >= 0.8 or "no A met" in sel["rationale"]


def test_runner_caches_and_reruns_on_config_change(tmp_path):
    coords, _, _ = _coords(N=300, K=2, seed=6)
    cfg = replace(StrataConfig(), output_dir=tmp_path, figure_dir=tmp_path / "fig")
    runner = StrataRunner(cfg)
    stage = next(s for s in cfg.stage_plan if s.kind == "structure")
    runner.run_stage(stage, {"coords": coords})
    assert (tmp_path / "structure" / "manifest.json").exists()
    assert runner._cache_ok(tmp_path / "structure", stage)          # second call would hit cache
    other = replace(cfg, region_reg=9e-3)                            # a model-affecting change
    assert not StrataRunner(other)._cache_ok(tmp_path / "structure", stage)
