"""Structural tests for the continuous-core engine (S1 + S2 cell wiring).

Fast: exercises `prepare()` only (no sampling, no PyMC build). Guards the invariants that matter
for the staged continuation:
  * S1 stays exactly what certified (J=68, simple structure + bifactor-G, Phi=I).
  * S2 = S1 + inter-dimension Phi + MADRS/QIDS/STAI window cross-loadings, with S1 a strict
    structural subset (S2 deforms the certified S1 fit, never replaces it).
  * The specific<->specific (metabolic<->inflammatory) cross-loadings are OFF by default at S2
    (rotationally aliased with Phi); they exist only in the explicit `specific_cross=True`
    sensitivity arm, and there they are metabolic<->inflammatory only and ridge-guarded.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

pytestmark = pytest.mark.skipif(
    not (REPO / "data" / "processed" / "baseline_v0.parquet").exists(),
    reason="needs data/processed/baseline_v0.parquet (scripts/01_build_data.py)")

from face.models.bayesian.continuous_core import WINDOWS, prepare  # noqa: E402


def _kinds(prep):
    return Counter(prep.kind.values())


def test_s1_is_simple_structure_phi_identity():
    p = prepare()                                   # S1 defaults
    assert p.correlated is False
    assert p.factor_cols == ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep"]
    k = _kinds(p)
    assert set(k) == {"g_anchor", "primary", "bifactor_G"}   # home + bifactor-G only
    assert len(p.pos_cells) == k["g_anchor"] + k["primary"]
    assert not any(it in WINDOWS for it in p.items)


def test_s2_default_adds_phi_and_windows_no_specific_cross():
    p1, p2 = prepare(), prepare(correlated=True, windows=True)   # the S2 production config
    assert p2.correlated is True
    k1, k2 = _kinds(p1), _kinds(p2)
    # S1 home + bifactor structure preserved verbatim (strict subset)
    for kind in ("g_anchor", "primary", "bifactor_G"):
        assert k2[kind] == k1[kind]
    # adds the 3 window items + their cross-loadings; NO specific<->specific crosses by default
    assert k2["window"] > 0
    assert k2["cross"] == 0
    assert p2.M.shape[1] == p1.M.shape[1] + len(WINDOWS)
    assert all(w in p2.items for w in WINDOWS)


def test_s2_windows_land_on_g_cognition_sleep():
    p = prepare(correlated=True, windows=True)
    landed: dict[str, set] = {w: set() for w in WINDOWS}
    for (j, c), kind in p.kind.items():
        if kind == "window":
            landed[p.items[j]].add(p.factor_cols[c])
    assert landed["madrs"] == {"overall_severity", "cognition", "sleep"}
    assert landed["qidsr120"] == {"overall_severity", "cognition", "sleep"}
    assert landed["staya"] == {"overall_severity", "sleep"}


def test_specific_cross_arm_is_metabolic_inflammatory_only_and_ridge_guarded():
    p = prepare(correlated=True, windows=True, specific_cross=True, cross_sd_scale=0.25)
    cross = {(p.items[j], p.factor_cols[c]) for (j, c) in p.kind if p.kind[(j, c)] == "cross"}
    home = {p.items[j]: h for j, h in enumerate(p.home)}
    pair = {"metabolic", "inflammatory"}
    assert cross, "specific_cross=True should free the metabolic<->inflammatory cells"
    for item, factor in cross:
        assert {home[item], factor} <= pair, f"unexpected cross {item}->{factor}"
    # ridge guard: specific crosses tighter than the (full-sd) window cells
    cross_sd = [sd for (j, c, mu, sd) in p.sgn_cells if p.kind[(j, c)] == "cross"]
    win_sd = [sd for (j, c, mu, sd) in p.sgn_cells if p.kind[(j, c)] == "window"]
    assert max(cross_sd) < min(win_sd)


def test_phi_is_valid_pd_correlation_with_g_orthogonal():
    """Regression guard for the LKJCorr bug: pm.LKJCorr returns the Cholesky factor L, so Φ = L Lᵀ.
    Checked at the 6-specific (S3) scale, where the wrong `tril+tril.T+I` reconstruction is reliably
    INDEFINITE (the cause of the S3a NaN). Φ must be PD, unit-diagonal, with G orthogonal."""
    import numpy as np
    import pymc as pm
    from face.models.bayesian.continuous_core import S3_FACTORS, build_marginalized
    prep = prepare(S3_FACTORS, correlated=True, windows=True, n_subsample=400)
    model = build_marginalized(prep)
    g = prep.factor_cols.index("overall_severity")
    for P in pm.draw(model["Phi"], draws=6, random_seed=0):
        assert np.allclose(np.diag(P), 1.0, atol=1e-6)                       # true correlation
        assert np.linalg.eigvalsh(P).min() > 1e-8, "Φ not PD (LKJCorr must be C = L Lᵀ)"
        assert np.allclose(np.delete(P[g], g), 0.0, atol=1e-6), "G must be ⊥ specifics"
