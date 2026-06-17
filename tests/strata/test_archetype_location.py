"""Guard for archetype location (anchor) uncertainty (P3-04/05).

``archetype_location_uncertainty`` re-fits the anchors across draws + bootstraps, Hungarian-aligns them,
and returns per-(archetype, dim) mean + HDI + a stability metric. On clean, well-separated planted
corners the anchors must be stable and the HDIs valid.
"""
from __future__ import annotations

import numpy as np

from face.strata.archetypes import archetype_location_uncertainty


def test_location_uncertainty_shapes_valid_hdi_and_stable():
    rng = np.random.default_rng(0)
    corners = np.array([[4.0, 0.0], [0.0, 4.0], [-3.0, -3.0]])
    W = rng.dirichlet([0.4, 0.4, 0.4], 250)
    X = W @ corners + rng.normal(0, 0.1, (250, 2))
    draws = X[None].repeat(8, 0) + rng.normal(0, 0.1, (8, 250, 2))     # synthetic posterior draws

    res = archetype_location_uncertainty(X, draws, [0, 1], A=3, n_draw=6, n_boot=6, seed=0)
    assert res["Z_ref"].shape == (3, 2)
    assert res["Z_lo"].shape == (3, 2) and res["Z_hi"].shape == (3, 2)
    assert (res["Z_lo"] <= res["Z_hi"] + 1e-9).all()                  # valid HDIs
    assert res["min_tucker_per_arch"].shape == (3,)
    assert res["n_refits"] == 12
    assert res["min_tucker_per_arch"].min() > 0.7                     # well-separated corners -> stable
