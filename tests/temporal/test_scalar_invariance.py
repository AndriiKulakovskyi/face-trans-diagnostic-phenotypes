"""Guard for the scalar-invariance (intercept-drift) ANCOVA (P4-01).

``intercept_drift`` must clear when an item's mean change is fully explained by the latent (scalar
invariant) and flag when the item intercept shifts beyond the latent change.
"""
from __future__ import annotations

import numpy as np

from face.temporal.invariance import intercept_drift


def _data(seed, v2_shift):
    rng = np.random.default_rng(seed)
    n = 400
    latent = rng.normal(0, 1, n * 3)
    visit = np.array(["V0"] * n + ["V1"] * n + ["V2"] * n)
    y = 0.8 * latent + rng.normal(0, 0.3, n * 3) + (visit == "V2") * v2_shift
    return y, latent, visit


def test_scalar_invariant_item_has_negligible_drift():
    # No intercept shift -> Δα ~ 0 (magnitude is the robust signal; a 94% interval false-flags ~6% by design).
    y, latent, visit = _data(0, v2_shift=0.0)
    r = intercept_drift(y, latent, visit)
    assert abs(r["V1"]["delta_alpha"]) < 0.1 and abs(r["V2"]["delta_alpha"]) < 0.1


def test_intercept_drift_is_detected():
    y, latent, visit = _data(1, v2_shift=1.0)
    r = intercept_drift(y, latent, visit)
    assert r["V2"]["excludes_zero"]
    assert abs(r["V2"]["delta_alpha"]) > 0.5            # a real ~1-SD intercept shift
