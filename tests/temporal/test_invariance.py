"""M3 G1 — invariance congruence/license logic (pure; no sampling)."""
from __future__ import annotations

import numpy as np

from face.temporal.invariance import axis_license, congruence_over_visits, tucker_phi


def test_tucker_phi_identical_and_scale_invariant():
    a = [0.8, 0.6, 0.7]
    assert tucker_phi(a, a) == 1.0
    assert tucker_phi(a, list(2 * np.array(a))) == 1.0      # scale-invariant
    assert abs(tucker_phi([1, 0], [0, 1])) < 1e-9           # orthogonal → 0


def _fits():
    V0 = {"a": ("metabolic", 0.8, 100), "b": ("metabolic", 0.6, 100),
          "c": ("cognition", 0.7, 100), "d": ("cognition", 0.5, 100)}
    V1 = {"a": ("metabolic", 0.79, 100), "b": ("metabolic", 0.61, 100),      # metabolic ~unchanged
          "c": ("cognition", 0.1, 100), "d": ("cognition", -0.7, 100)}       # cognition pattern flipped
    V2 = {"a": ("metabolic", 0.82, 100), "b": ("metabolic", 0.58, 100),
          "c": ("cognition", 0.68, 100), "d": ("cognition", 0.52, 100)}      # cognition back to V0-like
    return {("V0", 0): V0, ("V1", 0): V1, ("V2", 0): V2}


def test_congruence_detects_changed_pattern():
    cong = congruence_over_visits(_fits(), ["metabolic", "cognition"], ["V0", "V1", "V2"], [0], "V0")
    met = cong[(cong.factor == "metabolic")]
    cog_v1 = float(cong[(cong.factor == "cognition") & (cong.visit == "V1")]["phi_mean"].iloc[0])
    assert (met["phi_mean"] > 0.99).all()                  # metabolic stable both visits
    assert cog_v1 < 0.85                                    # cognition flipped at V1 → low φ


def test_axis_license_thresholds():
    cong = congruence_over_visits(_fits(), ["metabolic", "cognition"], ["V0", "V1", "V2"], [0], "V0")
    lic = axis_license(cong).set_index("axis")
    assert lic.loc["metabolic", "license"] == "invariant"          # min φ ≥ 0.95
    assert lic.loc["cognition", "license"] == "non-invariant"      # worst (V1) φ < 0.85


def test_congruence_respects_min_obs():
    fits = {("V0", 0): {"a": ("metabolic", 0.8, 100), "b": ("metabolic", 0.6, 10)},   # b under MIN_OBS
            ("V1", 0): {"a": ("metabolic", 0.8, 100), "b": ("metabolic", 0.6, 100)}}
    cong = congruence_over_visits(fits, ["metabolic"], ["V0", "V1"], [0], "V0")
    assert cong.empty                                       # only 1 testable item (a) → < 2, no φ
