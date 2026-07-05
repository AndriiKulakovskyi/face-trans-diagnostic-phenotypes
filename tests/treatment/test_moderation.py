"""M5.2b — the MDE / power guard (the bounds-and-defends re-scope) + the projector MDE wiring.

Pure-arithmetic tests: the minimum-detectable-effect helper, the ETI→SD recovery that lets the MDE be
computed from already-serialized posteriors (no refit), and that the projector emits MDE columns from the
exact posterior SD when present and falls back to the ETI otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from face.treatment.engine import TreatmentProjector
from face.treatment.moderation import e_value, mde, sd_from_eti

Z975, Z80, Z97 = 1.959963985, 0.841621234, 1.880793608


def test_mde_is_z_sum_times_se():
    assert mde(0.1) == np.float64((Z975 + Z80) * 0.1) or abs(mde(0.1) - (Z975 + Z80) * 0.1) < 1e-9
    assert abs(mde(0.0) - 0.0) < 1e-12
    assert abs(mde(-0.2) - mde(0.2)) < 1e-12                          # uses |se|


def test_sd_from_eti_recovers_gaussian_sd():
    # a unit-SD Gaussian has 94% ETI [-z97, +z97]; sd_from_eti must return ~1.0
    assert abs(sd_from_eti(-Z97, Z97) - 1.0) < 1e-6
    assert abs(sd_from_eti(-0.128, 0.123) - 0.0667) < 1e-3           # the lithium-BP ATE ETI


def test_mde_separates_bounded_from_underpowered():
    # lithium-BP functioning ATE ETI (tight) vs clozapine-SZ (wide): the MDE must be far smaller for lithium
    lithium = mde(sd_from_eti(-0.128, 0.123))                        # ~0.187 — bounded
    clozap = mde(sd_from_eti(-0.238, 0.288))                         # ~0.39 — underpowered
    assert lithium < 0.22 and clozap > 0.35
    assert clozap > 1.8 * lithium                                     # qualitatively different power


def test_projector_mde_exact_se_matches_eti_fallback():
    proj = TreatmentProjector()
    # one synthetic moderation row with BOTH exact SEs and the ETI strings
    exact = pd.Series({"ate": -0.003, "ate_lo": -0.128, "ate_hi": 0.123, "ate_se": 0.0667,
                       "int_his": "[-0.069,+0.225];[-0.075,+0.182]", "int_ses": "0.0781;0.0683"})
    eti_only = exact.drop(labels=["ate_se", "int_ses"])
    me, mf = proj._mde(exact), proj._mde(eti_only)
    assert abs(me["ate_mde"] - 0.187) < 5e-3
    assert me["int_mde_max"] > me["int_mde_min"] > 0
    # exact-SE and ETI-derived MDE agree (the posterior is ~Gaussian) within rounding
    for k in ("ate_mde", "int_mde_min", "int_mde_max"):
        assert abs(me[k] - mf[k]) < 0.02


def test_summary_emits_mde_and_verdict_columns():
    proj = TreatmentProjector()
    mod = pd.DataFrame([
        {"question": "lithium_bp", "mode": "active_comparator", "outcome": "functioning",
         "representation": "durable", "n": 663, "ate": -0.003, "e_value": 1.06,
         "ate_lo": -0.128, "ate_hi": 0.123, "ate_se": 0.0667,
         "moderation_d_elpd": -2.44, "moderation_se": 1.55, "moderation_any_axis": False,
         "int_his": "[-0.069,+0.225];[-0.075,+0.182];[-0.080,+0.179]", "int_ses": "0.0781;0.0683;0.0688"},
    ])
    s = proj.summary({"moderation": mod})
    assert {"ate_mde", "int_mde_min", "int_mde_max", "moderation_verdict"} <= set(s.columns)
    assert s.iloc[0]["moderation_verdict"] == "no moderation (bounded null)"   # MDE small → bounded
    assert 0.15 < s.iloc[0]["ate_mde"] < 0.25                         # bounded null


def test_verdict_separates_bounded_from_underpowered():
    proj = TreatmentProjector()
    # same no-interaction structure, but a wide (underpowered) interaction → "underpowered" verdict
    mod = pd.DataFrame([
        {"question": "clozapine_sz", "mode": "on_off", "outcome": "functioning", "representation": "durable",
         "n": 516, "ate": 0.021, "e_value": 1.16, "ate_lo": -0.238, "ate_hi": 0.288, "ate_se": 0.1417,
         "moderation_d_elpd": -3.1, "moderation_se": 0.82, "moderation_any_axis": False,
         "int_his": "[-0.287,+0.293];[-0.315,+0.231];[-0.305,+0.193]", "int_ses": "0.1544;0.1450;0.1321"},
    ])
    s = proj.summary({"moderation": mod})
    assert s.iloc[0]["moderation_verdict"] == "no moderation (underpowered)"   # MDE large → underpowered
    assert s.iloc[0]["int_mde_max"] > 0.30


def test_e_value_known_point():
    # E-value of a null effect is 1.0; grows with |d|
    assert abs(e_value(0.0) - 1.0) < 1e-9
    assert e_value(0.5) > e_value(0.2) > 1.0
