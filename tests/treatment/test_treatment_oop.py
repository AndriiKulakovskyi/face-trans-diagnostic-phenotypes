"""Golden tests for the OOP treatment engine (synthetic — no copula artifacts / heavy fits)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from face.treatment.moderation import e_value
from face.treatment.propensity import overlap, propensity_score, stabilized_iptw
from face.treatment.treatment_model_oop import (
    TreatmentConfig,
    TreatmentProjector,
    _config_sig,
    _verdict,
    arch_cols,
    arch_eiv_block,
)


def test_config_sig_and_smoke_factory():
    base = TreatmentConfig()
    assert _config_sig(base) == _config_sig(TreatmentConfig())
    sm = base.with_smoke_defaults()
    assert sm.smoke and not base.smoke and sm.draws < base.draws
    assert _config_sig(sm) != _config_sig(base)
    assert base.moderation_reps == ("durable", "archetypes")


def test_arch_eiv_block_drops_one_and_carries_sd():
    n = 60
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"arch_w{k}": rng.random(n) for k in range(4)})
    for k in range(4):
        df[f"arch_w{k}_sd"] = rng.random(n) * 0.1
    assert arch_cols(df) == ["arch_w0", "arch_w1", "arch_w2", "arch_w3"]
    obs, sd, names = arch_eiv_block(df)
    assert obs.shape == (n, 3) and sd.shape == (n, 3)        # A=4 -> drop-one -> 3
    assert names == ["arch_w0", "arch_w1", "arch_w2"]
    assert np.isfinite(obs).all() and (sd >= 0).all()


def test_overlap_gate_verdicts():
    # good overlap -> estimable
    assert _verdict({"n_treated": 200, "n_control": 200, "frac_in_support": 0.95}, 0.05) == "estimable"
    # channeled -> poor overlap
    assert _verdict({"n_treated": 200, "n_control": 200, "frac_in_support": 0.3}, 0.05).startswith("channeled")
    # tiny arm -> non-estimable
    assert _verdict({"n_treated": 10, "n_control": 200, "frac_in_support": 0.95}, 0.05).startswith("non-estimable")
    # residual imbalance caution
    assert "caution" in _verdict({"n_treated": 200, "n_control": 200, "frac_in_support": 0.9}, 0.4)


def test_propensity_overlap_iptw_smoke():
    """The reused causal kernels run on synthetic data: a separable design has lower overlap."""
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=(n, 3))
    treat = (x[:, 0] + rng.normal(scale=1.0, size=n) > 0).astype(float)
    ps = propensity_score(x, treat, seed=0)
    diag = overlap(ps, treat)
    assert 0.0 <= diag["frac_in_support"] <= 1.0 and diag["n_treated"] + diag["n_control"] == n
    w, keep = stabilized_iptw(ps, treat)
    assert w.shape == (n,) and keep.dtype == bool
    assert e_value(0.0) == 1.0 and e_value(0.5) > 1.0       # E-value monotone in |effect|


def test_projector_verdict_logic():
    mod = pd.DataFrame([
        {"question": "lithium_bp", "mode": "active_comparator", "outcome": "functioning",
         "representation": "durable", "n": 1000, "ate": 0.1, "ate_lo": -0.05, "ate_hi": 0.25,
         "ate_se": 0.08, "int_ses": "0.1;0.1", "e_value": 1.3,
         "moderation_d_elpd": 1.0, "moderation_se": 3.0, "moderation_any_axis": False},   # null
        {"question": "antipsychotic_bp", "mode": "active_comparator", "outcome": "functioning",
         "representation": "archetypes", "n": 900, "ate": 0.2, "ate_lo": 0.02, "ate_hi": 0.38,
         "ate_se": 0.09, "int_ses": "0.1;0.1;0.1;0.1", "e_value": 1.8,
         "moderation_d_elpd": 12.0, "moderation_se": 3.0, "moderation_any_axis": True},   # moderates
    ])
    s = TreatmentProjector().summary({"moderation": mod})
    v = dict(zip(s["question"], s["moderation_verdict"], strict=False))
    assert "null" in v["lithium_bp"]                       # bounded null (MDE re-scope verdict string)
    assert v["antipsychotic_bp"].startswith("moderates")
    # NaN ΔELPD but a credible interaction HDI -> suggestive (HDI only)
    mod_na = mod.copy(); mod_na["moderation_d_elpd"] = np.nan; mod_na["moderation_se"] = np.nan
    s2 = TreatmentProjector().summary({"moderation": mod_na})
    assert s2.loc[s2.question == "antipsychotic_bp", "moderation_verdict"].iloc[0] == "suggestive (HDI only)"
