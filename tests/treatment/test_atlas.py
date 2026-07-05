"""M5 atlas — the treatment-course atlas stage (Wilson rates + the proof gates) on a synthetic frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from face.treatment.engine import TreatmentAtlas, TreatmentConfig, _wilson


def test_wilson_known_points():
    lo, hi = _wilson(50, 100)
    assert lo < 0.5 < hi and 0.39 < lo < 0.41 and 0.59 < hi < 0.61      # 50/100 ≈ [0.40, 0.60]
    assert all(np.isnan(x) for x in _wilson(0, 0))                       # empty cell → NaN
    lo0, hi0 = _wilson(0, 30)
    assert lo0 == 0.0 or lo0 < 0.01                                      # zero events → tight lower bound


def _synthetic(n=480, seed=0):
    rng = np.random.default_rng(seed)
    corner = rng.integers(0, 4, n)
    w = rng.dirichlet(np.ones(4), n)
    for k in range(4):
        w[np.arange(n), corner] += 1.5                                   # make a dominant corner
    w = w / w.sum(1, keepdims=True)
    dom = w.argmax(1)
    # biological corner (0) → higher resistance/side-effects, lower response (a real gradient to detect)
    base = {0: 0.45, 1: 0.20, 2: 0.38, 3: 0.33}
    df = pd.DataFrame({
        "cohort": rng.choice(["bp", "sz"], n),
        "arm": rng.choice(["arm_a", "arm_b", "arm_c"], n),
        "arch_dominant": dom,
        "age": rng.normal(40, 12, n), "sex": rng.integers(0, 2, n).astype(float),
        "cgi_s__V0": rng.normal(4.2, 1.0, n), "substance__mean": rng.normal(0, 1, n),
        "ep_resistance": rng.binomial(1, [base[c] for c in dom]),
        "ep_response": rng.binomial(1, [0.6 - 0.15 * (c == 0) for c in dom]),
        "ep_side_effects": rng.binomial(1, [0.12 + 0.12 * (c == 0) for c in dom]),
    })
    for k in range(4):
        df[f"arch_w{k}"] = w[:, k]
        df[f"arch_w{k}_sd"] = 0.01
    return df


def test_atlas_structure_and_gates():
    atlas, gates = TreatmentAtlas(TreatmentConfig().with_smoke_defaults()).run(_synthetic())
    # rates valid, 4 corners × 3 endpoints present
    assert set(atlas["endpoint"]) == {"ep_resistance", "ep_response", "ep_side_effects"}
    assert atlas["archetype"].nunique() == 4
    pooled = atlas[atlas.cohort == "pooled"]
    assert ((pooled["rate"] >= 0) & (pooled["rate"] <= 1)).all()
    assert (pooled["lo"] <= pooled["rate"]).all() and (pooled["rate"] <= pooled["hi"]).all()
    # gates carry the proof columns
    for col in ("corner_beyond_sev_subst_demo_p", "composition_share", "cohort_interaction_p",
                "delta_auc", "delta_auc_perm_p"):
        assert col in gates.columns
    assert len(gates) == 3
    # perm p is a probability
    pp = gates["delta_auc_perm_p"].dropna()
    assert ((pp >= 0) & (pp <= 1)).all()
