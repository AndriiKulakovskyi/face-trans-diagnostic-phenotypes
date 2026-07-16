"""Golden tests for the OOP temporal engine (synthetic — no copula artifacts / heavy fits)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from face.measurement.engine import copula_invert
from face.temporal.engine import (
    FrozenCovariateDesign,
    TemporalConfig,
    _config_sig,
    _uid,
    copula_forward,
)
from face.temporal.persistence import membership_persistence


def test_config_sig_and_smoke_factory():
    base = TemporalConfig()
    assert _config_sig(base) == _config_sig(TemporalConfig())
    sm = base.with_smoke_defaults()
    assert sm.smoke and not base.smoke and sm.proj_draws < base.proj_draws
    assert _config_sig(sm) != _config_sig(base)
    assert base.A == 5   # A=5 archetypes on the 8-factor map


def test_copula_forward_monotone_and_inverts():
    """The new forward map is monotone and is the inverse of measurement.engine.copula_invert."""
    rng = np.random.default_rng(0)
    sorted_values = np.sort(rng.normal(size=80))
    sorted_z = np.sort(rng.normal(size=80))
    y = np.linspace(sorted_values.min(), sorted_values.max(), 25)
    z = copula_forward(y, sorted_values, sorted_z)
    assert np.all(np.diff(z) >= -1e-9)                         # monotone non-decreasing
    back = copula_invert(z, sorted_values, sorted_z)
    assert np.nanmax(np.abs(back - y)) < 1e-6                  # round-trip
    # NaN in -> NaN out (never imputed); out-of-support clamps to the V0 endpoints
    assert np.isnan(copula_forward(np.array([np.nan]), sorted_values, sorted_z))[0]
    assert copula_forward(np.array([1e9]), sorted_values, sorted_z)[0] == sorted_z[-1]


def test_uid_construction():
    idx = pd.MultiIndex.from_arrays([["bp", "sz"], ["p1", "p2"]], names=["cohort", "patient_id"])
    assert list(_uid(idx)) == ["bp|p1", "sz|p2"]


def test_membership_persistence_a4():
    """The G4 persistence kernel works at the copula granularity A=4 (not the native 8)."""
    rng = np.random.default_rng(1)
    n, A = 200, 4
    W = rng.dirichlet([0.5] * A, n)
    rows = []
    for v in ("V0", "V2"):
        for i in range(n):
            r = {"patient_uid": f"u{i}", "visit": v, "n_visits": 2}
            w = W[i] if v == "V0" else np.clip(W[i] + rng.normal(0, 0.03, A), 1e-6, None)
            w = w / w.sum()
            for k in range(A):
                r[f"archB_w{k}"] = w[k]
            r["archB_dominant"] = int(np.argmax(w))
            rows.append(r)
    panel = pd.DataFrame(rows)
    res = membership_persistence(panel, arm="archB", A=A, s="V0", t="V2")
    assert res["transition"].shape == (A, A)
    assert 0.0 <= res["dominant_agree"] <= 1.0
    assert res["cos_median"] > 0.8                            # near-identical weights -> high cosine


def test_frozen_covariate_design_reuses_training_transform():
    """Follow-up design uses frozen knots and scales rather than refitting."""
    from sklearn.preprocessing import SplineTransformer

    rng = np.random.default_rng(2)
    n = 300
    idx = pd.MultiIndex.from_arrays([rng.choice(["bp", "sz"], n), [f"p{i}" for i in range(n)]],
                                    names=["cohort", "patient_id"])
    age = rng.normal(40, 10, n)
    cov = pd.DataFrame({"age": age, "sex": rng.integers(0, 2, n),
                        "education_years": rng.normal(12, 3, n)}, index=idx)
    spline = SplineTransformer(n_knots=4, degree=3, include_bias=False).fit(age.reshape(-1, 1))
    ab = spline.transform(age.reshape(-1, 1))
    ab_mean, ab_sd = ab.mean(0), ab.std(0)
    edu = cov["education_years"].to_numpy()
    edu_mean, edu_sd = float(edu.mean()), float(edu.std())
    names = [f"age_spline_{k}" for k in range(ab.shape[1])]
    names += ["sex", "education_years"]
    names += [f"age_spline_{k}:sex" for k in range(ab.shape[1])]
    metadata = {
        "mode": "in_likelihood",
        "names": names,
        "transform": {
            "numeric": {
                "age": {"fill": float(age.mean())},
                "sex": {"fill": float(cov["sex"].mean())},
                "education_years": {
                    "fill": edu_mean,
                    "center": [edu_mean],
                    "scale": [edu_sd],
                },
            },
            "age_spline": {
                "degree": 3,
                "include_bias": False,
                "knot_vector": np.asarray(spline.bsplines_[0].t).tolist(),
                "center": ab_mean.tolist(),
                "scale": ab_sd.tolist(),
            },
        },
    }
    fcd = FrozenCovariateDesign(metadata=metadata)
    got = fcd.design(cov)
    ab_std = (ab - ab_mean) / ab_sd
    expected = np.column_stack(
        [
            ab_std,
            cov["sex"].to_numpy(),
            (edu - edu_mean) / edu_sd,
            ab_std * cov["sex"].to_numpy()[:, None],
        ]
    )
    assert np.allclose(got, expected, atol=1e-12)
