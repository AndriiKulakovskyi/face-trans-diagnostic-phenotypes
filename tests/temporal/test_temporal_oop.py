"""Golden tests for the OOP temporal engine (synthetic — no copula artifacts / heavy fits)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from face.models.bayesian.measurement_model_oop import copula_invert
from face.temporal.persistence import membership_persistence
from face.temporal.temporal_model_oop import (
    CANON,
    FrozenCovariateDesign,
    TemporalConfig,
    _config_sig,
    _uid,
    copula_forward,
)


def test_config_sig_and_smoke_factory():
    base = TemporalConfig()
    assert _config_sig(base) == _config_sig(TemporalConfig())
    sm = base.with_smoke_defaults()
    assert sm.smoke and not base.smoke and sm.proj_draws < base.proj_draws
    assert _config_sig(sm) != _config_sig(base)
    assert base.A == 4


def test_copula_forward_monotone_and_inverts():
    """The new forward map is monotone and is the inverse of measurement_model_oop.copula_invert."""
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


def test_frozen_covariate_design_residualizes():
    """FrozenCovariateDesign.residualize subtracts the frozen per-item fit (FWL) using visit covariates."""
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
    item = "x"
    z0 = pd.DataFrame({item: 0.5 * (age - age.mean()) / age.std() + rng.normal(0, 0.5, n)}, index=idx)
    # build a frozen design directly and fit betas
    A0 = np.column_stack([np.ones(n), (ab - ab.mean(0)) / (ab.std(0) + 1e-9),
                          cov["sex"].to_numpy(), (cov["education_years"] - cov["education_years"].mean()).to_numpy()
                          / cov["education_years"].std(),
                          ((ab - ab.mean(0)) / (ab.std(0) + 1e-9)) * cov["sex"].to_numpy()[:, None]])
    beta, *_ = np.linalg.lstsq(A0, z0[item].to_numpy(), rcond=None)
    fcd = FrozenCovariateDesign(spline=spline, age_basis_mean=ab.mean(0), age_basis_sd=ab.std(0),
                                edu_mean=float(cov["education_years"].mean()),
                                edu_sd=float(cov["education_years"].std()), edu_name="education_years",
                                site_columns=[], names=[], betas={item: beta})
    resid = fcd.residualize(z0, cov)
    # residual should be ~uncorrelated with age (the covariate fit was removed)
    r = np.corrcoef(resid[item].to_numpy(), age)[0, 1]
    assert abs(r) < abs(np.corrcoef(z0[item].to_numpy(), age)[0, 1])     # correlation reduced
