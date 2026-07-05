"""M3 stage 32 — the V0 standardization spec. The round-trip test is the load-bearing guard: if the
frozen spec does not reproduce `prepare()`'s V0 matrix, every follow-up level claim is mis-scaled."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from face.temporal.standardize import V0StdSpec, apply_spec, load_spec, save_spec

_PROC = Path(__file__).resolve().parents[2] / "data" / "processed"


# ---- the critical guard: apply_spec(V0) == prepare().M (needs the persisted V0 baseline) ----

@pytest.mark.skipif(not (_PROC / "baseline_v0.parquet").exists(), reason="baseline_v0.parquet not built")
def test_v0_roundtrip_reproduces_prepare():
    from face.measurement.kernel import S5_FACTORS, prepare
    from face.temporal.standardize import capture_v0_spec

    spec = capture_v0_spec(S5_FACTORS)
    prep = prepare(S5_FACTORS, correlated=True, windows=True)
    B0 = pd.read_parquet(_PROC / "baseline_v0.parquet")
    M_spec = apply_spec(spec, B0)[prep.items].to_numpy()
    M_prep = prep.M
    assert (np.isnan(M_prep) == np.isnan(M_spec)).all()          # identical missingness masks
    assert np.nanmax(np.abs(M_spec - M_prep)) < 1e-6             # identical values on observed cells
    assert list(spec.items) == list(prep.items)


# ---- pure unit tests (synthetic spec, no data) ----

def _toy_spec():
    # a: gaussian identity (sign +1, mean 0, sd 1); b: lognormal with V0 min>0 (plain-log branch)
    return V0StdSpec(items=["a", "b", "c"],
                     family={"a": "gaussian", "b": "lognormal", "c": "gaussian"},
                     sign={"a": 1, "b": -1, "c": 1},
                     logmin={"a": None, "b": 2.0, "c": None},
                     mean={"a": 0.0, "b": 0.0, "c": 0.0}, sd={"a": 1.0, "b": 1.0, "c": 1.0})


def _toy_B(rows=("p1", "p2")):
    idx = pd.MultiIndex.from_tuples([("bp", r) for r in rows], names=["cohort", "patient_id"])
    return pd.DataFrame({"a": [1.0, -3.0], "b": [5.0, -1.0]}, index=idx)   # 'c' column absent on purpose


def test_apply_gaussian_identity():
    out = apply_spec(_toy_spec(), _toy_B())
    assert list(out.columns) == ["a", "b", "c"]                 # spec order; absent 'c' included
    np.testing.assert_allclose(out["a"].to_numpy(), [1.0, -3.0])   # sign +1, mean 0, sd 1 → identity


def test_apply_missing_column_is_all_nan():
    out = apply_spec(_toy_spec(), _toy_B())
    assert out["c"].isna().all()                                # item not collected at this visit → NaN


def test_apply_lognormal_out_of_support_is_nan():
    out = apply_spec(_toy_spec(), _toy_B())
    # b: plain log (V0 min>0). p1 raw 5 → -log(5); p2 raw -1 → log(-1)=NaN (outside V0 support, not imputed)
    assert out["b"].iloc[0] == pytest.approx(-np.log(5.0))
    assert np.isnan(out["b"].iloc[1])


def test_apply_lognormal_zero_maps_to_nan():
    # a lognormal 0 → log(0) = -inf → must become NaN (the V2 lym_lbstresc bug; never a -inf cell)
    idx = pd.MultiIndex.from_tuples([("bp", "z")], names=["cohort", "patient_id"])
    out = apply_spec(_toy_spec(), pd.DataFrame({"a": [1.0], "b": [0.0]}, index=idx))
    assert np.isnan(out["b"].iloc[0]) and np.isfinite(out["a"].iloc[0])


def test_save_load_roundtrip(tmp_path):
    spec = _toy_spec()
    p = tmp_path / "spec.json"
    save_spec(spec, p)
    back = load_spec(p)
    assert back.items == spec.items and back.family == spec.family
    assert back.sign == spec.sign and back.logmin == spec.logmin
    assert back.mean == spec.mean and back.sd == spec.sd
