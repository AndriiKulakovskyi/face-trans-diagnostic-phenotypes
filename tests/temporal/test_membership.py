"""M3 stage 34 — archetype membership projection + the mixed-prep structure guard."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from face.temporal.membership import _entropy, archetype_membership

_PROC = Path(__file__).resolve().parents[2] / "data" / "processed"


def test_entropy_bounds():
    pure = np.array([[1.0, 0.0, 0.0]])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert _entropy(pure)[0] == pytest.approx(0.0, abs=1e-6)
    assert _entropy(uniform)[0] == pytest.approx(1.0, abs=1e-6)


def test_archetype_membership_shapes_and_simplex():
    Z = np.array([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])      # 3 archetypes in 2-D
    X = np.array([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])      # 3 patients = the 3 corners
    draws = np.repeat(X[None], 8, axis=0).astype("float32")   # [8, 3, 2], no variation
    m = archetype_membership(X, draws, [0, 1], Z, ["A", "B", "C"], prefix="t",
                             index=pd.RangeIndex(3), n_draw=8)
    wcols = [f"t_w{a}" for a in range(3)]
    assert set(wcols + ["t_dominant", "t_dominant_name", "t_entropy"]).issubset(m.columns)
    np.testing.assert_allclose(m[wcols].to_numpy().sum(1), 1.0, atol=1e-6)   # simplex
    assert list(m["t_dominant"]) == [0, 1, 2]                  # each corner → its own archetype
    assert list(m["t_dominant_name"]) == ["A", "B", "C"]
    assert (m[[f"t_w{a}_sd" for a in range(3)]].to_numpy() == 0).all()       # no draw variation → sd 0


@pytest.mark.skipif(not (_PROC / "baseline_v0.parquet").exists(), reason="baseline_v0 not built")
def test_prep_visit_mixed_reproduces_v0_structure():
    # the follow-up mixed prep, applied to V0, must reproduce M2.0's arrays bit-for-bit (else V1/V2 mis-scaled)
    from face.models.bayesian.continuous_core import S5_FACTORS, prepare_mixed
    from face.strata.scoring import align_ordinals_to_fit
    from face.temporal.standardize import capture_v0_spec, prep_visit_mixed

    expl = ["overall_severity", "suicidality", "developmental_risk", "substance"]
    B0 = pd.read_parquet(_PROC / "baseline_v0.parquet")
    ref = prepare_mixed(S5_FACTORS, explicit_factors=expl, min_cohorts=2)
    mpC = prepare_mixed(S5_FACTORS, explicit_factors=expl, min_cohorts=2, balanced=True,
                        n_subsample=2000, seed=20260605)
    align_ordinals_to_fit(ref, mpC.base.index, B0)
    mp_v0 = prepare_mixed(S5_FACTORS, explicit_factors=expl, min_cohorts=2)
    mine = prep_visit_mixed(capture_v0_spec(), mp_v0, B0, cert_index=mpC.base.index, B_v0=B0)

    def eq(a, b):
        return np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9))
    assert eq(ref.base.M, mine.base.M) and eq(ref.Bin, mine.Bin)
    assert eq(ref.Cnt, mine.Cnt) and eq(ref.Ord, mine.Ord)
    assert list(ref.ord_K) == list(mine.ord_K)
