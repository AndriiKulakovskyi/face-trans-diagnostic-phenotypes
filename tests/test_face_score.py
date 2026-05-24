"""Tests for the FACE profile scoring (trans_diag.face_score)."""
import numpy as np
import pandas as pd

from trans_diag import FACE_D_ITEMS, FACE_M_ITEMS, compute_face_scores


def _domains(n=500, seed=0):
    rng = np.random.default_rng(seed)
    cols = list(FACE_D_ITEMS) + list(FACE_M_ITEMS) + ["other_a", "other_b"]
    return pd.DataFrame(rng.standard_normal((n, len(cols))), columns=cols)


def test_columns_and_alignment():
    d = _domains()
    f = compute_face_scores(d)
    assert list(f.columns) == ["FACE_D", "FACE_M"]
    assert f.index.equals(d.index)


def test_face_d_is_signed_mean_of_components():
    d = _domains()
    f = compute_face_scores(d, standardize=False)
    assert np.allclose(f["FACE_D"], d[list(FACE_D_ITEMS)].mean(axis=1))   # all signs +1


def test_masked_under_observation_is_nan_not_imputed():
    d = _domains(50)
    d.loc[d.index[0], list(FACE_D_ITEMS)[1:]] = np.nan      # only 1 of 3 observed
    f = compute_face_scores(d, min_obs=2)
    assert np.isnan(f["FACE_D"].iloc[0])                    # under-observed → NaN (no imputation)
    assert np.isfinite(f["FACE_D"].iloc[1])


def test_higher_components_give_higher_score():
    d = _domains(200)
    hi = d.copy()
    hi[list(FACE_D_ITEMS)] += 1.0
    lo_m = compute_face_scores(d, standardize=False)["FACE_D"].mean()
    hi_m = compute_face_scores(hi, standardize=False)["FACE_D"].mean()
    assert hi_m > lo_m                                      # higher = more severe (sign +1)


def test_missing_component_columns_tolerated():
    d = _domains().drop(columns=["cholesterol"])            # FACE-M loses one item
    f = compute_face_scores(d)
    assert np.isfinite(f["FACE_M"]).all()                   # still scored from the remaining two
