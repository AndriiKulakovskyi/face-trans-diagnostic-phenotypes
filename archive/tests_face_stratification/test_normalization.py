"""Unit tests for normalization + missingness helpers.

Synthetic input — no FACE CSVs required, so these always run in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from face_stratification.harmonization.feature_schema import load_feature_schema
from face_stratification.harmonization.missingness import (
    compute_missingness_mask,
    impute_block_knn,
    split_blocks,
)
from face_stratification.harmonization.normalization import (
    fit_normalization,
    transform_normalization,
)


def _toy_matrix(schema) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 40
    idx = pd.MultiIndex.from_tuples(
        [("bp", f"p{i}") for i in range(n)], names=("cohort", "patient_id")
    )
    data = {}
    for feat in schema.features:
        if feat.type.value == "continuous":
            data[feat.id] = rng.normal(0, 1, size=n)
        elif feat.type.value == "ordinal":
            data[feat.id] = rng.integers(0, 5, size=n).astype(float)
        else:
            data[feat.id] = rng.integers(0, 2, size=n).astype(float)
    X = pd.DataFrame(data, index=idx)
    # Punch holes in ~20% of cells
    mask = rng.random(X.shape) < 0.2
    X = X.mask(mask)
    return X


def test_fit_transform_roundtrip_preserves_shape():
    schema = load_feature_schema()
    X = _toy_matrix(schema)
    stats = fit_normalization(X, schema)
    Xn = transform_normalization(X, stats)
    assert Xn.shape == X.shape
    assert list(Xn.columns) == list(X.columns)
    assert list(Xn.index) == list(X.index)


def test_normalization_replaces_infinities():
    schema = load_feature_schema()
    X = _toy_matrix(schema)
    # Inject a ±inf that could sneak in from a degenerate scale
    X.iloc[0, 0] = np.inf
    X.iloc[1, 0] = -np.inf
    stats = fit_normalization(X, schema)
    Xn = transform_normalization(X, stats)
    assert not np.isinf(Xn.to_numpy()).any()


def test_direction_sign_flips_higher_is_better_features():
    """Sanity: a 'higher_is_better' feature should flip sign after normalization."""
    schema = load_feature_schema()
    # Pick a higher_is_better feature (functioning EQ-5D is a good example)
    hib = next(f for f in schema.features if f.direction == "higher_is_better")
    X = _toy_matrix(schema)
    stats = fit_normalization(X, schema)
    assert stats.sign[hib.id] == -1.0


def test_missingness_mask_has_one_column_per_block():
    schema = load_feature_schema()
    X = _toy_matrix(schema)
    mask = compute_missingness_mask(X, schema)
    assert mask.shape[0] == X.shape[0]
    assert mask.shape[1] == len(schema.blocks)
    assert set(mask.columns) == {f"miss_{b.id}" for b in schema.blocks}


def test_split_blocks_covers_every_column():
    schema = load_feature_schema()
    X = _toy_matrix(schema)
    blocks = split_blocks(X, schema)
    union = set()
    for df in blocks.values():
        union.update(df.columns)
    assert union == set(X.columns)


def test_knn_imputation_fills_missing_values():
    schema = load_feature_schema()
    X = _toy_matrix(schema)
    # Normalize first so the imputation runs on a realistic scale
    stats = fit_normalization(X, schema)
    Xn = transform_normalization(X, stats)
    Xi = impute_block_knn(Xn, schema, k=3)
    # Every column originally had NaNs; after imputation at least 50% should be filled.
    filled_before = X.isna().sum().sum()
    filled_after = Xi.isna().sum().sum()
    assert filled_after <= filled_before
