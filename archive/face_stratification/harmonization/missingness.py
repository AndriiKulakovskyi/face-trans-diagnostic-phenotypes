"""Missingness handling for the unified V1 feature matrix.

Design
------
- **Mask features are first-class.** For every block we emit an extra binary
  column ``miss_<block>`` equal to 1 if *any* feature in that block is missing
  for this patient. This lets downstream models exploit "missing-not-at-random"
  patterns (e.g. ASP patients are systematically missing PANSS) instead of
  hiding them behind imputation.

- **No imputation on the default graph path.** The masked-similarity graph
  builder (see :mod:`face_stratification.graph.patient_similarity`) computes
  pairwise-complete distances directly on NaN-containing matrices — it never
  needs imputed values, and in fact *relies* on NaNs to mark unobserved
  measurements so the semantic overlap edge constraint can fire.

- **Opt-in block-local KNN imputation** is still provided via
  :func:`impute_block_knn` for users who want to compare the masked-graph
  approach against the imputation-based baseline. It is not used anywhere in
  the default pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd

from face_stratification.harmonization.feature_schema import FeatureSchema

logger = logging.getLogger(__name__)


def compute_missingness_mask(
    X: pd.DataFrame, schema: FeatureSchema
) -> pd.DataFrame:
    """Return a DataFrame of per-block missingness indicators.

    The output has one column per block named ``miss_<block>`` and the same
    index as ``X``. A value of ``1.0`` means "at least one feature in this
    block is missing for this patient". Blocks with no features in ``X`` are
    skipped.
    """
    cols: dict[str, pd.Series] = {}
    for block_id, feats in schema.features_by_block().items():
        feat_ids = [f.id for f in feats if f.id in X.columns]
        if not feat_ids:
            continue
        sub = X[feat_ids]
        cols[f"miss_{block_id}"] = sub.isna().any(axis=1).astype("float64")
    return pd.DataFrame(cols, index=X.index)


def impute_block_knn(
    X: pd.DataFrame,
    schema: FeatureSchema,
    *,
    k: int = 5,
    min_valid_rows: int = 30,
) -> pd.DataFrame:
    """Block-local kNN imputation (OPT-IN; NOT used by the default graph path).

    This function is retained as a baseline for methodological comparisons
    against the masked-similarity graph. In the default pipeline, imputed
    matrices should **never** feed the graph builder — pairwise-complete
    masked similarity handles NaN natively without fabricating values.

    Parameters
    ----------
    X:
        Unified matrix, post-normalization (but imputation works on raw too).
    schema:
        Feature schema — used to enumerate blocks.
    k:
        Number of neighbours for ``KNNImputer``. Ignored when sklearn is
        missing or the block has too few valid rows.
    min_valid_rows:
        If a block has fewer valid (non-all-NaN) rows than this, fall back to
        column-wise median imputation to avoid unstable kNN.
    """
    try:
        from sklearn.impute import KNNImputer  # type: ignore

        have_sklearn = True
    except ImportError:
        logger.warning(
            "sklearn not installed; falling back to median imputation. "
            "Install the 'stratification' extra to enable KNN imputation."
        )
        have_sklearn = False

    # Replace any inf with NaN so both KNNImputer and median imputation work.
    out = X.replace([np.inf, -np.inf], np.nan).copy()
    for block_id, feats in schema.features_by_block().items():
        feat_ids = [f.id for f in feats if f.id in out.columns]
        if not feat_ids:
            continue
        block = out[feat_ids]
        valid_rows = block.dropna(how="all")
        if len(valid_rows) < min_valid_rows or not have_sklearn:
            out[feat_ids] = block.fillna(block.median(numeric_only=True))
            continue

        # Split block columns into "fully missing" (can't impute) and
        # "has some data" (eligible for KNN). KNNImputer chokes on an input
        # matrix that contains an all-NaN column, so we feed it only the
        # usable subset and leave the rest untouched.
        col_has_data = block.notna().any(axis=0)
        usable = [c for c, ok in col_has_data.items() if ok]
        if not usable:
            continue  # nothing to impute in this block

        imputer = KNNImputer(n_neighbors=min(k, max(1, len(valid_rows) - 1)))
        try:
            imputed = imputer.fit_transform(block[usable].values)
            imputed_df = pd.DataFrame(imputed, index=block.index, columns=usable)
            out.loc[:, usable] = imputed_df
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "KNN imputation failed for block %s (%s); using median fallback.",
                block_id,
                exc,
            )
            out[feat_ids] = block.fillna(block.median(numeric_only=True))

    return out


def split_blocks(
    X: pd.DataFrame, schema: FeatureSchema
) -> dict[str, pd.DataFrame]:
    """Return the subset of ``X`` for each block, in schema order.

    Helper used by the graph builder so it can compute a separate similarity
    graph per block without re-walking the schema.
    """
    out: dict[str, pd.DataFrame] = {}
    for block_id, feats in schema.features_by_block().items():
        feat_ids = [f.id for f in feats if f.id in X.columns]
        if feat_ids:
            out[block_id] = X[feat_ids]
    return out


def _iter_block_feature_ids(
    schema: FeatureSchema, only_present: Iterable[str]
) -> dict[str, list[str]]:
    """Return {block_id -> [feature_ids present in ``only_present``]}."""
    present = set(only_present)
    return {
        bid: [f.id for f in feats if f.id in present]
        for bid, feats in schema.features_by_block().items()
    }


# ---------------------------------------------------------------------------
# Missingness characterization & experimental treatment strategies
# ---------------------------------------------------------------------------


def characterize_missingness(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    schema: Any,
) -> dict[str, Any]:
    """Characterize the missingness structure of the harmonized dataset.

    Returns a dict with:
    - 'per_feature_rates': DataFrame with per-feature, per-cohort missingness rates
    - 'correlation_matrix': correlation matrix of missingness indicators
    - 'mcar_test_per_block': dict mapping block_id -> (chi2, p_value, is_mcar)
    - 'mechanism_summary': dict mapping block_id -> 'MCAR' | 'MAR' | 'MNAR'
    """
    cohort_col = metadata["cohort"] if "cohort" in metadata.columns else None

    # Per-feature, per-cohort missingness rates
    miss_mask = X.isna().astype(float)
    rates: dict[str, pd.Series] = {}
    if cohort_col is not None:
        for cohort in sorted(cohort_col.unique()):
            idx = cohort_col == cohort
            rates[cohort] = miss_mask.loc[idx].mean()
        rates["overall"] = miss_mask.mean()
    else:
        rates["overall"] = miss_mask.mean()
    per_feature_rates = pd.DataFrame(rates)

    # Missingness correlation matrix (only informative columns)
    miss_indicators = X.isna().astype(float)
    variable_cols = [
        c for c in miss_indicators.columns if 0 < miss_indicators[c].mean() < 1
    ]
    corr_matrix = miss_indicators[variable_cols].corr() if variable_cols else pd.DataFrame()

    # Little's MCAR test per block (simplified chi-squared approach)
    feature_blocks: dict[str, list[str]] = {}
    for f in schema.features:
        feature_blocks.setdefault(f.block, []).append(f.id)

    mcar_results: dict[str, tuple[float, float, bool]] = {}
    mechanism_summary: dict[str, str] = {}
    for block_id, feat_ids in feature_blocks.items():
        cols = [c for c in feat_ids if c in X.columns]
        if not cols or len(cols) < 2:
            continue
        block_data = X[cols]
        miss_rate = block_data.isna().mean().mean()

        if miss_rate == 0 or miss_rate == 1:
            mcar_results[block_id] = (0.0, 1.0, True)
            mechanism_summary[block_id] = "MCAR"
            continue

        if cohort_col is not None:
            any_missing = block_data.isna().any(axis=1).astype(int)
            try:
                from scipy.stats import chi2_contingency

                contingency = pd.crosstab(cohort_col, any_missing)
                if contingency.shape[1] == 2:
                    chi2, p, _dof, _ = chi2_contingency(contingency)
                    is_mcar = p > 0.05
                    mcar_results[block_id] = (float(chi2), float(p), is_mcar)
                    if is_mcar:
                        mechanism_summary[block_id] = "MCAR"
                    elif miss_rate > 0.8:
                        mechanism_summary[block_id] = "MNAR"
                    else:
                        mechanism_summary[block_id] = "MAR"
                else:
                    mcar_results[block_id] = (0.0, 1.0, True)
                    mechanism_summary[block_id] = "MCAR"
            except Exception:  # noqa: BLE001
                mcar_results[block_id] = (np.nan, np.nan, False)
                mechanism_summary[block_id] = "unknown"
        else:
            mcar_results[block_id] = (np.nan, np.nan, False)
            mechanism_summary[block_id] = "unknown"

    return {
        "per_feature_rates": per_feature_rates,
        "correlation_matrix": corr_matrix,
        "mcar_test_per_block": mcar_results,
        "mechanism_summary": mechanism_summary,
    }


def augment_with_missingness_indicators(
    X: pd.DataFrame,
    schema: Any,
) -> pd.DataFrame:
    """Append binary missingness indicators per clinical block.

    For each block in the schema, adds a column ``miss_<block_id>`` that is 1
    if ANY feature in that block is missing for the patient, 0 otherwise.
    Also adds per-feature indicators ``miss_feat_<feature_id>`` for features
    with >5% and <95% missingness (informative indicators only).
    """
    result = X.copy()

    feature_blocks: dict[str, list[str]] = {}
    for f in schema.features:
        feature_blocks.setdefault(f.block, []).append(f.id)

    for block_id, feat_ids in feature_blocks.items():
        cols = [c for c in feat_ids if c in X.columns]
        if cols:
            result[f"miss_{block_id}"] = X[cols].isna().any(axis=1).astype(float)

    for col in X.columns:
        rate = X[col].isna().mean()
        if 0.05 < rate < 0.95:
            result[f"miss_feat_{col}"] = X[col].isna().astype(float)

    return result


def impute_block_mice(
    X: pd.DataFrame,
    schema: Any,
    n_imputations: int = 5,
    random_state: int = 42,
) -> list[pd.DataFrame]:
    """Block-wise MICE imputation, returning M imputed datasets.

    Uses sklearn ``IterativeImputer`` within each block independently.
    Returns a list of *n_imputations* DataFrames, each fully imputed.
    """
    try:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        have_iterative = True
    except ImportError:
        logger.warning(
            "sklearn IterativeImputer not available; returning median-imputed copy"
        )
        have_iterative = False

    if not have_iterative:
        filled = X.fillna(X.median())
        return [filled] * n_imputations

    feature_blocks: dict[str, list[str]] = {}
    for f in schema.features:
        feature_blocks.setdefault(f.block, []).append(f.id)

    imputed_datasets: list[pd.DataFrame] = []
    for m in range(n_imputations):
        result = X.copy()
        for _block_id, feat_ids in feature_blocks.items():
            cols = [c for c in feat_ids if c in X.columns]
            if not cols:
                continue
            block_data = X[cols].values
            if np.isnan(block_data).sum() == 0:
                continue
            if np.isnan(block_data).all():
                continue
            imp = IterativeImputer(
                max_iter=10,
                random_state=random_state + m,
                sample_posterior=True,
            )
            try:
                imputed = imp.fit_transform(block_data)
                result[cols] = imputed
            except Exception:  # noqa: BLE001
                result[cols] = result[cols].fillna(result[cols].median())
        imputed_datasets.append(result)

    return imputed_datasets
