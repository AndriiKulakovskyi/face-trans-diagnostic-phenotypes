"""Robust normalization for the unified V1 feature matrix.

Applies a per-feature transformation that respects the feature's statistical
type:

- ``continuous`` → winsorize (1st / 99th percentile) + robust z-score (median,
  MAD-derived scale),
- ``ordinal`` / ``binary`` / ``categorical`` → pass through unchanged (they are
  already on comparable numeric scales after adapter encoding),
- ``direction == "higher_is_better"`` → sign-flip after z-scoring so higher
  values always mean *worse* (more pathological).

Two normalization scopes are supported:

- **Global** (default, :func:`fit_normalization` + :func:`transform_normalization`)
  — fits one set of robust stats on the whole cohort. Preserves raw-score
  comparability across cohorts, which is what the transdiagnostic pipeline
  needs but introduces a mild bias toward the largest cohort.
- **Per-cohort** (:func:`fit_per_cohort_normalization` +
  :func:`transform_per_cohort_normalization`) — fits one set of robust stats
  *per cohort* and applies them only to rows of that cohort. Removes all
  per-cohort scale differences (good for within-cohort clustering) at the
  cost of destroying transdiagnostic signal tied to raw-score distributions.
  Primarily used for ablation studies — see
  :mod:`face_stratification.analysis.ablation`.

The transformation is fit on a training split and then replayed on any matrix
with the same columns — see :class:`NormalizationStats`.

Only pandas + numpy are used here; the rest of the sub-project may bring in
scikit-learn later, but this helper intentionally stays lightweight so it can
run in the minimal install.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from face_stratification.harmonization.feature_schema import (
    FeatureSchema,
    FeatureType,
)


_WINSOR_LOW = 0.01
_WINSOR_HIGH = 0.99


@dataclass
class NormalizationStats:
    """Fitted per-feature normalization parameters."""

    medians: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)  # MAD-derived scale
    lower_clip: dict[str, float] = field(default_factory=dict)
    upper_clip: dict[str, float] = field(default_factory=dict)
    sign: dict[str, float] = field(default_factory=dict)  # +1 or -1
    feature_types: dict[str, str] = field(default_factory=dict)


def fit_normalization(X: pd.DataFrame, schema: FeatureSchema) -> NormalizationStats:
    """Fit winsorization + robust z-score stats on ``X``.

    Continuous features get full winsorize+z; all other types are recorded with
    pass-through values (median=0, scale=1) so that :func:`transform_normalization`
    is a no-op for them while still being indexable in the same way.
    """
    by_id = schema.features_by_id()
    stats = NormalizationStats()

    for feat_id in X.columns:
        feat = by_id.get(feat_id)
        if feat is None:
            # Unknown column — leave as-is (but record pass-through params)
            stats.medians[feat_id] = 0.0
            stats.scales[feat_id] = 1.0
            stats.lower_clip[feat_id] = -np.inf
            stats.upper_clip[feat_id] = np.inf
            stats.sign[feat_id] = 1.0
            stats.feature_types[feat_id] = "unknown"
            continue

        stats.feature_types[feat_id] = feat.type.value

        col = X[feat_id].astype("float64")
        if feat.type is FeatureType.CONTINUOUS:
            # Suppress "All-NaN slice" / "Degrees of freedom <= 0" warnings for
            # degenerate columns — we handle them explicitly below.
            with np.errstate(invalid="ignore"):
                lo = float(col.quantile(_WINSOR_LOW))
                hi = float(col.quantile(_WINSOR_HIGH))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = -np.inf, np.inf
            clipped = col.clip(lower=lo, upper=hi)
            with np.errstate(invalid="ignore"):
                median_val = np.nanmedian(clipped) if clipped.notna().any() else 0.0
                mad_val = (
                    np.nanmedian(np.abs(clipped - median_val))
                    if clipped.notna().any()
                    else 0.0
                )
                std_val = float(np.nanstd(clipped)) if clipped.notna().any() else 0.0
            median = float(median_val) if np.isfinite(median_val) else 0.0
            scale = 1.4826 * float(mad_val) if mad_val and mad_val > 0 else (std_val or 1.0)
            scale = scale if scale > 0 else 1.0

            stats.medians[feat_id] = median
            stats.scales[feat_id] = scale
            stats.lower_clip[feat_id] = lo
            stats.upper_clip[feat_id] = hi
        else:
            # Pass-through (ordinal / binary / categorical)
            stats.medians[feat_id] = 0.0
            stats.scales[feat_id] = 1.0
            stats.lower_clip[feat_id] = -np.inf
            stats.upper_clip[feat_id] = np.inf

        # Sign flip: we want "higher = worse" everywhere so similarity metrics
        # treat the two ends consistently.
        stats.sign[feat_id] = -1.0 if feat.direction == "higher_is_better" else 1.0

    return stats


def transform_normalization(
    X: pd.DataFrame, stats: NormalizationStats
) -> pd.DataFrame:
    """Apply :class:`NormalizationStats` to ``X`` and return a new DataFrame.

    NaNs are preserved (never imputed here — imputation lives in
    :mod:`face_stratification.harmonization.missingness`). Any ``±inf`` values
    produced by degenerate columns (all-NaN, constant) are replaced with NaN so
    downstream imputers and kNN builders see only finite values.
    """
    out = X.copy()
    for feat_id in out.columns:
        if feat_id not in stats.medians:
            continue
        if stats.feature_types.get(feat_id) == "continuous":
            col = out[feat_id].astype("float64")
            lo = stats.lower_clip[feat_id]
            hi = stats.upper_clip[feat_id]
            # Only clip if bounds are finite; otherwise pass through
            if np.isfinite(lo) and np.isfinite(hi):
                col = col.clip(lower=lo, upper=hi)
            col = (col - stats.medians[feat_id]) / stats.scales[feat_id]
            out[feat_id] = col * stats.sign[feat_id]
        else:
            out[feat_id] = out[feat_id].astype("float64") * stats.sign[feat_id]

    # Any lingering ±inf → NaN (safe for all downstream consumers)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ─── Per-cohort normalization (ablation only) ─────────────────────────────────


@dataclass
class PerCohortNormalizationStats:
    """A collection of :class:`NormalizationStats`, one per cohort.

    Used exclusively by the ablation study comparing global vs per-cohort
    normalization. Not on the default Stage A path — global normalization
    is still the canonical choice for transdiagnostic analysis.
    """

    stats_by_cohort: dict[str, NormalizationStats] = field(default_factory=dict)


def _cohort_labels_of(df: pd.DataFrame) -> pd.Series:
    """Extract the cohort label as a Series indexed identically to ``df``.

    Works whether ``cohort`` lives on the MultiIndex or as a column.
    """
    if "cohort" in df.index.names:
        return pd.Series(df.index.get_level_values("cohort"), index=df.index, name="cohort")
    if "cohort" in df.columns:
        return df["cohort"]
    raise ValueError("DataFrame must expose a 'cohort' index level or column.")


def fit_per_cohort_normalization(
    X: pd.DataFrame,
    schema: FeatureSchema,
) -> PerCohortNormalizationStats:
    """Fit one :class:`NormalizationStats` per cohort present in ``X``.

    The resulting object behaves exactly like
    :class:`NormalizationStats`' collective for each cohort — call
    :func:`transform_per_cohort_normalization` to apply it.
    """
    cohort = _cohort_labels_of(X)
    out = PerCohortNormalizationStats()
    for name in pd.unique(cohort):
        mask = cohort == name
        sub = X.loc[mask]
        out.stats_by_cohort[str(name)] = fit_normalization(sub, schema)
    return out


def transform_per_cohort_normalization(
    X: pd.DataFrame,
    stats: PerCohortNormalizationStats,
) -> pd.DataFrame:
    """Apply :func:`fit_per_cohort_normalization`'s output, row by cohort.

    Rows whose cohort was not seen at fit time are left untouched and a
    warning is logged — this is never the case in the default ablation but
    the safety check prevents silent corruption.
    """
    import logging

    logger = logging.getLogger(__name__)

    cohort = _cohort_labels_of(X)
    # Start from a copy so untouched rows pass through.
    out = X.copy()

    for name in pd.unique(cohort):
        mask = cohort == name
        s = stats.stats_by_cohort.get(str(name))
        if s is None:
            logger.warning(
                "No per-cohort normalization stats for cohort %r; passing through.",
                name,
            )
            continue
        sub = X.loc[mask]
        transformed = transform_normalization(sub, s)
        out.loc[mask] = transformed.values

    # Any residual ±inf → NaN, same as the global variant.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
