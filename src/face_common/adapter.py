"""Adapt OUR unified frame into the engine's ``HarmonizedDataset``.

This is the bridge between our common-variables pipeline and the vendored
``face_stratification`` clustering engine. The engine consumes a
:class:`HarmonizedDataset`:

    X                : float matrix, MultiIndex[cohort, patient_id], NaN = missing
    metadata         : cohort + dsm_diagnosis per patient (never used by the math)
    feature_metadata : one row per column (block / type / direction / cohorts)
    schema           : a :class:`FeatureSchema`

We take a single visit (V0 by default) of our long-format frame, keep the
numeric feature columns, and reshape into that contract. Cohort labels are
lowercased (``BP``→``bp``) to match the engine's vocabulary, and the
within-cohort ``usubjid_patients`` becomes ``patient_id`` — the
``(cohort, patient_id)`` pair is globally unique even though ``usubjid_patients``
collides across cohorts.

No imputation is performed: missing values stay ``NaN`` and the engine's masked
pairwise-complete similarity handles them.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

from face_stratification.harmonization.harmonizer import HarmonizedDataset

from .filters import IDENTIFIER_COLUMNS
from .schema_gen import DEFAULT_SCHEMA_VERSION, build_feature_schema, feature_cohorts
from .variable import Variable

__all__ = [
    "to_harmonized_dataset",
    "normalize_for_embedding",
    "residualize_features",
    "COHORT_TO_CODE",
    "ADMINISTRATIVE_FEATURES",
    "CLINICAL_SECTIONS",
]

# our cohort labels → engine cohort codes
COHORT_TO_CODE = {"BP": "bp", "SZ": "sz", "DR": "dr"}

# Non-phenotype features that should not drive clustering (recruitment site is a
# confound, not a clinical axis). Scripts pass these via ``exclude=``.
ADMINISTRATIVE_FEATURES = frozenset({"siteid_city"})

# Dictionary sections that carry psychiatric phenotype (symptoms, illness course,
# functioning, history). Used via ``sections=`` to cluster on clinical signal and
# drop physiology (labs, vitals/anthropometry), cognition (neuropsych — strongly
# missingness/availability-confounded) and raw demographics, which otherwise
# dominate cosine similarity and produce sex×age strata rather than phenotypes.
CLINICAL_SECTIONS = frozenset({
    "AUTO-QUESTIONNAIRES",
    "HETERO-QUESTIONNAIRES",
    "SUICIDE",
    "EVALUATION MEDICALE",
    "SUBSTANCES",
    "ANTECEDENTS",
    "SOIN SUIVI HOSP ARRET TRAVAIL",
    "SOCIAL",
})

# A feature with more than this many distinct observed values is treated as
# continuous for scaling (the dictionary's dtype labels are unreliable — many
# continuous lab/neuropsych columns are tagged 'string'/'categorical').
_MIN_UNIQUE_CONTINUOUS = 10


def normalize_for_embedding(
    X: pd.DataFrame,
    *,
    min_unique: int = _MIN_UNIQUE_CONTINUOUS,
    winsor: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """Robust per-feature scaling so cosine similarity isn't dominated by scale.

    Cosine is scale-invariant *per patient* but not *per feature*: a raw column
    with large magnitude (a date encoded as 1e17, a lab count in the thousands)
    swamps binary 0/1 flags and z-scored scales. We rescale every empirically
    **continuous** column (> ``min_unique`` distinct values) by winsorizing to
    the 1st/99th percentile and robust z-scoring (median / MAD). Genuinely
    discrete columns (binary, ordinal, low-cardinality categorical) pass through
    on their native small scale. NaNs are preserved — no imputation.
    """
    lo_q, hi_q = winsor
    out = X.copy()
    for c in out.columns:
        col = out[c]
        if col.nunique(dropna=True) <= min_unique:
            continue  # discrete: leave on its native small scale
        lo, hi = col.quantile(lo_q), col.quantile(hi_q)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            col = col.clip(lower=lo, upper=hi)
        med = col.median()
        mad = (col - med).abs().median()
        scale = 1.4826 * mad if mad and mad > 0 else (col.std() or 1.0)
        out[c] = (col - med) / (scale if scale > 0 else 1.0)
    return out.replace([np.inf, -np.inf], np.nan)


def residualize_features(X: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """Remove the linear effect of ``covariates`` from each feature (OLS residuals).

    For every feature column we regress its observed values on ``[1, *covariates]``
    and replace them with residuals, so clustering reflects phenotype **net of**
    those covariates (here: age + sex). Covariate NaNs are mean-imputed for the
    design matrix only; feature NaNs are preserved (no imputation of features).
    """
    design = covariates.fillna(covariates.mean(numeric_only=True))
    A = np.column_stack([np.ones(len(design)), design.to_numpy(dtype="float64")])
    out = X.copy()
    for col in out.columns:
        y = out[col].to_numpy(dtype="float64")
        obs = np.isfinite(y)
        if int(obs.sum()) < A.shape[1] + 2:
            continue  # too few observations to fit reliably; leave as-is
        beta, *_ = np.linalg.lstsq(A[obs], y[obs], rcond=None)
        y[obs] = y[obs] - A[obs] @ beta
        out[col] = y
    return out


def _feature_metadata(schema) -> pd.DataFrame:
    """One row per schema feature (mirrors the engine's harmonizer output)."""
    records = [
        {
            "feature_id": f.id,
            "label_fr": f.label_fr,
            "block": f.block,
            "type": f.type.value,
            "temporal_scope": f.temporal_scope.value,
            "unit": f.unit,
            "direction": f.direction,
            "cohorts": ",".join(f.cohorts),
        }
        for f in schema.features
    ]
    return pd.DataFrame(records).set_index("feature_id")


def to_harmonized_dataset(
    df: pd.DataFrame,
    variables: Iterable[Variable],
    *,
    visit: str | None = "V0",
    min_coverage: float = 0.0,
    exclude: Iterable[str] | None = None,
    sections: Iterable[str] | None = None,
    residualize_on: Iterable[str] | None = None,
    normalize: bool = False,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
) -> HarmonizedDataset:
    """Build a :class:`HarmonizedDataset` from our unified long-format frame.

    Parameters
    ----------
    df:
        Output of :func:`build_unified_dataframe` (``format="long"``). Must carry
        ``cohort``, ``usubjid_patients`` and (if ``visit`` is set) a ``visit``
        column.
    variables:
        The dictionary (``load_variables(...)``), used to type features and to
        determine each feature's declared cohorts.
    visit:
        Restrict to this visit before reshaping (default ``"V0"``). Pass ``None``
        to use the frame as-is (caller must guarantee one row per patient).
    min_coverage:
        Optional global completeness floor: drop feature columns observed in
        fewer than this fraction of patients. Defaults to ``0.0`` (keep all;
        the engine applies its own per-partition coverage filter downstream).
    exclude:
        Canonical names to drop from the feature matrix (e.g. administrative /
        confound features such as recruitment site). See
        :data:`ADMINISTRATIVE_FEATURES`.
    sections:
        If given, keep only features whose dictionary ``section`` is in this set
        (e.g. :data:`CLINICAL_SECTIONS` to cluster on psychiatric phenotype and
        drop physiology / cognition / demographics).
    residualize_on:
        Canonical names of covariates (e.g. ``("age", "sex")``) to regress out of
        every feature via :func:`residualize_features`, so clustering reflects
        phenotype *net of* those covariates. The covariates are read before the
        ``sections`` filter (so they need not be kept) and are never themselves
        emitted as features.
    normalize:
        If ``True``, apply :func:`normalize_for_embedding` (robust per-feature
        scaling) so the cosine embedding isn't dominated by raw magnitude. Leave
        ``False`` to keep raw values (e.g. for rank-based enrichment, which is
        scale-invariant). Date-typed dictionary features are always dropped.
    schema_version:
        Version string stamped on the generated schema and embedding artifacts.
    """
    variables = list(variables)
    by_name = {v.canonical_name: v for v in variables}
    exclude_set = set(exclude or ())

    sub = df
    if visit is not None:
        if "visit" not in df.columns:
            raise ValueError("frame has no 'visit' column; pass visit=None")
        sub = df[df["visit"] == visit]
    sub = sub.reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"no rows for visit={visit!r}")

    for col in ("cohort", "usubjid_patients"):
        if col not in sub.columns:
            raise ValueError(f"frame missing required identifier column {col!r}")

    def _build_index(frame: pd.DataFrame):
        code = frame["cohort"].map(COHORT_TO_CODE)
        if code.isna().any():
            bad = sorted(set(frame.loc[code.isna(), "cohort"]))
            raise ValueError(f"unmapped cohort label(s): {bad}")
        pid = frame["usubjid_patients"].astype(str)
        idx = pd.MultiIndex.from_arrays(
            [code.to_numpy(), pid.to_numpy()], names=("cohort", "patient_id")
        )
        return code, pid, idx

    cohort_code, patient_id, index = _build_index(sub)

    # One row per (cohort, patient_id). V0 is the inclusion visit, so this should
    # already hold; drop accidental duplicates defensively and warn, then rebuild
    # the index (and all derived arrays) from the deduplicated frame.
    dup = index.duplicated(keep="first")
    if dup.any():
        warnings.warn(
            f"{int(dup.sum())} duplicate (cohort, patient_id) rows at visit={visit!r}; "
            "keeping first occurrence.",
            stacklevel=2,
        )
        sub = sub[~dup].reset_index(drop=True)
        cohort_code, patient_id, index = _build_index(sub)

    # Candidate feature columns: non-identifier, dictionary-backed, not excluded,
    # not a date column, with a source column in at least one cohort.
    feature_cols = [
        c
        for c in sub.columns
        if c not in IDENTIFIER_COLUMNS
        and c in by_name
        and c not in exclude_set
        and "date" not in by_name[c].dtype.lower()
        and feature_cohorts(by_name[c])
    ]

    # Numeric-coerce; keep columns with any observed value (string/date features
    # become all-NaN here and are dropped — they are not clustered on).
    numeric: dict[str, pd.Series] = {}
    for c in feature_cols:
        s = pd.to_numeric(sub[c], errors="coerce")
        if s.notna().any():
            numeric[c] = s.astype("float64")
    if not numeric:
        raise ValueError("no numeric feature columns survived coercion")

    full = pd.DataFrame(numeric)
    full.index = index

    # Covariates for residualization come from the full numeric set (before the
    # section filter), so e.g. age/sex are available even if PATIENT is dropped.
    covar_df = None
    if residualize_on:
        residualize_on = list(residualize_on)
        missing = [c for c in residualize_on if c not in full.columns]
        if missing:
            raise ValueError(f"residualize_on covariates unavailable as numeric: {missing}")
        covar_df = full[residualize_on]

    # Feature set: optional section filter, minus any covariate columns.
    keep = list(full.columns)
    if sections is not None:
        section_set = set(sections)
        keep = [c for c in keep if by_name[c].section in section_set]
    if residualize_on:
        keep = [c for c in keep if c not in set(residualize_on)]
    if not keep:
        raise ValueError("no feature columns left after section / covariate filtering")
    X = full[keep]

    if residualize_on:
        X = residualize_features(X, covar_df)

    if min_coverage > 0:
        coverage = X.notna().mean()
        X = X.loc[:, coverage[coverage >= min_coverage].index]
        if X.shape[1] == 0:
            raise ValueError(f"no feature met min_coverage={min_coverage}")

    if normalize:
        X = normalize_for_embedding(X)

    # Metadata: DSM diagnosis = arm subtype when present, else cohort code.
    if "arm" in sub.columns:
        arm = sub["arm"].astype("object").where(sub["arm"].notna(), None)
        dsm = [a if a else c.upper() for a, c in zip(arm, cohort_code)]
    else:
        dsm = [c.upper() for c in cohort_code]
    metadata = pd.DataFrame(
        {
            "cohort": cohort_code.to_numpy(),
            "patient_id": patient_id.to_numpy(),
            "dsm_diagnosis": dsm,
        },
        index=index,
    )

    schema = build_feature_schema(variables, list(X.columns), version=schema_version)
    feature_metadata = _feature_metadata(schema).reindex(X.columns)

    return HarmonizedDataset(
        X=X,
        metadata=metadata,
        feature_metadata=feature_metadata,
        schema=schema,
    )
