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

from .filters import IDENTIFIER_COLUMNS
from .harmonized_dataset import HarmonizedDataset
from .schema_gen import DEFAULT_SCHEMA_VERSION, build_feature_schema, feature_cohorts
from .skip_logic import decode_skip_logic
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

# Dictionary sections that carry psychiatric *symptom* phenotype (symptoms, illness
# course, functioning, history). Used via ``sections=`` to cluster on clinical signal and
# drop raw demographics, which otherwise dominate similarity and produce sex×age strata.
# Biology (labs/vitals) and cognition (neuropsych) are deliberately NOT in this set: each
# has its own curated-aggregation path (BIOLOGY_COMPOSITES, COGNITIVE_COMPOSITES in
# domains.py) so item-count and units don't distort them. Cognition was excluded entirely
# before 2026 because DR coverage was 0% (an extraction artifact); with DR recovered it now
# enters the model as curated constructs and its availability is checked explicitly (15).
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
    clip: float = 5.0,
) -> pd.DataFrame:
    """Type-aware per-feature scaling to a bounded **[-1, 1]** range, so the masked cosine
    embedding isn't dominated by scale and every feature is comparable & ML-ready. NaNs are
    preserved — no imputation.

    Per column, by empirical cardinality:
      * binary / ordinal / Likert (<= ``min_unique`` distinct) -> min-max to [-1, 1];
      * continuous (> ``min_unique``) -> winsorize (1/99) + robust z (median / MAD), clipped to
        ±``clip`` (guards explosive z on tiny-MAD log-normal columns such as prolactin), then
        divided by ``clip`` -> [-1, 1].
    All outputs lie in [-1, 1]; a constant column maps to 0 (NaN kept). Previously discrete
    columns were left on their native scale, mixing 0/1 flags with unbounded z-scores.
    """
    lo_q, hi_q = winsor
    out = X.copy()
    for c in out.columns:
        col = out[c]
        nu = int(col.nunique(dropna=True))
        if nu == 0:
            continue
        if nu <= min_unique:                        # binary / ordinal / Likert -> [-1, 1]
            lo, hi = col.min(), col.max()
            out[c] = (2.0 * (col - lo) / (hi - lo) - 1.0) if hi > lo else col * 0.0
        else:                                       # continuous -> robust-z-clip -> [-1, 1]
            lo, hi = col.quantile(lo_q), col.quantile(hi_q)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                col = col.clip(lower=lo, upper=hi)
            med = col.median()
            mad = (col - med).abs().median()
            scale = 1.4826 * mad if mad and mad > 0 else (col.std() or 1.0)
            z = (col - med) / (scale if scale > 0 else 1.0)
            out[c] = z.clip(lower=-clip, upper=clip) / clip
    return out.replace([np.inf, -np.inf], np.nan)


def _design_matrix(covariates: pd.DataFrame, spline_df: int) -> np.ndarray:
    """Confounder design: intercept + (spline or linear) covariates + interactions.

    Continuous covariates (> 10 distinct values) are natural-spline-expanded when
    ``spline_df > 0`` (degree-3 B-spline, ``spline_df`` knots); discrete ones stay
    linear. Each continuous spline block is also interacted with each discrete
    covariate (e.g. sex-specific age curves). Covariate NaNs are mean-imputed for
    the design only.
    """
    blocks: list[np.ndarray] = [np.ones((len(covariates), 1))]
    continuous: list[np.ndarray] = []
    discrete: list[np.ndarray] = []
    for name in covariates.columns:
        col = covariates[name]
        x = col.to_numpy(dtype="float64").reshape(-1, 1)
        mean = float(np.nanmean(x)) if np.isfinite(np.nanmean(x)) else 0.0
        x = np.nan_to_num(x, nan=mean)
        if spline_df > 0 and col.nunique(dropna=True) > 10:
            try:
                from sklearn.preprocessing import SplineTransformer
                basis = SplineTransformer(
                    n_knots=spline_df, degree=3, include_bias=False
                ).fit_transform(x)
            except Exception:  # pragma: no cover - fallback if sklearn too old
                basis = np.column_stack([x, x ** 2, x ** 3])
            continuous.append(basis)
            blocks.append(basis)
        else:
            discrete.append(x)
            blocks.append(x)
    for basis in continuous:          # continuous × discrete interactions
        for xb in discrete:
            blocks.append(basis * xb)
    return np.column_stack(blocks)


def residualize_features(
    X: pd.DataFrame,
    covariates: pd.DataFrame,
    *,
    spline_df: int = 0,
    cross_fit: int = 1,
    random_state: int = 0,
) -> pd.DataFrame:
    """Remove the effect of ``covariates`` from each feature (regression residuals).

    Clustering then reflects phenotype **net of** the covariates (here: age + sex).
    Feature NaNs are preserved (no imputation of features); covariate NaNs are
    mean-imputed for the design matrix only.

    Parameters
    ----------
    spline_df:
        ``0`` (default) → covariates enter linearly (plain OLS partialling-out,
        backward compatible). ``> 0`` → natural-spline-expand continuous
        covariates + add covariate×discrete interactions, i.e. **double-ML
        "partialling out" of nonlinear confounding** (e.g. nonlinear age, sex-
        specific age curves).
    cross_fit:
        ``1`` (default) → in-sample residuals. ``> 1`` → K-fold **cross-fitting**:
        each feature's residuals are computed out-of-fold to avoid overfitting /
        over-correction (Chernozhukov et al. double/debiased ML).
    """
    A = _design_matrix(covariates, spline_df)
    out = X.copy()
    min_obs = A.shape[1] + 2

    if cross_fit and cross_fit > 1:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cross_fit, shuffle=True, random_state=random_state)
        for col in out.columns:
            y = out[col].to_numpy(dtype="float64").copy()  # writable (newer numpy returns read-only)
            obs = np.where(np.isfinite(y))[0]
            if obs.size < max(min_obs, cross_fit):
                continue
            resid = y.copy()
            for tr, te in kf.split(obs):
                tr_i, te_i = obs[tr], obs[te]
                beta, *_ = np.linalg.lstsq(A[tr_i], y[tr_i], rcond=None)
                resid[te_i] = y[te_i] - A[te_i] @ beta
            out[col] = resid
    else:
        for col in out.columns:
            y = out[col].to_numpy(dtype="float64").copy()  # writable (newer numpy returns read-only)
            obs = np.isfinite(y)
            if int(obs.sum()) < min_obs:
                continue
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
    apply_skip_logic: bool = True,
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
    apply_skip_logic:
        If ``True`` (default), decode instrument skip-logic on the raw numeric
        matrix via :func:`~face.data.skip_logic.decode_skip_logic`: where a gate
        determines a conditional item and that item is missing, fill the
        structural zero (e.g. ISF05="never attempted" implies zero attempts;
        smoking status="never smoker" implies zero lifetime pack-years). This is
        not statistical imputation: only cells fixed by instrument logic are
        filled. Set ``False`` to keep the raw structural blanks as NaN.
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

    # Decode instrument skip-logic (gate=No -> structural 0) on the raw numeric
    # matrix, BEFORE the section/covariate filter and any scaling, so the
    # recovered zeros flow into both the direct feature matrix and the domain
    # scores (build_domain_scores consumes this X). No imputation: only cells the
    # instrument's own skip-logic determines are filled (see skip_logic.py).
    if apply_skip_logic:
        full, _ = decode_skip_logic(full)

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
        dsm = [a if a else c.upper() for a, c in zip(arm, cohort_code, strict=False)]
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
