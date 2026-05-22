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

__all__ = ["to_harmonized_dataset", "COHORT_TO_CODE"]

# our cohort labels → engine cohort codes
COHORT_TO_CODE = {"BP": "bp", "SZ": "sz", "DR": "dr"}


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
    schema_version:
        Version string stamped on the generated schema and embedding artifacts.
    """
    variables = list(variables)
    by_name = {v.canonical_name: v for v in variables}

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

    # Candidate feature columns: non-identifier, dictionary-backed, with a source
    # column in at least one cohort.
    feature_cols = [
        c
        for c in sub.columns
        if c not in IDENTIFIER_COLUMNS
        and c in by_name
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

    X = pd.DataFrame(numeric)
    X.index = index

    if min_coverage > 0:
        coverage = X.notna().mean()
        X = X.loc[:, coverage[coverage >= min_coverage].index]
        if X.shape[1] == 0:
            raise ValueError(f"no feature met min_coverage={min_coverage}")

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
