"""Orchestrates per-row profile extraction → unified V1 feature matrix.

Exposes a single top-level entry point, :func:`build_harmonized_dataset`,
which takes the four FACE cohort CSVs, runs the existing ``face_rlvr.profiles``
extractors on every row, passes each resulting ``PatientData`` through the
matching cohort adapter, and stacks everything into one DataFrame indexed by
``(cohort, patient_id)``.

The result is a :class:`HarmonizedDataset` tuple containing:

- ``X``: the unified V1 feature matrix as a ``pandas.DataFrame``
  with columns in the order declared by the schema.
- ``metadata``: a side DataFrame with the cohort label and a DSM diagnosis
  string per patient. This is **never** used by similarity / model code — it
  lives here purely so downstream comparison code can join it back after
  clustering.
- ``feature_metadata``: a DataFrame describing every column of ``X``
  (block, type, temporal scope, direction, which cohorts can provide it).

The pipeline is:

    CSV rows → face_rlvr.extract_*_patient → adapt_*_profile → dict → X row

No CSV parsing logic lives in this module. Failures on individual rows are
logged and skipped (the whole 9 k-row build must never crash on a single bad
patient).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from face_rlvr.profiles import (
    extract_asp_patient,
    extract_bp_patient,
    extract_dr_patient,
    extract_sz_patient,
)

from face_stratification.harmonization.cohort_adapters import (
    adapt_asp_profile,
    adapt_bp_profile,
    adapt_dr_profile,
    adapt_sz_profile,
)
from face_stratification.harmonization.feature_schema import (
    FeatureSchema,
    load_feature_schema,
)


logger = logging.getLogger(__name__)


# ─── Public types ─────────────────────────────────────────────────────────────


@dataclass
class HarmonizedDataset:
    """Result of :func:`build_harmonized_dataset`.

    Attributes
    ----------
    X:
        Unified V1 feature matrix. Index is a ``MultiIndex[cohort, patient_id]``.
        Columns are the feature ids in the order declared by the schema. Values
        are ``float`` (or ``NaN`` for missing / not-provided by the cohort).
    metadata:
        Side table indexed identically to ``X``, with columns ``cohort`` and
        ``dsm_diagnosis`` (plus any arm-level label available in the profile).
        Never used during similarity / model training — kept only for downstream
        comparison.
    feature_metadata:
        One row per column of ``X`` describing its schema entry.
    schema:
        The :class:`FeatureSchema` used to build this dataset.
    """

    X: pd.DataFrame
    metadata: pd.DataFrame
    feature_metadata: pd.DataFrame
    schema: FeatureSchema

    def __post_init__(self) -> None:
        # Strict cross-sectional invariant: one row per (cohort, patient_id).
        if not self.X.index.is_unique:
            raise ValueError(
                "Harmonized matrix has duplicate (cohort, patient_id) rows — "
                "V1-only builds must contain exactly one row per patient."
            )
        if list(self.X.index) != list(self.metadata.index):
            raise ValueError("X and metadata indices do not match.")

    @property
    def n_patients(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def cohort_counts(self) -> pd.Series:
        return self.metadata["cohort"].value_counts().sort_index()

    def feature_availability(self) -> pd.DataFrame:
        """For each feature, return coverage (% non-null) per cohort and overall."""
        rows: list[dict[str, Any]] = []
        for feat_id in self.X.columns:
            row: dict[str, Any] = {"feature_id": feat_id}
            for cohort, sub in self.X.groupby(level="cohort"):
                row[f"coverage_{cohort}"] = float(sub[feat_id].notna().mean())
            row["coverage_total"] = float(self.X[feat_id].notna().mean())
            rows.append(row)
        return pd.DataFrame(rows).set_index("feature_id")


# ─── Cohort dispatch tables ──────────────────────────────────────────────────


_Extractor = Callable[[pd.Series], Any]
_Adapter = Callable[[Any], dict[str, float | None]]


_COHORT_DISPATCH: dict[str, tuple[_Extractor, _Adapter]] = {
    "bp": (extract_bp_patient, adapt_bp_profile),
    "sz": (extract_sz_patient, adapt_sz_profile),
    "dr": (extract_dr_patient, adapt_dr_profile),
    "asp": (extract_asp_patient, adapt_asp_profile),
}


# ─── Diagnosis extraction (metadata only) ────────────────────────────────────


def _diagnosis_for(cohort: str, data: Any) -> str:
    """Return a coarse DSM-style diagnosis string for the metadata side table.

    These labels are never used by the harmonization / graph / model code — they
    exist purely so downstream comparison can measure cluster-vs-DSM agreement.
    """
    if cohort == "bp":
        arm = getattr(data.demographics, "arm", None)
        return f"BP:{arm}" if arm else "BP"
    if cohort == "sz":
        return "SZ"
    if cohort == "dr":
        return "DR"
    if cohort == "asp":
        dsm_label = getattr(data.autism_diagnosis, "dsm_type_label", None)
        return f"ASP:{dsm_label}" if dsm_label else "ASP"
    return cohort.upper()


# ─── Main entry point ────────────────────────────────────────────────────────


def build_harmonized_dataset(
    csv_paths: dict[str, str | Path],
    *,
    schema: FeatureSchema | None = None,
    max_rows_per_cohort: int | None = None,
    cohorts: tuple[str, ...] = ("bp", "sz", "dr", "asp"),
) -> HarmonizedDataset:
    """Build the unified V1 matrix from the four FACE cohort CSVs.

    Parameters
    ----------
    csv_paths:
        Mapping from cohort code (``"bp"`` / ``"sz"`` / ``"dr"`` / ``"asp"``) to
        the path of the cohort CSV. Missing keys are silently skipped.
    schema:
        Optional pre-loaded schema. Defaults to the shipped
        ``config/face_stratification/feature_schema.yaml``.
    max_rows_per_cohort:
        If set, read only this many rows per cohort — useful for smoke tests
        and notebooks. ``None`` means all rows.
    cohorts:
        Ordered tuple of cohort codes to include.

    Returns
    -------
    :class:`HarmonizedDataset`
    """
    schema = schema or load_feature_schema()
    feature_ids = list(schema.feature_ids())

    rows: list[dict[str, float | None]] = []
    meta_rows: list[dict[str, Any]] = []
    index_tuples: list[tuple[str, str]] = []

    for cohort in cohorts:
        if cohort not in csv_paths:
            logger.info("Cohort %s not in csv_paths, skipping", cohort)
            continue
        extractor, adapter = _COHORT_DISPATCH[cohort]
        path = Path(csv_paths[cohort])
        if not path.is_file():
            logger.warning("CSV for cohort %s not found at %s; skipping", cohort, path)
            continue

        logger.info("Loading cohort %s from %s", cohort, path)
        df = pd.read_csv(path, nrows=max_rows_per_cohort, low_memory=False)
        logger.info("Cohort %s: %d rows", cohort, len(df))

        for row_idx, csv_row in df.iterrows():
            try:
                data = extractor(csv_row)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Extraction failed for %s row %s: %s", cohort, row_idx, exc
                )
                continue

            try:
                feat_dict = adapter(data)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Adapter failed for %s patient %s: %s",
                    cohort,
                    getattr(data, "patient_id", row_idx),
                    exc,
                )
                continue

            pid = str(getattr(data, "patient_id", row_idx))
            idx_tuple = (cohort, pid)

            # Ensure every schema feature has a slot (None if adapter omitted it)
            normalized = {fid: feat_dict.get(fid) for fid in feature_ids}

            rows.append(normalized)
            meta_rows.append(
                {
                    "cohort": cohort,
                    "patient_id": pid,
                    "dsm_diagnosis": _diagnosis_for(cohort, data),
                }
            )
            index_tuples.append(idx_tuple)

    if not rows:
        raise RuntimeError("No patients successfully harmonized; check csv_paths.")

    index = pd.MultiIndex.from_tuples(index_tuples, names=("cohort", "patient_id"))

    X = pd.DataFrame(rows, index=index, columns=feature_ids, dtype="float64")

    metadata = pd.DataFrame(
        meta_rows, index=index, columns=["cohort", "patient_id", "dsm_diagnosis"]
    )
    # The cohort label lives both in the index and as a column for convenience.
    metadata["cohort"] = [c for c, _ in index_tuples]

    feature_metadata = _build_feature_metadata(schema)

    # V1 invariant: assert no longitudinal column slipped through
    _assert_v1_only(X)

    return HarmonizedDataset(
        X=X,
        metadata=metadata,
        feature_metadata=feature_metadata,
        schema=schema,
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _build_feature_metadata(schema: FeatureSchema) -> pd.DataFrame:
    records = []
    for feat in schema.features:
        records.append(
            {
                "feature_id": feat.id,
                "label_fr": feat.label_fr,
                "block": feat.block,
                "type": feat.type.value,
                "temporal_scope": feat.temporal_scope.value,
                "unit": feat.unit,
                "direction": feat.direction,
                "cohorts": ",".join(feat.cohorts),
            }
        )
    return pd.DataFrame(records).set_index("feature_id")


_FORBIDDEN_SUFFIXES = (
    "_n1",
    "_followup",
    "_delta",
    "_rci",
    "_change",
    "_v2",
    "_visit2",
)


def _assert_v1_only(X: pd.DataFrame) -> None:
    """Guardrail: reject any longitudinal column name that slipped into X."""
    offenders = [
        c for c in X.columns if any(c.endswith(suffix) for suffix in _FORBIDDEN_SUFFIXES)
    ]
    if offenders:
        raise ValueError(
            f"Harmonized matrix contains longitudinal-looking columns: {offenders}. "
            "face_stratification is V1-only — remove them from the schema."
        )
    # Also assert there's no 'dsm_diagnosis' column in X (it lives in metadata).
    if "dsm_diagnosis" in X.columns:
        raise ValueError(
            "'dsm_diagnosis' must not appear in the feature matrix — it belongs in metadata."
        )
