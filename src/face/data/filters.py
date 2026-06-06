"""Missingness filters and V0-anchored selection.

Two independent primitives:

  - `filter_variables(df, threshold, visit=None)` — drops feature columns
    whose completeness < threshold at the chosen visit (or across all rows
    if visit is None).
  - `filter_patients(df, threshold, visit=None, variables=None)` — drops
    patients whose completeness across the feature columns < threshold at
    the chosen visit. In anchor mode (`visit='V0', keep_other_visits=True`)
    the surviving patients keep all their visit rows.

Compose them into the pre-registered V0-anchored workflow via
`select_v0_anchor(df, ...)`, which returns a `V0Anchor` whose `.apply(df)`
projects the V0 selection onto any visit's data.

All functions operate on the long-format DataFrame produced by
`face.data.build_unified_dataframe(format='long')`. Identifier columns
(`usubjid_patients`, `cohort`, `arm`, `visit`, `visitnum`, `fondacode`,
`armcd`) are never dropped and never enter completeness calculations.

`FilterReport` instances are lightweight dataclasses that carry a full
per-element table (`pd.DataFrame`) for inspection, persistence to CSV, or
direct serialization to the audit/QA reports.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

IDENTIFIER_COLUMNS: frozenset[str] = frozenset({
    "patient_uid", "usubjid_patients", "fondacode", "cohort", "arm", "armcd",
    "visit", "visitnum",
})


def _patient_key(df: pd.DataFrame) -> pd.Series:
    """Globally-unique patient key.

    `usubjid_patients` is only unique within a cohort (ids are reused across
    BP/SZ/DR), so patient-level operations must key on (cohort, id). Prefer the
    precomputed `patient_uid` column; otherwise derive cohort::id; otherwise
    fall back to usubjid_patients alone (e.g., single-cohort test frames).
    """
    if "patient_uid" in df.columns:
        return df["patient_uid"]
    if "cohort" in df.columns:
        return (df["cohort"].astype(str) + "::"
                + df["usubjid_patients"].astype(str))
    return df["usubjid_patients"]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariableFilterReport:
    """Outcome of a variable-level missingness filter.

    Attributes
    ----------
    threshold : float
        Completeness threshold applied (0..1).
    visit : str or None
        Visit label used to compute completeness, or None when computed
        across all rows.
    n_rows_evaluated : int
        Number of rows the completeness was computed over.
    table : pd.DataFrame
        Columns: `variable`, `completeness` (float 0..1), `kept` (bool).
        Sorted by completeness ascending for quick triage.
    """
    threshold: float
    visit: str | None
    n_rows_evaluated: int
    table: pd.DataFrame

    @property
    def kept(self) -> list[str]:
        return self.table.loc[self.table["kept"], "variable"].tolist()

    @property
    def dropped(self) -> list[str]:
        return self.table.loc[~self.table["kept"], "variable"].tolist()

    def __str__(self) -> str:
        n_kept = int(self.table["kept"].sum())
        n_drop = int((~self.table["kept"]).sum())
        return (
            f"VariableFilterReport(threshold={self.threshold:.2f}, "
            f"visit={self.visit!r}, kept={n_kept}, dropped={n_drop}, "
            f"n_rows={self.n_rows_evaluated})"
        )


@dataclass(frozen=True)
class PatientFilterReport:
    """Outcome of a patient-level missingness filter.

    `table` has one row per (patient, visit) cell that was *evaluated* by
    the filter (so just V0 rows when called with `visit='V0'`, or all rows
    when called with `visit=None`). Columns: `patient_uid`,
    `usubjid_patients`, `visit`, `completeness`, `kept`.

    Patients are identified by `patient_uid` (globally unique, cohort::id) —
    NOT by `usubjid_patients`, which is only unique within a cohort.
    """
    threshold: float
    visit: str | None
    variables_used: tuple[str, ...]
    table: pd.DataFrame
    keep_other_visits: bool

    @property
    def kept_patient_uids(self) -> list:
        return self.table.loc[self.table["kept"], "patient_uid"].unique().tolist()

    @property
    def dropped_patient_uids(self) -> list:
        return self.table.loc[~self.table["kept"], "patient_uid"].unique().tolist()

    def __str__(self) -> str:
        n_kept = len(self.kept_patient_uids)
        n_drop = len(self.dropped_patient_uids)
        return (
            f"PatientFilterReport(threshold={self.threshold:.2f}, "
            f"visit={self.visit!r}, kept={n_kept}, dropped={n_drop}, "
            f"variables_used={len(self.variables_used)}, "
            f"keep_other_visits={self.keep_other_visits})"
        )


# ---------------------------------------------------------------------------
# Variable filter
# ---------------------------------------------------------------------------

def _validate_threshold(threshold: float) -> None:
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold must be in [0, 1], got {threshold!r}"
        )


def _candidate_feature_columns(
    df: pd.DataFrame,
    candidates: Sequence[str] | None = None,
) -> list[str]:
    if candidates is not None:
        missing = [c for c in candidates if c not in df.columns]
        if missing:
            raise ValueError(
                f"candidates not in df: {missing[:5]}"
                + (" …" if len(missing) > 5 else "")
            )
        return [c for c in candidates if c not in IDENTIFIER_COLUMNS]
    return [c for c in df.columns if c not in IDENTIFIER_COLUMNS]


def filter_variables(
    df: pd.DataFrame,
    threshold: float,
    *,
    visit: str | None = None,
    candidates: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, VariableFilterReport]:
    """Drop feature columns with completeness below `threshold`.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format unified frame (one row per patient × visit).
    threshold : float
        Completeness floor in [0, 1]. A column is kept iff
        ``1 - (NaN_count / n_rows_evaluated) >= threshold``.
    visit : str or None, keyword-only
        If given, completeness is computed only over rows where
        ``df['visit'] == visit``. Otherwise computed over all rows.
    candidates : sequence of str or None, keyword-only
        Restrict the set of columns considered for dropping. Defaults to
        all non-identifier columns. Identifiers are never dropped.

    Returns
    -------
    (filtered_df, report)
        `filtered_df` has identifiers + retained features (other columns
        in `df` that are not in `candidates` are also preserved unchanged).
    """
    _validate_threshold(threshold)
    if visit is not None and "visit" not in df.columns:
        raise ValueError("visit filter requires a 'visit' column in df")

    feature_candidates = _candidate_feature_columns(df, candidates)
    eval_df = df if visit is None else df[df["visit"] == visit]
    n_rows = len(eval_df)

    rows = []
    if n_rows == 0:
        # Degenerate: no rows to evaluate → no completeness → drop all candidates
        for col in feature_candidates:
            rows.append({"variable": col, "completeness": 0.0, "kept": False})
    else:
        for col in feature_candidates:
            completeness = 1.0 - (eval_df[col].isna().mean())
            kept = bool(completeness >= threshold)
            rows.append(
                {"variable": col, "completeness": float(completeness),
                 "kept": kept}
            )

    table = (pd.DataFrame(rows)
             if rows
             else pd.DataFrame(columns=["variable", "completeness", "kept"]))
    if not table.empty:
        table = table.sort_values("completeness", ascending=True).reset_index(drop=True)

    kept_features = set(table.loc[table["kept"], "variable"]) if not table.empty else set()
    # Preserve identifiers and any non-candidate columns
    preserved = [
        c for c in df.columns
        if c in IDENTIFIER_COLUMNS
        or c not in feature_candidates
        or c in kept_features
    ]
    filtered_df = df[preserved].copy()

    report = VariableFilterReport(
        threshold=threshold, visit=visit,
        n_rows_evaluated=n_rows, table=table,
    )
    return filtered_df, report


# ---------------------------------------------------------------------------
# Patient filter
# ---------------------------------------------------------------------------

def filter_patients(
    df: pd.DataFrame,
    threshold: float,
    *,
    visit: str | None = None,
    variables: Sequence[str] | None = None,
    keep_other_visits: bool = True,
) -> tuple[pd.DataFrame, PatientFilterReport]:
    """Drop patients whose completeness across `variables` falls below `threshold`.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format unified frame.
    threshold : float
        Completeness floor in [0, 1].
    visit : str or None, keyword-only
        If given, the completeness check is performed on each patient's
        row at that visit only. A patient survives iff that row's
        completeness ≥ threshold.

        If None, every row is evaluated independently: a (patient, visit)
        cell is kept iff its own completeness ≥ threshold.
    variables : sequence of str or None, keyword-only
        Columns that count toward completeness. Defaults to all non-
        identifier columns in `df`. Identifier columns are always ignored.
    keep_other_visits : bool, keyword-only
        Only meaningful when `visit` is set. If True (default), surviving
        patients keep ALL their visit rows (anchor mode — useful for
        downstream V1..V4 stability analysis). If False, only the `visit`
        rows of surviving patients are kept.

    Returns
    -------
    (filtered_df, report)
    """
    _validate_threshold(threshold)
    if "usubjid_patients" not in df.columns:
        raise ValueError("filter_patients requires 'usubjid_patients' column")
    if visit is not None and "visit" not in df.columns:
        raise ValueError("visit-based filtering requires a 'visit' column")

    feature_cols = list(_candidate_feature_columns(df, variables))
    pkey = _patient_key(df)
    if not feature_cols:
        # Nothing to evaluate — keep everyone (vacuous filter).
        empty_table = pd.DataFrame(
            {"patient_uid": [], "usubjid_patients": [], "visit": [],
             "completeness": [], "kept": []}
        )
        report = PatientFilterReport(
            threshold=threshold, visit=visit,
            variables_used=tuple(),
            table=empty_table,
            keep_other_visits=keep_other_visits,
        )
        return df.copy(), report

    feat = df[feature_cols]
    completeness_per_row = 1.0 - feat.isna().mean(axis=1)
    eval_rows = pd.DataFrame({
        "patient_uid": pkey.values,
        "usubjid_patients": df["usubjid_patients"].values,
        "visit": df["visit"].values if "visit" in df.columns else None,
        "completeness": completeness_per_row.values,
    })

    if visit is None:
        # Row-by-row filtering
        eval_rows["kept"] = eval_rows["completeness"] >= threshold
        keep_mask = eval_rows["kept"].values
        filtered_df = df.loc[keep_mask].copy()
    else:
        # Evaluate at the chosen visit; carry decision to other visits.
        # Key on patient_uid (globally unique) — NOT usubjid_patients, which
        # collides across cohorts. `df` may carry a non-contiguous index;
        # eval_rows has a fresh RangeIndex, so use positional masks.
        v_mask_series = df["visit"] == visit
        v_mask_pos = v_mask_series.values
        eval_rows = eval_rows[v_mask_pos].reset_index(drop=True)
        eval_rows["kept"] = eval_rows["completeness"] >= threshold
        kept_uids = set(eval_rows.loc[eval_rows["kept"], "patient_uid"])
        uid_in_kept = pkey.isin(kept_uids).values
        if keep_other_visits:
            mask = uid_in_kept
        else:
            mask = uid_in_kept & v_mask_pos
        filtered_df = df.loc[mask].copy()

    report = PatientFilterReport(
        threshold=threshold, visit=visit,
        variables_used=tuple(feature_cols),
        table=eval_rows.reset_index(drop=True),
        keep_other_visits=keep_other_visits,
    )
    return filtered_df, report


# ---------------------------------------------------------------------------
# V0 anchor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V0Anchor:
    """A locked V0 selection: feature columns + patient roster.

    Apply to any frame (e.g. the V1..V4 slices for stability analysis) via
    `apply(df, restrict_visits=...)`. The same `feature_columns` and
    `patient_uids` are reused unchanged — that is the *anchor* contract.

    Patients are identified by `patient_uids` (globally unique, cohort::id),
    never by `usubjid_patients` alone (which collides across cohorts).
    """
    feature_columns: tuple[str, ...]
    patient_uids: tuple
    variable_threshold: float
    patient_threshold: float
    variable_report: VariableFilterReport = field(repr=False)
    patient_report: PatientFilterReport = field(repr=False)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    @property
    def n_patients(self) -> int:
        return len(self.patient_uids)

    def apply(
        self,
        df: pd.DataFrame,
        *,
        restrict_visits: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Project the anchor onto another frame.

        Keeps:
          - rows whose patient key (cohort::id) is in the anchor's roster
            (and whose `visit` is in `restrict_visits` if provided);
          - identifier columns that exist in `df`;
          - feature columns that exist in `df` (silently drops any
            anchor feature that is missing from `df`).
        """
        if "usubjid_patients" not in df.columns:
            raise ValueError("anchor.apply requires 'usubjid_patients' column")
        identifiers = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
        feats = [c for c in self.feature_columns if c in df.columns]
        out_cols = identifiers + feats
        mask = _patient_key(df).isin(self.patient_uids).values
        if restrict_visits is not None:
            if "visit" not in df.columns:
                raise ValueError(
                    "restrict_visits requires a 'visit' column in df"
                )
            mask = mask & df["visit"].isin(restrict_visits).values
        return df.loc[mask, out_cols].copy()

    def __str__(self) -> str:
        return (
            f"V0Anchor(n_features={self.n_features}, "
            f"n_patients={self.n_patients}, "
            f"var_thr={self.variable_threshold}, "
            f"pt_thr={self.patient_threshold})"
        )


def select_v0_anchor(
    df: pd.DataFrame,
    *,
    variable_threshold: float = 0.75,
    patient_threshold: float = 0.75,
) -> tuple[pd.DataFrame, V0Anchor]:
    """V0-anchored two-step filter (variables first, then patients).

    Returns
    -------
    (v0_filtered, anchor)
        `v0_filtered` is the V0 slice containing surviving features and
        surviving patient rows only. `anchor.apply(df, restrict_visits=['V1'])`
        projects the same selection onto later visits for stability work.
    """
    if "visit" not in df.columns:
        raise ValueError("select_v0_anchor requires long-format frame with 'visit'")
    if "V0" not in set(df["visit"].dropna().unique()):
        raise ValueError("select_v0_anchor requires V0 rows in df")

    v0 = df[df["visit"] == "V0"]
    v0_after_vars, var_report = filter_variables(
        v0, threshold=variable_threshold, visit="V0",
    )
    feature_cols = list(var_report.kept)
    v0_after_patients, pt_report = filter_patients(
        v0_after_vars, threshold=patient_threshold, visit="V0",
        variables=feature_cols, keep_other_visits=False,
    )
    anchor = V0Anchor(
        feature_columns=tuple(feature_cols),
        patient_uids=tuple(pt_report.kept_patient_uids),
        variable_threshold=variable_threshold,
        patient_threshold=patient_threshold,
        variable_report=var_report,
        patient_report=pt_report,
    )
    return v0_after_patients, anchor
