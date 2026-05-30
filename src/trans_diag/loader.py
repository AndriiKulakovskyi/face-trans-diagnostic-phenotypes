from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd

from .rules import RULES, derive_siteid_city, identity_cast
from .variable import Variable, load_variables

YEARLY_VISIT_MAP: dict[str, str] = {
    "V0": "V0",
    "V1_an": "V1",
    "V2_ans": "V2",
    "V3_ans": "V3",
    "V4_ans": "V4",
    "V5_ans": "V5",
    "V6_ans": "V6",
    "V7_ans": "V7",
    "V8_ans": "V8",
    "V9_ans": "V9",
    "V10_ans": "V10",
}

_COHORT_FILES: list[tuple[str, str]] = [
    ("BP", "bipolar.csv"),
    ("SZ", "schizophrenia.csv"),
    ("DR", "depression.csv"),
]

_IDENTIFIER_CANONICALS: frozenset[str] = frozenset({
    "usubjid_patients", "fondacode", "arm", "armcd", "visitnum", "visit", "cohort",
})

_INFRASTRUCTURE_COLS: tuple[str, ...] = ("usubjid_patients", "visitnum", "visit", "arm")

# Canonicals whose harmonization needs a sibling raw column beyond their own
# source column. The loader reads the extra column and computes the feature via
# a dedicated deriver, bypassing the single-series RULES path. ``fondacode`` is
# also an identifier (carried as a label), so we name the *raw CSV column* here.
_SIBLING_RAW_COLS: dict[str, tuple[str, ...]] = {
    "siteid_city": ("fondacode",),
}

# Transformers introduced for the curated v2 dictionary (unit/scale fixes the v1
# pipeline never had). They are skipped when loading v1 so the legacy pipeline
# stays byte-identical; for v1 these canonicals fall back to identity_cast.
_V2_ONLY_RULES: frozenset[str] = frozenset({"qt", "rr", "qtc", "edulevel"})


def _matches_readiness(value: str, prefixes: Iterable[str]) -> bool:
    return any(value.startswith(p) for p in prefixes)


def _harmonize_series(
    var: Variable, series: pd.Series, cohort: str, use_v2_rules: bool = False
) -> pd.Series:
    transformer = RULES.get(var.canonical_name)
    if transformer is None or (
        var.canonical_name in _V2_ONLY_RULES and not use_v2_rules
    ):
        return identity_cast(
            series, cohort, var.dtype,
            canonical_name=var.canonical_name,
            unit_or_value_set=var.unit_or_value_set,
        )
    return transformer(series, cohort)


_NON_NUMERIC_DTYPES: frozenset[str] = frozenset({"string", "category"})


def _apply_sanity_bounds(
    series: pd.Series, var: Variable
) -> tuple[pd.Series, int]:
    """Null cells that fall outside the dictionary's [sanity_min, sanity_max].

    Out-of-range values are set to NA (treated as missing, never imputed and
    never clipped) — the bounds in face-common-vars-v2.xlsx are tuned to catch
    gross scale/unit errors, not to trim genuine clinical extremes. Returns the
    masked series and the number of cells nulled. A no-op when the variable has
    no bounds (v1, or a v2 row left blank) or is a non-numeric / date column.
    """
    lo, hi = var.sanity_min, var.sanity_max
    if lo is None and hi is None:
        return series, 0
    dtype_norm = (var.dtype or "").strip().lower()
    if dtype_norm in _NON_NUMERIC_DTYPES or dtype_norm.startswith("date"):
        return series, 0

    numeric = pd.to_numeric(series, errors="coerce")
    out_mask = pd.Series(False, index=series.index)
    if lo is not None:
        out_mask |= numeric < lo
    if hi is not None:
        out_mask |= numeric > hi
    out_mask &= numeric.notna()
    n_nulled = int(out_mask.sum())
    if n_nulled == 0:
        return series, 0
    return series.mask(out_mask), n_nulled


def _load_cohort(
    cohort: str,
    csv_path: Path,
    variables: list[Variable],
    sanity_report: dict[tuple[str, str], int] | None = None,
    use_fondacode_site: bool = False,
) -> pd.DataFrame:
    feature_vars = [
        v for v in variables
        if v.source_col(cohort) and v.canonical_name not in _IDENTIFIER_CANONICALS
    ]
    needed = set(_INFRASTRUCTURE_COLS) | {v.source_col(cohort) for v in feature_vars}
    # Pull any sibling raw columns required by multi-column derivers (e.g.
    # fondacode for siteid_city), even when they are otherwise identifiers.
    for v in feature_vars:
        for extra in _SIBLING_RAW_COLS.get(v.canonical_name, ()):
            needed.add(extra)

    raw = pd.read_csv(
        csv_path,
        usecols=lambda c: c in needed,
        low_memory=False,
        encoding="utf-8-sig",
    )
    missing_infra = [c for c in _INFRASTRUCTURE_COLS if c not in raw.columns]
    if missing_infra:
        raise ValueError(
            f"{csv_path.name}: missing infrastructure columns {missing_infra}"
        )

    raw = raw[raw["visit"].isin(YEARLY_VISIT_MAP)].copy()
    raw["visit"] = raw["visit"].map(YEARLY_VISIT_MAP).astype("string")

    pieces: list[pd.Series] = [raw[c].rename(c) for c in _INFRASTRUCTURE_COLS]
    seen: dict[str, int] = {}
    for v in feature_vars:
        if (
            use_fondacode_site
            and v.canonical_name == "siteid_city"
            and "fondacode" in raw.columns
        ):
            series = derive_siteid_city(
                raw[v.source_col(cohort)], raw["fondacode"], cohort
            )
        else:
            series = _harmonize_series(
                v, raw[v.source_col(cohort)], cohort,
                use_v2_rules=use_fondacode_site,
            )
        series, n_nulled = _apply_sanity_bounds(series, v)
        if sanity_report is not None and n_nulled:
            sanity_report[(cohort, v.canonical_name)] = n_nulled
        name = v.canonical_name
        if name in seen:
            # Multiple variable rows can collide on the same canonical name —
            # later wins; the first registration is overwritten silently.
            pieces[seen[name]] = series.rename(name)
        else:
            seen[name] = len(pieces)
            pieces.append(series.rename(name))

    out = pd.concat(pieces, axis=1)
    out["cohort"] = cohort
    return out


def build_unified_dataframe(
    data_dir: str | Path,
    dictionary_path: str | Path,
    readiness: list[str],
    format: Literal["long", "wide"] = "long",
    sanity_report: dict[tuple[str, str], int] | None = None,
) -> pd.DataFrame:
    """Build a unified longitudinal patient dataframe across BP, SZ, DR cohorts.

    Identifier columns ``usubjid_patients``, ``cohort``, ``arm`` are preserved as
    labels for downstream cluster evaluation — they are NOT clustering features.
    To extract the feature matrix:

        long:  df.drop(columns=['usubjid_patients','cohort','arm','visit','visitnum'])
        wide:  df.drop(columns=['usubjid_patients','cohort','arm'])

    Parameters
    ----------
    data_dir : path containing bipolar.csv, schizophrenia.csv, depression.csv.
    dictionary_path : path to face-common-vars.xlsx.
    readiness : list of cluster_readiness prefixes to keep, e.g. ['READY'] or
        ['READY', 'PARTIAL']. Required — no default.
    format : 'long' (one row per patient × visit) or 'wide' (one row per patient,
        feature columns suffixed _V0/_V1/...).
    sanity_report : optional dict; when provided, it is populated in place with
        ``{(cohort, canonical_name): n_cells_nulled}`` for every variable whose
        values fell outside the dictionary's sanity bounds. Empty when the v1
        dictionary (no bounds) is used.
    """
    if not readiness:
        raise ValueError("readiness must be a non-empty list of prefixes")
    if format not in {"long", "wide"}:
        raise ValueError(f"format must be 'long' or 'wide', got {format!r}")

    data_dir = Path(data_dir)
    variables = load_variables(dictionary_path)
    # The curated v2 dictionary carries per-variable sanity bounds; v1 does not.
    # That same provenance flag enables the fondacode-derived site code (a v2-only
    # harmonization), so the legacy v1 pipeline stays byte-identical.
    is_v2 = any(v.sanity_min is not None or v.sanity_max is not None for v in variables)
    variables = [v for v in variables if _matches_readiness(v.cluster_readiness, readiness)]

    frames = [
        _load_cohort(cohort, data_dir / fname, variables, sanity_report,
                     use_fondacode_site=is_v2)
        for cohort, fname in _COHORT_FILES
    ]
    merged = pd.concat(frames, axis=0, ignore_index=True)
    _enforce_dtypes(merged, variables)

    # `usubjid_patients` is only unique WITHIN a cohort (970 ids are reused
    # across BP/SZ/DR). The globally-unique patient key is (cohort, id);
    # expose it as `patient_uid` so every downstream stage (filtering,
    # clustering, projection, stability) identifies patients correctly.
    merged["patient_uid"] = (
        merged["cohort"].astype(str) + "::" + merged["usubjid_patients"].astype(str)
    )

    if format == "long":
        front = [c for c in ("patient_uid", "usubjid_patients", "cohort",
                             "arm", "visitnum", "visit")
                 if c in merged.columns]
        rest = [c for c in merged.columns if c not in front]
        return merged[front + rest]

    return _to_wide(merged)


_DTYPE_TO_PANDAS: dict[str, str] = {
    "float": "float64",
    "int8 binary": "Int8",
    "int8 ordinal": "Int16",
    "int8 categorical": "Int16",
    "string": "string",
    "category": "category",
    "date (YYYY-MM-DD)": "datetime64[ns]",
}


def _enforce_dtypes(df: pd.DataFrame, variables: list[Variable]) -> None:
    for v in variables:
        if v.canonical_name in _IDENTIFIER_CANONICALS:
            continue
        if v.canonical_name not in df.columns:
            continue
        target = _DTYPE_TO_PANDAS.get(v.dtype.strip())
        if target is None or str(df[v.canonical_name].dtype) == target:
            continue
        try:
            if target == "category":
                df[v.canonical_name] = df[v.canonical_name].astype("string").astype("category")
            elif target.startswith("datetime"):
                df[v.canonical_name] = pd.to_datetime(df[v.canonical_name], errors="coerce")
            else:
                df[v.canonical_name] = df[v.canonical_name].astype(target)
        except (TypeError, ValueError):
            pass


def _to_wide(merged: pd.DataFrame) -> pd.DataFrame:
    # Group on patient_uid (globally unique), NOT usubjid_patients (which
    # collides across cohorts). usubjid_patients/cohort/arm are carried as
    # time-invariant identifier columns.
    id_extras = [c for c in ("usubjid_patients", "cohort", "arm")
                 if c in merged.columns]
    drop_from_features = {"patient_uid", "usubjid_patients", "cohort", "arm",
                          "visit", "visitnum"}
    feature_cols = [c for c in merged.columns if c not in drop_from_features]

    id_frame = merged.groupby("patient_uid", as_index=True)[id_extras].first()

    if not feature_cols:
        return id_frame.reset_index()

    wide = merged.pivot_table(
        index="patient_uid",
        columns="visit",
        values=feature_cols,
        aggfunc="first",
    )
    wide.columns = [f"{feat}_{visit}" for feat, visit in wide.columns]

    return id_frame.join(wide).reset_index()
