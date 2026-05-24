"""Comprehensive per-variable audit of the unified dataframe.

For every dictionary row with cluster_readiness in {READY, PARTIAL} and at least
one cohort source column, this script verifies:

  1. The canonical_name appears as a column in the long-format output.
  2. The source CSV columns were dropped from the output (rename worked).
  3. The output dtype matches the dictionary's `Final dtype`.
  4. Non-null values conform to the `Final unit / value set` (if parseable).
  5. Per-cohort value sets are consistent across the cohorts that have the col.
  6. NaN rate per cohort is plausible (not 100% where the source col exists).
  7. The Variable instance carries every metadata field non-empty (where the
     dictionary populated it).

Findings are categorized PASS / WARN / FAIL and printed in a per-variable table.

Run: python3 scripts/audit.py
"""
from __future__ import annotations

import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from trans_diag import RULES, build_unified_dataframe, load_variables  # noqa: E402
from trans_diag.loader import YEARLY_VISIT_MAP  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
DICT_PATH = REPO_ROOT / "face-common-vars.xlsx"
RESULTS_DIR = REPO_ROOT / "results"
_YEARLY = set(YEARLY_VISIT_MAP)


DTYPE_EXPECTATIONS: dict[str, set[str]] = {
    "float": {"float64", "Float64"},
    "int8 binary": {"Int8"},
    "int8 ordinal": {"Int16", "Int8"},
    "int8 categorical": {"Int16", "Int8"},
    "string": {"string", "object"},
    "category": {"category"},
    "date (YYYY-MM-DD)": {"datetime64[ns]"},
    "[verify]": {"string", "object"},
}

_VALUE_SET_RE = re.compile(r"\{([^}]+)\}")


def parse_allowed(unit_or_value_set: str) -> set | None:
    if not unit_or_value_set:
        return None
    match = _VALUE_SET_RE.search(unit_or_value_set)
    if not match:
        return None
    out: set = set()
    for tok in match.group(1).split(","):
        first = tok.split("=", 1)[0].strip()
        if not first:
            continue
        if first.upper() in {"NA", "UNKNOWN", "NA=UNKNOWN"}:
            continue
        try:
            out.add(int(first))
        except ValueError:
            try:
                out.add(float(first))
            except ValueError:
                out.add(first)
    return out or None


def fmt_status(label: str) -> str:
    return {"PASS": " PASS", "WARN": " WARN", "FAIL": " FAIL"}.get(label, label)


def _load_raw_yearly_nan_rates(variables: list) -> dict[tuple[str, str], float]:
    """For every (canonical_name, cohort) where a source col exists, compute
    the raw NaN rate at yearly visits BEFORE harmonization.

    Returns {(canonical_name, cohort): raw_nan_rate}.
    """
    rates: dict[tuple[str, str], float] = {}
    for cohort, fname in (("BP", "bipolar.csv"),
                          ("SZ", "schizophrenia.csv"),
                          ("DR", "depression.csv")):
        needed = {"visit"}
        for v in variables:
            src = v.source_col(cohort)
            if src:
                needed.add(src)
        df = pd.read_csv(DATA_DIR / fname,
                         usecols=lambda c: c in needed,
                         low_memory=False, encoding="utf-8-sig")
        df_y = df[df["visit"].isin(_YEARLY)]
        for v in variables:
            src = v.source_col(cohort)
            if src and src in df_y.columns:
                rates[(v.canonical_name, cohort)] = df_y[src].isna().mean()
    return rates


def main() -> int:
    print("Loading dictionary and unified dataframe (READY + PARTIAL, long)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(
            DATA_DIR, DICT_PATH,
            readiness=["READY", "PARTIAL"], format="long",
        )

    variables = load_variables(DICT_PATH)
    usable = [
        v for v in variables
        if v.cluster_readiness.startswith(("READY", "PARTIAL"))
    ]
    identifiers = {
        "usubjid_patients", "fondacode", "arm", "armcd",
        "visitnum", "visit", "cohort",
    }
    feature_vars = [v for v in usable if v.canonical_name not in identifiers]

    print(f"\nDictionary: {len(usable)} READY+PARTIAL rows "
          f"({len(feature_vars)} non-identifier).")
    print(f"Output:     {df.shape[0]:,} rows × {df.shape[1]} cols\n")

    print("Computing raw yearly-visit NaN rates (pre-harmonization)...")
    raw_nan = _load_raw_yearly_nan_rates(feature_vars)

    # ------------------------------------------------------------------
    # Per-variable checks
    # ------------------------------------------------------------------
    rows = []
    cohort_set = ["BP", "SZ", "DR"]
    df_by_cohort = {c: df[df["cohort"] == c] for c in cohort_set}

    counts = Counter()
    fail_examples: dict[str, list[str]] = defaultdict(list)

    for v in feature_vars:
        checks: list[tuple[str, str, str]] = []  # (check_name, level, detail)

        # 1. canonical_name present
        present = v.canonical_name in df.columns
        if not present:
            checks.append(("present", "FAIL", "column missing from output"))
        else:
            checks.append(("present", "PASS", ""))

        # 2. source-col rename — none of the cohort source-col names should remain
        leaked_sources = [
            v.source_col(c) for c in cohort_set
            if v.source_col(c) and v.source_col(c) != v.canonical_name
            and v.source_col(c) in df.columns
        ]
        if leaked_sources:
            checks.append(("renamed", "FAIL",
                          f"source cols leaked: {leaked_sources}"))
        else:
            checks.append(("renamed", "PASS", ""))

        if not present:
            for name in ("dtype", "values", "consistency", "nan"):
                checks.append((name, "WARN", "skipped (column missing)"))
        else:
            series = df[v.canonical_name]

            # 3. Dtype
            expected = DTYPE_EXPECTATIONS.get(v.dtype.strip(), set())
            actual_dtype = str(series.dtype)
            if not expected:
                checks.append(("dtype", "WARN",
                              f"unknown expected dtype {v.dtype!r}; got {actual_dtype}"))
            elif actual_dtype in expected:
                checks.append(("dtype", "PASS", actual_dtype))
            else:
                checks.append(("dtype", "FAIL",
                              f"expected one of {sorted(expected)}, got {actual_dtype}"))

            # 4. Value-set conformance
            allowed = parse_allowed(v.unit_or_value_set)
            if allowed and v.dtype.strip() in {"int8 binary", "int8 ordinal", "int8 categorical"}:
                non_null = series.dropna()
                obs_values = set()
                for x in non_null.unique():
                    try:
                        obs_values.add(int(x))
                    except (TypeError, ValueError):
                        obs_values.add(x)
                unexpected = obs_values - allowed
                if unexpected:
                    checks.append(("values", "WARN",
                                  f"outside {sorted(allowed)}: {sorted(unexpected)[:5]}"))
                else:
                    checks.append(("values", "PASS", ""))
            else:
                checks.append(("values", "PASS", "n/a"))

            # 5. Per-cohort encoding consistency for shared variables
            cohorts_with_col = [c for c in cohort_set if v.source_col(c)]
            if len(cohorts_with_col) >= 2 and v.dtype.strip() in {"int8 binary", "int8 ordinal", "int8 categorical"}:
                per_cohort_sets = {}
                for c in cohorts_with_col:
                    s = df_by_cohort[c][v.canonical_name].dropna()
                    per_cohort_sets[c] = set(int(x) for x in s.unique() if pd.notna(x))
                union = set().union(*per_cohort_sets.values())
                divergent = [c for c, vs in per_cohort_sets.items()
                            if vs and vs != union and len(union - vs) > 0]
                if divergent and union:
                    checks.append(("consistency", "WARN",
                                  f"cohort value sets diverge: {per_cohort_sets}"))
                else:
                    checks.append(("consistency", "PASS", ""))
            else:
                checks.append(("consistency", "PASS", "n/a"))

            # 6. NaN rate per cohort — distinguish parsing bugs from true data gaps:
            #     - parsing bug: raw < 99% NaN, harmonized ≥ 99.9% NaN (we destroyed data)
            #     - data gap:    raw ≥ 99% NaN already (column is empty at yearly visits)
            parsing_bugs = []
            data_gaps = []
            for c in cohort_set:
                if not v.source_col(c):
                    continue
                harmonized_nan = df_by_cohort[c][v.canonical_name].isna().mean()
                if harmonized_nan < 0.999:
                    continue
                raw = raw_nan.get((v.canonical_name, c))
                if raw is None or raw >= 0.99:
                    data_gaps.append(f"{c}=raw{raw:.2f}" if raw is not None else f"{c}=?")
                else:
                    parsing_bugs.append(f"{c}=raw{raw:.2f}→100%")
            if parsing_bugs:
                checks.append(("nan", "FAIL",
                              f"parsing destroyed data: {parsing_bugs}"))
            elif data_gaps:
                checks.append(("nan", "WARN",
                              f"sparse/empty source: {data_gaps}"))
            else:
                checks.append(("nan", "PASS", ""))

        # 7. Metadata completeness
        meta_missing = []
        for field in ("canonical_name", "dtype", "unit_or_value_set",
                      "cluster_readiness", "section", "label"):
            if not getattr(v, field):
                meta_missing.append(field)
        if meta_missing:
            checks.append(("metadata", "WARN",
                          f"empty fields: {meta_missing}"))
        else:
            checks.append(("metadata", "PASS", ""))

        worst = "PASS"
        for _, level, _ in checks:
            if level == "FAIL":
                worst = "FAIL"; break
            if level == "WARN":
                worst = "WARN"
        counts[worst] += 1

        if worst != "PASS":
            issues = [(n, lvl, d) for n, lvl, d in checks if lvl != "PASS"]
            for n, lvl, d in issues:
                fail_examples[f"{lvl}:{n}"].append(f"{v.canonical_name}: {d}")

        rows.append({
            "canonical_name": v.canonical_name,
            "section": v.section,
            "dtype": v.dtype,
            "readiness": v.cluster_readiness.split(" ")[0],
            "rule": "registered" if v.canonical_name in RULES else "identity",
            "worst": worst,
            **{name: f"{lvl}{':'+detail if detail else ''}"[:60]
               for name, lvl, detail in checks},
        })

    audit_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 78)
    print("PER-VARIABLE AUDIT SUMMARY")
    print("=" * 78)
    print(f"  PASS:  {counts['PASS']:>4} / {len(feature_vars)}")
    print(f"  WARN:  {counts['WARN']:>4} / {len(feature_vars)}")
    print(f"  FAIL:  {counts['FAIL']:>4} / {len(feature_vars)}")
    print()

    print("Findings by check (showing up to 8 examples per category):")
    for key in sorted(fail_examples.keys()):
        examples = fail_examples[key]
        print(f"\n  [{key}]  ({len(examples)} variable(s))")
        for ex in examples[:8]:
            print(f"    - {ex}")
        if len(examples) > 8:
            print(f"    ...and {len(examples) - 8} more")

    # ------------------------------------------------------------------
    # Per-section breakdown
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PER-SECTION SUMMARY")
    print("=" * 78)
    section_table = audit_df.groupby("section")["worst"].value_counts().unstack(fill_value=0)
    for col in ("PASS", "WARN", "FAIL"):
        if col not in section_table.columns:
            section_table[col] = 0
    section_table = section_table[["PASS", "WARN", "FAIL"]]
    print(section_table.to_string())

    # ------------------------------------------------------------------
    # Per-dtype breakdown
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PER-DTYPE SUMMARY")
    print("=" * 78)
    dtype_table = audit_df.groupby("dtype")["worst"].value_counts().unstack(fill_value=0)
    for col in ("PASS", "WARN", "FAIL"):
        if col not in dtype_table.columns:
            dtype_table[col] = 0
    dtype_table = dtype_table[["PASS", "WARN", "FAIL"]]
    print(dtype_table.to_string())

    # ------------------------------------------------------------------
    # Identifier verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("IDENTIFIER COLUMNS")
    print("=" * 78)
    for col in ("usubjid_patients", "cohort", "arm", "visit", "visitnum"):
        present = col in df.columns
        unique_count = df[col].nunique() if present else "—"
        sample = list(df[col].dropna().unique())[:3] if present else []
        print(f"  {col:<20} present={present}  uniques={unique_count}  sample={sample}")

    # ------------------------------------------------------------------
    # Patient count check
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PATIENT COUNTS PER COHORT")
    print("=" * 78)
    pat_counts = df.groupby("cohort")["usubjid_patients"].nunique()
    print(pat_counts.to_string())

    # Write the full table to disk for downstream inspection
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "audit_report.csv"
    audit_df.to_csv(out_path, index=False)
    print(f"\nFull audit table written to: {out_path}")

    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
