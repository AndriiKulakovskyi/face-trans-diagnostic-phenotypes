#!/usr/bin/env python3
"""Audit substance-questionnaire routing without emitting patient-level data.

The two lifetime SUD summaries were added to the common dictionary by matching
column names.  This audit checks the summaries against their parent question,
unpacked substance checkboxes, detailed SZ criteria, and later SZ checkbox text.
It also verifies the smoking-status gate used to recover structural zero
pack-years.  Only aggregate counts are written.

Run from the repository root:

    PYTHONPATH=src python scripts/00_audit_substance_harmonization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.data import (  # noqa: E402
    build_unified_dataframe,
    load_variables,
    to_harmonized_dataset,
)

DATA = REPO / "data"
REPORTS = REPO / "reports"
YES = {"1", "Y", "Yes", "YES", "Oui", "OUI", "oui"}

SUD_CHILDREN = (
    "suoccur_alcool",
    "suoccur_sedatif",
    "suoccur_cannabis",
    "suoccur_stimulants",
    "suoccur_opiaces",
    "suoccur_cocaine",
    "suoccur_hallucinogenes",
    "suoccur_autre",
)


def _raw(name: str, columns: list[str]) -> pd.DataFrame:
    path = DATA / f"{name}.csv"
    available = set(pd.read_csv(path, nrows=0).columns)
    return pd.read_csv(
        path,
        usecols=[column for column in columns if column in available],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def _yes(series: pd.Series) -> pd.Series:
    return series.isin(YES)


def main() -> None:
    records: list[dict] = []

    def add(cohort: str, family: str, metric: str, value: int, status: str, detail: str) -> None:
        records.append({
            "cohort": cohort,
            "family": family,
            "metric": metric,
            "value": int(value),
            "status": status,
            "detail": detail,
        })

    bp = _raw(
        "bipolar",
        [
            "visit",
            "rad_tb_subst",
            *SUD_CHILDREN,
            "abudep_alcool",
            "abudep_cannabis",
            "agedebut_alcool",
            "agedebut_cannabis",
        ],
    )
    bp = bp[bp["visit"].eq("V0")]
    bp_any_child = bp[list(SUD_CHILDREN)].isin(YES).any(axis=1)
    bp_parent_yes = _yes(bp["rad_tb_subst"])
    add("BP", "lifetime_sud", "parent_positive", bp_parent_yes.sum(), "info", "Any lifetime SUD.")
    add("BP", "lifetime_sud", "any_child_positive", bp_any_child.sum(), "info", "Union of unpacked substance flags.")
    add(
        "BP",
        "lifetime_sud",
        "parent_positive_without_child",
        (bp_parent_yes & ~bp_any_child).sum(),
        "fail",
        "Parent-positive rows were not assigned to a substance by the unpacked fields.",
    )
    for substance in ("alcool", "cannabis"):
        summary = bp[f"suoccur_{substance}"]
        current = bp[f"abudep_{substance}"]
        onset = bp[f"agedebut_{substance}"]
        add(
            "BP",
            substance,
            "current_positive_but_lifetime_summary_zero",
            ((summary == "0") & _yes(current)).sum(),
            "fail",
            "The current-symptom branch contradicts the unpacked lifetime negative.",
        )
        add(
            "BP",
            substance,
            "onset_recorded_but_lifetime_summary_zero",
            ((summary == "0") & onset.ne("")).sum(),
            "fail",
            "A recorded disorder-onset age contradicts the unpacked lifetime negative.",
        )

    criterion_columns = [
        f"{stem}{letter}1"
        for stem in ("alcool", "cannab")
        for letter in "abcdefghijkl"
    ]
    sz = _raw(
        "schizophrenia",
        [
            "visit",
            "rad_tb_subst",
            "chk_substances_type",
            *SUD_CHILDREN,
            *criterion_columns,
        ],
    )
    sz_v0 = sz[sz["visit"].eq("V0")]
    add("SZ", "lifetime_sud", "parent_positive", _yes(sz_v0["rad_tb_subst"]).sum(), "info", "Any lifetime SUD.")
    for substance, stem in (("alcool", "alcool"), ("cannabis", "cannab")):
        criteria = sz_v0[[f"{stem}{letter}1" for letter in "abcdefghijkl"]].apply(
            pd.to_numeric, errors="coerce"
        )
        criterion_positive = criteria.eq(1).sum(axis=1).ge(2)
        summary_positive = _yes(sz_v0[f"suoccur_{substance}"])
        add(
            "SZ",
            substance,
            "lifetime_summary_positive",
            summary_positive.sum(),
            "fail" if summary_positive.sum() == 0 and criterion_positive.any() else "info",
            "The V0 summary is constant No despite positive diagnostic branches.",
        )
        add(
            "SZ",
            substance,
            "rows_with_two_or_more_lifetime_criteria",
            criterion_positive.sum(),
            "evidence",
            "Contradiction evidence only; not used to reconstruct a diagnosis.",
        )

    followup = sz[sz["visit"].ne("V0") & sz["chk_substances_type"].ne("")]
    for substance, token in (("alcool", "Alcool"), ("cannabis", "Cannabis")):
        selected = followup["chk_substances_type"].str.contains(token, regex=False)
        unpacked = _yes(followup[f"suoccur_{substance}"])
        add(
            "SZ",
            substance,
            "followup_checkbox_selected_but_unpacked_negative",
            (selected & ~unpacked).sum(),
            "fail",
            "Later visits retain checkbox text, exposing false negatives in unpacked flags.",
        )

    variables = load_variables(DATA / "face-common-vars.xlsx")
    unified = build_unified_dataframe(
        DATA,
        DATA / "face-common-vars.xlsx",
        readiness=["READY", "PARTIAL"],
        format="long",
    )
    ds_off = to_harmonized_dataset(
        unified, variables, visit="V0", normalize=False, apply_skip_logic=False
    )
    ds_on = to_harmonized_dataset(
        unified, variables, visit="V0", normalize=False, apply_skip_logic=True
    )
    for cohort in ("bp", "sz", "dr"):
        off = ds_off.X.xs(cohort, level="cohort")
        on = ds_on.X.xs(cohort, level="cohort")
        never = off["suncf_cigarettes_lt"].eq(1)
        recovered = never & off["sudose_cigarettes_lt"].isna()
        current = off["suncf_cigarettes_lt"].eq(3)
        add(
            cohort.upper(),
            "smoking",
            "never_smoker_pack_year_zeros_recovered",
            recovered.sum(),
            "pass" if on.loc[recovered, "sudose_cigarettes_lt"].eq(0).all() else "fail",
            "Structural zero recovery; existing values are not overwritten.",
        )
        add(
            cohort.upper(),
            "smoking",
            "current_smokers_with_fagerstrom_score",
            (current & off["fagers"].notna()).sum(),
            "info",
            "Fagerstrom is current-smoker-only; skipped scores remain missing, not zero.",
        )

    out = pd.DataFrame.from_records(records)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS / "00_substance_harmonization_audit.csv", index=False)

    failures = out[out["status"].eq("fail")]
    lines = [
        "# Substance harmonization audit",
        "",
        "This report contains aggregate counts only. It does not export patient identifiers.",
        "",
        out.to_markdown(index=False),
        "",
        "## Decision",
        "",
        "- `suoccur_alcool` and `suoccur_cannabis` are not valid M1 baseline indicators in the current export.",
        "- Do not replace them with `suoccur_*lt`: those fields measure lifetime exposure, not disorder.",
        "- Do not derive diagnoses from criterion counts without a validated scoring/window rule.",
        "- Quarantine both SUD indicators from the primary M1 fit pending a corrected upstream export or adjudication.",
        "- Retain Fagerstrom missingness outside current smokers; recover zero pack-years only for known never-smokers.",
        "",
        f"Failing aggregate checks: **{len(failures)}**.",
    ]
    (REPORTS / "00_substance_harmonization_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(out.to_string(index=False))
    print(f"\nWrote {REPORTS / '00_substance_harmonization_audit.csv'}")
    print(f"Wrote {REPORTS / '00_substance_harmonization_audit.md'}")


if __name__ == "__main__":
    main()
