from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_HEADER_MAP = {
    "canonical_name": "Canonical name (merged single-cohort)",
    "bp_csv_col": "BP column in CSV",
    "sz_csv_col": "SZ column in CSV",
    "dr_csv_col": "DR column in CSV",
    "dtype": "Final dtype",
    "unit_or_value_set": "Final unit / value set",
    "cluster_readiness": "Cluster readiness",
    "clinical_rationale": "Why retained? Clinical and scientific meaning and significance",
    "rule": "Rule / action (to make data comparable)",
    "section": "Section",
    "label": "Label",
    "findings": "Findings (cross-pathology comparability)",
}


@dataclass(frozen=True)
class Variable:
    canonical_name: str
    bp_csv_col: str | None
    sz_csv_col: str | None
    dr_csv_col: str | None
    dtype: str
    unit_or_value_set: str
    cluster_readiness: str
    clinical_rationale: str
    rule: str
    section: str
    label: str
    findings: str

    def source_col(self, cohort: str) -> str | None:
        return {
            "BP": self.bp_csv_col,
            "SZ": self.sz_csv_col,
            "DR": self.dr_csv_col,
        }[cohort]


def _na_to_none(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def load_variables(path: str | Path) -> list[Variable]:
    df = pd.read_excel(path, sheet_name="Sheet1")
    missing = [h for h in _HEADER_MAP.values() if h not in df.columns]
    if missing:
        raise ValueError(f"Sheet1 missing expected headers: {missing}")

    variables: list[Variable] = []
    for _, row in df.iterrows():
        canonical = _na_to_none(row[_HEADER_MAP["canonical_name"]])
        if canonical is None:
            continue
        variables.append(
            Variable(
                canonical_name=canonical,
                bp_csv_col=_na_to_none(row[_HEADER_MAP["bp_csv_col"]]),
                sz_csv_col=_na_to_none(row[_HEADER_MAP["sz_csv_col"]]),
                dr_csv_col=_na_to_none(row[_HEADER_MAP["dr_csv_col"]]),
                dtype=_na_to_none(row[_HEADER_MAP["dtype"]]) or "",
                unit_or_value_set=_na_to_none(row[_HEADER_MAP["unit_or_value_set"]]) or "",
                cluster_readiness=_na_to_none(row[_HEADER_MAP["cluster_readiness"]]) or "",
                clinical_rationale=_na_to_none(row[_HEADER_MAP["clinical_rationale"]]) or "",
                rule=_na_to_none(row[_HEADER_MAP["rule"]]) or "",
                section=_na_to_none(row[_HEADER_MAP["section"]]) or "",
                label=_na_to_none(row[_HEADER_MAP["label"]]) or "",
                findings=_na_to_none(row[_HEADER_MAP["findings"]]) or "",
            )
        )
    return variables
