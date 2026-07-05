"""Per-visit model-ready tables — generalize `scripts/01_build_data` to any visit.

Same data contract as `baseline_v0.parquet`: index MultiIndex[cohort, patient_id], columns = the modeled
indicators, values = raw harmonized (NaN = missing, never imputed; skip-logic decoded). Standardization to
the frozen V0 scale happens later via `face.temporal.standardize.apply_spec` — these tables hold raw values,
exactly like V0 (the engine z-scores at scoring time). The longitudinal panel ASSEMBLY (the tidy
(patient_uid, visit) coordinate+membership table) lands here at stage 34.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from face.data import build_unified_dataframe, load_variables, to_harmonized_dataset

REPO = Path(__file__).resolve().parents[3]
XLSX = REPO / "data" / "face-common-vars.xlsx"
MATRIX = REPO / "configs" / "loading_matrix.csv"


def modeled_items() -> list[str]:
    """The modeled indicator set declared in the prior loading matrix (same set `01_build_data` persists)."""
    return pd.read_csv(MATRIX).drop_duplicates("item")["item"].tolist()


def load_long(readiness: tuple[str, ...] = ("READY", "PARTIAL")) -> pd.DataFrame:
    """The harmonized long frame (all visits) — load once, reuse across visits (the CSV read is the cost)."""
    return build_unified_dataframe("data", str(XLSX), readiness=list(readiness), format="long")


def build_visit_table(long_df: pd.DataFrame, visit: str, *, variables=None,
                      items: list[str] | None = None) -> pd.DataFrame:
    """Raw harmonized modeled-indicator matrix at `visit` (MultiIndex[cohort, patient_id], NaN = missing).

    Identical pipeline to `01_build_data` (skip-logic on, no normalization), just parameterized by visit.
    Pass `variables`/`items` preloaded to avoid re-reading the dictionary per visit.
    """
    variables = variables if variables is not None else load_variables(str(XLSX))
    items = items if items is not None else modeled_items()
    ds = to_harmonized_dataset(long_df, variables, visit=visit, normalize=False, apply_skip_logic=True)
    present = [it for it in items if it in ds.X.columns]
    return ds.X[present].apply(pd.to_numeric, errors="coerce")
