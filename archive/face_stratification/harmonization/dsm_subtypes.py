"""DSM-5 subtype extraction for fine-grained diagnostic labels.

Extracts clinically meaningful subtypes from raw patient data to replace
coarse cohort labels (bp/sz/dr/asp) with DSM-5-informed sub-categories.

Subtypes are extracted from:
  BP: ``arm`` column (Bipolaire de type 1 / type 2 / non spécifié)
  SZ: PANSS positive vs negative dominance profile
  DR: Treatment resistance staging (Sachs score + epi_resist)
  ASP: DSM diagnostic category (dsmtype) + EGF functioning level
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_dsm_subtypes_from_raw(
    harmonized_metadata: pd.DataFrame,
    csv_data: dict[str, pd.DataFrame],
) -> pd.Series:
    """Extract DSM-5 subtypes directly from raw CSV data.

    Parameters
    ----------
    harmonized_metadata:
        DataFrame indexed by ``(cohort, patient_id)`` with columns
        ``cohort``, ``patient_id``.
    csv_data:
        ``{cohort: DataFrame}`` — the raw CSV DataFrames keyed by cohort.

    Returns
    -------
    ``pd.Series`` with same index as ``harmonized_metadata``.
    """
    subtypes = pd.Series("unknown", index=harmonized_metadata.index, dtype=str)

    bp_map = _build_bp_subtype_map(csv_data.get("bp"))
    sz_map = _build_sz_subtype_map(csv_data.get("sz"))
    dr_map = _build_dr_subtype_map(csv_data.get("dr"))
    asp_map = _build_asp_subtype_map(csv_data.get("asp"))

    cohort_maps = {"bp": bp_map, "sz": sz_map, "dr": dr_map, "asp": asp_map}

    for idx in harmonized_metadata.index:
        cohort, pid = idx
        cmap = cohort_maps.get(cohort, {})
        subtypes.at[idx] = cmap.get(str(pid), f"{cohort}_unspecified")

    counts = subtypes.value_counts()
    logger.info("DSM subtypes extracted: %d categories, %d patients", len(counts), len(subtypes))
    return subtypes


def _build_bp_subtype_map(df: pd.DataFrame | None) -> dict[str, str]:
    """BP-I vs BP-II from the ``arm`` column."""
    if df is None:
        return {}
    out: dict[str, str] = {}
    for row_idx, row in df.iterrows():
        pid = str(row_idx)
        arm = str(row.get("arm", "")).strip().lower()
        if "type 1" in arm:
            out[pid] = "BP-I"
        elif "type 2" in arm:
            out[pid] = "BP-II"
        elif "spécifié" in arm or "specifie" in arm:
            out[pid] = "BP-NOS"
        else:
            out[pid] = "BP-NOS"
    return out


def _build_sz_subtype_map(df: pd.DataFrame | None) -> dict[str, str]:
    """SZ symptom profile from PANSS positive/negative dominance."""
    if df is None:
        return {}
    out: dict[str, str] = {}
    for row_idx, row in df.iterrows():
        pid = str(row_idx)
        pp = _safe_float(row.get("panssp"))
        pn = _safe_float(row.get("panssn"))

        if pp is None and pn is None:
            out[pid] = "SZ-unspecified"
            continue

        if pp is not None and pn is not None:
            if pp > pn + 5:
                out[pid] = "SZ-positive"
            elif pn > pp + 5:
                out[pid] = "SZ-negative"
            else:
                out[pid] = "SZ-mixed"
        elif pp is not None:
            out[pid] = "SZ-positive" if pp > 20 else "SZ-low-symptoms"
        else:
            out[pid] = "SZ-negative" if pn > 20 else "SZ-low-symptoms"
    return out


def _build_dr_subtype_map(df: pd.DataFrame | None) -> dict[str, str]:
    """TRD staging from ``sachs_`` score and ``epi_resist`` flag."""
    if df is None:
        return {}
    out: dict[str, str] = {}
    for row_idx, row in df.iterrows():
        pid = str(row_idx)
        sachs = _safe_float(row.get("sachs_"))
        resist = _safe_float(row.get("epi_resist"))

        if sachs is not None:
            if sachs >= 30:
                out[pid] = "DR-high-resistance"
            elif sachs >= 15:
                out[pid] = "DR-moderate-resistance"
            else:
                out[pid] = "DR-low-resistance"
        elif resist is not None:
            if resist >= 2:
                out[pid] = "DR-high-resistance"
            elif resist >= 1:
                out[pid] = "DR-partial-resistance"
            else:
                out[pid] = "DR-non-resistant"
        else:
            out[pid] = "DR-unspecified"
    return out


def _build_asp_subtype_map(df: pd.DataFrame | None) -> dict[str, str]:
    """ASP subtype from ``dsmtype`` (diagnostic category) + ``egfval`` (functioning)."""
    if df is None:
        return {}
    out: dict[str, str] = {}
    dsm_labels = {1.0: "Autism", 2.0: "Asperger", 3.0: "PDD-NOS", 4.0: "Other-ASD"}

    for row_idx, row in df.iterrows():
        pid = str(row_idx)
        dsm_code = _safe_float(row.get("dsmtype"))
        egf = _safe_float(row.get("egfval"))

        dsm_label = dsm_labels.get(dsm_code, "ASD")

        if egf is not None:
            if egf >= 61:
                level = "high-func"
            elif egf >= 41:
                level = "moderate-func"
            else:
                level = "low-func"
            out[pid] = f"ASP-{dsm_label}-{level}"
        else:
            out[pid] = f"ASP-{dsm_label}"
    return out


# Legacy interface for backward compatibility
def extract_dsm_subtypes(
    harmonized_metadata: pd.DataFrame,
    patient_data_by_id: dict[str, Any] | None = None,
) -> pd.Series:
    """Fallback that returns cohort-level labels when raw CSV data is unavailable."""
    subtypes = pd.Series(index=harmonized_metadata.index, dtype=str)
    for idx in harmonized_metadata.index:
        cohort = idx[0] if isinstance(idx, tuple) else harmonized_metadata.at[idx, "cohort"]
        subtypes.at[idx] = f"{cohort}_unspecified"
    return subtypes


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
