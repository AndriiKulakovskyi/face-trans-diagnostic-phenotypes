"""M5.1 — the treatment-exposure harmonization layer.

Treatment is captured by different mechanisms per cohort (M5.0 audit); this module reduces all three to
**common drug-class exposures** at a given visit (default V0 = the moderation "assignment" baseline):

  SZ — `med_psy_code_atc` : per-visit list of ATC codes        -> classes (current)
  DR — `psycho_act_cmclas` (+ `psy_lifetime_cmclas`)            -> classes (current [+ lifetime])
  BP — `cmoccur_*` Y/N lifetime flags + `lithiumplasma`         -> classes (lifetime)

ATC prefixes and French class strings both map to the five common classes; SZ/DR are current/per-visit,
BP is lifetime (illness-history-confounded — flagged for the causal design). Identifier = `usubjid_patients`
(== the panel `patient_id`); raw `visit` mapped via the loader's `YEARLY_VISIT_MAP`. No imputation: a
patient with no medication record at the visit is NaN, not 0. Methods: docs/TREATMENT_MODEL.md (§2, §4).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from face.data.loader import YEARLY_VISIT_MAP

CLASSES = ("antipsychotic", "antidepressant", "mood_stabilizer", "lithium", "anxiolytic")

# ATC prefix -> class (longest-prefix first: lithium N05AN before antipsychotic N05A)
_ATC = [("N05AN", "lithium"), ("N05A", "antipsychotic"), ("N06A", "antidepressant"),
        ("N03A", "mood_stabilizer"), ("N05B", "anxiolytic"), ("N05C", "anxiolytic")]
# French class-string fragment -> class (DR `cmclas` vocab completed: normothymique, tranquillisant)
_NAME = [("lithium", "lithium"), ("normothymique", "mood_stabilizer"), ("thymorégulateur", "mood_stabilizer"),
         ("thymoregulateur", "mood_stabilizer"), ("anticonvulsant", "mood_stabilizer"),
         ("antidépresseur", "antidepressant"), ("antidepresseur", "antidepressant"),
         ("antipsychotique", "antipsychotic"), ("neuroleptique", "antipsychotic"),
         ("benzodiazépine", "anxiolytic"), ("anxiolytique", "anxiolytic"), ("hypnotique", "anxiolytic"),
         ("tranquillisant", "anxiolytic")]
# BP structured lifetime flags
_BP_CMOCCUR = {"antipsychotic": ["cmoccur_antip", "cmoccur_neuro"], "antidepressant": ["cmoccur_antid"],
               "mood_stabilizer": ["cmoccur_thymo"], "lithium": ["cmoccur_lithi"], "anxiolytic": ["cmoccur_benzo"]}
_COHORT_FILE = {"bp": "bipolar.csv", "sz": "schizophrenia.csv", "dr": "depression.csv"}


def _listify(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple)):
        return list(cell)
    s = str(cell)
    if s.strip().startswith("["):
        try:
            return list(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            return [s]
    return [s]


def atc_to_classes(cell) -> set[str]:
    out = set()
    for a in _listify(cell):
        a = str(a).upper().strip()
        for pre, cls in _ATC:
            if a.startswith(pre):
                out.add(cls)
                break
    return out


def classstr_to_classes(cell) -> set[str]:
    out = set()
    for item in _listify(cell):
        s = str(item).lower()
        out |= {cls for frag, cls in _NAME if frag in s}
    return out


def _yes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["y", "yes", "oui", "1", "1.0"])


def _flags(series: pd.Series, parser) -> pd.DataFrame:
    sets = series.map(parser)
    df = pd.DataFrame({cls: sets.map(lambda s, c=cls: 1.0 if c in s else 0.0) for cls in CLASSES},
                      index=series.index)
    return df.where(series.notna())


def _at_visit(raw: pd.DataFrame, visit: str) -> pd.DataFrame:
    r = raw.copy()
    r["visit"] = r["visit"].map(YEARLY_VISIT_MAP)
    r = r[r["visit"] == visit]
    r["patient_id"] = r["usubjid_patients"].astype(str)
    return r.drop_duplicates("patient_id", keep="first").set_index("patient_id")


def extract_exposures(cohort: str, raw: pd.DataFrame, *, visit: str = "V0") -> pd.DataFrame:
    """Per-`patient_id` common-class exposure flags at `visit`, plus cohort extras
    (`on_clozapine` for SZ, `lithium_plasma` for BP). Columns: `on_{class}` (+ extras)."""
    r = _at_visit(raw, visit)
    out = pd.DataFrame(index=r.index)
    if cohort == "sz" and "med_psy_code_atc" in r.columns:
        fl = _flags(r["med_psy_code_atc"], atc_to_classes)
        out[[f"on_{c}" for c in CLASSES]] = fl.values
        out["on_clozapine"] = _yes(r["rad_clozapine"]).astype(float) if "rad_clozapine" in r.columns else np.nan
        out["temporality"] = "current"
    elif cohort == "dr" and "psycho_act_cmclas" in r.columns:
        fl = _flags(r["psycho_act_cmclas"], classstr_to_classes)
        out[[f"on_{c}" for c in CLASSES]] = fl.values
        out["temporality"] = "current"
    elif cohort == "bp":
        for cls, cols in _BP_CMOCCUR.items():
            present = [c for c in cols if c in r.columns]
            out[f"on_{cls}"] = (np.logical_or.reduce([_yes(r[c]) for c in present]).astype(float)
                                if present else np.nan)
            # preserve NaN where the source is missing (no imputation)
            if present:
                out.loc[r[present].isna().all(axis=1), f"on_{cls}"] = np.nan
        out["lithium_plasma"] = pd.to_numeric(r.get("lithiumplasma"), errors="coerce")
        out["temporality"] = "lifetime"
    else:
        for c in CLASSES:
            out[f"on_{c}"] = np.nan
        out["temporality"] = "none"
    return out


def build_treatment_exposures(data_dir: str | Path = "data", *, visit: str = "V0") -> pd.DataFrame:
    """Harmonized exposure table: one row per (cohort, patient_id) with `on_{class}` at `visit` +
    `on_clozapine` / `lithium_plasma` + `temporality`. Reads the raw per-cohort CSVs."""
    data_dir = Path(data_dir)
    frames = []
    for cohort, fname in _COHORT_FILE.items():
        raw = pd.read_csv(data_dir / fname, low_memory=False)
        ex = extract_exposures(cohort, raw, visit=visit).reset_index()
        ex.insert(0, "cohort", cohort)
        frames.append(ex)
    out = pd.concat(frames, ignore_index=True)
    cols = ["cohort", "patient_id", *[f"on_{c}" for c in CLASSES], "on_clozapine", "lithium_plasma", "temporality"]
    return out.reindex(columns=cols)
