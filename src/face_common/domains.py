"""Aggregate item-level features into balanced clinical/biological domain scores.

Motivation (LABBOOK E5): cosine treats every column as one equal dimension, so a
construct measured by many items dominates — in the 129-feature clinical set,
SUICIDE alone was 39 dims (30%). Aggregating items → one score per
instrument/domain makes each construct count once, fixes the item-count
weighting, and reduces missingness (a domain score needs only *some* of its
items observed). No imputation: a patient with too few observed items for a
domain gets NaN for that domain.

Two kinds of domain:

- **Symptom instruments** — features in the clinical psychiatric sections are
  auto-grouped by canonical *stem* (e.g. ``isf01a``/``isf02`` → ``isf``;
  ``madrs`` items → ``madrs``). One domain per instrument: the masked mean of
  its robust-z-scored items.
- **Biology composites** — curated, clinically-motivated combinations of labs /
  vitals with explicit sign (e.g. metabolic syndrome: BMI↑, waist↑, triglycerides↑,
  HDL↓, glucose↑, blood pressure↑). Only well-defined composites are emitted;
  raw labs that are pure demographics proxies (haemoglobin, etc.) or treatment /
  pregnancy markers are intentionally *not* aggregated.

Each item is robust-z-scored (winsorize 1/99 + median/MAD) and oriented so
"higher = more pathological" before averaging, so members on different units
(mmol/L vs g/L vs mmHg) combine sensibly.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .adapter import CLINICAL_SECTIONS
from .variable import Variable

__all__ = [
    "build_domain_scores",
    "BIOLOGY_COMPOSITES",
    "DOMAIN_SECTIONS",
    "BIOLOGY_SECTIONS",
]

BIOLOGY_SECTIONS = frozenset({"BILAN BIOLOGIQUE", "CONSTANTES ET ECG"})

# Sections to pull raw features from when building domain scores (symptom +
# biology). Pass this as `sections=` to to_harmonized_dataset (raw, no residualize).
DOMAIN_SECTIONS = frozenset(CLINICAL_SECTIONS) | BIOLOGY_SECTIONS

# Curated biology composites: domain -> [(canonical, sign), ...].
# sign = +1 if higher is more pathological, -1 if lower is more pathological.
# Members absent from the matrix are skipped (masked).
BIOLOGY_COMPOSITES: dict[str, list[tuple[str, int]]] = {
    "metabolic_syndrome": [
        ("bmi", +1), ("wstcir", +1), ("trig", +1), ("hdl", -1),
        ("gluc", +1), ("hba1c", +1), ("sysbpsupine", +1), ("diabpsupine", +1),
    ],
    "cholesterol": [("chol", +1), ("ldl", +1), ("cholhdl_lbstresc", +1)],
    "inflammation": [("crp", +1), ("wbc", +1), ("neut", +1)],
    "prolactin": [("prolctn", +1)],
    "hepatic": [
        ("ast_lbstresc", +1), ("alt_lbstresc", +1), ("ggt_lbstresc", +1),
        ("alp_lbstresc", +1), ("bili_lbstresc", +1),
    ],
    "renal": [
        ("creat_lbstresc", +1), ("urea_lbstresc", +1), ("urate", +1),
        ("creatclr_lbstresc", -1),
    ],
    "cardiac_qtc": [("qtc", +1)],
}

_STEM_RE = re.compile(r"\d+[a-z]*$")


def instrument_stem(name: str) -> str:
    """Instrument stem of a canonical name: strip a trailing item number.

    ``isf01a`` → ``isf``; ``cssrs3`` → ``cssrs``; ``psqi11`` → ``psqi``;
    ``madrs`` → ``madrs``; ``agedebut_cigarettes_lt`` → unchanged (no trailing
    item number, so it stays its own single-item domain).
    """
    stem = _STEM_RE.sub("", name).rstrip("_")
    return stem or name


def _robust_z(s: pd.Series) -> pd.Series:
    """Winsorize (1/99) + robust z-score (median / 1.4826·MAD). NaN preserved."""
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    if pd.notna(lo) and pd.notna(hi) and hi > lo:
        s = s.clip(lower=lo, upper=hi)
    med = s.median()
    mad = (s - med).abs().median()
    scale = 1.4826 * mad if mad and mad > 0 else (s.std() or 1.0)
    return (s - med) / (scale if scale > 0 else 1.0)


def build_domain_scores(
    X: pd.DataFrame,
    variables: Iterable[Variable],
    *,
    symptom_sections: Iterable[str] = CLINICAL_SECTIONS,
    biology: dict[str, list[tuple[str, int]]] | None = None,
    min_items_frac: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate ``X`` item columns into domain scores.

    Parameters
    ----------
    X:
        Raw V0 feature matrix (patients × items), indexed by ``[cohort,
        patient_id]``. Must contain the symptom-section items and any biology
        composite members you want aggregated.
    variables:
        The dictionary, used for each feature's ``section``.
    symptom_sections:
        Sections whose features are auto-grouped by instrument stem.
    biology:
        Curated biology composites (defaults to :data:`BIOLOGY_COMPOSITES`).
    min_items_frac:
        A domain score is ``NaN`` for a patient unless at least this fraction of
        the domain's member items are observed (no imputation).

    Returns
    -------
    (scores, meta):
        ``scores`` is patients × domains (float, NaN where under-observed);
        ``meta`` is one row per domain (kind, n_items, members, coverage).
    """
    by_name = {v.canonical_name: v for v in variables}
    biology = BIOLOGY_COMPOSITES if biology is None else biology
    symptom_sections = set(symptom_sections)

    # domain -> list of (column, sign)
    groups: dict[str, list[tuple[str, int]]] = {}
    kinds: dict[str, str] = {}

    # symptom instruments (auto-grouped by stem)
    for c in X.columns:
        v = by_name.get(c)
        if v is None or v.section not in symptom_sections:
            continue
        dom = instrument_stem(c)
        groups.setdefault(dom, []).append((c, +1))
        kinds[dom] = "symptom"

    # biology composites (curated)
    for dom, members in biology.items():
        present = [(c, sgn) for c, sgn in members if c in X.columns]
        if present:
            groups[dom] = present
            kinds[dom] = "biology"

    scores: dict[str, pd.Series] = {}
    meta_rows: list[dict] = []
    for dom, members in groups.items():
        z = pd.DataFrame(
            {c: _robust_z(X[c]) * sgn for c, sgn in members}, index=X.index
        )
        obs_frac = z.notna().mean(axis=1)
        score = z.mean(axis=1)
        score = score.where(obs_frac >= min_items_frac, np.nan)
        scores[dom] = score
        meta_rows.append({
            "domain": dom,
            "kind": kinds[dom],
            "n_items": len(members),
            "coverage": float(score.notna().mean()),
            "members": ",".join(c for c, _ in members),
        })

    scores_df = pd.DataFrame(scores, index=X.index)
    meta_df = pd.DataFrame(meta_rows).set_index("domain").sort_values(
        ["kind", "n_items"], ascending=[True, False]
    )
    return scores_df, meta_df
