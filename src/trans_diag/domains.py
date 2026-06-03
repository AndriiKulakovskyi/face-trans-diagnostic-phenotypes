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
    "COGNITIVE_COMPOSITES",
    "DOMAIN_SECTIONS",
    "BIOLOGY_SECTIONS",
    "COGNITION_SECTIONS",
]

BIOLOGY_SECTIONS = frozenset({"BILAN BIOLOGIQUE", "CONSTANTES ET ECG"})

# Cognition (neuropsychology) lives in its own section group, parallel to biology, so its
# items aggregate into curated cognitive constructs (COGNITIVE_COMPOSITES) rather than the
# generic symptom-stem path (which would let WAIS sub-items dominate by count — LABBOOK E17).
# DR neuropsychology was recovered 2026-05 (scripts/build_dr_neuropsych_mapping.py); only the
# constructs measured in all three cohorts enter the model.
COGNITION_SECTIONS = frozenset({"NEUROPSYCHOLOGIE"})

# Sections to pull raw features from when building domain scores (symptom + biology +
# cognition). Pass this as `sections=` to to_harmonized_dataset (raw, no residualize).
DOMAIN_SECTIONS = frozenset(CLINICAL_SECTIONS) | BIOLOGY_SECTIONS | COGNITION_SECTIONS

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

# Curated cognitive constructs: construct -> [(feature, sign), ...].
# sign = +1 if higher = better cognition (WAIS standard scores), -1 for TMT times (higher = worse).
#
# v2 (2026-05-30, from docs/neuropsy_features.yaml): the NEUROPSYCHOLOGIE features are already
# construct-level WAIS *standard* scores (1-19) / processing-speed indices / TMT seconds — not
# raw items — so each construct maps directly to its feature(s); no item->stem aggregation is
# needed. Cross-cohort comparability comes from using standard scores (edition-independent) and
# per-cohort source columns (e.g. SZ uses SDMT for processing speed). Processing speed pools
# WAIS-coding + IVT index. Sign convention: +1 if higher = better cognition (WAIS standard scores,
# recall/fluency counts), -1 for TMT times (higher = worse). Members absent from the matrix are
# skipped (masked), so 2-cohort constructs (e.g. verbal_memory = CVLT, BP/SZ only — DR has no CVLT)
# yield NaN for the absent cohort and the masked FA handles it. Unlike v1, processing speed and
# executive are INCLUDED as candidates; the v2 dimensional model re-derives structure from zero and
# the confound battery (15_review_checks) re-tests each axis; no v1 cognition result is assumed.
#
# 2026-06-03 (dictionary review): added verbal_memory (CVLT total/short/long-delay recall, BP/SZ)
# and verbal_fluency (phonemic + semantic, 3-cohort) — the battery previously had no verbal episodic
# memory and a thin flexibility/EF measure. The remaining 2-cohort sensitivity tests (WAIS matrices,
# arithmetic, symbols) stay OUT (NOT USABLE in the dictionary).
COGNITIVE_COMPOSITES: dict[str, list[tuple[str, int]]] = {
    "verbal_reasoning": [("wais_similitudes_std", +1)],
    "working_memory": [("wais_digitspan_std", +1)],
    "processing_speed": [("wais_code_std", +1), ("wais_ivt_index", +1)],
    "psychomotor_speed": [("tmt_a_time_sec", -1)],
    "executive": [("tmt_b_time_sec", -1)],
    "verbal_memory": [
        ("cvlt_total_recall", +1),
        ("cvlt_short_delay_free_recall", +1),
        ("cvlt_long_delay_free_recall", +1),
    ],
    "verbal_fluency": [
        ("verbal_fluency_phonemic", +1),
        ("verbal_fluency_semantic", +1),
    ],
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


def _robust_z(s: pd.Series, clip: float = 5.0) -> pd.Series:
    """Winsorize (1/99) + robust z-score (median / 1.4826·MAD), clipped to ±``clip``. NaN preserved.

    Heavy right-skewed non-negative columns (log-normal labs — prolactin, CRP, triglycerides,
    counts) are log1p-compressed first: otherwise their tiny MAD lets a single high value reach
    z≈100 and dominate every correlation/aggregate (the ``prolactin`` explosion). The ±``clip``
    is a final guard against any residual blow-up. Both keep the masked, no-imputation design.
    """
    x = s.dropna()
    if len(x) and (x >= 0).all() and x.median() > 0 and x.quantile(0.99) > 10 * x.median():
        s = np.log1p(s.clip(lower=0))
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    if pd.notna(lo) and pd.notna(hi) and hi > lo:
        s = s.clip(lower=lo, upper=hi)
    med = s.median()
    mad = (s - med).abs().median()
    scale = 1.4826 * mad if mad and mad > 0 else (s.std() or 1.0)
    return ((s - med) / (scale if scale > 0 else 1.0)).clip(lower=-clip, upper=clip)


def _masked_mean(z: pd.DataFrame, *, min_frac: float | None = None,
                 min_count: int | None = None) -> pd.Series:
    """Row-wise mean of a signed-z frame, NaN unless enough members are observed.

    Gate on a fraction of members (``min_frac``) or an absolute count (``min_count``).
    No imputation: unobserved members are simply excluded from the mean.
    """
    n_obs = z.notna().sum(axis=1)
    score = z.mean(axis=1)
    if min_count is not None:
        return score.where(n_obs >= min_count, np.nan)
    obs_frac = z.notna().mean(axis=1)
    return score.where(obs_frac >= (min_frac if min_frac is not None else 0.5), np.nan)


def build_domain_scores(
    X: pd.DataFrame,
    variables: Iterable[Variable],
    *,
    symptom_sections: Iterable[str] = CLINICAL_SECTIONS,
    biology: dict[str, list[tuple[str, int]]] | None = None,
    cognition: dict[str, list[tuple[str, int]]] | None = None,
    cognition_sections: Iterable[str] = COGNITION_SECTIONS,
    cognition_min_items: int = 1,
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
    cognition:
        Curated cognitive constructs (e.g. :data:`COGNITIVE_COMPOSITES`). When given,
        ``NEUROPSYCHOLOGIE``-section items are aggregated two-level — items → instrument
        stems → constructs — so a heavily-itemized test counts once. ``None`` (default)
        leaves cognition out, reproducing the pre-2026 symptom+biology matrix.
    cognition_sections:
        Sections whose items feed the cognitive constructs (default
        :data:`COGNITION_SECTIONS`).
    cognition_min_items:
        A construct score is ``NaN`` unless at least this many of its member instrument
        stems are observed (default 1). Lenient because instrument coverage is
        heterogeneous across cohorts (e.g. DR has digit span but not the full WAIS
        working-memory battery); a single observed indicator still estimates the ability.
    min_items_frac:
        A symptom/biology/instrument-stem score is ``NaN`` for a patient unless at least
        this fraction of its member items are observed (no imputation).

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

    # cognition constructs (curated, two-level: items → instrument stems → constructs).
    # Kept separate from the symptom-stem path so heavily-itemized tests count once, and so
    # the construct floor can be lenient (≥ cognition_min_items observed stems) under the
    # cross-cohort instrument heterogeneity of the neuropsych battery.
    if cognition:
        cog_sections = set(cognition_sections)
        referenced = {stem for members in cognition.values() for stem, _ in members}
        stem_items: dict[str, list[str]] = {}
        for c in X.columns:
            v = by_name.get(c)
            if v is None or v.section not in cog_sections:
                continue
            stem = instrument_stem(c)
            if stem in referenced:
                stem_items.setdefault(stem, []).append(c)
        # Stem aggregation is also lenient (>= cognition_min_items observed items): the
        # neuropsych battery is partially administered across cohorts (e.g. DR has verbal
        # fluency items fv01-07 but not fv08-19), so a fractional floor over all items of a
        # stem would null whichever cohort ran the shorter version.
        stem_scores = {
            stem: _masked_mean(
                pd.DataFrame({c: _robust_z(X[c]) for c in cols}, index=X.index),
                min_count=cognition_min_items)
            for stem, cols in stem_items.items()
        }
        stem_df = pd.DataFrame(stem_scores, index=X.index)
        for dom, members in cognition.items():
            present = [(s, sgn) for s, sgn in members if s in stem_df.columns]
            if not present:
                continue
            z = pd.DataFrame({s: _robust_z(stem_df[s]) * sgn for s, sgn in present},
                             index=X.index)
            score = _masked_mean(z, min_count=cognition_min_items)
            scores[dom] = score
            meta_rows.append({
                "domain": dom, "kind": "cognition", "n_items": len(present),
                "coverage": float(score.notna().mean()),
                "members": ",".join(s for s, _ in present),
            })

    scores_df = pd.DataFrame(scores, index=X.index)
    meta_df = pd.DataFrame(meta_rows).set_index("domain").sort_values(
        ["kind", "n_items"], ascending=[True, False]
    )
    return scores_df, meta_df
