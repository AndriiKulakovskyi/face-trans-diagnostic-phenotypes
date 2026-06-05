"""Phenotype atlas — named, reproducible factor scores for use as **predictive features**.

This is the *feature* view of the v2 dimensional analysis (companion: ``docs/PHENOTYPE_ATLAS.md``,
LABBOOK V2-23). The structural model (manuscript) is **3 weakly-correlated trans-diagnostic axes**
(internalizing · cognition · cardiometabolic). The bootstrap robustness analysis showed those 3 are
the only K-invariant *correlated* structure; beyond them the data contains several **reproducible but
mutually-orthogonal standalone dimensions** (illness-course, substance-use, mania, suicidality,
childhood-adversity), each ~100 % bootstrap-stable but independent of the backbone and of each other.

For prediction we want exactly those orthogonal, non-redundant directions. Each factor below is scored
as the **masked mean of its sign-oriented, standardized first-order construct scores** (no imputation;
observed support only) — a transparent, K-independent definition (not a fragile single-K rotation
slot). Single-construct standalones pass their construct score through directly.

Member lists were derived from a K=7 varimax solution (constructs |loading| > 0.40) and lightly
curated for clinical coherence (e.g. the cardiometabolic block keeps glucose/BP/cholesterol even where
they sit just below 0.40). Signs orient each member so the factor's stated pole is the **high** end.

Caveats that matter for using these as features (full detail in ``docs/PHENOTYPE_ATLAS.md``):
  * **internalizing** mood scales are 0 % observed in FACE-SZ; SZ is scored only via QoL/functioning
    proxies (EQ-5D, GAF, CGI) → in SZ the score means "poor functioning", not "depression".
  * **cognition** is memory-anchored (CVLT) in BP/SZ but executive/fluency-based in DR (no CVLT there).
  * **illness_course** is fixed-historical (baseline-only) and only ~half-covered in FACE-DR.
  * **substance_use** is BP/SZ only (never measured in FACE-DR).
Use the returned per-factor *coverage* fraction to gate which feature is usable for which patient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# factor -> [(construct, sign)]; sign aligns members so HIGHER = the factor's stated pole.
PHENOTYPE_FACTORS: dict[str, list[tuple[str, int]]] = {
    # ── correlated trans-diagnostic backbone (3 weakly-correlated axes) ───────────────────
    "internalizing": [
        ("qidsr", +1), ("madrs", +1), ("staya", +1), ("fast", +1), ("eq5d", -1), ("eq", -1),
        ("qids_anhedonia_interest", +1), ("egf", -1), ("prism", +1), ("cgi_severity", +1),
        ("psqi", +1), ("mars", -1),
    ],  # higher = more distress / symptom severity / worse functioning
    "cognition": [
        ("cvlt_total_recall", +1), ("cvlt_long_delay_free_recall", +1),
        ("cvlt_short_delay_free_recall", +1), ("executive", -1), ("processing_speed", -1),
        ("verbal_fluency_semantic", +1), ("psychomotor_speed", -1), ("working_memory", -1),
        ("perceptual_reason", -1), ("verbal_fluency_phonemic", +1), ("edulevel", +1),
    ],  # higher = BETTER cognitive performance (memory-anchored)
    "cardiometabolic": [
        ("inflammation", +1), ("lipids_hdl", +1), ("cholesterol", +1), ("adiposity", +1),
        ("glycemia", +1), ("blood_pressure", +1), ("autonomic_hr", +1), ("bio_lym_lbstresc", +1),
    ],  # higher = worse cardiometabolic / inflammatory load
    # ── orthogonal standalone dimensions (reproducible, independent of the backbone) ───────
    "illness_course": [
        ("agedebut_hospitalisation", +1), ("agetrt", +1), ("agedebutpremier_episode", +1),
        ("nboccur_hospitalisation_lt", -1), ("hodur_hospitalisation_lt", -1),
    ],  # higher = LATER onset + LOWER hospitalization burden (milder / later course)
    "substance_use": [("substance_use_disorder", +1)],     # lifetime alcohol/cannabis use disorder (BP/SZ)
    "mania": [("mania_activation", +1)],                   # Altman + YMRS
    "suicidality": [("suicidal_ideation", +1)],            # ISF ideation
    "childhood_adversity": [("wurs", +1), ("ctq", +1)],    # childhood ADHD + trauma (weak / exploratory)
}

# the weakly-correlated backbone vs the orthogonal standalones (see docs/PHENOTYPE_ATLAS.md)
AXES: tuple[str, ...] = ("internalizing", "cognition", "cardiometabolic")
STANDALONES: tuple[str, ...] = (
    "illness_course", "substance_use", "mania", "suicidality", "childhood_adversity",
)

# per-factor metadata for documentation + downstream gating
FACTOR_META: dict[str, dict] = {
    "internalizing":       dict(kind="axis", direction="higher = more distress / severity",
                                temporal="state", cohorts="BP, DR (SZ = proxy only)"),
    "cognition":           dict(kind="axis", direction="higher = better cognition",
                                temporal="baseline-anchored", cohorts="BP, SZ, DR (DR = no CVLT)"),
    "cardiometabolic":     dict(kind="axis", direction="higher = worse cardiometabolic load",
                                temporal="trait", cohorts="BP, SZ, DR"),
    "illness_course":      dict(kind="standalone", direction="higher = later onset / milder course",
                                temporal="fixed-historical", cohorts="BP, SZ (DR ~half)"),
    "substance_use":       dict(kind="standalone", direction="higher = lifetime alcohol/cannabis SUD",
                                temporal="lifetime", cohorts="BP, SZ"),
    "mania":               dict(kind="standalone", direction="higher = more activation/mania",
                                temporal="state", cohorts="BP, SZ, DR"),
    "suicidality":         dict(kind="standalone", direction="higher = more suicidal ideation",
                                temporal="state", cohorts="BP, SZ, DR"),
    "childhood_adversity": dict(kind="standalone", direction="higher = more childhood adversity",
                                temporal="fixed-historical", cohorts="BP, SZ, DR (weak signal)"),
}


def build_phenotype_factors(
    construct_scores: pd.DataFrame, min_coverage: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the phenotype-atlas factors from first-order construct scores (no imputation).

    Parameters
    ----------
    construct_scores : DataFrame
        Stage-2 construct scores, index ``[cohort, patient_id]`` × construct (``stage2_scores.pkl``).
    min_coverage : float
        Set a factor's score to NaN for any patient observing fewer than this *fraction* of the
        factor's member constructs (default 0 = score on any observed member).

    Returns
    -------
    (scores, coverage) : tuple of DataFrame
        ``scores`` = patient × factor masked-mean score (z-scaled members, sign-oriented).
        ``coverage`` = patient × factor fraction of member constructs observed (in ``[0, 1]``).
    """
    S = construct_scores
    scores: dict[str, pd.Series] = {}
    coverage: dict[str, pd.Series] = {}
    for fac, members in PHENOTYPE_FACTORS.items():
        cols = [(c, s) for c, s in members if c in S.columns]
        if not cols:
            continue
        sub = S[[c for c, _ in cols]]
        sd = sub.std(ddof=0).replace(0, np.nan)
        z = (sub - sub.mean()) / sd                                  # standardize each construct
        signed = z * np.array([s for _, s in cols], dtype=float)     # orient to the factor's pole
        frac = signed.notna().sum(axis=1) / len(cols)                # coverage fraction
        sc = signed.mean(axis=1)                                     # masked mean (pandas skips NaN)
        sc = sc.where(frac >= min_coverage)
        scores[fac] = sc
        coverage[fac] = frac
    return (pd.DataFrame(scores, index=S.index), pd.DataFrame(coverage, index=S.index))


__all__ = ["PHENOTYPE_FACTORS", "FACTOR_META", "AXES", "STANDALONES", "build_phenotype_factors"]
