"""The FACE profile — two clinically-actionable trans-diagnostic indices (translational proposal).

Compact, fixed-weight summaries of the six-axis dimensional model (MANUSCRIPT §3.9), computable
from a handful of routine instruments for clinical follow-up (*suivi*). NOT a replacement for the
6-axis model — two actionable readouts of it, one per use case:

  FACE-D (affective-distress / functional severity): a 3-item proxy of the depression/internalizing
         axis (reproduces it at r≈0.97) — the strongest predictor of patient-reported outcomes;
         for tracking symptom burden and treatment response across BP/SZ/DR on one scale.
  FACE-M (cardiometabolic load): a proxy of the metabolic axis (r≈0.88) — a trait-stable
         physical-health risk flag for monitoring (mortality-relevant, partly iatrogenic).

Both are **sign-oriented so higher = more severe / more risk**, computed as a masked mean of
standardized component domains (a score is NaN if fewer than ``min_obs`` components are observed —
no imputation). They are standardized to the reference cohort (z); for clinical reporting rescale
to T-scores, ``T = 50 + 10*z``. FACE-D deliberately uses *symptom* scales only (not the FAST/EQ-5D
functioning/QoL items the depression axis also loads on) so it can predict those outcomes without
circularity.

These are a *proposal* requiring prospective validation — not a validated clinical instrument.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["FACE_D_ITEMS", "FACE_M_ITEMS", "compute_face_scores"]

# component domains and clinical sign (+1: higher value → more severe / more risk)
FACE_D_ITEMS: dict[str, int] = {"qidsr": +1, "madrs": +1, "staya": +1}            # depression + anxiety
FACE_M_ITEMS: dict[str, int] = {"metabolic_syndrome": +1, "cholesterol": +1, "inflammation": +1}


def _masked_index(z: pd.DataFrame, items: dict[str, int], min_obs: int) -> np.ndarray:
    """Sign-oriented masked mean of the observed component columns (NaN if < min_obs observed)."""
    cols = [c for c in items if c in z.columns]
    if not cols:
        return np.full(len(z), np.nan)
    M = z[cols].to_numpy(float) * np.array([items[c] for c in cols], dtype=float)
    obs = np.isfinite(M)
    n = obs.sum(axis=1)
    s = np.where(obs, M, 0.0).sum(axis=1) / np.maximum(n, 1)
    return np.where(n >= min(min_obs, len(cols)), s, np.nan)


def compute_face_scores(domains: pd.DataFrame, *, standardize: bool = True,
                        min_obs: int = 2) -> pd.DataFrame:
    """Compute the FACE-D and FACE-M indices from (residualized) domain scores.

    Parameters
    ----------
    domains : DataFrame [patients × domain scores] — the construct-level domain scores used by the
              dimensional model (e.g. ``results/cluster_domains_scores.parquet``).
    standardize : z-score each component domain over the input cohort before averaging (default
              True; set False if the domains are already standardized).
    min_obs : minimum observed components for a non-missing score (default 2 of 3).

    Returns
    -------
    DataFrame [FACE_D, FACE_M] aligned to ``domains.index`` (NaN where under-observed).
    """
    z = domains
    if standardize:
        z = (domains - domains.mean()) / domains.std(ddof=0)
    return pd.DataFrame(
        {"FACE_D": _masked_index(z, FACE_D_ITEMS, min_obs),
         "FACE_M": _masked_index(z, FACE_M_ITEMS, min_obs)},
        index=domains.index,
    )
