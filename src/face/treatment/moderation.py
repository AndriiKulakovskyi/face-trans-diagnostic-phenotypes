"""M5.2b — treatment-effect moderation helpers (E-value + the propensity-restricted sample).

The moderation estimand is the `treat × durable-axis` interaction in the EIV outcome GLM (the formal
"does the map change who benefits"), fit on the propensity common-support sample with stabilized IPTW +
covariate adjustment (doubly robust). This module supplies the confounding-sensitivity **E-value**
(VanderWeele: how strong an unmeasured confounder would have to be, on both treatment and outcome, to
explain away an association) and the sample loader that joins the M5 frame to the 55 propensity output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def e_value(standardized_effect: float) -> float:
    """Approximate E-value for a standardized mean difference `d` (Chinn: RR ≈ exp(0.91·d))."""
    rr = float(np.exp(0.91 * abs(float(standardized_effect))))
    if rr < 1:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def _z(p: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(p))


def mde(se: float, *, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect (two-sided) for a coefficient with posterior SD / standard error `se`,
    at significance `alpha` and `power`:  ``MDE = (z_{1-alpha/2} + z_power)·se`` — the smallest true effect
    the design resolves at the stated power. The yardstick that distinguishes a **bounded** null (small
    MDE, the design could have seen a meaningful effect and didn't) from an **underpowered** one (large
    MDE). `se` is the interaction/ATE posterior SD already produced by `fit_glm` (`_summarize`'s `sd`)."""
    return (_z(1.0 - alpha / 2.0) + _z(power)) * abs(float(se))


def sd_from_eti(lo: float, hi: float, *, mass: float = 0.94) -> float:
    """Recover an (approximately Gaussian) posterior SD from an equal-tailed credible interval of total
    `mass`:  ``SD ≈ (hi − lo) / (2·z_{(1+mass)/2})``. Lets the MDE be computed from already-serialized
    ETIs with no model refit (the project's 94% intervals → z_{0.97} ≈ 1.881)."""
    return (float(hi) - float(lo)) / (2.0 * _z((1.0 + mass) / 2.0))


def load_moderation_sample(question: str, mode: str, m5_dir: str | Path) -> pd.DataFrame:
    """The M5 frame joined to the 55 propensity output (treat, IPTW), restricted to common support."""
    m5_dir = Path(m5_dir)
    frame = pd.read_parquet(m5_dir / "analysis_frame.parquet")
    frame["patient_id"] = frame["patient_id"].astype(str)
    ps = pd.read_parquet(m5_dir / f"propensity_{question}_{mode}.parquet")
    ps["patient_id"] = ps["patient_id"].astype(str)
    ps = ps[ps["in_support"]]
    return frame.merge(ps[["cohort", "patient_id", "treat", "iptw"]], on=["cohort", "patient_id"], how="inner")
