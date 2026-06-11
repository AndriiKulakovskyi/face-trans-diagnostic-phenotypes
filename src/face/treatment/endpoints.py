"""M5 treatment-response endpoints — binary state from the raw CGI response signals.

Built on the standard CGI codings (0 = not-assessed → NaN, never imputed): CGI-Improvement
(`cgi02`, 1 = very-much-improved … 7 = very-much-worse), therapeutic effect (`cgi03a`, 1 = marked …
4 = unchanged/worse), side-effects (`cgi03b`, 1 = none … 4 = outweigh), CGI-S (`cgi01`), adherence
(`mars`, 0–10). Methods: docs/TREATMENT_MODEL.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Endpoint:
    """An M5 treatment-response endpoint. `polarity` = good (desirable: response, therapeutic effect)
    or poor (adverse: resistance, side-effects, low adherence)."""

    name: str
    label: str
    polarity: str
    role: str


ENDPOINTS = (
    Endpoint("response", "CGI-Improvement responder (much/very-much improved)", "good", "primary"),
    Endpoint("therapeutic_effect", "CGI therapeutic effect marked/moderate", "good", "primary"),
    Endpoint("resistance", "Treatment-resistant (CGI-S≥4 & not improved)", "poor", "primary"),
    Endpoint("side_effects", "Significant side-effects (interfere+)", "poor", "primary"),
    Endpoint("low_adherence", "Low adherence (MARS ≤ threshold)", "poor", "secondary"),
)


def _valid(df: pd.DataFrame, col: str, lo: float, hi: float) -> pd.Series:
    """Numeric series clipped to the valid CGI range; out-of-range (incl. 0 = not-assessed) → NaN."""
    s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    return s.where((s >= lo) & (s <= hi))


def build_endpoints(df: pd.DataFrame, *, mars_low: float = 5, resistance_cgis: float = 4) -> pd.DataFrame:
    """Add the `ep_{name}` binary columns (float 0/1, NaN where the needed signal is missing/not-assessed).
    Requires the raw response signals (`cgi02`, `cgi03a`, `cgi03b`, `cgi01`, `mars`) as columns."""
    f = df.copy()
    cgi02 = _valid(f, "cgi02", 1, 7)
    cgi03a = _valid(f, "cgi03a", 1, 4)
    cgi03b = _valid(f, "cgi03b", 1, 4)
    cgi01 = _valid(f, "cgi01", 1, 7)
    mars = pd.to_numeric(f["mars"], errors="coerce") if "mars" in f.columns else pd.Series(np.nan, index=f.index)

    f["ep_response"] = (cgi02 <= 2).where(cgi02.notna()).astype(float)
    f["ep_therapeutic_effect"] = (cgi03a <= 2).where(cgi03a.notna()).astype(float)
    f["ep_side_effects"] = (cgi03b >= 3).where(cgi03b.notna()).astype(float)
    f["ep_resistance"] = ((cgi01 >= resistance_cgis) & (cgi02 >= 3)).where(cgi01.notna() & cgi02.notna()).astype(float)
    f["ep_low_adherence"] = (mars <= mars_low).where(mars.notna()).astype(float)
    return f


def load_m5_config(path: str | Path) -> dict:
    """Parse configs/m5_outcomes.yaml (meta + signals + endpoint registry)."""
    return yaml.safe_load(Path(path).read_text())
