"""Clinical event surrogates — recover actionable 'events' from the repeated scales.

M4's outcomes are repeated clinical scales, not a hospitalization/relapse register. But clinically
meaningful **state transitions** can be defined from the V0→V1→V2 scales — functional remission,
functional deterioration, sustained impairment, recovery, a CGI-S 'relapse surrogate', sustained
illness. These binaries are more actionable than a z-scored level, they concentrate the prognostic
signal, and they carry intuitive metrics (rates, AUC, NNT). Thresholds are the standard GAF/CGI-S
anchors (GAF≥71 ≈ mild/no symptoms; GAF<61 ≈ moderate impairment; CGI-S≤2 ≈ remission; CGI-S≥4 ≈
moderately ill; a ≥10-pt GAF drop / ≥2-pt CGI-S rise = clinically meaningful change). Nothing is
imputed — an endpoint is NaN whenever any visit it needs is missing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Endpoint:
    """A binary clinical endpoint. `polarity` = 'good' (a desirable outcome) or 'poor' (an adverse
    one) — for sign-correct framing. `needs` lists the visits required (else NaN, never imputed)."""

    name: str
    label: str
    outcome: str
    needs: tuple
    polarity: str


ENDPOINTS = (
    Endpoint("egf_remission", "Functional remission (GAF ≥ 71 at 2y)", "egf", ("V0", "V2"), "good"),
    Endpoint("egf_recovery", "Functional recovery (impaired → GAF ≥ 71)", "egf", ("V0", "V2"), "good"),
    Endpoint("egf_deterioration", "Functional deterioration (GAF drop ≥ 10)", "egf", ("V0", "V2"), "poor"),
    Endpoint("egf_sustained_impair", "Sustained impairment (GAF < 61 at V1 & V2)", "egf",
             ("V0", "V1", "V2"), "poor"),
    Endpoint("cgi_remission", "Symptomatic remission (CGI-S ≤ 2 at 2y)", "cgi_s", ("V0", "V2"), "good"),
    Endpoint("cgi_relapse", "Clinical worsening (CGI-S rise ≥ 2 — relapse surrogate)", "cgi_s",
             ("V0", "V2"), "poor"),
    Endpoint("cgi_sustained_severe", "Sustained illness (CGI-S ≥ 4 at V1 & V2)", "cgi_s",
             ("V0", "V1", "V2"), "poor"),
)


def build_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the `ep_{name}` binary columns (float 0/1, NaN where any needed visit is missing). Recovery
    is defined only among the baseline-impaired (GAF<61) — its denominator is those with room to recover."""
    f = frame.copy()
    egf0, egf1, egf2 = f["egf__V0"], f["egf__V1"], f["egf__V2"]
    cgi0, cgi1, cgi2 = f["cgi_s__V0"], f["cgi_s__V1"], f["cgi_s__V2"]
    have_egf02 = egf0.notna() & egf2.notna()
    have_egf012 = have_egf02 & egf1.notna()
    have_cgi02 = cgi0.notna() & cgi2.notna()
    have_cgi012 = have_cgi02 & cgi1.notna()

    f["ep_egf_remission"] = (egf2 >= 71).where(have_egf02).astype(float)
    f["ep_egf_recovery"] = ((egf0 < 61) & (egf2 >= 71)).where(have_egf02 & (egf0 < 61)).astype(float)
    f["ep_egf_deterioration"] = (egf2 <= egf0 - 10).where(have_egf02).astype(float)
    f["ep_egf_sustained_impair"] = ((egf1 < 61) & (egf2 < 61)).where(have_egf012).astype(float)
    f["ep_cgi_remission"] = (cgi2 <= 2).where(have_cgi02).astype(float)
    f["ep_cgi_relapse"] = (cgi2 >= cgi0 + 2).where(have_cgi02).astype(float)
    f["ep_cgi_sustained_severe"] = ((cgi1 >= 4) & (cgi2 >= 4)).where(have_cgi012).astype(float)
    return f


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial rate (robust at small n / extreme p). Returns (lo, hi)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_table(frame: pd.DataFrame, ep: str, by: str) -> pd.DataFrame:
    """Per-`by`-group rate of endpoint `ep_{ep}` with Wilson CI and N (non-missing only)."""
    col = f"ep_{ep}"
    rows = []
    for g, sub in frame.groupby(by, sort=False):
        s = sub[col].dropna()
        n, k = int(len(s)), int(s.sum())
        lo, hi = wilson_ci(k, n)
        rows.append({by: g, "n": n, "k": k, "rate": (k / n if n else float("nan")),
                     "lo": lo, "hi": hi})
    return pd.DataFrame(rows)
