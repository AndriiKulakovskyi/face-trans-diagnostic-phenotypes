"""M5 treatment-response-heterogeneity engine — does the durable map predict response to TAU?

The classic M5 (does a stratum *moderate* treatment X vs Y?) is not answerable in FACE: there is no
treatment-assignment variable. Re-scoped (methods of record: docs/TREATMENT_MODEL.md) to **treatment-
response heterogeneity** — does a baseline coordinate or stratum predict who **responds to**, is
**resistant to**, **tolerates**, and **adheres to** treatment-as-usual, incrementally beyond DSM-5
diagnosis + baseline severity? Response *stratification*, not treatment *selection* (TAU is unobserved).

M5 is a consumer of the fixed M1/M2/M3 objects (panel, draws, strata, IPW) and **reuses the M4 engine**
(`face.prognosis.{glm,reference,compare,clinical_value,robustness}`); its own modules add the response
endpoints and the response-signal extraction. The load-bearing hazard is that response is
severity-confounded, so the *beyond-severity* gate (Q2) is make-or-break.

Modules (built incrementally):
  endpoints — response / resistance / tolerability / adherence binaries from the raw CGI signals
  frame     — the M5 analysis frame (panel coords + strata + covariates + the response endpoints + IPW)  [stage 51]
"""
from __future__ import annotations

from face.prognosis import (  # one source of dim order, durable axes, grid
    CANON,
    DURABLE,
    SPINE,
    VISITS,
)

# Raw harmonized response signals (NOT in the processed tables; extracted native via the data layer).
RESPONSE_SIGNALS: tuple[str, ...] = ("cgi02", "cgi03a", "cgi03b", "cgi01", "mars")

__all__ = ["CANON", "VISITS", "DURABLE", "SPINE", "RESPONSE_SIGNALS"]
