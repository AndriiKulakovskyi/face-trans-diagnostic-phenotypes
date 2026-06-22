"""M4 prognosis engine — does the durable map *predict*, beyond diagnosis + severity?

M3 showed the map and the strata cohere and persist over time; M4 tests whether a baseline
coordinate or stratum **forecasts a future clinical outcome, incrementally beyond DSM-5 diagnosis
+ baseline severity + the baseline value of that same outcome** — the durable *trait* axes
(cognition, metabolic, inflammatory) predicting the moving *state* outcomes (functioning,
severity). "Persists" != "predicts".

M4 is a **consumer** of the fixed M1/M2/M3 objects: it never re-discovers or re-scores. It joins
the M3 panel (baseline coordinates + their per-patient measurement error) to the outcomes
re-administered at V1/V2, and fits one new statistical object — an errors-in-variables Bayesian GLM
that propagates the M1 uncertainty (via the panel draws) and corrects for attrition (the M3 IPW
weights). Methods of record: docs/PROGNOSIS_MODEL.md. Scope: V0 -> V2 (primary) / V1 (replication);
**internal incremental association only** ("predicts" != "causes"); the outcomes are re-administered
clinical scales, **not** incident events (no hospitalization/relapse/attempt register exists).

Modules (built incrementally, one per concern; mirrors `face.temporal`):
  frame          — outcome registry + the EIV analysis frame (coords+sd + strata + covariates + IPW + outcomes)
  reference      — the diagnosis+severity reference design (the bar every increment must beat)  [stage 42]
  glm            — the errors-in-variables Bayesian outcome GLM (draws-marginalized)            [stage 43]
  compare        — nested model comparison (delta-ELPD / LOO, coefficient table)                [stage 43]
  profile        — confirmatory Bayesian profile regression                                     [stage 44]
  transdiagnostic— within-cohort + head-to-head vs DSM-5 (the G5 test M3 deferred)              [stage 45]
  robustness     — IPW / EIV / leave-one-cohort-out / regression-to-the-mean sensitivity        [stage 46]
"""
from __future__ import annotations

from face.temporal import CANON, VISITS  # single source of dim order + the follow-up grid

# The durable trait axes M3 certified (high ICC, licensed invariant) — the stratify-on / predict-on
# biology corner. M4's headline predictors. See docs/TEMPORAL_OOP_FINDINGS.md §8.
DURABLE: tuple[str, ...] = ("cognition", "metabolic", "inflammatory")

# The general factor (functional-burden / impairment axis). It is the error-aware baseline-severity
# term in the reference model, and is NEVER a predictor of a functioning/severity outcome — G is
# anchored on the EGF/CGI-S/FAST/EQ-5D items, so "G predicts future functioning" is autoregression.
SPINE: str = "overall_severity"

# Baseline coordinates are always read at V0; outcomes at the horizons (meta.primary_horizon).
PREDICTOR_VISIT: str = "V0"

__all__ = ["CANON", "VISITS", "DURABLE", "SPINE", "PREDICTOR_VISIT"]
