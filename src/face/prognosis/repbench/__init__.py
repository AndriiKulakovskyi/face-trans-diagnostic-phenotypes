"""M4 representation benchmark — does the M1/M2 Gaussian-copula latent map (coordinates + uncertainty +
A=4 archetypes) match raw indicators at predicting 2-year functional outcomes, while winning on data
efficiency, transportability, honesty, and interpretability?

The hypothesis is representation *quality*, not accuracy supremacy (the coordinates are a compression of the
raw cells, so raw weakly dominates in-distribution asymptotically):
  H1 sufficiency  — raw adds ~nothing over the latent map (a tie is the win);
  H2 efficiency   — the 9-dim bottleneck generalises better at small N;
  H3 uncertainty  — per-patient sd / draws add over mean-only and over raw;
  H4 transport    — the low-dim map transfers across cohorts (LOCO) and time better than a raw black box.

Methods of record: docs/M4_REPRESENTATION_BENCHMARK.md. Phase 0 ships the data wiring (copula objects),
proper-score metrics (CRPS, calibration), and the identical-across-arms CV folds; model fits land in P1+.
"""
from __future__ import annotations

CANON = (
    "overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
    "mania_activation", "suicidality", "developmental_risk", "substance",
)
ARCH = ("arch_w0", "arch_w1", "arch_w2", "arch_w3")
SEED = 20260610
