"""M1 — the transdiagnostic dimensional map (the most complex, load-bearing milestone).

One global, missingness-aware Bayesian sparse-bifactor / ESEM model with mixed likelihoods and a
Gaussian-copula (rank-INT) continuous block, marginalized via the Woodbury identity, with a
regularized-horseshoe prior on the off-home cross-loadings. Fit at full N = 9,013, cohort-weighted.

  * ``engine``    — the M1 model + staged runner + patient projector + visualizer.
  * ``kernel``    — the marginalized Woodbury likelihood + rank-INT/copula transforms (golden-tested).
  * ``sampling``  — the NUTS sampling runner (per-cohort invariance fits).
  * ``confirm``   — in-engine confirmation (prior-free refit + PPC + WAIC).
  * ``synthetic`` — the invertible copula generator (synthetic cohorts for golden tests).

Methods of record: docs/methods/m1_measurement.md.
"""
