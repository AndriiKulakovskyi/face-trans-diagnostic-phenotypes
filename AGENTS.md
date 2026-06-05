# AGENTS.md

This project's guide for AI assistants and collaborators is **[CLAUDE.md](CLAUDE.md)** — read it for
the **V3** overview, the four-layer precision-psychiatry framing, the **instructions for future
agents** (ignore V2 roadmap assumptions · preserve no-naive-imputation · use observed-likelihood
modeling · keep diagnosis as covariate/validation, not a clustering feature · produce V3
decision-modeling outputs), the data-processing foundation, repo layout, and status.
(AGENTS.md is intentionally a thin pointer, to avoid two guides drifting apart.)

**One-line status.** The **V3 precision-psychiatry plan is the single source of truth**
([docs/V3_PLAN.md](docs/V3_PLAN.md)): diagnostic cohorts → hybrid transdiagnostic dimension discovery
(patient-level Bayesian/FIML latent model, soft priors, mixed likelihoods) → validated probabilistic
patient strata → prognosis/treatment decision models. The completed **V2** dimensional study is a
**benchmark / reference arm only** ([docs/legacy_v2/](docs/legacy_v2/README.md)); the runnable
`src/` + `scripts/01–15` are the V2 implementation, and the V3 discovery engine is **not yet built**.
The pre-V2 v1 study is archived at git tag `v1-archive-2026-05-30`.
