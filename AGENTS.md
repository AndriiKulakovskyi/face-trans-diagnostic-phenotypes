# AGENTS.md

This project's guide for AI assistants and collaborators is **[CLAUDE.md](CLAUDE.md)** — read it for
the **V3** overview, the four-layer precision-psychiatry framing, the **instructions for future
agents** (preserve no-naive-imputation · use observed-likelihood modeling · keep diagnosis as
covariate/validation, not a clustering feature · produce V3 decision-modeling outputs), the
data-processing foundation, repo layout, and status.
(AGENTS.md is intentionally a thin pointer, to avoid two guides drifting apart.)

**One-line status.** The **V3 precision-psychiatry plan is the single source of truth**
([docs/V3_PLAN.md](docs/V3_PLAN.md)): diagnostic cohorts → hybrid transdiagnostic dimension discovery
(patient-level marginalized Bayesian / FIML latent model, soft priors, mixed likelihoods) → validated
probabilistic patient strata → prognosis/treatment decision models. The **certified measurement model**
is built (data layer `src/v3/data/`, pipeline `scripts/v3/`, outputs `results/v3/`; see
[docs/V3_RESULTS.md](docs/V3_RESULTS.md)); the downstream strata / prognosis / treatment layers are
**not yet built**.
