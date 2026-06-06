# Legacy — Engine A (superseded)

These four scripts are the **first-generation V3 Bayesian measurement engine**. They are
**superseded** by the config-first soft-prior ESEM-bifactor engine
(`src/v3/latent_models/bayesian/`, run via `scripts/v3/03_build_prior_matrix.py` and
`scripts/v3/04_fit_measurement.py`). Kept for provenance — **do not build on them.**

**Why superseded.** The model spec was hard-coded (`SPEC` / `FACTORS` dicts) and omitted the
severity/functioning indicators, which produced a premature **"no general factor"** conclusion. The
config-first engine reproduces these scripts exactly at its **Stage 0**, then overturns that headline
at **Stage 1** (a general factor identifies — a functional-impairment/distress axis ⊥ biology). The
one novel finding here — objective-sleep as canonical, from `06` — is folded into the new engine's
config (`sleep: objective`). See [`docs/STATE.md`](../../../docs/STATE.md).

| script | was | superseded by |
|--------|-----|---------------|
| `03_bayesian_core.py` | marginalized 4-factor continuous core | engine Stage 0 |
| `04_extended_model.py` | +affective +mixed-likelihood suicidality (5 factors) | engine Stages 1–4 |
| `05_visualize.py` | figures for the above | (regenerate from the engine) |
| `06_sleep_affect_sensitivity.py` | PSQI objective-vs-subjective sleep sensitivity | folded into config |
