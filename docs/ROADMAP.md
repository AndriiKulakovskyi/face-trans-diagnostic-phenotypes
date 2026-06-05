# ROADMAP — FACE V3 precision psychiatry

> Single source of truth for *what* we are doing and *why*. The detailed plan of record is
> [`V3_PLAN.md`](V3_PLAN.md); the target pipeline is [`PIPELINE.md`](PIPELINE.md). The completed **V2**
> dimensional study is the **benchmark / reference arm** only — see [`legacy_v2/`](legacy_v2/README.md).
> The pre-V2 **v1** study is at git tag `v1-archive-2026-05-30`.

## Objective

Turn the FACE 3-cohort psychiatric data into a **precision-psychiatry stratification and
decision-modeling framework**, in four layers that must not be collapsed:

```text
diagnostic cohorts (BP · SZ · DR)            ← entry + validation metadata, NEVER clustering features
  → transdiagnostic dimension discovery       ← patient-level, missingness-aware latent measurement
  → validated patient strata                   ← probabilistic decision regions, not natural subtypes
  → prognosis / treatment decision models      ← the precision-psychiatry objective
```

## Primary question

Do patient-level latent dimensions and the strata derived from them add clinically meaningful
predictive or decision value **beyond** `diagnosis + age + sex + site + baseline severity`?

## Design principles (hold everywhere)

- **No naive imputation, ever.** No completed-data / mean / KNN / MICE matrix before discovery or
  clustering. Use **observed-data likelihood** (FIML / Bayesian) over each patient's observed cells,
  with **posterior uncertainty**, and explicit **missingness models** where missingness is informative.
  Keep deterministic **skip-logic** structural-zero decoding (it is not imputation).
- **Diagnosis is a covariate / validation target, not a clustering feature.** Strata and dimensions are
  derived without DSM labels; DSM is used to *validate* (η², coverage, confounding), not to define.
- **V0 anchor.** Dimensions are defined at baseline V0; later visits (V1–V4) test temporal coherence
  and supply outcomes — they never define the structure.
- **Soft starting ontology.** The 10 candidate dimensions seed **soft priors**, not hand-tagged scores;
  the data may **confirm, split, merge, reject, downgrade, or cross-load** any of them.
- **Estimator hierarchy.** **Primary discovery engine** = patient-level **Bayesian sparse bifactor /
  ESEM-like** model with **mixed likelihoods** + soft loading priors. **Confirmatory benchmark** =
  **FIML SEM/ESEM**. **Reproducibility baseline** = the **V2 masked-correlation** factors.
- **Utility, not elegance.** Every accepted dimension/stratum must show a downstream value (calibration,
  discrimination, decision-curve net benefit, subgroup prognosis, or treatment-effect heterogeneity).

## The 10 candidate dimensions (soft ontology → adjudicated)

Impulsivity · Cognitive flexibility · Negative symptoms · Anhedonia · Metabolism/immunometabolism ·
Sleep/circadian · Overall clinical severity · Sensory abnormalities · Neurodevelopment · Suicidality.

Starting status (eligibility *before* modeling; full table in [`V3_PLAN.md`](V3_PLAN.md) §0B):
**core** = severity (`G`), cognition, metabolism (test split), sleep, suicidality; **extension** =
anhedonia; **proxy/module** = impulsivity, neurodevelopment, negative symptoms; **unsupported unless
direct indicators exist** = sensory abnormalities. Each is then adjudicated to
{confirmed · split · merged · module · proxy · unsupported} by the latent model.

## Phases (see [`V3_PLAN.md`](V3_PLAN.md) for the full A–T plan)

| Phase | Theme | Status |
|---|---|---|
| **A** | Foundation: freeze V2 as benchmark; V3 data contract; harmonization/units/direction; skip-logic | ✅ done — V3-1 (`scripts/v3/01`) |
| **B** | Missingness atlas + mechanism classification + measurement eligibility | ✅ done — V3-2 (`scripts/v3/02`) |
| **C** | Soft-prior construct map (V2 constructs + 10 candidates) | ✅ done — V3-1 (`configs/`) |
| **D** | V2 masked-estimator replication on V3 data (reproducibility baseline) | ◻ V2 code exists (`scripts/01–06`) |
| **E** | FIML SEM/ESEM benchmark + general-vs-specific test | ⬜ planned |
| **F** | **Bayesian sparse bifactor + mixed likelihoods (primary discovery engine)** | 🟢 certified core — V3-5 (`scripts/v3/03`, marginalized; R-hat 1.01, 0 div) |
| **G** | Model comparison + dimension adjudication + retest V2 claims | ⬜ planned |
| **H** | Measurement invariance / transdiagnostic validity / DIF | ⬜ planned |
| **I** | Posterior patient-level dimension scores + V3 phenotype atlas | ⬜ planned |
| **J** | Probabilistic strata as decision regions | ⬜ planned |
| **K** | Strata validation (stability · artefact · clinical · longitudinal) | ⬜ planned |
| **L** | Prognosis model ladder (M0→M6) + missingness-aware learners | ⬜ planned |
| **M** | Treatment & decision modeling (target-trial emulation, CATE by stratum) | ⬜ planned |
| **N** | Clinical interpretation: dimension/stratum/model cards (TRIPOD-AI, PROBAST-AI) | ⬜ planned |
| **O–T** | Repo structure · acceptance criteria · deliverables · risk register · V2→V3 management · decision tree | ⬜ planned |

**Progress (2026-06-05):** Phases **A·B·C done**, Phase **F core CERTIFIED** — V3 code is in
`scripts/v3/` (`01` eligibility · `02` missingness · `03` Bayesian core), outputs in `results/v3/`,
step journal in [`LABBOOK_V3.md`](LABBOOK_V3.md). Current state: a **certified** marginalized
correlated-factor model on the continuous core (cohort-balanced 500/cohort; R-hat 1.01, 0 div) —
weakly-correlated factors, **no general factor**, cognition≈⊥biology, metabolic/inflammatory separable
(mean |Φ|≈0.09). Next: extend to suicidality + affective/anhedonia + the cognition MNAR arm. `◻` = the existing
`src/trans_diag/` + `scripts/01–15` are the **V2 benchmark implementation** (Phase D baseline).
Phases E, G–T are **not yet built**.

## What "done" looks like

Three manuscripts: (1) the patient-level missingness-aware **measurement model**; (2) **dimensions →
validated patient strata**; (3) **precision-psychiatry decision modeling**. Each V3 claim must beat or
defensibly refine the V2 benchmark — reproducing V2 with heavier machinery is **not** success.
