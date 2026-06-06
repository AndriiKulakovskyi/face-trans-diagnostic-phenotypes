# RESULTS — empirical findings log (M1, the measurement map)

> Curated, article-grade record of the measurement-model results and their interpretation. Methods of
> record: [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md); the prior (theory) map: [`PRIOR_ATLAS.md`](PRIOR_ATLAS.md);
> machine-generated per-stage tables: `reports/04_stage*_{report.md,loadings.csv}`. Every number here is
> reproducible from `scripts/01_build_data.py` → `scripts/04_fit.py`. Accumulates per stage (S1 → S5).
>
> **Scope discipline:** staged fits S1–S4 are convergence checkpoints + partial reads; the *reported map*
> is the global fit S5. S1 below is the **continuous backbone** only — read its boundaries (§S1.6).

---

## S1 — continuous core (G + cognition · metabolic · inflammatory · sleep)

**Headline.** On the full FACE V0 sample (N = 9,013, no completeness selection, no imputation), a certified
bifactor measurement model shows a clean **general factor G = transdiagnostic functional burden**, and —
the load-bearing result — **biological load (metabolic, inflammatory) is approximately orthogonal to G**,
whereas cognition and sleep partly track it.

### S1.1 Goal

Three objectives, deliberately narrow (S1 is the "first stable fit", not the final map):
1. Establish the **general factor G** (overall illness burden) on the continuous backbone.
2. Establish the four **continuous specific dimensions**: cognition, metabolic, inflammatory, sleep.
3. **Feasibility gate** — prove the model can be fit on the *entire* sample with *no imputation* and
   *certify*. If it could not, the whole no-imputation/full-sample approach would be in doubt.

### S1.2 Method

- **Data:** all 9,013 V0 patients (BP 6,252 · SZ 2,209 · DR 552); the 68 **continuous** indicators
  homing on G + the four specific factors; skip-logic structural-zeros decoded; **missing left missing**.
- **Model:** **bifactor** — each indicator's variance = (loading on G)² + (loading on its specific
  factor)² + noise. **G anchored only by functioning / global-severity measures** (FAST, EGF, EQ-5D,
  CGI-S) so it denotes "overall impairment/severity", not any single symptom. Independent specifics
  (Φ = I) and simple structure (no cross-loadings) — the minimal first fit; correlations + cross-loadings
  are S2.
- **Estimation:** observed-data likelihood (each patient contributes only observed cells — the
  no-imputation principle) on the **full sample** (no most-complete subsampling → no selection bias). Fit
  via the **marginalized (Woodbury, low-rank) parameterization** (latents integrated out) with NumPyro/JAX
  on the Mac M4 (CPU). The mathematically-equivalent **explicit-latent** parameterization was also run and
  **reproduced the loadings** — the result is not an artifact of one estimator.

### S1.3 Certification — **CERTIFIED**

`N = 9,013` · `J = 68` continuous indicators · `415,531` observed cells ·
**max R-hat 1.010 · min ESS 1,939 · 0 divergences · no Heywood** (gates: R-hat ≤ 1.01, ESS ≥ 400, div = 0).
Source: `reports/04_stage1_report.md`.

### S1.4 Loadings

**G — functional burden (anchors load on G only):**

| indicator | loading | | indicator | loading |
|---|---:|---|---|---:|
| FAST (total) | 1.04 | | CGI-S | 0.54 |
| FAST-25…30 (components) | 0.71–0.80 | | EQ-5D VAS | 0.53 |
| EGF (functioning) | 0.69 | | FAST-28 | 0.47 |
| EQ-5D | 0.58 | | subjective-illness (`lvsbjind`) | 0.01 |

**Specific factors — mean primary home loading:** cognition 0.57 · sleep 0.50 · inflammatory 0.38 ·
metabolic 0.32. (Biology pools are heterogeneous — e.g. BMI 0.92 but many labs 0.2–0.4 — so the factors
are identified but indicator quality varies.)

**Bifactor — mean |loading on G| of each specific factor's indicators (the orthogonality test):**

| domain | mean \|loading on G\| | reading |
|---|---:|---|
| **metabolic** | **0.08** | ≈ independent of overall burden |
| **inflammatory** | **0.07** | ≈ independent of overall burden |
| sleep | 0.22 | moderately tracks burden |
| cognition | 0.27 | moderately tracks burden |

### S1.5 Interpretation (in project context)

1. **Methodological result — the map can be built honestly, at scale.** A bifactor model certifies on
   *all* 9,013 patients with no imputation and no completeness selection. This removes the central
   validity threat (and the specific flaw behind the earlier engine's "no general factor" claim, which
   was fit on a completeness-selected subsample).
2. **G is a clean functional-burden axis.** It is anchored by impairment (FAST), functioning (EGF/EQ-5D),
   and clinician severity (CGI-S) — and *not* by any specific symptom. (`lvsbjind`, subjective illness,
   ≈ 0 — patients' subjective rating barely tracks the objective burden axis.) G is the map's principal
   axis: "how impaired/severe is this patient overall."
3. **Biology is a separate axis from clinical severity — the project's load-bearing premise.** Metabolic
   (0.08) and inflammatory (0.07) load ≈ 0 on G, so biological load varies *independently* of overall
   burden: two patients equally impaired clinically can differ sharply in metabolic/inflammatory load. If
   biology merely rose with severity it would be redundant and useless for stratification; because it is
   roughly orthogonal, **biological strata can capture heterogeneity that severity alone misses** — which
   is the whole point of the precision-psychiatry layer to come. It also means a patient **cannot be
   reduced to one severity number**; the multidimensional profile is the object the stratification layer
   will act on. Cognition (0.27) and sleep (0.22) partly *are* "being unwell" (they move with burden) —
   clinically sensible.

**Theory → data check.** The prior atlas (`PRIOR_ATLAS.md`) *hypothesised* biology on its own
metabolic/inflammatory factors with only a weak *possible* cross-link to G; the data **confirmed** biology
sits off the burden axis. On the continuous backbone, the hybrid model did its job — theory proposed, the
FACE data confirmed.

### S1.6 Boundaries — what S1 does **not** yet show

- **Continuous backbone only.** The symptom/behavioural dimensions — suicidality, the depression/anxiety
  composites (MADRS/QIDS/STAI, modelled as cross-loading "windows"), developmental-risk, mania, substance,
  anhedonia — are **not in S1**. So "biology ⊥ burden" concerns the *functional-burden G*, not symptom dimensions.
- **Independent specifics (Φ = I).** Inter-dimension correlations (e.g. metabolic ↔ inflammatory) are **not
  estimated yet** — that is S2. No between-dimension claims may be drawn from S1.
- **Simple structure.** No cross-loadings yet (S2 frees them).
- **V0 baseline, internal validity only** — no temporal persistence (V1–V4), no external cohort, no
  cross-cohort measurement-invariance test yet.
- It is a **measurement** result — not strata, not prognosis.

### S1.7 Position in the roadmap

```
cohorts → DIMENSIONS (M1, building) → strata (M2) → prognosis (M4) / treatment (M5)
               ▲
          S1 = G + continuous backbone, CERTIFIED   →   S2 cross-loadings + windows + Φ
          → S3 mixed-likelihood (suicidality, developmental) → S4 anhedonia
          → S5 GLOBAL = the reported map → FIML confirmation → adjudication → empirical atlas
```

S1 is the first certified piece of the dimensions layer, and it already delivers the project's
load-bearing hypothesis on the backbone: **biology is its own axis, not a proxy for severity** — exactly
what would make biological strata worth drawing later.
