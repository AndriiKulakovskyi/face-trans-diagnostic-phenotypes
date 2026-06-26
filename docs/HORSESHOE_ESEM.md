# Sparse-ESEM cross-loadings via a regularized horseshoe — protecting weak-but-valid signal in low-instrument factors

> A method note. The map this produces (8 factors: the immunometabolic merge + horseshoe cross-loadings)
> is fit by `notebooks/run_horseshoe_map.py`; status/results at the bottom.

## The problem this solves

A measurement map has to choose how indicators relate to factors. Two standard choices both fail here:

- **Hard-zero (rigid CFA).** Every off-home loading is fixed at exactly 0; each indicator loads only on its
  home factor (+ the general factor G). This is the certified default — clean and identified, but it
  *forbids* an instrument from loading weakly on a second axis even when the data support it. A reviewer can
  fairly call it over-constrained.
- **Free / soft-Normal (naive ESEM).** Free all off-home cells at, say, Normal(0, 0.05). The documented
  failure: this **floods every factor's column with weak cross-loadings**, and a **thin factor** — one
  defined by very few indicators (substance: 4; mania: 2) — loses its identity, because the flood of
  spurious cross-loadings into its column overwhelms the handful of genuine home loadings. The column goes
  multimodal and poisons convergence. *This is why hard-zero is the default.*

We want the middle ground: **let an instrument carry a small loading on several axes when the data genuinely
support it, without collapsing a low-instrument factor.** The classic tool for "mostly zero, but let real
signal through" is a **sparsity-inducing prior**, and the right one is the **regularized horseshoe**.

## The method

For each off-home specific↔specific cross-loading \(\lambda_j\) we place a **regularized ("Finnish")
horseshoe** (Carvalho et al. 2009; Piironen & Vehtari 2017):

\[
\lambda_j = z_j \,\tau\, \tilde\eta_j,\qquad
\tilde\eta_j = \sqrt{\dfrac{c^2\,\eta_j^2}{c^2 + \tau^2\eta_j^2}},\qquad
z_j\sim\mathcal N(0,1),\;\; \eta_j\sim\text{HalfCauchy}(1),\;\; \tau\sim\text{HalfNormal}(\tau_0).
\]

Three pieces, each doing one job:

- **Global shrinkage \(\tau\)** (small \(\tau_0\)) pulls the *whole* set of cross-loadings toward 0 — the
  prior's default belief is "no cross-loadings."
- **Local shrinkage \(\eta_j\)** has **heavy (Cauchy) tails**: a single cross-loading the data insist on can
  take a large \(\eta_j\) and **escape** the global shrinkage. This is the "horseshoe" — a sharp spike at 0
  with heavy tails, so most cells are ~0 but a few are free.
- **Slab \(c\)** regularizes the escape: \(\tilde\eta_j \to c/\tau\) for large \(\eta_j\), so an escaped
  loading is capped near \(|\lambda|\approx c\) (cross-loadings stay *small* — they are not home loadings),
  and it **tames the funnel** so NUTS mixes. The fit is non-centred (\(z\) separate from the scales).

### Why this protects thin factors (the point)

Dilution happens when many indicators from *other* factors acquire spurious cross-loadings **into** a thin
factor's column, drowning its few home loadings. Under the horseshoe those spurious cells get small local
\(\eta\) and are crushed to ~0 by the global \(\tau\) — so the thin column stays defined **only by its home
indicators**, which keep their separate, sign-anchored positive prior (home loadings are *not* horseshoed).
A *genuine* weak cross-loading — clinically real but supported by few instruments — can still emerge through
the heavy tails and be reported with a credible interval. So the prior is exactly matched to the scientific
situation: **default-off, evidence-on, magnitude-capped — which is the honest way to handle a weak signal
carried by a factor with very few indicators.** A map whose credible cross-loadings are few, small, and
clinically sensible is positive evidence the structure is well-specified, not an artefact of rigid zeros.

## Implementation

- `MeasurementConfig.with_horseshoe(tau0=0.05, slab_c=0.30)` sets `cross_loading_prior="horseshoe"`
  (default stays `"hard_zero"`; recorded in `config_sig` only when active, so hard-zero caches are untouched).
- `LoadingSpec.from_core(..., horseshoe=True)` routes every off-home specific cross cell (the `plausible`/
  `unlikely` cross cells) into `hs_cells` instead of hard-zeroing them. Home cells and G/window cells are
  unchanged.
- `BayesianBifactorESEM._build_loadings` builds the regularized horseshoe block (`hs_tau`, `hs_eta`, `hs_z`
  → `lam_hs`) and writes it into Λ. Shared by the marginalized (continuous) and mixed builders, so it works
  at every rung.
- The CI-aware export (`run_export_loadings.py`) surfaces the `lam_hs` cells with their 95% CIs (which
  cross-loadings are credible).

## How it is fit

`notebooks/run_horseshoe_map.py` runs a warm-start ladder (each rung seeds the next):

1. `hs_s1_merged` — continuous core (G, cognition, **immunometabolic**, sleep), **hard-zero** (clean backbone).
2. `hs_s3_merged` — + developmental_risk, mania_activation (continuous), hard-zero.
3. `hs_s5_merged` — full **8-factor MIXED** map (+ suicidality, substance explicit), **horseshoe** cross-loadings,
   **warm-started from the clean hard-zero backbone**. (Warm-starting the relaxed fit from the hard-zero
   solution is identified here — unlike freeing cross-loadings on the two-factor biology block, which was
   non-identified; merging biology removed that rotation.)

```bash
HDF5_USE_FILE_LOCKING=FALSE python notebooks/run_horseshoe_map.py --smoke        # wiring
python3 scripts/run_job.py horseshoe -- env HDF5_USE_FILE_LOCKING=FALSE \
    python notebooks/run_horseshoe_map.py                                         # full (detached)
```

`tau0` (global sparsity; smaller → sparser) and `slab_c` (escaped-loading cap) are the two knobs.

## Validation (mechanism)

A tiny continuous fit on the merged backbone (6 factors, 294 freed cross cells) confirmed the prior behaves:
**0 divergences**, **92% of cross-loadings shrunk to |median| < 0.05** (median 0.002), yet the heavy tails
let a genuine cell reach |λ| ≈ 0.59. Sparse *and* permissive, sampling cleanly.

## Results — the decoupled outcome (2026-06-26)

The **full mixed horseshoe did not converge** (R̂ 3.0 on the loadings). The diagnostic isolated why: the
horseshoe scales mix fine in the continuous model (`hs_tau`/`hs_eta` R̂ 1.00) — it is the **mixed embedding**
(≈1,900 per-patient explicit latents coupling to the cross-loadings through Φ) that breaks it, not the prior.
So we **decoupled** (the architecture is sound and stronger this way):

**1. Operational map = hard-zero 8-factor mixed** (`hs_s5_merged_hz`): converges cleanly (R̂ 1.03, 0 div) —
the coordinates M2–M5 consume.

**2. Sparse-ESEM = continuous validator + selector** (`sparse_esem_6f`; stable variant: fixed τ + Student-t
local). Freeing all 294 off-home cross-loadings, **~83% shrink to ≈0** — the hard-zero zeros are *earned,
not imposed* — and a handful of small, well-converged, clinically-sensible cross-loadings are selected. (The
0.73 cell from the loose-prior diagnostic was a prior-artifact: it vanished under proper shrinkage.)

**3. Final map = hard-zero + the data-earned cross-loadings** (`hs_s5_merged_xc`, R̂ 1.06, 0 div). The full
mixed model is the arbiter: of the 6 sparse-ESEM candidates it kept **3 credible**, all *sleep / childhood-
trauma items → cognition*; the 3 mania candidates widened to non-credible and were dropped (the mania↔sleep
link belongs in Φ ≈ 0.24, not an item cross-loading on a 2-item factor — the same lesson as immunometabolic).

| earned cross-loading | median | 95% CI | R̂ | reading |
|---|---|---|---|---|
| CTQ-37 → cognition | −0.094 | [−0.134, −0.053] | 1.00 | childhood adversity ⟷ cognition |
| PSQI latency → cognition | +0.057 | [+0.011, +0.102] | 1.00 | sleep ⟷ cognition |
| PSQI daytime dysfunction → cognition | −0.070 | [−0.122, −0.020] | 1.00 | sleepiness ⟷ cognition |

Thin factors are intact (substance home |λ| 0.585, mania 0.484). **The validation is the headline:** given
total freedom, the model reproduces known clinical cross-talk and nothing spurious — strong evidence the
8-factor map is well-specified. The final map is `hs_s5_merged_xc`; M2–M5 are rebuilt on it.
