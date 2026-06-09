# M1 measurement map explained

> ⚠️ **STATUS NOTE (2026-06-09): this teaching guide predates M1 completion and is STALE on results/status.**
> It describes a **7-dimension, provisional** map with "FIML confirmation ahead" and "→ GPU" — all
> **superseded**. M1 is **complete**: a **certified 9-dimension** map (mania + substance added), no GPU used,
> FIML reframed to an in-engine bundle (§5). The **engine intuition + math below remain valid**; for current
> results/status read [`ADJUDICATION.md`](ADJUDICATION.md), [`STATE.md`](STATE.md), [`RESULTS.md`](RESULTS.md).
>
> A companion guide to `MEASUREMENT_MODEL.md`.
>
> The methods document is the formal plan of record. This document is a teaching
> guide: it explains the intuition, math, code path, and compute strategy behind
> Milestone 1, especially the S1 marginalized Woodbury engine.

Equations are written in fenced `math` blocks with LaTeX-style notation so the
document remains readable in Markdown previews that do not render raw `$$`
delimiters, while still being easy to copy into a manuscript or slide deck.

> ### Status & as-built reconciliation (read before §10)
>
> This guide explains the **concepts** and the **marginalized Woodbury engine** — those parts are correct
> and timeless. A few specifics have **drifted from the as-built code/results**; the canonical sources are
> [`RESULTS.md`](RESULTS.md) (what the data showed, S1→S5), [`STATE.md`](STATE.md) (current state), and
> [`MEASUREMENT_MODEL.md`](MEASUREMENT_MODEL.md) (methods of record). The deltas, all expanded in
> **Part C (Appendix) at the end**:
>
> 1. **Entry point.** The real pipeline is `scripts/04_fit.py` → `src/face/models/bayesian/continuous_core.py`
>    (`prepare()`, `build_marginalized()`, `build_mixed()`). There is **no** `04_fit_measurement_map.py` (§10).
> 2. **The engine is now a *grouped*-GEMM Woodbury** — the per-patient `A_i` Cholesky in §9–§10 is built as
>    one BLAS GEMM and factorized **once per unique observed-pattern**, then gathered to patients (≈ 2.75×
>    faster, log-likelihood-identical). §10's code block is the older per-patient form.
> 3. **Φ from `pm.LKJCorr` is the Cholesky factor `L`** → the correlation is `Φ = L Lᵀ` (§5.3's math is
>    right; we hit and fixed a bug that symmetrized `L` instead — it was indefinite at 6 factors).
> 4. **S2 as-built** = inter-dimension Φ + the MADRS/QIDS/STAI **window** cross-loadings; the
>    *specific↔specific* (metabolic↔inflammatory) cross-loadings are **not** freed — rotationally aliased
>    with Φ, so **Φ carries that association**. So §2's "S2 = ESEM cross-loadings" is only partly as-built.
> 5. **S3 was split** into **S3a** (developmental-risk, continuous, *certified*) and **S3b** (suicidality,
>    mixed-likelihood, *provisional*). **S4 → anhedonia rejected.** **S5** ran on a random N=5,000 subsample
>    (the §3.6 frontier fallback), provisional; full-N + certification → GPU.
> 6. **Compute tuning:** the no-funnel marginalized geometry wants **lower** `target_accept` (0.85–0.90) +
>    a tree-depth cap, not the "raise to 0.99" reflex in §5.2 (which *slows* a non-pathological posterior).
> 7. **Priors as-built** match §5.1 (primary `N⁺(0.60,0.30²)`, window cross `N(0,0.25²)`, unlikely
>    `N(0,0.05²)`, `σ=0.05+HalfNormal(1)`); §3.1's `(0.70,0.25)` is the methods-doc spec, not the matrix.

## 0. The short idea

M1 asks a measurement question:

> Given many partially observed clinical, cognitive, behavioral, and biological
> indicators, can we infer a smaller set of continuous dimensions that explain
> how patients vary across diagnoses?

The answer is not a cluster label. It is a **coordinate system**. Each dimension
is an axis, each instrument has a loading on one or more axes, and each patient
gets a position on the axes with uncertainty.

The method gives three different kinds of output:

1. **A scientific map.** Which empirical dimensions exist, and which indicators
   define them.
2. **A patient representation.** For each patient, estimated coordinates on the
   surviving dimensions, with reliability/uncertainty.
3. **A theory check.** A comparison between the prior clinical ontology and the
   posterior structure learned from FACE data.

The reason this is hard is that the FACE matrix is not a neat rectangular table
where every patient has every variable. It is sparse and structured by cohort,
site, instrument availability, and clinical skip logic. M1 therefore uses an
observed-data likelihood: each patient contributes only the cells actually
observed for them.

## 1. What M1 is trying to build

Milestone 1 builds the **measurement map**. The map is not a clustering result
yet. It is the coordinate system on which later clustering, prognosis, and
treatment questions will sit.

The intended order is:

```text
diagnostic cohorts -> transdiagnostic dimensions -> strata -> prognosis / treatment
   BP / SZ / DR           M1 measurement map          later          later
```

The input is the FACE baseline visit (`V0`) across bipolar disorder,
schizophrenia, and depression. The output of M1 should be:

1. A posterior loading matrix: which indicators define each dimension.
2. A factor correlation matrix: how the dimensions relate to each other.
3. Per-patient dimension coordinates with uncertainty.
4. A prior-to-posterior atlas: what theory expected versus what the data
   supported.
5. Adjudication verdicts for candidate dimensions: confirmed, split, merged,
   proxy, rejected, or not testable.

The core principle is that diagnosis is **metadata**, not a measurement
indicator. A patient is not placed on the map because they are BP/SZ/DR; they
are placed on the map because of their observed clinical, cognitive, behavioral,
and biological measurements.

Analogy: the diagnoses are like the city labels on a map. They help us interpret
where people came from, but they are not the latitude and longitude system
itself.

## 2. The stages S1-S5

The final scientific object is the global model, but fitting it cold is hard.
The stages are a continuation strategy: start with a small stable model, then
add one source of difficulty at a time.

The staging logic is not "S1 is the result, then S2 is another result." The
logic is "S1 is the first stable piece of the final model; each later stage
keeps the certified pieces and adds one new difficulty."

| Stage | Adds | Why it exists | What it gives |
|---|---|---|---|
| S1 | Continuous core: `G`, cognition, metabolic, inflammatory, sleep | Proves the model can fit full `N = 9,013` with no imputation. | Stable backbone loadings and the first test of whether biology is separate from functional burden. |
| S2 | ESEM cross-loadings | Lets indicators load on plausible secondary factors. Tests whether simple structure is too rigid. | A more realistic loading matrix: instruments can be multidimensional instead of forced into one box. |
| S3 | Mixed-likelihood blocks such as suicidality and developmental-risk | Adds binary, ordinal, and count indicators that cannot be Gaussian-marginalized as simply. | Brings clinically important non-continuous constructs into the same map. |
| S4 | Thin/partial factors such as anhedonia | Tests whether weakly measured factors identify, merge, or reject. | Evidence for whether thin constructs are real standalone dimensions or should be folded into other axes. |
| S5 | All dimensions jointly, plus sensitivity variants | The reported global map: posterior loadings, correlations, adjudication, and scores. | The final M1 object: empirical atlas, factor correlations, patient coordinates, and verdicts. |

S1-S4 are checkpoints. Their results can be used diagnostically, but the final
reported measurement map should come from S5.

The important discipline is that a stage can fail usefully. If S2 fails but S1
certified, the failure tells us something specific: freeing cross-loadings
created an identification or geometry problem. If S3 fails, the difficulty is
probably the added mixed likelihoods or the way explicit patient latents enter.
This is why staged continuation is an engineering strategy and a scientific
quality-control strategy at the same time.

Analogy: building S1-S5 is like constructing a bridge in phases. S1 checks that
the main pillars stand. S2 adds flexible joints. S3 adds heavy traffic. S4 tests
a narrow side span. S5 is the full bridge under the intended load.

## 3. The basic latent-factor model

Let:

```math
i = 1,\ldots,N \quad \text{index patients}
```
```math
j = 1,\ldots,J \quad \text{index indicators}
```
```math
f = 1,\ldots,F \quad \text{index latent factors}
```
For S1:

```math
F = 5
```
```math
z_i =
\begin{bmatrix}
G_i \\
\text{cognition}_i \\
\text{metabolic}_i \\
\text{inflammatory}_i \\
\text{sleep}_i
\end{bmatrix}
```
`z_i` is the unobserved factor vector for patient `i`. In S1, the factors are
independent:

```math
z_i \sim \mathcal{N}(0, I_F)
```
The observed indicators are modeled as:

```math
x_i = \Lambda z_i + \epsilon_i
```
```math
\epsilon_i \sim \mathcal{N}(0, \Psi)
```
where:

```math
\Lambda \in \mathbb{R}^{J \times F}
```
is the loading matrix, and:

```math
\Psi = \operatorname{diag}(\sigma_1^2,\ldots,\sigma_J^2)
```
is the diagonal residual variance matrix.

In words:

```text
observed score = shared latent dimensions * loadings + item-specific noise
```

This equation is the measurement-map idea in one line. The patient does not
"have" a FAST score because FAST is a dimension. FAST is an instrument. The
model asks how much FAST reflects the general burden factor, how much it reflects
other dimensions if allowed, and how much residual item-specific variation is
left after the shared dimensions are accounted for.

The loading `lambda_jk` says how strongly indicator `j` measures factor `k`.
For example, in S1:

- FAST and EGF anchor `G`, the general functional-burden factor.
- Neuropsychology indicators anchor cognition.
- BMI, waist, glucose, lipids anchor metabolic.
- CRP, WBC, neutrophils, platelets anchor inflammatory.
- PSQI indicators anchor sleep.

Analogy: a thermometer reading is not body temperature itself; it is an observed
instrument response. A factor is like the underlying temperature, while the
loading says how sensitive that instrument is to it.

The method gives more than a naming exercise. If cognition and sleep partly load
on `G`, then part of their variance is shared with overall burden. If metabolic
and inflammatory markers do not load much on `G`, then biology is not merely a
severity proxy. That distinction is exactly what later stratification needs: two
patients can be equally impaired on `G` but biologically different.

## 3.1 Why soft priors are used

The prior loading matrix encodes clinical theory without turning theory into a
fixed score. A primary indicator has a prior that expects a positive loading on
its home factor; a plausible cross-loading has a prior centered near zero but
wide enough to move if the data support it; an unlikely cross-loading is shrunk
tightly toward zero.

In schematic LaTeX form:

```math
\lambda_{jf} \mid \text{primary}
\sim
\mathcal{N}^{+}(0.70,0.25^2)
```
```math
\lambda_{jf} \mid \text{plausible cross}
\sim
\mathcal{N}(0,0.25^2)
```
```math
\lambda_{jf} \mid \text{unlikely}
\sim
\mathcal{N}(0,0.05^2)
```
Why this helps:

1. It anchors the model enough to avoid arbitrary rotations and label switching.
2. It lets the data disagree with theory where the covariance supports it.
3. It gives a formal prior-to-posterior comparison: the atlas can show what
   clinical theory predicted and what FACE actually supported.

## 4. No imputation: observed-cell likelihood

Each patient has a different subset of observed variables. Let:

```math
O_i \subseteq \{1,\ldots,J\}
```
be the set of indicators observed for patient \(i\).

The likelihood uses only those observed cells:

```math
\log p(X_{\mathrm{obs}} \mid \theta)
= \sum_{i=1}^{N} \log p(x_{i,O_i} \mid \theta)
```
Missing cells contribute no likelihood term. They are not filled with means,
KNN values, MICE values, or posterior guesses.

This matters because FACE missingness is structured: some instruments are absent
by cohort, site, design, skip logic, or clinical routing. If we filled missing
values, we would create artificial covariance. That would be especially harmful
for a study whose goal is to estimate covariance structure.

Analogy: if a patient did not take a lab test, we do not pretend their lab value
was average. We simply say that this test gives us no information about this
patient's latent position.

What this gives scientifically: the fitted map is not learned from a selected
"most complete" subgroup. Every patient can contribute whatever information they
actually have. That protects M1 from a major selection-bias trap: if the map were
fit only on highly complete cases, it would describe the patients and cohorts
with the best measurement coverage, not the full FACE baseline population.

## 5. The Bayesian posterior and MCMC

The unknown quantities are collectively called \(\theta\). In the continuous
Woodbury stages, the main unknowns are the loadings and residual variances:

```math
\theta = \{\Lambda,\sigma\}
```
In S2A this means:

```math
\theta_{\mathrm{S2A}}
=
\{
\lambda_{\mathrm{G\ anchor}},
\lambda_{\mathrm{G\ cross}},
\lambda_{\mathrm{home}},
\lambda_{\mathrm{ESEM\ cross}},
\sigma
\}
```
In S2B, the specific-factor correlation matrix is added:

```math
\theta_{\mathrm{S2B}}
=
\theta_{\mathrm{S2A}}
\cup
\{\Phi_{\mathrm{specific}}\}
```
In an explicit-latent model, the patient factor scores are also sampled:

```math
\theta_{\mathrm{explicit}}
= \{\Lambda, \sigma, z_1,\ldots,z_N\}
```
The marginalized Woodbury model keeps the patient factors in the generative
model, but integrates them out during fitting. Therefore the MCMC state is much
smaller: it samples the map parameters, not every patient's latent position.

### 5.1 Likelihood, priors, and posterior density

Bayes' rule gives the posterior:

```math
p(\theta \mid X_{\mathrm{obs}})
\propto
p(X_{\mathrm{obs}} \mid \theta)\,p(\theta)
```
The first term is the likelihood: how well the model explains the observed
data. The second term is the prior: the soft loading ontology, residual priors,
and identification constraints.

For the continuous marginalized block, patient \(i\)'s likelihood is:

```math
x_{i,O_i}
\mid
\theta
\sim
\mathcal{N}
\left(
0,
\Sigma_{O_i,O_i}
\right)
```
where:

```math
\Sigma
=
\Lambda\Phi\Lambda^\top+\Psi
```
and:

```math
\Psi
=
\operatorname{diag}(\sigma_1^2,\ldots,\sigma_J^2)
```
For S2A:

```math
\Phi = I
```
so:

```math
\Sigma
=
\Lambda\Lambda^\top+\Psi
```
The per-patient log likelihood is:

```math
\ell_i(\theta)
=
-\frac{1}{2}
\left[
k_i\log(2\pi)
+
\log|\Sigma_{O_i,O_i}|
+
x_{i,O_i}^{\top}
\Sigma_{O_i,O_i}^{-1}
x_{i,O_i}
\right]
```
where:

```math
k_i = |O_i|
```
The full observed-cell log likelihood is:

```math
\log p(X_{\mathrm{obs}}\mid\theta)
=
\sum_{i=1}^{N}\ell_i(\theta)
```
The priors add scientific structure and identification. In the current S1/S2
continuous engine:

```math
\lambda_{jf}\mid\mathrm{primary\ or\ G\ anchor}
\sim
\mathcal{N}^{+}(0.60,0.30^2)
```
```math
\lambda_{jf}\mid\mathrm{G\ bifactor/window\ cross}
\sim
\mathcal{N}(0,0.25^2)
```
```math
\lambda_{jf}\mid\mathrm{non\mbox{-}G\ plausible\ ESEM\ cross}
\sim
\mathcal{N}(0,(0.25\times0.60)^2)
```
```math
\sigma_j
=
0.05+\operatorname{HalfNormal}(1)
```
The \(0.05\) residual-scale floor is a guard against residual variance collapse.
The non-G ESEM cross-loading scale is tightened in S2 so that cross-loadings can
move when the data support them, but do not smear weakly across every possible
dimension.

The log posterior density used by NUTS is:

```math
\log p(\theta\mid X_{\mathrm{obs}})
=
\sum_i \ell_i(\theta)
+
\log p(\Lambda)
+
\log p(\sigma)
+
\log p(\Phi)
+
C
```
For S2A, the \(\log p(\Phi)\) term is absent because \(\Phi=I\) is fixed. For
S2B, \(\Phi_{\mathrm{specific}}\) receives an LKJ prior.

Important: the constant \(C\) is not needed for MCMC. NUTS only needs to compare
posterior densities and compute gradients, so constants that do not depend on
\(\theta\) can be ignored.

This posterior is what M1 needs. A single maximum-likelihood or maximum-posterior
solution would give one map, but not enough information about whether a loading
is stable, whether two dimensions are weakly identified, or whether a patient
score is reliable. The Bayesian posterior gives a distribution over maps, and
that distribution is what later adjudication and scoring should use.

### MCMC is not ordinary optimization

It is tempting to say that MCMC "optimizes" the model, but that is not quite
right.

Classical optimization searches for one best point:

```math
\hat{\theta}
= \arg\max_{\theta}
\log p(\theta \mid X_{\mathrm{obs}})
```
MCMC instead samples many plausible points from the posterior:

```math
\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(S)}
\sim
p(\theta \mid X_{\mathrm{obs}})
```
From those samples we estimate:

- posterior means,
- credible intervals,
- uncertainty in loadings,
- convergence diagnostics such as R-hat and ESS.

For example, a loading is not just reported as:

```math
\hat{\lambda}_{jf}=0.42
```
It can be summarized as a posterior distribution:

```math
\lambda_{jf}^{(1)},\ldots,\lambda_{jf}^{(S)}
\quad\Rightarrow\quad
\mathbb{E}[\lambda_{jf}\mid X_{\mathrm{obs}}],
\quad
\operatorname{HDI}_{95\%}(\lambda_{jf})
```
This matters for adjudication. A factor with several strong loadings whose
credible intervals stay away from zero is very different from a factor whose
average loading looks acceptable but whose uncertainty is wide.

The code uses PyMC/NumPyro NUTS. NUTS is a Hamiltonian Monte Carlo method. During
tuning, it adapts technical settings such as step size and mass matrix so that
posterior exploration is efficient. After tuning, it draws posterior samples.

The code does perform numerical work that looks optimization-like during tuning:
NUTS adapts step sizes and a mass matrix, and gradient evaluations guide the
trajectory through posterior space. But the output is not "the optimized
parameters." The output is a sample-based approximation to the posterior.

Analogy: optimization asks, "Where is the highest mountain peak?" MCMC asks,
"What does the whole mountain range look like, and how much terrain lies in each
region?"

### 5.2 What NUTS does

The code uses PyMC with the NumPyro implementation of **NUTS**, the No-U-Turn
Sampler. NUTS is a Hamiltonian Monte Carlo method.

The key idea is that NUTS does not wander randomly like a simple random-walk
sampler. It uses the gradient of the log posterior density:

```math
\nabla_{\theta}\log p(\theta\mid X_{\mathrm{obs}})
```
to propose long, informed moves through parameter space.

NUTS introduces an auxiliary momentum variable \(r\) and simulates movement
through an energy landscape:

```math
H(\theta,r)
=
-\log p(\theta\mid X_{\mathrm{obs}})
+
\frac{1}{2}r^\top M^{-1}r
```
where \(M\) is the mass matrix. The first term is the posterior energy; the
second term is kinetic energy. A good trajectory moves through posterior space
while approximately preserving \(H\).

During **warmup** or **tuning**, NUTS adapts:

- the step size \(\epsilon\), which controls how far each numerical leapfrog
  step moves;
- the mass matrix \(M\), which rescales parameters so the sampler can move more
  evenly across loadings, residuals, and correlations;
- the trajectory length, by growing a path until it would start doubling back,
  hence "No-U-Turn."

After warmup, the sampler keeps posterior draws:

```math
\theta^{(1)},\ldots,\theta^{(S)}
```
Those draws are what we summarize as loadings, credible intervals, diagnostics,
and later score uncertainty.

What NUTS gives us:

1. It explores a high-dimensional posterior much more efficiently than a random
   walk.
2. It reports when the posterior geometry is difficult through divergences and
   energy diagnostics.
3. It gives uncertainty, not only point estimates.

What NUTS does **not** give automatically:

1. It does not prove the model is scientifically correct.
2. It does not remove the need for priors and identification constraints.
3. It does not make a short run trustworthy. We still need R-hat, ESS,
   divergence, and Heywood checks.

### 5.3 Why Woodbury helps NUTS

NUTS needs many evaluations of:

```math
\log p(\theta\mid X_{\mathrm{obs}})
```
and its gradient. Each evaluation requires the observed-data Gaussian
likelihood. Without Woodbury, patient \(i\)'s likelihood needs determinant and
inverse operations on:

```math
\Sigma_{O_i,O_i}
```
which lives in observed-indicator space. For S2A this can be as large as
approximately \(71\times71\) for a highly observed patient.

Woodbury rewrites the hard parts using:

```math
A_i
=
I_F+\Lambda_{O_i}^{\top}W_i\Lambda_{O_i}
```
where \(A_i\) is only \(F\times F\). For S2A:

```math
F = 5
```
So the expensive inverse/determinant work moves from item space to factor
space:

```math
71\times71
\quad\longrightarrow\quad
5\times5
```
This does not change the posterior target. It makes each likelihood and
gradient evaluation cheaper. Since NUTS may evaluate the log posterior many
times for every retained draw, this saving compounds across chains, warmup, and
sampling.

For S2B, where the specific factors can be correlated, the model uses:

```math
\Sigma
=
\Lambda\Phi\Lambda^\top+\Psi
```
The same low-rank idea still applies by writing:

```math
\Phi = L_{\Phi}L_{\Phi}^{\top}
```
and defining:

```math
\Lambda_{\mathrm{eff}}
=
\Lambda L_{\Phi}
```
Then:

```math
\Lambda\Phi\Lambda^\top
=
\Lambda L_{\Phi}L_{\Phi}^{\top}\Lambda^\top
=
\Lambda_{\mathrm{eff}}\Lambda_{\mathrm{eff}}^\top
```
So the S1/S2A Woodbury machinery can be reused with
\(\Lambda_{\mathrm{eff}}\). This is why the marginalized engine can support
correlated specifics without returning to patient-level latent sampling.

### 5.4 Diagnostics and certification metrics

The diagnostics answer a different question from the likelihood. The likelihood
asks:

> How well does this parameter value explain the observed data?

The diagnostics ask:

> Did the sampler explore the posterior reliably enough for us to trust the
> summaries?

#### R-hat

R-hat compares variation **within** chains to variation **between** chains. If
all chains explore the same posterior, they should have similar means and
variances.

In simplified form:

```math
\widehat{R}
\approx
\sqrt{
\frac{\widehat{\operatorname{Var}}^{+}(\theta)}
{W}
}
```
where \(W\) is within-chain variance and
\(\widehat{\operatorname{Var}}^{+}\) combines within-chain and between-chain
variation.

Interpretation:

| R-hat | Meaning |
|---|---|
| \(1.00\) to \(1.01\) | Good / certification range |
| \(1.01\) to \(1.02\) | Borderline; often acceptable for smoke diagnosis, not final certification |
| \(>1.05\) | Chains are probably not mixed |
| \(>1.10\) | Strong warning; posterior summaries are not trustworthy |

For M1 certification we use:

```math
\max \widehat{R} \le 1.01
```

#### ESS

ESS means **effective sample size**. MCMC draws are autocorrelated: consecutive
draws are not independent. ESS estimates how many independent samples the
correlated draws are worth.

For a chain with \(S\) draws and autocorrelations \(\rho_t\), the idea is:

```math
\operatorname{ESS}
\approx
\frac{S}
{1+2\sum_{t=1}^{\infty}\rho_t}
```
If autocorrelation is high, ESS is much smaller than the raw number of draws.

Interpretation:

| ESS | Meaning |
|---|---|
| Very low, e.g. \(<100\) | Posterior summaries are noisy |
| \(100\) to \(400\) | Useful for smoke diagnosis, weak for certification |
| \(\ge 400\) | M1 certification gate |
| Much higher | Better precision for posterior means/intervals |

For M1 certification we use:

```math
\min \operatorname{ESS} \ge 400
```

#### Divergences

A divergence means the Hamiltonian simulation could not accurately follow the
posterior geometry. This often happens when the posterior has narrow funnels,
sharp curvature, or badly scaled parameters.

Divergences are more serious than low ESS. Low ESS often means "run longer."
Divergences can mean:

- the posterior geometry is pathological;
- the parameterization is poor;
- the sampler may be missing important regions;
- posterior summaries may be biased.

For M1 certification we require:

```math
\operatorname{divergences}=0
```
If divergences appear, the first response is usually to raise
`target_accept`, for example from `0.95` to `0.99`. If divergences remain, the
model likely needs reparameterization or stronger identification.

#### Heywood checks

A Heywood case is a factor-analysis pathology where the model explains too much
variance with a loading or collapses residual variance. In classical factor
analysis this can appear as negative residual variance. In this Bayesian code,
we guard against residual collapse using:

```math
\sigma_j = 0.05+\operatorname{HalfNormal}(1)
```
and we also flag implausibly large loadings:

```math
|\lambda_{jf}| > 2.5
```
For M1 certification we require:

```math
\operatorname{Heywood}=\mathrm{False}
```

#### How to read a run

A healthy smoke run usually looks like:

```text
divergences = 0
Heywood = False
R-hat improving toward 1.00
ESS increasing as draws/tune increase
```
This means the model geometry is probably acceptable and the next step is a
longer run or full cohort run.

An unhealthy run looks like:

```text
divergences > 0
or Heywood = True
or R-hat stays high despite long chains
```
That means we should not simply keep running forever. We should diagnose which
parameters are causing the problem, then adjust target acceptance, priors,
parameterization, or the stage split.

## 6. Why the explicit S1 model is slow

The explicit S1 model is:

```math
z_i \sim \mathcal{N}(0, I_F)
```
```math
x_i \mid z_i,\Lambda,\Psi
\sim
\mathcal{N}(\Lambda z_i,\Psi)
```
For `N = 9,013` and `F = 5`, direct sampling introduces:

```math
9{,}013 \times 5 = 45{,}065
```
patient-level latent variables, before counting loadings and residual
variances.

That is expensive for two reasons:

1. There are many more parameters for NUTS to move through.
2. Hierarchical latent-variable models can create difficult posterior geometry,
   often called a funnel.

The problem is not just arithmetic. The sampler has to explore a high-dimensional
space where patient scores, loadings, and residual variances interact. This is
why the same statistical model can be much slower or less stable under a bad
parameterization.

What this tells us: acceleration is not only about faster hardware. It is about
choosing a mathematically equivalent parameterization that removes unnecessary
sampling burden. If a latent variable can be integrated out exactly, sampling it
directly is often wasted work for the purpose of estimating loadings.

## 7. The S1 marginalized model

S1 can avoid sampling patient factor scores because all S1 indicators are treated
as Gaussian after orientation, log transforms where needed, and z-scoring.

The explicit model is:

```math
z_i \sim \mathcal{N}(0,I)
```
```math
x_i \mid z_i,\Lambda,\Psi
\sim
\mathcal{N}(\Lambda z_i,\Psi)
```
We can integrate out `z_i` analytically:

```math
p(x_i \mid \Lambda,\Psi)
=
\int
p(x_i \mid z_i,\Lambda,\Psi)\,p(z_i)\,dz_i
```
Because both pieces are Gaussian, the integral is closed form:

```math
x_i \sim
\mathcal{N}\left(0,\Lambda\Lambda^\top+\Psi\right)
```
More generally, if factors have covariance `Phi`:

```math
x_i \sim
\mathcal{N}\left(0,\Lambda\Phi\Lambda^\top+\Psi\right)
```
In S1, `Phi = I`, so:

```math
\Sigma = \Lambda\Lambda^\top+\Psi
```
The current S1 marginalized code works on oriented, log-transformed where
needed, z-scored indicators and uses a zero-mean marginal likelihood. The
formal explicit reference model includes item intercepts; after centering, those
intercepts are not the load-bearing part of the S1 fit. A more general
marginalized implementation can include a mean vector:

```math
x_i \sim
\mathcal{N}\left(\mu_i,\Lambda\Phi\Lambda^\top+\Psi\right)
```
where `mu_i` may contain item intercepts and covariate effects. The Woodbury
acceleration concerns the covariance part, so it is compatible with either
zero-mean or mean-adjusted Gaussian likelihoods.

For a patient with observed indicators `O_i`, the likelihood is:

```math
x_{i,O_i}
\sim
\mathcal{N}\left(0,\Sigma_{O_i,O_i}\right)
```
This is the same observed-data Gaussian likelihood used by FIML. The Bayesian
version adds priors over `Lambda` and `Psi`; the likelihood geometry is the same.

Analogy: suppose a patient's latent factor score is a hidden wind that pushes
several weather vanes. The explicit model estimates the wind for every patient.
The marginalized model asks a different question: "If winds vary normally in the
population, what covariance pattern should we see among the weather vanes?" For
estimating the vanes' sensitivities, we do not need to name the exact wind for
every person during fitting.

## 8. Marginalization derivation

Start with:

```math
z \sim \mathcal{N}(0,I)
```
```math
\epsilon \sim \mathcal{N}(0,\Psi)
```
```math
x = \Lambda z + \epsilon
```
The mean is:

```math
\mathbb{E}[x]
=
\mathbb{E}[\Lambda z+\epsilon]
=
\Lambda\mathbb{E}[z]+\mathbb{E}[\epsilon]
=
0
```
The covariance is:

```math
\operatorname{Var}(x)
=
\operatorname{Var}(\Lambda z+\epsilon)
=
\Lambda\operatorname{Var}(z)\Lambda^\top
+ \operatorname{Var}(\epsilon)
```
Because \(\operatorname{Var}(z)=I\) and
\(\operatorname{Var}(\epsilon)=\Psi\):

```math
\operatorname{Var}(x)
=
\Lambda I\Lambda^\top+\Psi
=
\Lambda\Lambda^\top+\Psi
```
Therefore:

```math
x \sim
\mathcal{N}\left(0,\Lambda\Lambda^\top+\Psi\right)
```
For observed cells only:

```math
x_O \sim
\mathcal{N}\left(0,\Lambda_O\Lambda_O^\top+\Psi_O\right)
```
where `Lambda_O` is the subset of loading rows for the observed indicators and
`Psi_O` is the corresponding residual diagonal.

The log likelihood for patient `i` is:

```math
\ell_i
=
-\frac{1}{2}
\left[
k_i\log(2\pi)
+ \log |C_i|
+ x_{i,O_i}^{\top}C_i^{-1}x_{i,O_i}
\right]
```
with:

```math
C_i
=
\Psi_{O_i}
+ \Lambda_{O_i}\Lambda_{O_i}^{\top}
```
and:

```math
k_i = |O_i|
```
Naively, computing `log |C_i|` and `C_i^-1` for every patient can be expensive,
because `C_i` lives in observed-indicator space. In S1, that can be up to 68 x
68 per patient.

The Woodbury trick moves the hard part into factor space, which is only 5 x 5.

The key intuition is this: the observed indicators may be numerous, but their
shared covariance is generated by only a few latent factors. The residual
variance \(\Psi\) is diagonal, so it is cheap. The difficult part,
\(\Lambda\Lambda^\top\), is low-rank because it is built from only \(F\)
columns of \(\Lambda\). Woodbury exploits exactly that structure.

## 9. The Woodbury identity and determinant lemma

The covariance has this form:

```math
C = \Psi+\Lambda\Lambda^\top
```
\(\Psi\) is diagonal, so it is cheap to invert.
\(\Lambda\Lambda^\top\) is low-rank because there are only \(F\) factors.

The Woodbury identity says:

```math
(\Psi+\Lambda\Lambda^\top)^{-1}
=
\Psi^{-1}
-
\Psi^{-1}\Lambda
\left(I+\Lambda^\top\Psi^{-1}\Lambda\right)^{-1}
\Lambda^\top\Psi^{-1}
```
The matrix determinant lemma says:

```math
|\Psi+\Lambda\Lambda^\top|
=
|\Psi|\,
\left|I+\Lambda^\top\Psi^{-1}\Lambda\right|
```
Define:

```math
W = \Psi^{-1}
```
```math
A = I+\Lambda^\top W\Lambda
```
```math
b = \Lambda^\top W x
```
Interpretation of these objects:

- \(W\) says how much precision each observed indicator contributes.
- \(A\) is the posterior precision of the latent factor vector after seeing the
  observed indicators, up to the prior identity precision.
- \(b\) is the information that the observed indicator values provide about the
  latent factors.

Then:

```math
\log |C|
=
\log |\Psi|+\log |A|
```
```math
x^\top C^{-1}x
=
x^\top W x
-
b^\top A^{-1}b
```
So the patient log likelihood becomes:

```math
\ell_i
=
-\frac{1}{2}
\left[
k_i\log(2\pi)
+ \log |\Psi_i|
+ \log |A_i|
+ x_i^\top W_i x_i
- b_i^\top A_i^{-1}b_i
\right]
```
The expensive inverse/determinant is now on:

```math
A_i
=
I_F+\Lambda^\top W_i\Lambda
```
which is `F x F`. In S1, `F = 5`.

That is the entire acceleration: same Gaussian marginal likelihood, cheaper
linear algebra, much smaller MCMC state.

## 10. How the code implements S1

The S1 code path is:

```text
scripts/01_build_data.py
  -> data/processed/baseline_v0.parquet

scripts/04_fit.py --stage 1            # (the entrypoint for every stage)
  -> face.models.bayesian.continuous_core.prepare()
  -> face.models.bayesian.continuous_core.build_marginalized()
  -> pm.sample(..., nuts_sampler="numpyro", nuts_sampler_kwargs={"max_tree_depth": 8})
  -> reports/04_stage1_report.md
  -> reports/04_stage1_loadings.csv
```

`scripts/04_fit.py` is the single entrypoint for all stages: `--stage {1,2,3}`, plus `--mixed`
(the S3b/S5 non-Gaussian block), `--g-correlated` (the S5 sensitivity), `--subsample`, `--seed`,
`--draws/--tune/--chains`, `--label`. The marginalized engine lives in
`src/face/models/bayesian/continuous_core.py`.

### `prepare()`

`prepare()` reads:

```text
data/processed/baseline_v0.parquet
configs/prior_loading_matrix_v3.csv
```

It selects the S1 continuous items whose home factors are:

```text
overall_severity, cognition, metabolic, inflammatory, sleep
```

Then it:

1. Applies log transforms for lognormal indicators.
2. Orients each item so higher means more burden/dysfunction.
3. Z-scores continuous indicators.
4. Preserves missing values as `NaN`.
5. Resolves loading priors from the prior matrix.

The result is a `CorePrep` dataclass:

```text
M:           N x J matrix, with NaN for missing
items, home: item names and each item's home factor ("" for the window items)
factor_cols: column order of the loading matrix, [G, *specifics]
pos_cells:   [(j, c, mu, sd)] cells with a TruncatedNormal>0 prior (primary / G-anchor)
sgn_cells:   [(j, c, mu, sd)] cells with a signed Normal prior (bifactor-G / window / specific cross)
kind:        {(j, c): "g_anchor|primary|bifactor_G|window|cross"} — the role of each loading cell
correlated:  whether Phi is estimated (LKJ) vs fixed to I
```

(An earlier version exposed `g_anchor_items`/`spec_items`/`cellG`/`cellGx`/`cellHome`; the engine was
refactored to the generic `pos_cells`/`sgn_cells`/`kind` representation above, which the same builder
reuses across S1–S5.)

### `build_model()`: explicit latent reference model

`build_model()` constructs the direct model:

```math
G_i \sim \mathcal{N}(0,1)
```
```math
D_{ik} \sim \mathcal{N}(0,1)
```
```math
x_{ij}
\sim
\mathcal{N}
\left(
\alpha_j
+ \lambda_{jG}G_i
+ \sum_k \lambda_{jk}D_{ik},
\sigma_j^2
\right)
```
This model is conceptually straightforward and is useful as a reference, but it
samples the patient latents.

### `build_marginalized()`: the certified S1 engine

`build_marginalized()` integrates `G` and `D` out:

```math
x_{i,O_i}
\sim
\operatorname{MVN}
\left(0,\Lambda_{O_i}\Lambda_{O_i}^{\top}
+ \operatorname{diag}(\psi_{O_i})\right)
```
The code variables map to the math like this:

| Code variable | Math meaning |
|---|---|
| `M` | `X`, the oriented/z-scored patient x indicator matrix |
| `mask` | observed-cell indicator; 1 if observed, 0 if missing |
| `x` | `M` with missing cells set to 0 only for arithmetic masking |
| `kobs` | `k_i`, number of observed cells per patient |
| `lamG`, `lamS` | general and specific loadings |
| `Lam` | full loading matrix `Lambda` |
| `sigma` | residual standard deviations |
| `psi` | residual variances `sigma^2` |
| `W` | masked precision, `mask / psi` |
| `A` | \(I+\Lambda^\top W\Lambda\) |
| `b` | \(\Lambda^\top W x\) |
| `logdetA` | \(\log |A|\) |
| `quadA` | \(b^\top A^{-1}b\) |
| `term1` | \(x^\top W x\) |
| `logdetPsi` | \(\log |\Psi_{\mathrm{observed}}|\) |
| `ll` | per-patient marginalized log likelihood |

Two implementation details are easy to misread:

1. `x = np.nan_to_num(M, nan=0.0)` is **not imputation**. The missing cells are
   also multiplied by `mask = 0`, so they contribute zero precision and zero
   likelihood. The zero is only a computational placeholder.
2. `W = mask / psi[None, :]` is the observed-cell precision matrix. If an
   indicator is missing for patient \(i\), then \(W_{ij}=0\). If it is observed,
   then \(W_{ij}=1/\psi_j\).

The important code block is:

```python
W = mask / psi[None, :]
P = W[:, :, None] * Lam[None, :, :]
A = eye(F)[None, :, :] + (P.transpose(0, 2, 1) @ Lam)
b = (W * x) @ Lam

logdetA = 2 * log(diag(cholesky(A))).sum(-1)
quadA = solve_cholesky(A, b)' solve_cholesky(A, b)
term1 = (W * x**2).sum(1)
logdetPsi = (mask * log(psi)[None, :]).sum(1)

ll = -0.5 * (kobs*log(2*pi) + logdetPsi + logdetA + term1 - quadA)
```

> **As-built (grouped-GEMM, the production form).** The code above is the readable *per-patient* version.
> The shipped engine computes the **same** `ll` two ways faster: (i) `A` is assembled as one BLAS
> matrix-multiply `A = I + mask @ Q`, where `Q[j] = (Lt_j Lt_jᵀ)/ψ_j` flattened — no `[N,J,F]` intermediate;
> and (ii) because `A_i` depends on the row only through its **observed pattern**, the (expensive) Cholesky
> is done **once per unique pattern** (≈ half as many as patients) and gathered back to patients. The factor
> correlation is folded in via `Lt = Λ · chol(Φ)` (so `Λ Φ Λᵀ + Ψ = Lt Ltᵀ + Ψ`). Result: identical
> log-likelihood (verified to 1e-4), ≈ 2.75× faster. See **Part C**.

The model adds:

```python
pm.Potential("obs_ll", ll.sum())
```

That line is the likelihood contribution. PyMC/NUTS still samples the posterior,
but the posterior now contains only the loading/residual parameters, not
patient-level S1 factor scores.

From the MCMC perspective, every proposed draw of \(\Lambda\) and \(\psi\)
requires evaluating this likelihood. Woodbury makes each evaluation much
cheaper. Since NUTS evaluates the log probability and its gradient many times
per posterior sample, reducing the cost of one likelihood evaluation compounds
across the whole run.

## 11. Is Woodbury equivalent to MCMC?

No. They are different kinds of things.

```text
MCMC      = an algorithm for sampling from the posterior distribution.
Woodbury  = a linear-algebra identity for computing a Gaussian likelihood faster.
```

The S1 marginalized model still uses MCMC. More precisely:

```math
\text{PyMC/NumPyro NUTS samples }
p(\Lambda,\Psi\mid X_{\mathrm{obs}})
```
Woodbury only makes each likelihood evaluation cheaper and better conditioned.

The useful equivalence is not "MCMC equals Woodbury." The useful equivalence is:

```math
z_i \sim \mathcal{N}(0,I),
\qquad
x_i \mid z_i,\Lambda,\Psi
\sim
\mathcal{N}(\Lambda z_i,\Psi)
```
has the same marginal likelihood for \(\Lambda\) and \(\Psi\) as:

```math
x_i
\sim
\mathcal{N}\left(0,\Lambda\Lambda^\top+\Psi\right)
```
So, under the same priors and same S1 assumptions, the explicit and marginalized
parameterizations target the same posterior for the structural parameters
`Lambda` and `Psi`.

In the current code, "same S1 assumptions" means the centered/z-scored,
zero-mean continuous block. The explicit reference path has an `alpha` parameter;
the marginalized certified path fixes the centered mean at zero. That difference
does not change the conceptual marginalization argument, but it is worth keeping
in mind when comparing exact code paths line by line.

What changes is the route:

| Question | Explicit latent model | Marginalized Woodbury model |
|---|---|---|
| Are patient factors part of the model? | Yes | Yes, integrated out |
| Are patient factors sampled by MCMC? | Yes | No |
| Are loadings sampled by MCMC? | Yes | Yes |
| Does it estimate uncertainty? | Yes | Yes |
| Is the likelihood mathematically the same for Gaussian S1? | Yes | Yes |
| Is it faster? | Usually much faster | Yes, because fewer sampled variables and cheaper linear algebra |

What this gives in practice: if the explicit and marginalized versions produce
the same loading pattern, that is strong reassurance that the S1 result is not
an artifact of one computational representation. The marginalized version is the
efficient production route; the explicit version is a useful conceptual and
sanity-check route.

## 12. Where patient scores go in a marginalized model

Because the marginalized S1 fit does not sample `z_i` directly, it does not
automatically produce patient factor scores during MCMC.

But scores can be computed after fitting. For a fixed posterior draw of
`Lambda` and `Psi`, the conditional distribution of `z_i` given observed
continuous data is Gaussian:

```math
z_i
\mid
x_{i,O_i},\Lambda,\Psi
\sim
\mathcal{N}(m_i,V_i)
```
For S1 with `Phi = I`:

```math
A_i
=
I+\Lambda_{O_i}^{\top}W_i\Lambda_{O_i}
```
```math
b_i
=
\Lambda_{O_i}^{\top}W_i x_{i,O_i}
```
```math
V_i = A_i^{-1}
```
```math
m_i = A_i^{-1}b_i
```
So the same `A_i` and `b_i` used in the Woodbury likelihood also give the
posterior mean and uncertainty of patient scores.

This is why fitting and scoring can be separated:

1. Fit the map once to estimate `Lambda` and `Psi`.
2. Project each patient's observed cells onto that map to get scores and score
   uncertainty.

Analogy: first calibrate the coordinate grid; then place each patient on it.

## 13. Why S1 certified on the Mac

S1 certified because it combined three good properties:

1. **Gaussian continuous block.** This allows exact marginalization.
2. **Low-rank factor structure.** There are 5 factors, much fewer than 68
   indicators.
3. **No patient-level latent sampling.** NUTS does not move through 45,065
   patient-score parameters.

The S1 result used:

```text
N = 9,013
J = 68 continuous indicators
observed cells = 415,531
max R-hat = 1.010
min ESS = 1,939
divergences = 0
```

That is the central computational lesson: the hard scientific invariant
full-sample/no-imputation was preserved, but the parameterization made it
tractable.

## 14. Why later stages are harder

S2 is still mostly a Gaussian problem if it only frees continuous cross-loadings.
That means it should be possible to extend the S1 Woodbury approach to S2.

S3-S5 become harder because they add non-Gaussian indicators:

```text
binary         -> Bernoulli-logit
ordinal        -> ordered logistic
count          -> negative binomial
skewed biology -> lognormal / heavy-tailed variants
```

For binary, ordinal, and count likelihoods, the integral:

```math
\int p(y_i \mid z_i)\,p(z_i)\,dz_i
```
usually does not have the same simple closed form. That means some latent
variables may have to re-enter the sampler, or the model needs a more advanced
partial-collapse strategy.

## 14.1 What the completed M1 method gives

When S5 is certified, the method gives a measured coordinate system rather than
a hand-made score sheet.

### Loading matrix

The posterior loading matrix answers:

> Which indicators define each dimension, and how strongly?

For each indicator \(j\) and factor \(f\), the model estimates:

```math
p(\lambda_{jf}\mid X_{\mathrm{obs}})
```
The posterior mean gives the central loading estimate; the posterior interval
shows uncertainty. This is how the model can say, for example, whether a
depression/anxiety composite behaves as a sleep window, a suicidality window, a
burden window, or some mixture.

### Factor correlation matrix

The factor correlation matrix answers:

> Are dimensions statistically independent, overlapping, or almost redundant?

In the general form:

```math
z_i \sim \mathcal{N}(0,\Phi)
```
\(\Phi\) is the latent factor covariance/correlation matrix. It is not the same
as correlating crude scale totals. It is the model's estimate of how the latent
dimensions relate after accounting for measurement error, missingness, and
cross-loadings.

### Patient coordinates

The patient score layer answers:

> Where is patient \(i\) on each surviving dimension, and how certain are we?

For each patient and factor:

```math
p(z_{if}\mid x_{i,O_i},\widehat{\Lambda},\widehat{\Psi})
```
or, more fully, using posterior draws:

```math
p(z_{if}\mid X_{\mathrm{obs}})
```
The important point is uncertainty. A patient with six observed cognition
indicators should not be treated as equally well-characterized as a patient with
one weak cognition indicator. The map should carry that reliability forward into
stratification and prognosis.

### Adjudication

The adjudication layer answers:

> Which clinical candidate constructs survived contact with the FACE data?

Possible verdicts include confirmed, split, merged, proxy, rejected, and
not-testable. This is a scientific output, not an administrative label. For
example, if immunometabolism splits into metabolic and inflammatory dimensions,
that split is part of what the data taught us. If sensory abnormalities have no
usable indicators, the correct verdict is not-testable, not an invented proxy.

## 15. How proposed optimizations speed up the calculation

The safest acceleration strategy is not one trick. It is a stack of compatible
changes that preserve the statistical target.

A useful way to classify speedups is:

1. **Algebraic speedups.** Same model, same likelihood, fewer operations.
   Woodbury is in this class.
2. **Parameterization speedups.** Same statistical target, easier posterior
   geometry. Marginalizing patient factors in S1 is in this class.
3. **Workflow speedups.** Same final model, faster iteration while developing.
   Smoke runs and cached Parquet are in this class.
4. **Approximate speedups.** Faster but changes the inferential target. ADVI or
   aggressive subsampling are in this class and need explicit labeling.

### 15.1 Extend Woodbury marginalization to S2

S2 adds cross-loadings, but if the indicators are still continuous Gaussian, the
marginal form remains:

```math
\Sigma = \Lambda\Phi\Lambda^\top+\Psi
```
Only `Lambda` becomes less sparse, and `Phi` may become non-identity for
specific factors. The Woodbury identity still applies:

```math
\left(\Psi+\Lambda\Phi\Lambda^\top\right)^{-1}
```
can be computed in factor space using the same low-rank idea.

Expected benefit:

- avoids patient-level latent sampling in S2;
- keeps full `N`;
- tests cross-loadings with the same efficient likelihood class as S1.

What this gives scientifically: S2 can answer "which instruments are genuinely
multidimensional?" without paying the full cost of sampling every patient's
latent coordinates. If S2 remains continuous-Gaussian, the cross-loading test is
still a loading/covariance problem and should exploit the same low-rank
structure as S1.

### 15.2 Use sparse active loading cells

The prior matrix has many possible item x factor cells, but the primary fit
should free only:

```text
primary cells
G bifactor cells
theory-motivated plausible_cross cells
```

The `unlikely_cross` cells can stay fixed or very tightly shrunk in the main
fit. Freeing all of them is a sensitivity analysis, not the default.

Expected benefit:

- fewer loading parameters;
- less rotational ambiguity;
- faster NUTS adaptation;
- better R-hat and ESS.

Why this works: ESEM models can become weakly identified when too many
cross-loadings are free at once. The clinical prior matrix is already telling us
which cross-loadings are plausible. Freeing those first is not hiding from the
data; it is using the planned ontology to keep the main posterior identifiable.

### 15.3 Warm-start each stage from the previous stage

Stages should not start cold. If S1 has already estimated stable loadings and
residual variances, S2 should initialize from them. S3 should initialize from
S2, and so on.

Expected benefit:

- fewer failed warmup trajectories;
- less label/sign switching;
- shorter tuning;
- easier diagnosis when a stage fails.

Mathematically, warm starts do not change the posterior:

```math
p(\theta\mid X_{\mathrm{obs}})
```
is the same posterior regardless of the initial point. But they can make NUTS
reach the typical set faster. The typical set is the high-probability region
where valid posterior samples live. Starting S2 near the certified S1 loadings
is much better than asking NUTS to discover the factor orientation from scratch.

### 15.4 Cache preprocessing and read Parquet

`scripts/01_build_data.py` already persists:

```text
data/processed/baseline_v0.parquet
data/processed/indicator_metadata.parquet
```

Higher-stage fit code should consume these files directly rather than rebuilding
the harmonized dataframe from raw CSV each run.

Expected benefit:

- faster iteration;
- less repeated I/O and harmonization;
- fewer opportunities for accidental data-layer drift.

This is a workflow speedup, not a statistical shortcut. The Parquet files are
the same model-ready baseline table; reading them directly avoids repeating
dictionary parsing, harmonization, sanity-bound application, and skip-logic
decoding on every model run.

### 15.5 Use smoke runs for code development

Development runs should use:

```text
small random subsample
fewer draws
fewer tune steps
2 chains
```

Certification runs should use:

```text
full N
4 chains
diagnostic gates
posterior predictive checks
```

Expected benefit:

- code/debug cycles finish quickly;
- full expensive runs are reserved for scientific certification.

This changes workflow speed, not the final model.

The rule should be: smoke runs are for debugging code and geometry; certified
runs are for scientific claims. A smoke run can tell us that the model compiles,
that dimensions are oriented correctly, or that a proposed S2 parameterization is
hopeless. It cannot certify the measurement map.

### 15.6 Separate exact acceleration from approximation

Some speedups preserve the exact target:

- Gaussian marginalization;
- Woodbury identity;
- JAX vectorization;
- warm starts;
- caching;
- sparse but predeclared loading cells.

Some speedups change or approximate the target:

- ADVI as final inference;
- aggressive subsampling;
- complete-case selection;
- naive imputation;
- dropping difficult indicators.

The second group can be useful for diagnostics or sensitivity checks, but should
not silently become the final reported model.

### 15.7 Consider partial collapse for S3-S5

For mixed likelihood stages, exact full marginalization is not automatic because
non-Gaussian indicators depend on latent factors.

A possible advanced strategy is partial collapse:

```math
p(x_{\mathrm{cont}},y_{\mathrm{explicit}}\mid\theta)
=
p(x_{\mathrm{cont}}\mid\theta)
\int
p(y_{\mathrm{explicit}}\mid z,\theta)
p(z\mid x_{\mathrm{cont}},\theta)
\,dz
```
The continuous block is collapsed with Woodbury, and the explicit block samples
latent scores from their Gaussian conditional distribution given the continuous
data. This can be exact if implemented carefully, but it is more complex than
S1.

Expected benefit:

- continuous data no longer create a funnel;
- explicit latent variables are better informed;
- S3-S5 may mix better than a fully explicit model.

Risk:

- more engineering;
- more complicated validation;
- easy to accidentally double-count the continuous likelihood if the
  factorization is implemented incorrectly.

## 16. What should be optimized first

The highest-value implementation order is:

1. **S2 Woodbury engine.** Generalize the S1 marginalized code to support
   continuous cross-loadings and, if needed, correlated specifics.
2. **Stage warm starts.** Persist and reuse posterior means for loadings and
   residual variances.
3. **Parquet-first loaders.** Make every current-stage fit consume
   `data/processed/*.parquet`.
4. **Unified smoke/certify interface.** Make `--smoke`, `--subsample`,
   `--draws`, `--tune`, `--chains`, and `--init-from-stage` consistent.
5. **Hybrid S3 design.** Decide whether S3 should use explicit latents,
   partial collapse, or a GPU-heavy direct model.

This order gives the biggest speedup while staying closest to the certified S1
logic.

## 17. Common misunderstandings

### "Woodbury replaces MCMC"

No. Woodbury accelerates the likelihood. MCMC still samples the posterior.

### "Marginalized means patient scores disappear"

No. Patient scores are integrated out during fitting. They can be recovered
afterward as posterior conditional scores.

### "Marginalized means approximate"

No, not in S1. For Gaussian latent factors and Gaussian observed indicators, the
marginal likelihood is exact.

### "No imputation means ignoring missingness"

No. Missingness defines each patient's observed set `O_i`. The model evaluates
the likelihood only on observed cells. Missing cells are not ignored as a data
problem; they are handled by the observed-data likelihood.

### "S1 proves the final map"

No. S1 proves that the continuous backbone can certify at full N and gives an
important provisional finding. The final map is S5.

## 18. One-page summary

S1 has two mathematically equivalent views:

Explicit latent view:

```math
z_i \sim \mathcal{N}(0,I),
\qquad
x_i\mid z_i
\sim
\mathcal{N}(\Lambda z_i,\Psi)
```
Marginalized view:

```math
x_i
\sim
\mathcal{N}\left(0,\Lambda\Lambda^\top+\Psi\right)
```
The marginalized view is faster because:

1. it removes tens of thousands of patient-level latent variables from MCMC;
2. Woodbury computes the Gaussian likelihood in factor space (`5 x 5`) rather
   than indicator space (`up to 68 x 68`);
3. JAX/NumPyro vectorizes the batched linear algebra;
4. missing cells are handled by masks, not imputation.

The proposed optimization strategy is to reuse this principle wherever the math
allows it, especially S2, and to reserve explicit latent sampling for the
non-Gaussian pieces that genuinely need it.

---

# Part C — Appendix: what we actually built and found (S1 → S5)

*The body above is the conceptual guide and the planned strategy. This appendix records the **as-built**
outcome. The canonical, numbers-with-context record is [`RESULTS.md`](RESULTS.md); this is the short version.*

## C.1 The reported map (7 dimensions)

| dimension | anchored by | loading |
|---|---|---|
| **G — functional burden** | FAST, EGF, EQ-5D, CGI-S (functioning + global severity only) | FAST 0.90, EGF 0.73 |
| cognition | CVLT, WAIS, fluency, TMT-B | 0.57 |
| metabolic | BMI, waist, BP, glucose, lipids | 0.32 |
| inflammatory | CRP, WBC, neutrophils, platelets | 0.39 |
| sleep | PSQI objective sub-scores | 0.48 |
| developmental-risk | CTQ, age-of-onset, WURS, perinatal, family history | 0.42 |
| suicidality | ISF binary ideation/attempt + count (mixed-likelihood) | 2.7–3.4 (logit) |

Inter-dimension correlations **Φ are weak** (mean |off-diag| ≈ 0.10) — the specifics are **distinct axes**.
Only sensible couplings: metabolic–inflammatory 0.19, suicidality–developmental 0.23, sleep–developmental
0.19. Depression/anxiety (MADRS/QIDS/STAI) are **cross-loading windows onto G** (0.6–0.8), *not* a dimension.

## C.2 What each stage actually did

- **S1** — continuous core, full N = 9,013, **certified** (R-hat 1.010, ESS 1,939, 0 div, ~72 min). Provisional
  read: biology direct-loads ≈ 0 on G.
- **S2** — added Φ (LKJ over specifics, G ⊥) + the MADRS/QIDS/STAI windows. The metabolic↔inflammatory
  *specific* cross-loadings were found **rotationally aliased with Φ** (intractable + non-identified), so they
  are **dropped** — Φ carries that association. Full N, **certified** (R-hat 1.010).
- **S3a** — +developmental-risk via continuous anchors, **certified** (subsample). **S3b** — +suicidality via
  the explicit non-Gaussian block (binary/count ISF), coupled to the marginalized specifics through Φ by the
  conditional decomposition `f_m|f_e`; **0 divergences**, *provisional* (slow-mixing cross-loadings).
- **S4** — anhedonia (thin, BP/DR, one indicator) tested and **rejected** (R-hat 1.54; redundant with G + the
  depression window). A *result*, not a failure.
- **S5** — the global 7-dimension fit, the **reported map** (N=5,000 subsample, provisional R-hat 1.04, 0 div).

## C.3 Two bugs we found and fixed (both with measurement, both regression-guarded)

1. **Φ correctness.** `pm.LKJCorr` returns the **Cholesky factor `L`**, so the correlation is `Φ = L Lᵀ`.
   The code had symmetrized `L`'s lower triangle instead — *indefinite* at the 6-factor scale (→ NaN), and a
   wrong-but-coincidentally-close Φ at 4 factors. (An independent FIML/lavaan fit would have caught this — one
   reason FIML confirmation is worth running.)
2. **Performance.** The per-patient `A_i` Cholesky (§9–§10) is the real cost driver — *every gradient step*
   does N small Cholesky decompositions, which is why even trivial-geometry S1 took ~72 min. Fixed with the
   **grouped-GEMM Woodbury** (Cholesky once per observed-pattern + `A` as one GEMM): **2.75×**, log-likelihood
   identical to the dense form.

## C.4 The compute reality (updates §5.2, §15, §16)

- **Tuning, corrected:** the marginalized posterior has **no funnel** (latents integrated out) → shallow NUTS
  trees (~21 leapfrogs/iter), 0 divergences. So the right move is **lower** `target_accept` (0.85–0.90) **+ a
  `max_tree_depth` cap of 8** (bounds the expensive deep *warmup* trajectories) — *not* "raise to 0.99," which
  shrinks the step size and slows a non-pathological posterior. Measured: cap+0.85 ≈ 2.7× faster at 7 factors.
- **Profiled and rejected** (slower, not faster): a dense mass matrix; forcing parallel chains via
  `XLA_FLAGS` host devices. Default (sequential chains, each multi-threaded) wins on this Mac.
- **The S3+ mixed-likelihood frontier** exceeds the Mac full-N ceiling, so S3–S5 ran on **random subsamples**
  (N = 4,000–5,000, realistic missingness, *not* completeness-selected — §3.6). Full-N + full certification of
  the reported S5 is reserved for the **GPU** (§4.5). Full-N S1/S2 ≈ 1 h each.
- **§16's #5 (the "hybrid S3 design") was resolved:** the *conditional decomposition* — continuous specifics
  marginalized (Woodbury), only G + the non-Gaussian factors explicit, coupled through Φ — is exactly the
  "partial collapse" of §15.7, and it works (0 divergences).

## C.5 The load-bearing premise, refined

The "**biology ⊥ G**" headline was *tested under both identifications* (the §3.1 correlated-G variant — run it,
it is cheap and decisive): bifactor direct-G loadings ≈ 0 (metabolic 0.08, inflammatory 0.07), but the
correlated-G *factor* correlations are **metabolic 0.28 / inflammatory 0.14** (vs cognition 0.35, sleep 0.47).
So biology is the **least severity-entangled** domain (~92–98% of its variance independent of G), **but not
strictly orthogonal** — the exact claim is *"largely severity-independent,"* not *"orthogonal."* This is the
kind of honest correction the dual-identification test exists to produce.

## C.6 What is left for M1

The map exists; these finish/harden it (none started): **FIML confirmation** (continuous backbone — classical
fit indices + an independent-estimator check that the structure is not a prior artefact) · **measurement
invariance** across BP/SZ/DR (the key open validity check) · **full-N certified S5 on the GPU** ·
**prior→posterior empirical atlas** · per-patient **scoring** at scale · the formal **adjudication** write-up.
