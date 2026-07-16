"""OOP implementation of the FACE M1 Bayesian sparse bifactor/ESEM model.

This module intentionally lives beside, not inside, the original ``measurement.kernel``
engine.  It reads the same processed artifacts but rebuilds the measurement model
with a small set of explicit classes:

* ``MeasurementDataset``: data loading, transforms, covariate design, masks.
* ``LoadingSpec``: item x factor prior cells -> sparse loading parameters.
* ``BayesianBifactorESEM``: PyMC model builders and Woodbury likelihood kernel.
* ``StageRunner``: cached staged sampling with diagnostics.
* ``PatientProjector`` and ``MeasurementVisualizer``: scoring and summary figures.

The corrected canonical fit keeps ``unlikely_cross`` cells at exact zero, estimates
covariate effects jointly in every observation-family likelihood, routes every retained
indicator exactly once, and uses an LKJ prior for the free factor-correlation block.
Continuous observations are Gaussian after rank-INT transformation; native binary,
ordinal, and count observations retain their corresponding link-scale likelihoods.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import arviz as az  # type: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingImports]
import pymc as pm  # type: ignore[reportMissingImports]
import pytensor.tensor as pt  # type: ignore[reportMissingImports]
from scipy.interpolate import BSpline  # type: ignore[reportMissingImports]
from scipy.linalg import solve_triangular  # type: ignore[reportMissingImports]
from scipy.special import logsumexp  # type: ignore[reportMissingImports]
from scipy.stats import multivariate_normal, norm, rankdata  # type: ignore[reportMissingImports]
from sklearn.preprocessing import SplineTransformer  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parents[3]
PROC = Path(os.environ.get("FACE_DATA_DIR", str(REPO / "data" / "processed")))
MATRIX = REPO / "configs" / "loading_matrix.csv"
RESULTS = REPO / "results" / "m1_measurement"
FIGURES = REPO / "docs" / "figures" / "m1_measurement"
MODEL_VERSION = "m1_measurement_v4_correlated_substance_tiered_certification"

G_KEY = "overall_severity"

# Factor lists define the staged continuation ladder.
#
# The order is scientific and computational:
# 1. Start with the stable continuous backbone (S1_FACTORS).
# 2. Add non-Gaussian dimensions later, when the continuous geometry is known.
# 3. End at the full 9D map (S5_FACTORS).
#
# ``overall_severity`` is the bifactor general factor G.  The remaining entries
# are specific axes.  In the primary bifactor model, G is held orthogonal to the
# specifics; the specifics may correlate with each other through Phi.
S1_FACTORS = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep"]
S3_FACTORS = S1_FACTORS + ["suicidality", "developmental_risk"]
S5_FACTORS = S3_FACTORS + ["mania_activation", "substance"]

# Continuous-only intermediate rung of the ladder.  developmental_risk (CTQ / WURS /
# age-of-onset / perinatal) and mania_activation carry enough Gaussian/log-Gaussian
# anchors to be marginalized here, so the mixed S5 fit can warm-start its continuous
# backbone from this rung.  suicidality and substance are binary-dominated (they only
# identify in the explicit mixed block), so they stay out of this continuous rung and
# start fresh in S5 — mirroring the certified engine's S3a.
S3_CONT_FACTORS = S1_FACTORS + ["developmental_risk", "mania_activation"]

# Factors listed here must be explicit in the mixed-likelihood model.
#
# Why?  Gaussian/continuous indicators can be analytically marginalized.  Binary,
# ordinal, and count indicators cannot be collapsed into the same closed-form
# Gaussian covariance.  For factors carrying those indicators we keep patient
# latent coordinates in the sampler and attach Bernoulli / ordered-logistic /
# negative-binomial likelihoods to them.
DEFAULT_EXPLICIT_FACTORS = [
    "overall_severity",
    "immunometabolic",
    "suicidality",
    "developmental_risk",
    "substance",
]

# Clinically, these scales are broad composites. They mix several things:
# - general illness burden / distress
# - sleep disturbance
# - cognitive slowing
# - sometimes suicidality or anhedonia in the theory matrix
# So if we forced MADRS to define a separate "depression factor,"
# the model would likely rediscover diagnosis-flavored depression severity,
# which the M1 design explicitly tries to avoid. The point is not to create a DSM depression dimension.
# The point is to let depression/anxiety scales act as windows onto the transdiagnostic dimensions.
# So conceptually MADRS / QIDS / STAI:
#   are not dimensions
#   are not anchors
#   are symptom windows
#   are allowed to say: this patient has burden that maps partly to G, sleep, cognition, etc.
WINDOWS = ["madrs", "qidsr120", "staya"]

CONTINUOUS_FAMILIES = {"gaussian", "lognormal", "student_t"}
LOG2PI = float(np.log(2.0 * np.pi))


def _rank_int(x: np.ndarray) -> np.ndarray:
    """Rank-based inverse-normal (the Gaussian-copula marginal transform): u = rank/(n+1),
    z = Phi^-1(u), average ranks for ties. ``x`` is the observed (finite) 1-D array."""
    r = rankdata(x)
    return norm.ppf(r / (x.size + 1.0))


def _cohort_weights(cohort: np.ndarray) -> np.ndarray:
    """Per-patient §3.6 cohort weights: w_i = N / (K * n_cohort_i), so each of the K cohorts has
    total weight N/K (equal influence) and sum(w) = N (total information preserved -> precision stays
    order-correct, not inflated)."""
    cohort = np.asarray(cohort)
    n = cohort.size
    cohorts = list(dict.fromkeys(cohort))
    k = len(cohorts)
    counts = {c: int((cohort == c).sum()) for c in cohorts}
    return np.array([n / (k * counts[c]) for c in cohort], dtype="float64")


def copula_invert(z: np.ndarray, sorted_values: np.ndarray, sorted_z: np.ndarray) -> np.ndarray:
    """Gaussian-copula inverse: map latent z back to the ORIENTED original scale via the stored
    empirical map (``CoreData.copula[item]``), y = F_j^-1(Phi(z)) by monotone interpolation on
    (sorted_z -> sorted_values).  Clipped to the V0 support (np.interp does not extrapolate).  With
    the per-item sign the raw indicator is ``raw = sign * y``.  Generative recipe for synthetic
    patients: eta ~ N(0, Phi); z ~ N(Lam eta, Psi); y = copula_invert(z, *copula[item])."""
    return np.interp(np.asarray(z, dtype="float64"), sorted_z, sorted_values)


@dataclass(frozen=True)
class StageDefinition:
    """One staged continuation target.

    Think of a stage as a recipe for one posterior fit: which dimensions are
    present, whether Phi is correlated, whether the window items are added, and
    how long NUTS should run.  Staging is a homotopy/continuation strategy: solve
    an easier model first, then add complexity after the simpler geometry has
    been checked.
    """

    name: str
    factors: list[str]
    correlated: bool = False
    windows: bool = False
    mixed: bool = False
    # Empty means "derive from retained native loading support".  A nonempty list
    # is an assertion checked against that derived set, never the routing source.
    explicit_factors: list[str] = field(default_factory=list)
    min_cohorts: int = 3
    n_subsample: int | None = None
    balanced: bool = False
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    seed: int = 20260605
    # Cross-loading arm: free the theory-motivated ``plausible_cross`` specific<->specific cells
    # (the immunometabolic metabolic<->inflammatory bridge in matrix v3) instead of hard-zeroing them.
    # ``cross_sd_scale`` multiplies their prior_sd (0.25 in the matrix), so 1.0 -> Normal(0, 0.25).
    # Default False/0.25 reproduces the certified hard-zero map exactly.
    specific_cross: bool = False
    cross_sd_scale: float = 0.25
    g_correlated: bool = False
    hurdle_counts: bool = False
    enforce_gates: bool = False


@dataclass(frozen=True)
class MeasurementConfig:
    """Paths and modeling switches for the parallel implementation.

    The default object is a hard-zero, joint-covariate model: unlikely
    cross-loadings are fixed at exactly zero and item-specific covariate effects
    are estimated inside the likelihood.  The canonical M1 runner additionally
    selects the Gaussian-copula family and corrected eight-factor prior map.

    Why hard-zero is the default (empirically established 2026-06-19): freeing the
    ~980 ``unlikely_cross`` cells as soft Normal(0, 0.05) FLOODS every factor's
    column with weak cross-loadings.  Well-anchored factors (many home items)
    shrug this off, but a THIN factor (few home items, e.g. substance / smoking)
    gets its identity diluted — its home loadings collapse toward 0 and the column
    becomes multimodal, which then poisons the global R-hat through Phi.  Hard-zero
    leaves each factor defined only by its named home + bifactor-G + window cells,
    so thin factors identify cleanly and mixing improves across the board
    (full-9D mixed fit: ESS rose ~6x, the substance factor went from unpinnable to
    well-mixed).  ``with_soft_unlikely()`` re-enables the soft cells as the
    documented sensitivity arm (congruent with the hard-zero map for the
    well-anchored backbone -- see ``scripts/10b``).
    """

    processed_dir: Path = PROC
    prior_matrix: Path = MATRIX
    output_dir: Path = RESULTS
    figure_dir: Path = FIGURES
    psi_floor: float = 0.05
    # ``soft_unlikely`` / ``soft_g_anchor_specific``: when True the cells flagged
    # ``unlikely_cross`` / ``g_anchor_on_specific`` in the prior matrix become free
    # (Normal(0, 0.05) / near-zero) instead of exact 0.  DEFAULT FALSE (hard-zero
    # primary) -- see the class docstring for why; True is the soft sensitivity arm.
    lkj_eta: float = 2.0
    soft_unlikely: bool = False
    soft_g_anchor_specific: bool = False
    # Covariates calibrate item means before the latent coordinate is read.
    #   * ``"in_likelihood"`` (default): estimate item intercepts and covariate
    #     slopes jointly for Gaussian and every native family.  This is the M1
    #     estimand and propagates covariate-effect uncertainty.
    #   * ``"residualize"``: legacy Gaussian-only FWL shortcut retained for
    #     controlled sensitivities; it is not the corrected mixed-family model.
    #   * ``"none"``: no covariate adjustment.
    # ``include_covariates`` is the master on/off; when False no covariates are used
    # regardless of ``covariate_mode``.
    include_covariates: bool = True
    covariate_mode: str = "in_likelihood"
    covariate_missingness: str = "mean_indicator"
    include_cohort_covariates: bool = False
    age_spline_knots: int = 4
    fast_mode: bool = False
    max_tree_depth: int = 8
    # ``likelihood_mode``: "native" (default) = the certified tiered mixed likelihood
    # (Gaussian/log-Gaussian continuous + Bernoulli/ordered-logistic/neg-binomial explicit);
    # "gaussian_copula" = the acceleration vertical -- map Gaussianizable indicators through their
    # empirical CDF (rank-INT: z = Phi^-1(F_j(y))) and run the SAME marginalized Woodbury model on z
    # (a semiparametric Gaussian copula factor model; invertible for synthetic generation).  Tiering:
    # continuous always Gaussianized; ordinal/count promoted to the marginalized block iff
    # n_distinct >= copula_min_distinct AND modal_frequency < copula_max_modal_frac; binary and
    # low-cardinality ordinal keep their native discrete likelihood.
    likelihood_mode: str = "native"
    copula_min_distinct: int = 8
    copula_max_modal_frac: float = 0.5
    # ``cohort_weighted`` (methods §3.6): weight each patient's likelihood by w_i = N/(K*n_cohort_i)
    # so every cohort contributes equally (transdiagnostic estimand) while using ALL patients, in a
    # single coherent posterior -- the faithful way to "use full N" without BP-dominance.  Weights sum
    # to N (total information preserved, so precision is order-correct, not falsely inflated).  This is
    # a composite/pseudo-likelihood: the point estimate is the balanced estimand; the posterior SD is
    # order-correct but not fully Bayesian-calibrated (validate via cross-seed congruence).
    cohort_weighted: bool = False
    # ``exclude_items``: drop these indicators from the model entirely (matrix rows + data columns), so
    # they contribute no loadings and no likelihood terms.  For sensitivity arms that ask whether a factor
    # survives removing specific indicators -- e.g. the immunometabolic "minus anthropometry" refit
    # (exclude bmi/weight/wstcir) testing whether a coherent cardiometabolic-inflammatory axis remains once
    # the body-size anchors are gone.  Not a primary fit: the canonical map keeps all indicators.
    exclude_items: tuple[str, ...] = ()
    # ``orthogonal_factors``: optional sensitivity restriction pinning named
    # specific factors outside the correlated block (G is always pinned by the
    # bifactor parameterization).  The primary model leaves this empty so all
    # seven specifics, including Substance, remain in one LKJ block.
    orthogonal_factors: tuple[str, ...] = ()
    # ``equal_home_loading_factors``: identify a thin factor by constraining all
    # of its retained Gaussian home indicators to share one positive loading.
    # This is a measurement restriction only; it does not constrain Phi.  The
    # repaired primary uses it for the two-indicator substance factor until
    # validated alcohol/cannabis indicators are restored.
    equal_home_loading_factors: tuple[str, ...] = ()
    # ``cross_loading_prior``: how the off-home specific<->specific cross-loadings are treated.
    #   "hard_zero" (default): fixed at exactly 0 (the certified rigid map).
    #   "horseshoe": freed under a REGULARIZED (Finnish) horseshoe — a sparsity-inducing prior with a
    #     sharp spike at 0 (shrinks the many noise cross-loadings hard, so a thin factor's column is not
    #     diluted and its identity is protected) AND heavy tails (so a GENUINE small cross-loading can
    #     escape the shrinkage and become credible).  This is the principled way to let instruments load
    #     weakly on several axes without collapsing low-instrument factors (substance n=4, mania n=2) --
    #     see docs/HORSESHOE_ESEM.md.  ``hs_tau0`` is the global-shrinkage scale (smaller -> sparser);
    #     ``hs_slab_c`` is the slab width that caps an escaped cross-loading's magnitude (and stabilizes NUTS).
    cross_loading_prior: str = "hard_zero"
    hs_tau0: float = 0.05
    hs_slab_c: float = 0.30
    # ``hs_fixed_tau``: fix the global shrinkage to ``hs_tau0`` instead of sampling it. This deletes the
    # global funnel (the worst mixing pathology), at the cost of choosing the sparsity level a priori.
    # ``hs_local_df``: degrees of freedom of the local shrinkage's HalfStudentT (nu=1 -> HalfCauchy, the
    # canonical horseshoe; nu~3 keeps heavy-but-finite tails that still let a genuine cross-loading escape
    # while mixing far better).  Defaults reproduce the canonical sampled-tau Cauchy horseshoe.
    hs_fixed_tau: bool = False
    hs_local_df: float = 1.0
    rhat_max: float = 1.01
    # Intercepts and covariate calibration slopes are high-dimensional nuisance
    # terms, not the latent measurement map.  They retain an explicit convergence
    # gate, but a small number may lie between the strict map threshold and this
    # continuation threshold without vetoing an otherwise aligned factor map.
    nuisance_rhat_max: float = 1.02
    ess_min: float = 400.0
    # Substance remains in the scientific covariance model, but its current
    # two-indicator measurement basis receives a separate *provisional* Monte
    # Carlo gate.  Passing these limits permits continuation; it does not confer
    # strict Substance certification, which still uses rhat_max/ess_min.
    substance_rhat_max: float = 1.05
    substance_ess_min: float = 100.0
    bfmi_min: float = 0.30
    max_depth_fraction: float = 0.05
    loading_congruence_min: float = 0.95
    loading_sign_threshold: float = 0.10

    def with_horseshoe(self, *, tau0: float = 0.05, slab_c: float = 0.30,
                       fixed_tau: bool = False, local_df: float = 1.0) -> MeasurementConfig:
        """Free the off-home specific cross-loadings under a regularized horseshoe (sparse ESEM).

        ``fixed_tau=True`` + ``local_df~3`` is the stable variant (no global funnel, lighter local tails)
        for the validation fit; the canonical sampled-tau Cauchy horseshoe is the default."""
        return replace(self, cross_loading_prior="horseshoe", hs_tau0=tau0, hs_slab_c=slab_c,
                       hs_fixed_tau=fixed_tau, hs_local_df=local_df)

    def with_soft_unlikely(self) -> MeasurementConfig:
        """Return the soft-prior sensitivity variant: free the ``unlikely_cross`` and
        ``g_anchor_on_specific`` cells (Normal(0, 0.05) / near-zero) instead of exact 0.

        This is a sensitivity arm, not the primary: it is congruent with the
        hard-zero map for the well-anchored backbone but dilutes thin factors (see
        the class docstring).  Use it to confirm the backbone is robust to the
        hard-vs-soft choice, not to fit the thin factors."""
        return replace(self, soft_unlikely=True, soft_g_anchor_specific=True)

    def with_fast_mode(self) -> MeasurementConfig:
        """Return a speed-oriented variant (hard-zero, fast-mode flag set)."""
        return replace(self, soft_unlikely=False, soft_g_anchor_specific=False, fast_mode=True)

    def with_gaussian_copula(self) -> MeasurementConfig:
        """Return the Gaussian-copula acceleration variant (rank-INT Gaussianize the continuous +
        high-cardinality ordinal/count block, marginalized via Woodbury; binary + low-cardinality
        ordinal stay native).  An acceleration vertical, not the certified faithful default."""
        return replace(self, likelihood_mode="gaussian_copula")

    def with_substance_orthogonal(self) -> MeasurementConfig:
        """Optional sensitivity arm pinning Substance outside the specific LKJ block."""
        return replace(self, orthogonal_factors=tuple(sorted(set(self.orthogonal_factors) | {"substance"})))

    def with_equal_home_loadings(self, *factors: str) -> MeasurementConfig:
        """Share one positive home loading within each named thin factor.

        Factor variances remain fixed to one and the full configured correlation
        block remains free.  The constraint is therefore a tau-equivalent
        measurement identification restriction, not a covariance restriction.
        """
        return replace(
            self,
            equal_home_loading_factors=tuple(
                sorted(set(self.equal_home_loading_factors) | set(factors))
            ),
        )

    def with_cohort_weighted(self) -> MeasurementConfig:
        """Return the §3.6 cohort-weighted variant: use ALL patients with per-patient weights that
        equalize each cohort's influence (transdiagnostic estimand) in one coherent posterior.
        Pair with full-N stages (no subsample)."""
        return replace(self, cohort_weighted=True)

    def with_excluded_items(self, *items: str) -> MeasurementConfig:
        """Return a sensitivity variant that drops ``items`` from the model entirely.

        The named indicators are removed from the prior matrix and the data block, so they contribute
        no loadings and no likelihood terms.  Used e.g. for the immunometabolic "minus anthropometry"
        refit (``with_excluded_items("bmi", "weight", "wstcir")``) -- a sensitivity arm, not the
        canonical map (which keeps all indicators)."""
        return replace(self, exclude_items=tuple(dict.fromkeys((*self.exclude_items, *items))))

    def with_smoke_defaults(self) -> MeasurementConfig:
        """Return a notebook-smoke variant optimized for fast wiring checks.

        Smoke mode is not the scientific model: it disables the most expensive
        theory-faithful pieces so users can validate imports, masking, model
        construction, caching, and plotting before starting medium/production fits.
        """
        return replace(
            self,
            soft_unlikely=False,
            soft_g_anchor_specific=False,
            include_covariates=False,
            fast_mode=True,
            max_tree_depth=6,
        )

    @property
    def stage_plan(self) -> list[StageDefinition]:
        """Default staged continuation ladder (each rung warm-starts the next).

        S1 estimates the continuous backbone with independent specifics.
        S2 keeps the same backbone but adds Phi and the symptom windows.
        S3 adds the marginalizable developmental_risk + mania specifics, so the
            mixed fit has a basin for its continuous backbone.
        S5 adds the full 9D mixed-likelihood model on a balanced subsample,
            because explicit patient latents for binary/ordinal/count items are the
            computational frontier.

        target_accept ladder (convergence-tested 2026-06-19): the continuous rungs
        relax 0.95 -> 0.90 -> 0.85 as the geometry improves, but the mixed S5 uses
        0.95 (NOT 0.90).  The mixed fit's explicit developmental_risk latents can fall
        into a collapsed-factor mode during warmup; the higher target_accept (smaller
        steps + more thorough warmup) keeps chains from sticking there -- multi-seed
        S5 mixing went from a hard-stuck R-hat 1.55 (ta 0.90) to a mild ~1.06-1.14
        (ta 0.95, tune 2000), 0 divergences, with the substance + developmental-CTQ
        loadings reproducible across seeds.
        """
        return [
            StageDefinition("s1_core", S1_FACTORS, draws=1000, tune=1000, target_accept=0.95),
            StageDefinition(
                "s2_esem",
                S1_FACTORS,
                correlated=True,
                windows=True,
                draws=1000,
                tune=1000,
                target_accept=0.9,
            ),
            StageDefinition(
                "s3_continuous",
                S3_CONT_FACTORS,
                correlated=True,
                windows=True,
                draws=1000,
                tune=1000,
                target_accept=0.85,
            ),
            StageDefinition(
                "s5_9dim_mixed",
                S5_FACTORS,
                correlated=True,
                windows=True,
                mixed=True,
                explicit_factors=list(DEFAULT_EXPLICIT_FACTORS),
                min_cohorts=2,
                n_subsample=2000,
                balanced=True,
                draws=1500,
                tune=2000,
                target_accept=0.95,
            ),
        ]

    def cross_loading_stage_plan(self) -> list[StageDefinition]:
        """Cross-loading arm (sensitivity): the certified S5 map PLUS the theory-motivated
        ``plausible_cross`` specific cells (the immunometabolic metabolic<->inflammatory bridge),
        freed at Normal(0, 0.25).  Warm-started from the certified ``s5_9dim_mixed`` so it starts
        in the hard-zero basin; otherwise identical to it (same N, draws/tune, target_accept, seed).
        A separate stage name (own output dir) keeps the canonical fit untouched."""
        s5 = self.stage_plan[-1]   # the certified s5_9dim_mixed mixed rung
        return [replace(s5, name="s5_xcross", specific_cross=True, cross_sd_scale=1.0)]

    @property
    def smoke_stage_plan(self) -> list[StageDefinition]:
        """Fast notebook plan for S1 wiring checks, not convergence evidence."""
        return [
            StageDefinition(
                "smoke_s1_wiring_v2",
                S1_FACTORS,
                n_subsample=300,
                draws=300,
                tune=500,
                chains=4,
                target_accept=0.95,
                seed=20260605,
            ),
        ]

    @property
    def mixed_smoke_stage_plan(self) -> list[StageDefinition]:
        """Optional mixed-likelihood smoke; useful for wiring, too short for diagnostics."""
        return [
            StageDefinition(
                "smoke_s5_9dim_mixed_v2",
                S5_FACTORS,
                correlated=True,
                windows=True,
                mixed=True,
                explicit_factors=list(DEFAULT_EXPLICIT_FACTORS),
                min_cohorts=2,
                n_subsample=300,
                balanced=True,
                draws=150,
                tune=250,
                chains=2,
                target_accept=0.95,
                seed=20260605,
            ),
        ]


@dataclass
class CoreData:
    """Encoded continuous block consumed by the marginalized likelihood.

    ``M`` is the patient x continuous-indicator matrix after burden orientation,
    optional log transforms, and z-scoring.  Missing values remain ``NaN``.  The
    likelihood later turns this into a binary mask; it never fills those cells.

    ``covariates`` is aligned row-for-row to ``M``.  It adjusts item means but is
    not part of the factor space.
    """

    M: np.ndarray
    covariates: np.ndarray
    covariate_names: list[str]
    items: list[str]
    home: list[str]
    factor_cols: list[str]
    spec_factors: list[str]
    g_col: int
    cohort: np.ndarray
    index: pd.Index
    families: dict[str, str]
    signs: dict[str, int]
    moments: dict[str, tuple[str, int, float | None, float, float]] = field(default_factory=dict)
    # Gaussian-copula inversion map (copula likelihood_mode only): item -> (sorted oriented
    # observed values, sorted rank-INT z), aligned by rank, for y = F_j^-1(Phi(z)).  None in native mode.
    copula: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    covariate_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MixedData:
    """Hybrid mixed-likelihood inputs.

    The mixed model splits the latent factors into two blocks:
    * ``e_cols``: explicit factors sampled per patient because they touch
      Bernoulli / ordinal / count indicators.
    * ``m_cols``: continuous-only factors that can still be marginalized.

    This is the key hybrid trick: keep the fast Woodbury likelihood for the
    Gaussian part while still giving non-Gaussian clinical variables their
    proper likelihoods.
    """

    base: CoreData
    e_cols: list[int]
    m_cols: list[int]
    bin_items: list[str]
    Bin: np.ndarray
    ord_items: list[str]
    Ord: np.ndarray
    ord_K: list[int]
    cnt_items: list[str]
    Cnt: np.ndarray
    ng_home: dict[str, int]
    ng_hp: dict[str, tuple[float, float]]
    ng_gp: dict[str, tuple[float, float]]
    ng_index: dict[str, int]
    routing_report: list[dict[str, Any]] = field(default_factory=list)

    @property
    def native_items(self) -> list[str]:
        """Stable native-parameter row order shared by fit, export, and scoring."""
        return self.bin_items + self.cnt_items + self.ord_items


@dataclass
class LoadingSpec:
    """Sparse loading matrix specification.

    ``pos_cells`` are sign-anchored home loadings.  They use a positive
    truncated normal prior so each factor has a stable orientation: higher score
    means more burden after item-level sign flips.

    ``signed_cells`` are cross-loadings.  They may be positive or negative and
    are usually centered at zero.  This is what makes the model ESEM rather than
    a rigid CFA: theory suggests where cross-loadings are plausible, but the
    posterior decides how much signal they carry.

    ``kind`` is not used by PyMC directly; it is an audit map that tells us why a
    loading exists: primary, G bifactor loading, window, unlikely soft-zero, etc.
    """

    pos_cells: list[tuple[int, int, float, float]]
    signed_cells: list[tuple[int, int, float, float]]
    kind: dict[tuple[int, int], str]
    factor_cols: list[str]
    items: list[str]
    home: list[str]
    # Off-home specific<->specific cells freed under the regularized horseshoe (cross_loading_prior
    # == "horseshoe"); empty in the hard-zero / soft-unlikely modes.  Just (row, col) — no fixed prior.
    hs_cells: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def from_core(
        cls,
        core: CoreData,
        matrix: pd.DataFrame,
        *,
        windows: bool,
        soft_unlikely: bool,
        soft_g_anchor_specific: bool,
        specific_cross: bool = False,
        horseshoe: bool = False,
        window_sd_scale: float = 1.0,
        cross_sd_scale: float = 0.25,
        bifactor_g_sd: dict[str, float] | None = None,
        flat: bool = False,
    ) -> LoadingSpec:
        cell = {
            (r.item, r.factor): (str(r.prior_type), float(r.prior_mean), float(r.prior_sd))
            for r in matrix.itertuples()
        }
        col = {f: i for i, f in enumerate(core.factor_cols)}
        pos_cells: list[tuple[int, int, float, float]] = []
        signed_cells: list[tuple[int, int, float, float]] = []
        hs_cells: list[tuple[int, int]] = []
        kind: dict[tuple[int, int], str] = {}
        for j, item in enumerate(core.items):
            h = core.home[j]
            for factor in core.factor_cols:
                c = col[factor]
                ptype, mu, sd = cell.get((item, factor), ("unlikely_cross", 0.0, 0.05))
                if factor == h:
                    # Home cell: this item is intended to define this factor.
                    # The positive constraint kills the sign-flip ambiguity.
                    # Example: CRP -> inflammatory, BMI -> metabolic.
                    pos_cells.append((j, c, mu, sd))
                    kind[(j, c)] = "g_anchor" if factor == G_KEY else "primary"
                elif ptype == "g_anchor_on_specific":
                    if soft_g_anchor_specific:
                        # G anchors (functioning/global severity) should not
                        # define the specific axes.  We keep the cell estimable
                        # with an extremely tight prior so this matches the
                        # theory ("near zero") without pretending the value is
                        # mathematically impossible.
                        signed_cells.append((j, c, mu, sd))
                        kind[(j, c)] = "g_anchor_on_specific"
                elif factor == G_KEY:
                    if h:
                        # Bifactor loading: every specific-domain item may also
                        # carry some general burden signal.  This is how the
                        # model learns whether an item is mostly "what kind" or
                        # also "how much overall".
                        g_type, g_mu, g_sd = cell.get((item, G_KEY), ("plausible_cross", 0.0, 0.25))
                        sd_g = bifactor_g_sd.get(h, g_sd) if bifactor_g_sd else g_sd
                        signed_cells.append((j, c, g_mu, sd_g))
                        kind[(j, c)] = "bifactor_G"
                    elif windows and ptype == "plausible_cross":
                        # A window has no home factor; it is a broad symptom
                        # surface that looks into existing dimensions.
                        signed_cells.append((j, c, mu, sd * window_sd_scale))
                        kind[(j, c)] = "window"
                elif ptype == "plausible_cross":
                    if h == "":
                        if windows:
                            signed_cells.append((j, c, mu, sd * window_sd_scale))
                            kind[(j, c)] = "window"
                    elif horseshoe:
                        # Sparse ESEM: every off-home specific cross-loading is freed under the
                        # regularized horseshoe (shrunk hard unless the data genuinely support it).
                        hs_cells.append((j, c))
                        kind[(j, c)] = "hs_cross"
                    elif specific_cross:
                        # Specific-to-specific cross-loadings are optional.
                        # They can be hard to distinguish from Phi correlations,
                        # so the default ladder keeps them off unless explicitly
                        # requested as a sensitivity arm.
                        signed_cells.append((j, c, mu, sd * cross_sd_scale))
                        kind[(j, c)] = "cross"
                elif ptype == "unlikely_cross":
                    if horseshoe:
                        # The many theory-"unlikely" cells are exactly the noise the horseshoe shrinks
                        # to ~0 (protecting thin factors), while leaving room for a genuine surprise.
                        hs_cells.append((j, c))
                        kind[(j, c)] = "hs_cross"
                    elif soft_unlikely:
                        # Theory says this relation is unlikely, not impossible.
                        # A tight Normal(0, 0.05) prior lets a strong data signal
                        # escape while shrinking noise back to zero.
                        signed_cells.append((j, c, mu, sd))
                        kind[(j, c)] = "unlikely"
        if flat:
            # Prior-free confirmation keeps only the identification constraints:
            # home cells remain positive, but their location/scale no longer
            # encode the ontology.  If this reproduces the soft-prior fit, the
            # map is earned by the likelihood rather than manufactured by priors.
            pos_cells = [(j, c, 0.0, 5.0) for j, c, _mu, _sd in pos_cells]
            signed_cells = [(j, c, 0.0, 5.0) for j, c, _mu, _sd in signed_cells]
        return cls(pos_cells, signed_cells, kind, core.factor_cols, core.items, core.home, hs_cells)

    @property
    def n_free(self) -> int:
        """Total number of loading parameters."""
        return len(self.pos_cells) + len(self.signed_cells) + len(self.hs_cells)


class MeasurementDataset:
    """Load and encode FACE V0 model-ready tables.

    This class is deliberately boring: its job is to convert the persisted data
    contract into arrays for PyMC.  It does not discover structure, and it does
    not impute.  If a cell is missing in ``baseline_v0.parquet``, it remains
    missing in ``CoreData.M``.
    """

    def __init__(self, config: MeasurementConfig | None = None):
        self.config = config or MeasurementConfig()
        self.matrix = pd.read_csv(self.config.prior_matrix)
        if self.config.exclude_items:
            # Sensitivity arm: drop named indicators entirely.  Filtering the matrix here propagates to
            # ``self.meta``/``self.home`` and to item selection in ``core()`` (which iterates ``self.home``),
            # so excluded items are never requested from the baseline parquet either.
            self.matrix = (
                self.matrix[~self.matrix["item"].isin(self.config.exclude_items)]
                .reset_index(drop=True)
            )
        self.meta = self.matrix.drop_duplicates("item").set_index("item")[
            ["likelihood_family", "modeling_block", "item_sign"]
        ]
        self.home = (
            self.matrix[self.matrix.prior_type.isin(["primary", "g_anchor"])]
            .drop_duplicates("item")
            .set_index("item")["factor"]
            .to_dict()
        )

    def core(
        self,
        factors: list[str] | None = None,
        *,
        correlated: bool = False,
        windows: bool = False,
        cohort_subset: list[str] | None = None,
        keep_index: np.ndarray | None = None,
        balanced: bool = False,
        n_subsample: int | None = None,
        seed: int = 20260605,
        include_covariates: bool | None = None,
        force_factors_continuous: list[str] | None = None,
        visit: str = "V0",
    ) -> CoreData:
        """Return encoded continuous data for a stage."""
        factors = factors or list(S1_FACTORS)
        # Factor columns are the columns of Lambda.  G is always first so the
        # later Phi construction can treat the G row/column specially.
        factor_cols = [G_KEY] + [f for f in factors if f != G_KEY]
        spec_factors = [f for f in factor_cols if f != G_KEY]
        force_cont = set(force_factors_continuous or [])

        # Start from home indicators for the factors active in this stage.  The
        # prior matrix is the single source of truth: if an item has a primary
        # or G-anchor row, it has a home factor.  For the marginalized block we
        # keep only continuous/modeling-block items, because the Gaussian
        # covariance identity applies only there.
        items = sorted(
            item
            for item, h in self.home.items()
            if h in factors
            and item in self.meta.index
            and (
                self.meta.loc[item, "modeling_block"] == "continuous"
                or h in force_cont
            )
        )
        if windows:
            # Windows are added only in ESEM stages.  They are not included in
            # S1 because S1 is the stable simple-structure starting point.
            window_items = [
                w
                for w in WINDOWS
                if w in self.meta.index
                and self.meta.loc[w, "modeling_block"] == "continuous"
                and any(self._cell_type(w, f) == "plausible_cross" for f in factor_cols)
            ]
            items = sorted(set(items) | set(window_items))

        baseline = pd.read_parquet(self.config.processed_dir / f"baseline_{visit.lower()}.parquet")
        items = [item for item in items if item in baseline.columns]
        if self.config.likelihood_mode == "gaussian_copula":
            # Promote high-cardinality ordinal/count items (home in this stage's factors) into the
            # Gaussianized marginalized block; binary + low-cardinality ordinal stay native/explicit.
            promoted = [
                it
                for it, h in self.home.items()
                if h in factors
                and it in self.meta.index
                and it in baseline.columns
                and self.meta.loc[it, "modeling_block"] == "explicit"
                and self._gaussianizable(it, baseline)
            ]
            items = sorted(set(items) | set(promoted))
        if cohort_subset is not None:
            cohort_mask = np.isin(
                np.asarray(baseline.index.get_level_values("cohort")), list(cohort_subset)
            )
            baseline = baseline.loc[cohort_mask]
        if keep_index is not None:
            baseline = baseline.iloc[keep_index]
        cohort = np.asarray(baseline.index.get_level_values("cohort"))

        copula_mode = self.config.likelihood_mode == "gaussian_copula"
        raw: dict[str, pd.Series] = {}
        moments: dict[str, tuple[str, int, float | None, float, float]] = {}
        for item in items:
            values = pd.to_numeric(baseline[item], errors="coerce").astype(float)
            family = str(self.meta.loc[item, "likelihood_family"])
            log_min: float | None = None
            if family == "lognormal" and not copula_mode:
                # Skewed positive labs are modeled on the log scale (native only).  The copula
                # path needs no log -- rank-INT is monotone-invariant -- and stores the raw
                # oriented values for clean inversion.  Deterministic transform, not imputation.
                log_min = float(np.nanmin(values.to_numpy()))
                values = (
                    np.log1p(values - log_min + 1e-6)
                    if np.isfinite(log_min) and log_min <= 0
                    else np.log(values)
                )
            sign = int(self.meta.loc[item, "item_sign"])
            # Orient every item so that larger encoded values mean "more burden".
            raw[item] = sign * values
            moments[item] = (family, sign, log_min, 0.0, 1.0)  # mu/sd set below in the native path
        raw_df = pd.DataFrame(raw, index=baseline.index)

        cov_enabled = self.config.include_covariates if include_covariates is None else include_covariates
        mode = self.config.covariate_mode if cov_enabled else "none"
        empty_cov = (np.zeros((len(raw_df), 0), dtype="float64"), [])
        design_metadata: dict[str, Any] = {}
        copula: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

        if copula_mode:
            # Semiparametric Gaussian copula: each item -> standard-normal marginals via the
            # empirical CDF (rank-INT).  The same zero-mean Woodbury kernel then applies to z.
            copula = {}
            enc: dict[str, pd.Series] = {}
            for item in items:
                v = raw_df[item].to_numpy("float64")
                obs = np.isfinite(v)
                z = np.full(v.shape, np.nan)
                xo = v[obs]
                if xo.size:
                    zo = _rank_int(xo)
                    z[obs] = zo
                    order = np.argsort(xo, kind="mergesort")
                    copula[item] = (xo[order], np.sort(zo))  # ascending oriented-value <-> z, by rank
                enc[item] = pd.Series(z, index=raw_df.index)
            Mdf = pd.DataFrame(enc, index=raw_df.index)
            if mode == "residualize":
                # FWL in z-space (keeps ~zero-mean); inversion uses the stored marginal copula map.
                Mdf = self._residualize_on_covariates(Mdf)
                covariates, covariate_names = empty_cov
            elif mode == "in_likelihood":
                covariates, covariate_names, design_metadata = self._covariate_design(
                    raw_df.index, with_metadata=True
                )
            else:
                covariates, covariate_names = empty_cov
        else:
            if mode == "residualize":
                # FWL: partial each item out on the covariate design before z-scoring, so the
                # marginalized likelihood stays zero-mean (no alpha/beta enter the sampler).
                raw_df = self._residualize_on_covariates(raw_df)
                covariates, covariate_names = empty_cov
            elif mode == "in_likelihood":
                covariates, covariate_names, design_metadata = self._covariate_design(
                    raw_df.index, with_metadata=True
                )
            else:
                covariates, covariate_names = empty_cov

            encoded: dict[str, pd.Series] = {}
            for item in items:
                values = raw_df[item]
                mu = float(values.mean())
                sd = float(values.std()) if float(values.std()) > 0 else 1.0
                # Standardize on observed cells only (pandas skips NaN; no missing value invented).
                encoded[item] = (values - mu) / sd
                family, sign, log_min, _m, _s = moments[item]
                moments[item] = (family, sign, log_min, mu, sd)
            Mdf = pd.DataFrame(encoded, index=baseline.index)
        if n_subsample and n_subsample < len(Mdf):
            rng = np.random.default_rng(seed)
            # Subsampling is a compute strategy, never a completeness filter.
            # Balanced subsamples equalize cohort influence for diagnostics;
            # unbalanced subsamples preserve the natural cohort proportions.
            ix = (
                _balanced_idx(cohort, n_subsample, rng)
                if balanced
                else np.sort(rng.choice(len(Mdf), size=n_subsample, replace=False))
            )
            Mdf = Mdf.iloc[ix]
            covariates = covariates[ix]
            cohort = cohort[ix]
        covariate_metadata = {
            "mode": mode,
            "missingness": self.config.covariate_missingness,
            "names": list(covariate_names),
            "age_spline_knots": int(self.config.age_spline_knots),
            "include_cohort_covariates": bool(self.config.include_cohort_covariates),
            "transform": design_metadata,
            "design_sha256": hashlib.sha256(
                np.ascontiguousarray(covariates, dtype="float64").tobytes()
            ).hexdigest(),
        }
        homes = [self.home.get(item, "") for item in items]
        families = {item: str(self.meta.loc[item, "likelihood_family"]) for item in items}
        signs = {item: int(self.meta.loc[item, "item_sign"]) for item in items}
        return CoreData(
            M=Mdf.to_numpy("float64"),
            covariates=covariates,
            covariate_names=covariate_names,
            items=items,
            home=homes,
            factor_cols=factor_cols,
            spec_factors=spec_factors,
            g_col=factor_cols.index(G_KEY),
            cohort=cohort,
            index=Mdf.index,
            families=families,
            signs=signs,
            moments=moments,
            copula=copula,
            covariate_metadata=covariate_metadata,
        )

    def mixed(
        self,
        factors: list[str] | None = None,
        *,
        explicit_factors: list[str] | None = None,
        specific_cross: bool = False,
        min_obs: int = 1500,
        min_cohorts: int = 3,
        balanced: bool = False,
        n_subsample: int | None = None,
        seed: int = 20260605,
        cohort_subset: list[str] | None = None,
    ) -> MixedData:
        """Return exhaustively routed hybrid mixed-likelihood inputs.

        Native items are selected before factors are partitioned.  The explicit
        factor set is then the union of the free loading columns touched by those
        items; all remaining factors are marginalized.  ``explicit_factors`` is
        therefore only an optional assertion supplied by a stage recipe.
        """
        factors = factors or list(S3_FACTORS)
        base = self.core(
            factors,
            correlated=True,
            windows=True,
            balanced=balanced,
            n_subsample=n_subsample,
            seed=seed,
            cohort_subset=cohort_subset,
        )
        matrix = self.matrix
        full = pd.read_parquet(self.config.processed_dir / "baseline_v0.parquet")
        cohort_full = np.asarray(full.index.get_level_values("cohort"))
        copula_mode = self.config.likelihood_mode == "gaussian_copula"
        routing_report: list[dict[str, Any]] = []
        candidates: list[str] = []
        for item, home in sorted(self.home.items()):
            if home not in base.factor_cols or item not in self.meta.index or item not in full.columns:
                continue
            if self.meta.loc[item, "modeling_block"] != "explicit":
                continue
            covered = self._covered(
                full[item], cohort_full, min_obs=min_obs, min_cohorts=min_cohorts
            )
            promoted = bool(copula_mode and self._gaussianizable(item, full))
            if covered and not promoted:
                route, reason = "native", "retained_native"
                candidates.append(item)
            elif promoted:
                route, reason = "gaussian", "promoted_gaussian_copula"
            else:
                route, reason = "excluded", "coverage_gate"
            routing_report.append(
                {
                    "item": item,
                    "home": home,
                    "family": str(self.meta.loc[item, "likelihood_family"]),
                    "route": route,
                    "reason": reason,
                    "n_observed": int(pd.to_numeric(full[item], errors="coerce").notna().sum()),
                }
            )

        overlap = set(base.items) & set(candidates)
        if overlap:
            raise ValueError(f"indicator routing overlap: {sorted(overlap)}")
        reported = {row["item"] for row in routing_report}
        for item in base.items:
            if item not in reported:
                routing_report.append(
                    {
                        "item": item,
                        "home": self.home.get(item, ""),
                        "family": str(self.meta.loc[item, "likelihood_family"]),
                        "route": "gaussian",
                        "reason": "continuous_gaussian_copula",
                        "n_observed": int(
                            pd.to_numeric(full[item], errors="coerce").notna().sum()
                        ),
                    }
                )

        allowed_native_types = {"primary", "g_anchor", "plausible_cross"}
        cell_rows = {
            (str(r.item), str(r.factor)): r
            for r in matrix.itertuples()
        }
        support: dict[str, list[str]] = {}
        for item in candidates:
            home = self.home[item]
            home_row = cell_rows.get((item, home))
            if home_row is None or str(home_row.prior_type) not in {"primary", "g_anchor"}:
                raise ValueError(f"native item {item!r} has no positive home/G anchor prior")
            free = [home]
            if home != G_KEY:
                g_row = cell_rows.get((item, G_KEY))
                if g_row is not None and str(g_row.prior_type) in allowed_native_types:
                    free.append(G_KEY)
            unsupported = []
            item_rows = matrix[matrix["item"] == item]
            for row in item_rows.itertuples():
                factor = str(row.factor)
                ptype = str(row.prior_type)
                if factor in base.factor_cols and factor not in free and ptype in allowed_native_types:
                    unsupported.append((factor, ptype))
            # A plausible-cross row is free only in an ESEM stage that explicitly
            # enables specific cross-loadings.  Simple-structure stages correctly
            # keep the same prior-map cell at zero.  Native ESEM cross-loadings
            # are not yet parameterized by the mixed likelihood, so fail closed
            # rather than silently dropping one when they are requested.
            if unsupported and specific_cross:
                raise ValueError(
                    f"native item {item!r} has unsupported free cross-loadings: {unsupported}"
                )
            support[item] = free

        explicit_set = {factor for free in support.values() for factor in free}
        e_cols = [i for i, factor in enumerate(base.factor_cols) if factor in explicit_set]
        m_cols = [i for i, factor in enumerate(base.factor_cols) if factor not in explicit_set]
        derived_explicit = [base.factor_cols[i] for i in e_cols]
        requested = [factor for factor in (explicit_factors or []) if factor in base.factor_cols]
        if requested and set(requested) != set(derived_explicit):
            raise ValueError(
                "stage explicit_factors does not match native routing closure: "
                f"requested={requested}, derived={derived_explicit}"
            )
        if candidates and G_KEY not in explicit_set:
            raise ValueError("native routing did not retain the general factor")
        e_idx = {base.factor_cols[c]: i for i, c in enumerate(e_cols)}

        for row in routing_report:
            if row["item"] in support:
                row["free_factors"] = support[row["item"]]

        stage = full.loc[base.index]
        families = {item: str(self.meta.loc[item, "likelihood_family"]) for item in candidates}
        bin_items = sorted(item for item in candidates if families[item] == "bernoulli")
        ord_items = sorted(item for item in candidates if families[item] == "ordered_logistic")
        cnt_items = sorted(item for item in candidates if families[item] == "neg_binomial")

        def grab(cols: list[str]) -> np.ndarray:
            return pd.DataFrame(
                {col: pd.to_numeric(stage[col], errors="coerce") for col in cols}, index=stage.index
            ).to_numpy("float64")

        Bin = grab(bin_items)
        Cnt = grab(cnt_items)
        Ord = grab(ord_items)
        ord_K: list[int] = []
        for k in range(Ord.shape[1]):
            # Ordered-logistic likelihoods expect categories 0..K-1.  The raw
            # harmonized codes may be 1/2/3 or other clinical codes, so we remap
            # observed categories to adjacent integers while preserving order.
            col = Ord[:, k]
            obs = np.isfinite(col)
            unique = np.unique(col[obs])
            mapping = {value: i for i, value in enumerate(unique)}
            for value, code in mapping.items():
                col[col == value] = code
            Ord[:, k] = col
            ord_K.append(max(2, len(unique)))

        ng_home: dict[str, int] = {}
        ng_hp: dict[str, tuple[float, float]] = {}
        ng_gp: dict[str, tuple[float, float]] = {}
        for item in bin_items + ord_items + cnt_items:
            home = self.home[item]
            ng_home[item] = e_idx[home]
            home_row = cell_rows[(item, home)]
            ng_hp[item] = (float(home_row.prior_mean), float(home_row.prior_sd))
            if home != G_KEY and G_KEY in support[item]:
                g_row = cell_rows[(item, G_KEY)]
                ng_gp[item] = (float(g_row.prior_mean), float(g_row.prior_sd))
        native_order = bin_items + cnt_items + ord_items
        ng_index = {item: i for i, item in enumerate(native_order)}
        return MixedData(
            base,
            e_cols,
            m_cols,
            bin_items,
            Bin,
            ord_items,
            Ord,
            ord_K,
            cnt_items,
            Cnt,
            ng_home,
            ng_hp,
            ng_gp,
            ng_index,
            routing_report,
        )

    def loading_spec(
        self,
        core: CoreData,
        *,
        windows: bool,
        flat: bool = False,
        specific_cross: bool = False,
        cross_sd_scale: float = 0.25,
        bifactor_g_sd: dict[str, float] | None = None,
    ) -> LoadingSpec:
        """Resolve a theory-faithful loading spec for an encoded core block."""
        spec = LoadingSpec.from_core(
            core,
            self.matrix,
            windows=windows,
            soft_unlikely=self.config.soft_unlikely,
            soft_g_anchor_specific=self.config.soft_g_anchor_specific,
            specific_cross=specific_cross,
            horseshoe=self.config.cross_loading_prior == "horseshoe",
            cross_sd_scale=cross_sd_scale,
            bifactor_g_sd=bifactor_g_sd,
            flat=flat,
        )
        free = [(j, c) for j, c, _mu, _sd in spec.pos_cells]
        free += [(j, c) for j, c, _mu, _sd in spec.signed_cells]
        free += list(spec.hs_cells)
        if len(free) != len(set(free)):
            raise ValueError("loading topology assigns more than one parameter to a cell")
        col = {factor: c for c, factor in enumerate(core.factor_cols)}
        positive = {(j, c) for j, c, _mu, _sd in spec.pos_cells}
        for j, home in enumerate(core.home):
            if home in col and (j, col[home]) not in positive:
                raise ValueError(
                    f"indicator {core.items[j]!r} lacks a positive home loading on {home!r}"
                )
        return spec

    def conceptual_loading_map(
        self, mixed: MixedData, spec: LoadingSpec
    ) -> list[dict[str, Any]]:
        """Return the auditable full retained-item by active-factor topology."""
        base = mixed.base
        pos = {
            (j, c): (k, mu, sd)
            for k, (j, c, mu, sd) in enumerate(spec.pos_cells)
        }
        signed = {
            (j, c): (k, mu, sd)
            for k, (j, c, mu, sd) in enumerate(spec.signed_cells)
        }
        hs = {(j, c): k for k, (j, c) in enumerate(spec.hs_cells)}
        gaussian_row = {item: j for j, item in enumerate(base.items)}
        route = {row["item"]: row for row in mixed.routing_report}
        matrix_rows = {
            (str(row.item), str(row.factor)): row for row in self.matrix.itertuples()
        }
        retained = list(base.items) + list(mixed.native_items)
        cells: list[dict[str, Any]] = []
        for item in retained:
            home = self.home.get(item, "")
            block = "gaussian" if item in gaussian_row else "native"
            free_native = set(route.get(item, {}).get("free_factors", []))
            for c, factor in enumerate(base.factor_cols):
                matrix_row = matrix_rows.get((item, factor))
                prior_type = (
                    str(matrix_row.prior_type) if matrix_row is not None else "unlikely_cross"
                )
                prior_mean = float(matrix_row.prior_mean) if matrix_row is not None else 0.0
                prior_sd = float(matrix_row.prior_sd) if matrix_row is not None else 0.05
                sign = (
                    str(matrix_row.sign_constraint) if matrix_row is not None else "signed"
                )
                parameter = None
                topology_role = "structural_zero"
                if block == "gaussian":
                    key = (gaussian_row[item], c)
                    if key in pos:
                        parameter = f"lam_pos[{pos[key][0]}]"
                        topology_role = spec.kind[key]
                    elif key in signed:
                        parameter = f"lam_cross[{signed[key][0]}]"
                        topology_role = spec.kind[key]
                    elif key in hs:
                        parameter = f"lam_hs[{hs[key]}]"
                        topology_role = spec.kind[key]
                elif factor in free_native:
                    if factor == home:
                        parameter = f"lh_{item}"
                        topology_role = "g_anchor" if factor == G_KEY else "primary"
                    elif factor == G_KEY:
                        parameter = f"lg_{item}"
                        topology_role = "bifactor_G"
                cells.append(
                    {
                        "item": item,
                        "block": block,
                        "family": str(self.meta.loc[item, "likelihood_family"]),
                        "home": home,
                        "factor": factor,
                        "parameter": parameter,
                        "topology_role": topology_role,
                        "prior_type": prior_type,
                        "prior_mean": prior_mean,
                        "prior_sd": prior_sd,
                        "sign_constraint": sign,
                        "structural_zero_reason": None if parameter else prior_type,
                    }
                )
        return cells

    def residualize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Public FWL covariate-residualization of gaussian-scale columns (delegates to
        :meth:`_residualize_on_covariates`).  Used by the parallel variational GLLVM engine
        so its covariate adjustment is byte-identical to this engine's; behavior-preserving,
        no cache/model impact."""
        return self._residualize_on_covariates(df)

    def _gaussianizable(self, item: str, baseline: pd.DataFrame) -> bool:
        """Copula-mode tiering. Continuous families are always Gaussianized (rank-INT). An
        ordinal/count item is promoted into the Gaussianized (marginalized) block iff it is
        high-cardinality and not point-mass-dominated; binary and low-cardinality ordinal stay
        in their native discrete likelihood."""
        fam = str(self.meta.loc[item, "likelihood_family"])
        if fam in CONTINUOUS_FAMILIES:
            return True
        if fam in ("ordered_logistic", "neg_binomial") and item in baseline.columns:
            v = pd.to_numeric(baseline[item], errors="coerce").to_numpy("float64")
            v = v[np.isfinite(v)]
            if v.size == 0:
                return False
            _vals, counts = np.unique(v, return_counts=True)
            return bool(
                _vals.size >= self.config.copula_min_distinct
                and counts.max() / v.size < self.config.copula_max_modal_frac
            )
        return False

    def _cell_type(self, item: str, factor: str) -> str:
        rows = self.matrix[(self.matrix.item == item) & (self.matrix.factor == factor)]
        return "" if rows.empty else str(rows.iloc[0]["prior_type"])

    def _residualize_on_covariates(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """OLS-partial each (pre-z-score) item on the covariate design over its
        observed rows; item NaNs are preserved.  For a Gaussian/log-Gaussian item
        this is Frisch-Waugh-Lovell-equivalent to the in-likelihood item-mean
        covariate adjustment, so Lambda/Phi are unchanged but no per-item alpha/beta
        is sampled (mirrors ``measurement.kernel._residualize_on_covariates``)."""
        design, _names = self._covariate_design(raw_df.index)
        A = np.column_stack([np.ones((len(raw_df), 1), dtype="float64"), design])
        out = raw_df.copy()
        min_obs = A.shape[1] + 2
        for col in out.columns:
            y = out[col].to_numpy("float64").copy()
            obs = np.isfinite(y)
            if int(obs.sum()) < min_obs:
                continue
            beta, *_ = np.linalg.lstsq(A[obs], y[obs], rcond=None)
            y[obs] = y[obs] - A[obs] @ beta
            out[col] = y
        return out

    def _covariate_design(
        self,
        index: pd.Index,
        *,
        with_metadata: bool = False,
    ) -> tuple[np.ndarray, list[str]] | tuple[np.ndarray, list[str], dict[str, Any]]:
        cov_path = self.config.processed_dir / "covariates_v0.parquet"
        site_path = self.config.processed_dir / "site_v0.parquet"
        cov = pd.read_parquet(cov_path).reindex(index) if cov_path.exists() else pd.DataFrame(index=index)
        blocks: list[np.ndarray] = []
        names: list[str] = []
        n = len(index)
        metadata: dict[str, Any] = {
            "source": {
                "covariates": str(cov_path),
                "site": str(site_path),
            }
        }

        if self.config.covariate_missingness != "mean_indicator":
            raise ValueError(
                "covariate_missingness must be 'mean_indicator' for the corrected M1 model"
            )

        def numeric_col(name: str) -> tuple[np.ndarray, np.ndarray, float]:
            if name in cov.columns:
                values = pd.to_numeric(cov[name], errors="coerce").to_numpy("float64")
            else:
                values = np.full(n, np.nan, dtype="float64")
            missing = ~np.isfinite(values)
            finite = values[np.isfinite(values)]
            mean = float(finite.mean()) if finite.size else 0.0
            filled = np.nan_to_num(values, nan=mean).reshape(-1, 1)
            return filled, missing.astype("float64").reshape(-1, 1), mean

        age, age_missing, age_fill = numeric_col("age")
        sex, sex_missing, sex_fill = numeric_col("sex")
        education_name = "edulevel" if "edulevel" in cov.columns else "education_years"
        education, education_missing, education_fill = numeric_col(education_name)
        spline = SplineTransformer(
            n_knots=self.config.age_spline_knots,
            degree=3,
            include_bias=False,
        )
        age_basis_raw = spline.fit_transform(age)
        age_center = np.mean(age_basis_raw, axis=0)
        age_scale = np.std(age_basis_raw, axis=0)
        age_scale[age_scale <= 0] = 1.0
        age_basis = (age_basis_raw - age_center) / age_scale
        education_center = np.mean(education, axis=0)
        education_scale = np.std(education, axis=0)
        education_scale[education_scale <= 0] = 1.0
        education = (education - education_center) / education_scale
        blocks.extend([age_basis, sex, education, age_basis * sex])
        names.extend([f"age_spline_{i}" for i in range(age_basis.shape[1])])
        names.extend(["sex", education_name])
        names.extend([f"age_spline_{i}:sex" for i in range(age_basis.shape[1])])
        for name, missing in (
            ("age_missing", age_missing),
            ("sex_missing", sex_missing),
            (f"{education_name}_missing", education_missing),
        ):
            if missing.any():
                blocks.append(missing)
                names.append(name)
        metadata["numeric"] = {
            "age": {"fill": age_fill},
            "sex": {"fill": sex_fill},
            education_name: {
                "fill": education_fill,
                "center": education_center.tolist(),
                "scale": education_scale.tolist(),
            },
        }
        metadata["age_spline"] = {
            "degree": 3,
            "n_knots": int(self.config.age_spline_knots),
            "include_bias": False,
            "knot_vector": np.asarray(spline.bsplines_[0].t).tolist(),
            "center": age_center.tolist(),
            "scale": age_scale.tolist(),
        }
        metadata["interactions"] = ["age_spline:sex"]
        if site_path.exists():
            site_raw = pd.to_numeric(
                pd.read_parquet(site_path)["siteid_city"].reindex(index), errors="coerce"
            )
            site_missing = site_raw.isna().to_numpy("float64").reshape(-1, 1)
            site = site_raw.round().astype("Int64")
            dummies = pd.get_dummies(site.astype("object"), prefix="site", dummy_na=False, drop_first=True)
            if dummies.shape[1]:
                blocks.append(dummies.to_numpy("float64"))
                names.extend(list(dummies.columns))
            if site_missing.any():
                blocks.append(site_missing)
                names.append("site_missing")
            site_levels = sorted(int(v) for v in site.dropna().unique())
            metadata["site"] = {
                "levels": site_levels,
                "reference": site_levels[0] if site_levels else None,
                "dummy_columns": list(dummies.columns),
                "missing_indicator": bool(site_missing.any()),
            }
        if self.config.include_cohort_covariates:
            cohort = pd.Series(index.get_level_values("cohort"), index=index)
            dummies = pd.get_dummies(cohort, prefix="cohort", drop_first=True)
            if dummies.shape[1]:
                blocks.append(dummies.to_numpy("float64"))
                names.extend(list(dummies.columns))
            cohort_levels = sorted(str(v) for v in cohort.unique())
            metadata["cohort"] = {
                "levels": cohort_levels,
                "reference": cohort_levels[0] if cohort_levels else None,
                "dummy_columns": list(dummies.columns),
            }
        if not blocks:
            result = (np.zeros((n, 0), dtype="float64"), [])
        else:
            result = (np.column_stack(blocks).astype("float64"), names)
        if with_metadata:
            return result[0], result[1], metadata
        return result

    @staticmethod
    def _covered(series: pd.Series, cohort: np.ndarray, *, min_obs: int, min_cohorts: int) -> bool:
        values = pd.to_numeric(series, errors="coerce")
        cohorts = ("bp", "sz", "dr")
        return bool(
            values.notna().sum() >= min_obs
            and sum((values[cohort == c].notna().sum()) > 0 for c in cohorts) >= min_cohorts
        )


class BayesianBifactorESEM:
    """PyMC builders and numerical utilities for the OOP measurement model.

    This class is where the mathematical model becomes a probabilistic program.
    PyMC variables define priors; ``pm.Potential`` injects the custom observed
    likelihood; ``pm.sample`` later uses gradients of the resulting posterior.
    """

    def __init__(self, config: MeasurementConfig | None = None):
        self.config = config or MeasurementConfig()

    def build_marginalized(
        self,
        core: CoreData,
        spec: LoadingSpec,
        *,
        correlated: bool,
        g_correlated: bool = False,
        weights: np.ndarray | None = None,
    ) -> pm.Model:
        """Build the marginalized Gaussian observed-data likelihood.

        This is the continuous-core factor model:

            x_i ~ Normal(mu_i, Lambda Phi Lambda' + Psi)

        after integrating out the patient latent factors.  The patient scores
        are not sampled here.  Instead, only structural parameters such as
        Lambda, Phi, sigma, alpha, and beta are sampled.  That is why the model
        is much smaller and faster than an explicit 9000-patient latent model.
        """
        M = core.M
        N, J = M.shape
        F = len(core.factor_cols)
        # ``mask`` is the entire missingness-aware trick in miniature.  Missing
        # cells are represented by 0 in ``x`` only so tensor arithmetic has a
        # finite value.  The mask then removes those cells from every likelihood
        # term.  This is not mean imputation, zero imputation, or single
        # imputation of any kind.
        mask = np.isfinite(M).astype("float64")
        x = np.nan_to_num(M, nan=0.0)
        kobs = mask.sum(1)
        pat_mask, pat_inv = self.patterns(mask)
        covariates = core.covariates
        P = covariates.shape[1]
        with pm.Model() as model:
            # Lambda is the loading matrix: rows are indicators, columns are
            # dimensions.  A loading answers "how strongly does this item move
            # when this latent dimension increases?"
            Lam = self._build_loadings(spec, J, F)
            pm.Deterministic("Lam", Lam)
            # Phi is the factor correlation matrix.  In the primary bifactor
            # version G is orthogonal to all specifics; only specifics correlate
            # with each other.
            Phi, R = self._build_phi(core, correlated=correlated, g_correlated=g_correlated)
            pm.Deterministic("Phi", Phi)
            # Alpha is each item's baseline expected value on the z-scored scale.
            # Beta is item-specific covariate calibration: age/site/etc can shift
            # the mean of an item without becoming a latent psychiatric dimension.
            # Item-mean terms are only sampled in ``in_likelihood`` covariate mode
            # (P > 0).  In the default ``residualize``/``none`` modes the data is
            # already zero-mean, so no alpha/beta enters the sampler — matching the
            # certified zero-mean Woodbury and removing J + J*P nuisance parameters.
            if P:
                alpha = pm.Normal("alpha", 0.0, 1.5, shape=J)
                beta = pm.Normal("beta", 0.0, 0.25, shape=(J, P))
                mu = alpha[None, :] + pt.as_tensor(covariates) @ beta.T
            else:
                mu = None
            sigma = self.config.psi_floor + pm.HalfNormal("sigma", 1.0, shape=J)
            # Reparameterize Lambda Phi Lambda' as (Lambda chol(Phi)) (.)'.
            # This lets the same Woodbury kernel handle independent and
            # correlated factors.
            Lt = Lam @ R
            residual = pt.as_tensor(x) if mu is None else pt.as_tensor(x) - mu
            ll = self.woodbury_potential(
                residual,
                mask,
                Lt,
                sigma**2,
                pat_mask,
                pat_inv,
                kobs,
                F,
            )
            total = (pt.as_tensor(weights) * ll).sum() if weights is not None else ll.sum()
            # The custom likelihood is added as a Potential because it is not a
            # built-in "observed RV"; it is the sum of per-patient observed-cell
            # log densities computed by the Woodbury identity.
            pm.Potential("obs_ll", total)
        return model

    def build_mixed(
        self,
        mixed: MixedData,
        spec: LoadingSpec,
        *,
        hurdle_counts: bool = False,
        g_correlated: bool = False,
        weights: np.ndarray | None = None,
    ) -> pm.Model:
        """Build the hybrid explicit/marginalized mixed-likelihood model.

        The mixed model keeps two ideas in one posterior:

        * Continuous items still use the fast marginalized Gaussian likelihood.
        * Binary/ordinal/count items attach to explicit patient latents ``f_e``.

        Mathematically, the continuous-only factors ``f_m`` are integrated out
        conditional on the explicit factors ``f_e``.  That conditional Gaussian
        decomposition is what the Phi_ee / Phi_mm / Phi_me block below builds.
        """
        base = mixed.base
        M = base.M
        N, J = M.shape
        F = len(base.factor_cols)
        # Per-patient likelihood weights (cohort-weighted §3.6 fit): when set, every term is
        # weighted by w_i so each cohort contributes equally (transdiagnostic estimand) using all
        # patients.  None -> the standard unweighted model (byte-identical native path).
        w = None if weights is None else np.asarray(weights, dtype="float64")
        Ke, Km = len(mixed.e_cols), len(mixed.m_cols)
        mask = np.isfinite(M).astype("float64")
        x = np.nan_to_num(M, nan=0.0)
        kobs = mask.sum(1)
        pat_mask, pat_inv = self.patterns(mask)
        Se = _selection(mixed.e_cols, F)
        Sm = _selection(mixed.m_cols, F)
        P = base.covariates.shape[1]
        with pm.Model() as model:
            Lam = self._build_loadings(spec, J, F)
            pm.Deterministic("Lam", Lam)
            Phi, _R = self._build_phi(
                base, correlated=True, g_correlated=g_correlated
            )
            pm.Deterministic("Phi", Phi)
            Se_t = pt.as_tensor(Se)
            Sm_t = pt.as_tensor(Sm)
            # Partition Phi into explicit and marginalized blocks.
            #
            #   Phi_ee: covariance among explicit factors
            #   Phi_mm: covariance among marginalized factors
            #   Phi_me: cross-covariance between them
            #
            # From multivariate-normal conditioning:
            #   f_m | f_e ~ Normal(Mmat f_e, Sres)
            #
            # The continuous likelihood then sees a mean contribution from f_e
            # and a residual low-rank covariance from the marginalized f_m.
            Phi_ee = Se_t @ Phi @ Se_t.T
            Phi_mm = Sm_t @ Phi @ Sm_t.T
            Phi_me = Sm_t @ Phi @ Se_t.T
            Mmat = pt.linalg.solve(Phi_ee, Phi_me.T).T
            Sres = Phi_mm - Mmat @ Phi_me.T
            C_S = pt.linalg.cholesky(Sres + 1e-8 * pt.eye(Km))
            L_ee = pt.linalg.cholesky(Phi_ee + 1e-8 * pt.eye(Ke))
            Lam_e = Lam @ Se_t.T
            Lam_m = Lam @ Sm_t.T
            Bmat = Lam_e + Lam_m @ Mmat
            Lt = Lam_m @ C_S
            # z_e is a non-centered parameterization.  NUTS samples standard
            # Normal coordinates, then we multiply by the Cholesky factor to get
            # correlated explicit latents with covariance Phi_ee.
            z = pm.Normal("z_e", 0.0, 1.0, shape=(N, Ke))
            f_e = pm.Deterministic("f_e", z @ L_ee.T)

            # Item-mean covariate terms only in ``in_likelihood`` mode (P > 0); the
            # default residualized path keeps the continuous block zero-mean.
            if P:
                alpha = pm.Normal("alpha", 0.0, 1.5, shape=J)
                beta = pm.Normal("beta", 0.0, 0.25, shape=(J, P))
                mu = alpha[None, :] + pt.as_tensor(base.covariates) @ beta.T
            else:
                mu = None
            if P:
                beta_native = pm.Normal(
                    "beta_native", 0.0, 0.25, shape=(len(mixed.native_items), P)
                )
                covariates_t = pt.as_tensor(base.covariates)
            else:
                beta_native = None
                covariates_t = None
            sigma = self.config.psi_floor + pm.HalfNormal("sigma", 1.0, shape=J)
            base_resid = pt.as_tensor(x) if mu is None else pt.as_tensor(x) - mu
            residual = base_resid - f_e @ Bmat.T
            ll = self.woodbury_potential(residual, mask, Lt, sigma**2, pat_mask, pat_inv, kobs, Km)
            pm.Potential("cont_ll", ll.sum() if w is None else (pt.as_tensor(w) * ll).sum())

            for k, item in enumerate(mixed.bin_items):
                # Bernoulli-logit item: probability of endorsement is a logistic
                # function of the patient's explicit home-factor coordinate and
                # their general burden coordinate G.
                y = mixed.Bin[:, k]
                obs = np.flatnonzero(np.isfinite(y))
                a = pm.Normal(f"a_{item}", 0.0, 1.5)
                lh = pm.TruncatedNormal(
                    f"lh_{item}",
                    mu=mixed.ng_hp[item][0],
                    sigma=mixed.ng_hp[item][1],
                    lower=0.0,
                )
                eta = a + lh * f_e[:, mixed.ng_home[item]][obs]
                if item in mixed.ng_gp:
                    lg = pm.Normal(
                        f"lg_{item}", mixed.ng_gp[item][0], mixed.ng_gp[item][1]
                    )
                    eta = eta + lg * f_e[:, 0][obs]
                if beta_native is not None and covariates_t is not None:
                    eta = eta + covariates_t[obs] @ beta_native[mixed.ng_index[item]]
                yv = y[obs].astype("int8")
                if w is None:
                    pm.Bernoulli(f"y_{item}", logit_p=eta, observed=yv)
                else:
                    lp = pm.logp(pm.Bernoulli.dist(logit_p=eta), pt.as_tensor(yv))
                    pm.Potential(f"y_{item}", (pt.as_tensor(w[obs]) * lp).sum())
            for k, item in enumerate(mixed.cnt_items):
                # Count item: use negative binomial so the variance can exceed
                # the mean, which is typical for clinical count variables.
                y = mixed.Cnt[:, k]
                obs = np.flatnonzero(np.isfinite(y))
                a = pm.Normal(f"a_{item}", 0.0, 1.5)
                lh = pm.TruncatedNormal(
                    f"lh_{item}",
                    mu=mixed.ng_hp[item][0],
                    sigma=mixed.ng_hp[item][1],
                    lower=0.0,
                )
                alpha = pm.HalfNormal(f"alpha_{item}", 2.0)
                eta = a + lh * f_e[:, mixed.ng_home[item]][obs]
                if item in mixed.ng_gp:
                    lg = pm.Normal(
                        f"lg_{item}", mixed.ng_gp[item][0], mixed.ng_gp[item][1]
                    )
                    eta = eta + lg * f_e[:, 0][obs]
                if beta_native is not None and covariates_t is not None:
                    eta = eta + covariates_t[obs] @ beta_native[mixed.ng_index[item]]
                if hurdle_counts:
                    psi = pm.Deterministic(f"psi_{item}", pm.math.sigmoid(pm.Normal(f"apsi_{item}", 0.0, 1.5)))
                    hl = _hurdle_nb_logp(pt.as_tensor(np.rint(y[obs]).astype("float64")), psi, pt.exp(eta), alpha)
                    pm.Potential(f"y_{item}", hl.sum() if w is None else (pt.as_tensor(w[obs]) * hl).sum())
                else:
                    yv = np.rint(y[obs]).astype("int64")
                    if w is None:
                        pm.NegativeBinomial(f"y_{item}", mu=pt.exp(eta), alpha=alpha, observed=yv)
                    else:
                        lp = pm.logp(pm.NegativeBinomial.dist(mu=pt.exp(eta), alpha=alpha), pt.as_tensor(yv))
                        pm.Potential(f"y_{item}", (pt.as_tensor(w[obs]) * lp).sum())
            for k, item in enumerate(mixed.ord_items):
                # Ordinal item: ordered-logistic thresholds map the continuous
                # latent predictor to ordered clinical categories.
                y = mixed.Ord[:, k]
                obs = np.flatnonzero(np.isfinite(y))
                K = int(mixed.ord_K[k])
                cut = pm.Normal(
                    f"c_{item}",
                    mu=np.linspace(-1.5, 1.5, K - 1),
                    sigma=2.0,
                    shape=K - 1,
                    transform=pm.distributions.transforms.ordered,
                )
                lh = pm.TruncatedNormal(
                    f"lh_{item}",
                    mu=mixed.ng_hp[item][0],
                    sigma=mixed.ng_hp[item][1],
                    lower=0.0,
                )
                eta = lh * f_e[:, mixed.ng_home[item]][obs]
                if item in mixed.ng_gp:
                    lg = pm.Normal(
                        f"lg_{item}", mixed.ng_gp[item][0], mixed.ng_gp[item][1]
                    )
                    eta = eta + lg * f_e[:, 0][obs]
                if beta_native is not None and covariates_t is not None:
                    eta = eta + covariates_t[obs] @ beta_native[mixed.ng_index[item]]
                yv = y[obs].astype("int32")
                if w is None:
                    pm.OrderedLogistic(f"y_{item}", eta=eta, cutpoints=cut, observed=yv, compute_p=False)
                else:
                    lp = pm.logp(pm.OrderedLogistic.dist(eta=eta, cutpoints=cut, compute_p=False), pt.as_tensor(yv))
                    pm.Potential(f"y_{item}", (pt.as_tensor(w[obs]) * lp).sum())
        return model

    def _build_loadings(self, spec: LoadingSpec, J: int, F: int):
        # We assemble Lambda from sparse cell lists instead of sampling a dense
        # J x F matrix.  Cells not present in ``pos_cells`` or ``signed_cells``
        # are exact zeros.  This keeps the model sparse and makes the prior
        # matrix auditable: every free loading exists for a named theoretical
        # reason.
        pr, pc, pmu, psd = _cell_arrays(spec.pos_cells)
        sr, sc, smu, ssd = _cell_arrays(spec.signed_cells)
        Lam = pt.zeros((J, F))
        if len(pr):
            # Positive home loadings orient the factors.  Without this, the same
            # factor could be multiplied by -1 and all its loadings flipped, with
            # the exact same likelihood.
            equal_groups = self._equal_pos_groups(spec)
            if equal_groups:
                grouped = {index for indices in equal_groups.values() for index in indices}
                free = np.array(
                    [index for index in range(len(pr)) if index not in grouped],
                    dtype="int64",
                )
                values = pt.zeros(len(pr))
                if len(free):
                    free_values = pm.TruncatedNormal(
                        "lam_pos_free",
                        mu=pmu[free],
                        sigma=psd[free],
                        lower=0.0,
                        shape=len(free),
                    )
                    values = pt.set_subtensor(values[free], free_values)
                for factor, indices in equal_groups.items():
                    index = np.asarray(indices, dtype="int64")
                    shared = pm.TruncatedNormal(
                        f"lam_equal_{factor}",
                        mu=float(np.mean(pmu[index])),
                        sigma=float(np.mean(psd[index])),
                        lower=0.0,
                    )
                    values = pt.set_subtensor(values[index], shared)
                # Preserve the expanded item-by-loading representation used by
                # the loading atlas and elementwise diagnostics.  Repeated
                # entries are the same sampled parameter by construction.
                values = pm.Deterministic("lam_pos", values)
            else:
                values = pm.TruncatedNormal(
                    "lam_pos", mu=pmu, sigma=psd, lower=0.0, shape=len(pr)
                )
            Lam = pt.set_subtensor(Lam[pr, pc], values)
        if len(sr):
            # Cross-loadings are signed because a broad instrument can relate to
            # another axis in either direction after burden orientation.
            values = pm.Normal("lam_cross", mu=smu, sigma=ssd, shape=len(sr))
            Lam = pt.set_subtensor(Lam[sr, sc], values)
        if spec.hs_cells:
            # Regularized ("Finnish") horseshoe on the off-home specific cross-loadings.  A spike at 0
            # (global tau x local eta, both shrinking) drives the many noise cross-loadings to ~0 -- so a
            # thin factor's column keeps its identity instead of being diluted -- while the heavy local
            # tails let a GENUINELY supported small cross-loading escape.  The slab c regularizes that
            # escape (caps |lambda| ~ c) and tames the funnel so NUTS mixes.  Non-centered (z separate
            # from the scales).  See docs/HORSESHOE_ESEM.md.
            hr = np.array([j for j, _c in spec.hs_cells], dtype="int64")
            hc = np.array([_c for _j, _c in spec.hs_cells], dtype="int64")
            n_hs = len(hr)
            tau0 = float(self.config.hs_tau0)
            if self.config.hs_fixed_tau:
                tau = pt.constant(tau0)                                                  # no global funnel
            else:
                tau = pm.HalfNormal("hs_tau", sigma=tau0)                                # global shrinkage
            nu = float(self.config.hs_local_df)                                          # nu=1 -> HalfCauchy
            eta = pm.HalfStudentT("hs_eta", nu=nu, sigma=1.0, shape=n_hs)                 # local (heavy tail)
            z = pm.Normal("hs_z", 0.0, 1.0, shape=n_hs)
            c2 = float(self.config.hs_slab_c) ** 2
            eta_reg = pt.sqrt(c2 * eta**2 / (c2 + (tau**2) * (eta**2)))                  # regularized scale
            lam_hs = pm.Deterministic("lam_hs", z * tau * eta_reg)
            Lam = pt.set_subtensor(Lam[hr, hc], lam_hs)
        return Lam

    def _equal_pos_groups(self, spec: LoadingSpec) -> dict[str, list[int]]:
        """Expanded ``lam_pos`` indices sharing one sampled home loading."""
        groups: dict[str, list[int]] = {}
        for factor in self.config.equal_home_loading_factors:
            if factor not in spec.factor_cols:
                continue
            factor_col = spec.factor_cols.index(factor)
            indices = [
                index
                for index, (_row, col, _mu, _sd) in enumerate(spec.pos_cells)
                if col == factor_col
            ]
            if len(indices) >= 2:
                groups[factor] = indices
        return groups

    def _build_phi(self, core: CoreData, *, correlated: bool, g_correlated: bool):
        # Phi is the latent-factor correlation matrix.  It controls how dimensions
        # co-vary before we look at indicators.  In a strict bifactor model, G
        # has unit variance and zero correlation with every specific; the
        # specifics may be independent or correlated depending on the stage.
        F = len(core.factor_cols)
        if not correlated or F <= 2:
            return pt.eye(F), pt.eye(F)
        if g_correlated:
            # Construct the LKJ prior through its independent vine partial
            # correlations.  This is distributionally equivalent to LKJCorr,
            # while avoiding a PyMC 6.0.1 cross-model initializer cache defect
            # when continuation stages have different matrix dimensions.
            Phi, R = self._lkj_vine_correlation(
                f"Phi_all_{F}", n=F, eta=float(self.config.lkj_eta)
            )
            return Phi, R
        # Primary bifactor variant: only the *correlated* specific block gets a free correlation.
        # G is always orthogonal (bifactor); ``orthogonal_factors`` optionally
        # pins additional sensitivity-arm factors outside the block.  The
        # primary configuration leaves it empty, so all specifics share the LKJ.
        orthogonal = {core.g_col}
        for f in self.config.orthogonal_factors:
            if f in core.factor_cols:
                orthogonal.add(core.factor_cols.index(f))
        corr_idx = [i for i in range(F) if i not in orthogonal]
        ns = len(corr_idx)
        if ns <= 1:
            return pt.eye(F), pt.eye(F)  # nothing left to correlate -> all factors orthogonal
        Cs_rv, Rs = self._lkj_vine_correlation(
            f"Phi_spec_{ns}", n=ns, eta=float(self.config.lkj_eta)
        )
        Cs = pm.Deterministic("Phi_spec", Cs_rv)
        E = np.zeros((F, ns))
        for k, i in enumerate(corr_idx):
            E[i, k] = 1.0
        diag_orth = np.zeros((F, F))
        for o in orthogonal:
            diag_orth[o, o] = 1.0
        # Embed the correlated block; orthogonal factors (G + any pinned) get unit diagonal, zero off.
        Phi = pt.as_tensor(E) @ Cs @ pt.as_tensor(E).T + pt.as_tensor(diag_orth)
        return Phi, pt.linalg.cholesky(Phi + 1e-8 * pt.eye(F))

    @staticmethod
    def _lkj_vine_correlation(name: str, *, n: int, eta: float):
        """Return an LKJ(eta) correlation and its Cholesky factor.

        In the C-vine representation, canonical partial correlations in column
        ``j`` are independent shifted symmetric Beta variables with shape
        ``eta + (n - j - 2) / 2``.  The row-wise recursion below maps those
        unconstrained partial correlations bijectively to a positive-definite
        unit-diagonal matrix.  It is the LKJ construction of Lewandowski,
        Kurowicka, and Joe (2009), not an approximation to an LKJ prior.
        """
        if n <= 1:
            return pt.eye(n), pt.eye(n)
        partials: list[pt.TensorVariable] = []
        for j in range(n - 1):
            alpha = float(eta) + 0.5 * (n - j - 2)
            unit = pm.Beta(
                f"{name}_partial_{j}",
                alpha=alpha,
                beta=alpha,
                shape=n - j - 1,
            )
            partials.append(2.0 * unit - 1.0)

        chol = pt.zeros((n, n))
        chol = pt.set_subtensor(chol[0, 0], 1.0)
        for i in range(1, n):
            remaining = pt.as_tensor_variable(1.0)
            for j in range(i):
                partial = partials[j][i - j - 1]
                chol = pt.set_subtensor(chol[i, j], partial * remaining)
                remaining = remaining * pt.sqrt(pt.clip(1.0 - partial**2, 1e-12, 1.0))
            chol = pt.set_subtensor(chol[i, i], remaining)
        corr = chol @ chol.T
        return corr, chol

    @staticmethod
    def patterns(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unique observed-cell patterns and per-row inverse map.

        Many patients share the same missingness pattern.  The expensive
        Woodbury matrix depends only on which columns are observed, not on the
        observed values themselves.  Grouping by pattern lets us do that matrix
        work once per pattern and then gather the result for all matching rows.
        """
        patterns, inv = np.unique(mask, axis=0, return_inverse=True)
        return patterns.astype("float64"), inv.reshape(-1).astype("int64")

    @staticmethod
    def woodbury_potential(
        residual,
        mask: np.ndarray,
        Lt,
        psi,
        pat_mask: np.ndarray,
        pat_inv: np.ndarray,
        kobs: np.ndarray,
        F: int,
    ):
        """Observed-cell MVN log-likelihood via Woodbury and pattern grouping.

        The dense covariance for the continuous model is:

            Sigma = Lt Lt' + diag(psi)

        where ``Lt = Lambda chol(Phi)``.  For each patient's observed columns,
        the log density requires logdet(Sigma_obs) and
        r_obs' Sigma_obs^{-1} r_obs.  Woodbury rewrites those terms using:

            A = I + Lt_obs' diag(1/psi_obs) Lt_obs

        A is F x F, where F is the number of factors.  That is the speed win:
        invert/cholesky a small factor matrix instead of a large item matrix.
        """
        J = pat_mask.shape[1]
        # Precompute each item's low-rank contribution Lt_j Lt_j' / psi_j, then
        # sum the contributions for the observed items of each missingness
        # pattern with one matrix multiplication.
        Qf = ((Lt[:, :, None] * Lt[:, None, :]) / psi[:, None, None]).reshape((J, F * F))
        A = (pt.eye(F).reshape((1, F * F)) + pt.as_tensor(pat_mask) @ Qf).reshape(
            (pat_mask.shape[0], F, F)
        )
        Lc = pt.linalg.cholesky(A)
        # Matrix determinant lemma:
        # logdet(Sigma_obs) = logdet(Psi_obs) + logdet(A)
        logdet_a = 2.0 * pt.log(pt.diagonal(Lc, axis1=-2, axis2=-1)).sum(-1)
        logdet_psi = (pt.as_tensor(pat_mask) * pt.log(psi)[None, :]).sum(1)
        # Woodbury inverse:
        # Sigma^{-1} = Psi^{-1} - Psi^{-1} Lt A^{-1} Lt' Psi^{-1}
        # The code below computes the quadratic term without materializing the
        # large inverse matrix.
        Wr = pt.as_tensor(mask) * residual / psi[None, :]
        b = Wr @ Lt
        sol = pt.linalg.solve_triangular(Lc[pat_inv], b[:, :, None], lower=True)[:, :, 0]
        quad_a = (sol**2).sum(-1)
        term1 = (Wr * residual).sum(1)
        return -0.5 * (pt.as_tensor(kobs) * LOG2PI + logdet_psi[pat_inv] + logdet_a[pat_inv] + term1 - quad_a)

    @staticmethod
    def dense_observed_loglik(M: np.ndarray, Lam: np.ndarray, Phi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Brute-force observed-cell MVN log-density, useful as a test reference."""
        Sigma = Lam @ Phi @ Lam.T + np.diag(sigma**2)
        out = np.zeros(M.shape[0])
        for i, row in enumerate(M):
            obs = np.flatnonzero(np.isfinite(row))
            if obs.size:
                out[i] = multivariate_normal.logpdf(row[obs], mean=np.zeros(obs.size), cov=Sigma[np.ix_(obs, obs)])
        return out


class StageRunner:
    """Cached staged fitting with diagnostics.

    This is the operational layer: it turns a stage definition into a PyMC model,
    calls NUTS, writes ``idata.nc`` and ``manifest.json``, and reuses compatible
    cached fits.  The statistical model lives above; this class manages compute.
    """

    def __init__(self, config: MeasurementConfig | None = None):
        self.config = config or MeasurementConfig()
        self.dataset = MeasurementDataset(self.config)
        self.builder = BayesianBifactorESEM(self.config)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run_stage(
        self,
        stage: StageDefinition,
        *,
        overwrite: bool = False,
        prev_stage: StageDefinition | None = None,
    ):
        """Run or load a stage and return ``(idata, payload)``.

        ``idata`` is an ArviZ InferenceData object.  It contains posterior draws
        for Lambda, Phi, sigma, and any explicit latents.  ``payload`` is the
        lightweight manifest used for progress reporting and cache validation.

        ``prev_stage`` enables the continuation warm-start: if its cached fit
        exists, this stage's loadings/residuals are initialized from that
        posterior (matched by name), so each rung of the ladder starts in the
        previous rung's basin instead of cold.
        """
        out = self.config.output_dir / stage.name
        out.mkdir(parents=True, exist_ok=True)
        idata_path = out / "idata.nc"
        manifest_path = out / "manifest.json"
        if idata_path.exists() and not overwrite:
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            if (
                manifest.get("model_version") == MODEL_VERSION
                and manifest.get("stage_spec") == _stage_spec(stage)
                and _cached_model_sig(manifest) == _config_sig(self.config)
            ):
                # Certification policy does not affect posterior draws.  Recompute
                # diagnostics from the cached NetCDF so a policy correction can be
                # applied without repeating a multi-hour fit.  Model-affecting
                # fields remain protected by the cache signature above.
                idata = az.from_netcdf(str(idata_path))
                cached_mixed = None
                if stage.mixed:
                    cached_mixed = self.dataset.mixed(
                        stage.factors,
                        explicit_factors=stage.explicit_factors,
                        specific_cross=stage.specific_cross,
                        min_cohorts=stage.min_cohorts,
                        balanced=stage.balanced,
                        n_subsample=stage.n_subsample,
                        seed=stage.seed,
                    )
                    cached_core = cached_mixed.base
                else:
                    cached_core = self.dataset.core(
                        stage.factors,
                        correlated=stage.correlated,
                        windows=stage.windows,
                        balanced=stage.balanced,
                        n_subsample=stage.n_subsample,
                        seed=stage.seed,
                    )
                cached_spec = self.dataset.loading_spec(
                    cached_core,
                    windows=stage.windows,
                    specific_cross=stage.specific_cross,
                    cross_sd_scale=stage.cross_sd_scale,
                )
                diag = self.diagnostics(
                    idata,
                    core=cached_core,
                    spec=cached_spec,
                    mixed=cached_mixed,
                )
                gate_pass = self._passes_gates(diag)
                manifest["config_sig"] = _config_sig(self.config)
                manifest["certification_policy"] = _certification_policy(self.config)
                manifest["diagnostics"] = diag
                manifest["certification"] = self._certification(stage, diag, gate_pass)
                manifest["revalidated_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(
                    f"[m1-stage] REVALIDATED {stage.name}: "
                    f"passed={gate_pass} diagnostics={diag}",
                    flush=True,
                )
                if stage.enforce_gates and not gate_pass:
                    raise RuntimeError(
                        f"cached stage {stage.name!r} failed certification gates; "
                        f"see {manifest_path}"
                    )
                return idata, manifest

        start = time.time()
        if stage.mixed:
            # Mixed stages need both the continuous block and explicit
            # binary/ordinal/count arrays.  They are heavier because they sample
            # patient-level explicit latents.
            data = self.dataset.mixed(
                stage.factors,
                explicit_factors=stage.explicit_factors,
                specific_cross=stage.specific_cross,
                min_cohorts=stage.min_cohorts,
                balanced=stage.balanced,
                n_subsample=stage.n_subsample,
                seed=stage.seed,
            )
            spec = self.dataset.loading_spec(
                data.base,
                windows=stage.windows,
                specific_cross=stage.specific_cross,
                cross_sd_scale=stage.cross_sd_scale,
            )
            weights = _cohort_weights(data.base.cohort) if self.config.cohort_weighted else None
            model = self.builder.build_mixed(
                data,
                spec,
                hurdle_counts=stage.hurdle_counts,
                g_correlated=stage.g_correlated,
                weights=weights,
            )
            payload_data = data.base
        else:
            # Continuous stages are the fast path: no patient latent coordinates
            # are sampled because they are marginalized by the Woodbury kernel.
            payload_data = self.dataset.core(
                stage.factors,
                correlated=stage.correlated,
                windows=stage.windows,
                balanced=stage.balanced,
                n_subsample=stage.n_subsample,
                seed=stage.seed,
            )
            spec = self.dataset.loading_spec(
                payload_data,
                windows=stage.windows,
                specific_cross=stage.specific_cross,
                cross_sd_scale=stage.cross_sd_scale,
            )
            weights = _cohort_weights(payload_data.cohort) if self.config.cohort_weighted else None
            model = self.builder.build_marginalized(
                payload_data,
                spec,
                correlated=stage.correlated,
                g_correlated=stage.g_correlated,
                weights=weights,
            )

        initvals = self._warmstart(payload_data, spec, prev_stage)
        job_name = os.environ.get("FACE_JOB_NAME")
        heartbeat_stop = threading.Event()
        heartbeat_thread: threading.Thread | None = None
        if job_name:
            from face.io import runstate

            runstate.merge_state(
                job_name,
                stage=stage.name,
                last_heartbeat=runstate.utcnow(),
                output_dir=str(out),
            )

            def heartbeat() -> None:
                while not heartbeat_stop.wait(30.0):
                    runstate.merge_state(
                        job_name, stage=stage.name, last_heartbeat=runstate.utcnow()
                    )

            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()
        print(
            f"[m1-stage] START {stage.name}: N={payload_data.M.shape[0]} "
            f"Jg={payload_data.M.shape[1]} factors={payload_data.factor_cols}",
            flush=True,
        )
        try:
            with model:
            # NUTS is Hamiltonian Monte Carlo with automatic path length choice.
            # ``tune`` adapts the step size / mass matrix; ``draws`` are kept as
            # posterior samples.  NumPyro/JAX executes the gradient-based sampler
            # faster than the default PyMC backend on this model.  ``nuts={...}`` is
            # the non-deprecated route that forwards ``max_tree_depth`` to numpyro
            # (PyMC routes it through ``_sample_external_nuts``); ``initvals`` is the
            # continuation warm-start (None on the first rung).
                idata = pm.sample(
                    draws=stage.draws,
                    tune=stage.tune,
                    chains=stage.chains,
                    target_accept=stage.target_accept,
                    random_seed=stage.seed,
                    nuts_sampler="numpyro",
                    initvals=initvals,
                    nuts={"max_tree_depth": self.config.max_tree_depth},
                    idata_kwargs={"log_likelihood": False},
                    progressbar=True,
                )
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=2.0)
        elapsed = time.time() - start
        # Persist completed chains before any reporting code runs.  A diagnostic
        # or manifest bug must never discard a multi-hour sampling result.
        tmp_idata_path = idata_path.with_suffix(".tmp.nc")
        idata.to_netcdf(str(tmp_idata_path))
        tmp_idata_path.replace(idata_path)
        diag = self.diagnostics(
            idata,
            core=payload_data,
            spec=spec,
            mixed=data if stage.mixed else None,
        )
        payload = {
            "model_version": MODEL_VERSION,
            "stage": stage.name,
            "stage_spec": _stage_spec(stage),
            "config_sig": _config_sig(self.config),
            "certification_policy": _certification_policy(self.config),
            "elapsed_sec": elapsed,
            "N": int(payload_data.M.shape[0]),
            "J": int(payload_data.M.shape[1]),
            "factors": payload_data.factor_cols,
            "covariates": payload_data.covariate_metadata,
            "diagnostics": diag,
        }
        if stage.mixed:
            cohort_totals = None
            if weights is not None:
                cohort_totals = {
                    str(c): float(weights[np.asarray(payload_data.cohort) == c].sum())
                    for c in dict.fromkeys(payload_data.cohort)
                }
            loading_cells = self.dataset.conceptual_loading_map(data, spec)
            payload["routing"] = {
                "J_gaussian": int(payload_data.M.shape[1]),
                "J_native": int(len(data.native_items)),
                "J_retained": int(payload_data.M.shape[1] + len(data.native_items)),
                "n_binary": int(len(data.bin_items)),
                "n_ordinal": int(len(data.ord_items)),
                "n_count": int(len(data.cnt_items)),
                "explicit_factors": [payload_data.factor_cols[c] for c in data.e_cols],
                "marginalized_factors": [payload_data.factor_cols[c] for c in data.m_cols],
                "items": data.routing_report,
                "sha256": _json_sha256(data.routing_report),
            }
            payload["loading_map"] = {
                "cells": loading_cells,
                "sha256": _json_sha256(loading_cells),
            }
            payload["likelihood"] = {
                "count_family": "hurdle_negative_binomial"
                if stage.hurdle_counts
                else "negative_binomial",
                "cohort_weighting": "generalized_posterior"
                if weights is not None
                else "ordinary_posterior",
                "weight_sum": None if weights is None else float(weights.sum()),
                "cohort_weight_totals": cohort_totals,
            }
        gate_pass = self._passes_gates(diag)
        payload["certification"] = self._certification(stage, diag, gate_pass)
        manifest_path.write_text(json.dumps(payload, indent=2))
        print(
            f"[m1-stage] END {stage.name}: elapsed={elapsed:.1f}s diagnostics={diag}",
            flush=True,
        )
        if stage.enforce_gates and not gate_pass:
            raise RuntimeError(
                f"stage {stage.name!r} failed certification gates; see {manifest_path}"
            )
        return idata, payload

    def _certification(
        self, stage: StageDefinition, diagnostics: dict[str, Any], gate_pass: bool
    ) -> dict[str, Any]:
        components = self._gate_components(diagnostics)
        nuisance_over_strict = int(
            diagnostics.get("nuisance_parameters_above_map_rhat_max", 0)
        )
        warnings = []
        if nuisance_over_strict:
            warnings.append(
                f"{nuisance_over_strict} nuisance calibration parameters exceed "
                f"the strict map R-hat threshold but remain subject to the "
                f"{self.config.nuisance_rhat_max:.3f} nuisance continuation gate"
            )
        substance_count = int(diagnostics.get("substance_parameter_count", 0))
        if substance_count and components["substance_provisional"] and not components[
            "substance_strict"
        ]:
            warnings.append(
                "Substance parameters satisfy only the provisional "
                f"R-hat <= {self.config.substance_rhat_max:.3f}, ESS >= "
                f"{self.config.substance_ess_min:.0f} continuation gate; Substance "
                "loadings and correlations are not strictly certified"
            )
        return {
            "scope": "strict_core_provisional_substance_geometry_and_chain_alignment",
            "enforced": bool(stage.enforce_gates),
            "passed": bool(gate_pass) if stage.enforce_gates else None,
            "core_certified": bool(components["core_passed"]),
            "substance_provisional_passed": bool(
                components["substance_provisional"]
                and components["substance_alignment"]
            ),
            "substance_strictly_certified": bool(
                components["substance_strict"]
                and components["substance_alignment"]
            ),
            "strict_sampling_certified": bool(
                components["strict_sampling_passed"]
            ),
            "gate_components": components,
            "full_m1_certified": False,
            "warnings": warnings,
            "pending": [
                "posterior_predictive_calibration",
                "patient_level_cohort_stratified_resampling",
            ],
        }

    def _stage_core(self, stage: StageDefinition) -> CoreData:
        """Reconstruct the encoded continuous block a stage was fit on (deterministic
        given the stage recipe), used to name-match the warm-start source loadings."""
        if stage.mixed:
            return self.dataset.mixed(
                stage.factors,
                explicit_factors=stage.explicit_factors,
                specific_cross=stage.specific_cross,
                min_cohorts=stage.min_cohorts,
                balanced=stage.balanced,
                n_subsample=stage.n_subsample,
                seed=stage.seed,
            ).base
        return self.dataset.core(
            stage.factors,
            correlated=stage.correlated,
            windows=stage.windows,
            balanced=stage.balanced,
            n_subsample=stage.n_subsample,
            seed=stage.seed,
        )

    def _warmstart(
        self,
        core: CoreData,
        spec: LoadingSpec,
        prev_stage: StageDefinition | None,
    ) -> dict[str, np.ndarray] | None:
        """Continuation init values from the previous stage's cached posterior.

        The previous fit's posterior-mean ``Lam`` and ``sigma`` are read from cache
        and mapped to this stage's free cells by ``(item, factor)`` name, so the
        hand-off survives item reordering and added factors.  Cells with no source
        match start at their prior mean (0 for signed); positive home cells are
        floored at 0.02; a near-zero warm-start residual SD is clipped up to 0.1 so
        it cannot blow the Woodbury precision to NaN.  Explicit-factor latents and
        per-item non-Gaussian parameters are left to the sampler's default init.
        """
        if prev_stage is None:
            return None
        prev_path = self.config.output_dir / prev_stage.name / "idata.nc"
        if not prev_path.exists():
            return None
        try:
            prev_idata = az.from_netcdf(str(prev_path))
            prev_core = self._stage_core(prev_stage)
            Lam_src = np.asarray(prev_idata.posterior["Lam"].mean(("chain", "draw")).values)
        except Exception:
            return None
        src_load = {
            (item, factor): float(Lam_src[j, c])
            for j, item in enumerate(prev_core.items)
            for c, factor in enumerate(prev_core.factor_cols)
        }
        init: dict[str, np.ndarray] = {}
        if spec.pos_cells:
            expanded = np.array(
                [max(0.02, src_load.get((spec.items[j], spec.factor_cols[c]), mu))
                 for j, c, mu, _sd in spec.pos_cells],
                dtype="float64",
            )
            equal_groups = self.builder._equal_pos_groups(spec)
            if equal_groups:
                grouped = {
                    index for indices in equal_groups.values() for index in indices
                }
                free = [
                    index for index in range(len(spec.pos_cells)) if index not in grouped
                ]
                if free:
                    init["lam_pos_free"] = expanded[free]
                for factor, indices in equal_groups.items():
                    init[f"lam_equal_{factor}"] = np.asarray(
                        float(np.mean(expanded[indices])), dtype="float64"
                    )
            else:
                init["lam_pos"] = expanded
        if spec.signed_cells:
            init["lam_cross"] = np.array(
                [src_load.get((spec.items[j], spec.factor_cols[c]), 0.0)
                 for j, c, _mu, _sd in spec.signed_cells],
                dtype="float64",
            )
        try:
            sig_src = np.asarray(prev_idata.posterior["sigma"].mean(("chain", "draw")).values)
            sig_map = {it: float(sig_src[k]) for k, it in enumerate(prev_core.items) if k < len(sig_src)}
            init["sigma"] = np.clip(
                np.array([sig_map.get(it, 0.8) for it in core.items], dtype="float64"), 0.1, 1.2
            )
        except Exception:
            pass
        if init:
            print(f"  warm-start from {prev_stage.name}: {sorted(init)}", flush=True)
        return init or None

    def diagnostics(
        self,
        idata,
        *,
        core: CoreData | None = None,
        spec: LoadingSpec | None = None,
        mixed: MixedData | None = None,
    ) -> dict[str, Any]:
        """Return tiered convergence and geometry diagnostics.

        Map-defining parameters (loadings, factor correlations, residual scales,
        and native-family dispersion terms) are separated from high-dimensional
        mean-calibration nuisance parameters (intercepts, thresholds, and
        covariate slopes).  Explicit patient coordinates are checked on a bounded
        panel.  The separation prevents one of thousands of nuisance slopes from
        being mislabeled as a failure of the latent factor map.
        """
        post = idata.posterior
        nuisance_names = [
            v for v in ["alpha", "beta", "beta_native"] if v in post
        ]
        nuisance_prefixes = ("a_", "c_")
        nuisance_names += [
            str(v)
            for v in post.data_vars
            if str(v).startswith(nuisance_prefixes) and str(v) not in nuisance_names
        ]

        def empty_summary() -> dict[str, Any]:
            return {
                "rhat": float("nan"),
                "ess_bulk": float("nan"),
                "ess_tail": float("nan"),
                "parameter_count": 0,
                "above_map_rhat_max": 0,
            }

        def summarize_names(names: list[str]) -> dict[str, Any]:
            if not names:
                return empty_summary()
            # ArviZ otherwise rounds R-hat to two decimals before the gate, which
            # turns a numerical diagnostic into an accidental 0.005-wide bin.
            summary = az.summary(idata, var_names=names, round_to="none")
            if "sd" in summary.columns:
                summary = summary[pd.to_numeric(summary["sd"], errors="coerce") > 0]
            rhat_col = "r_hat" if "r_hat" in summary.columns else "rhat"
            rhat = pd.to_numeric(summary[rhat_col], errors="coerce")
            bulk = pd.to_numeric(summary["ess_bulk"], errors="coerce")
            tail = pd.to_numeric(summary["ess_tail"], errors="coerce")
            return {
                "rhat": float(rhat.max()),
                "ess_bulk": float(bulk.min()),
                "ess_tail": float(tail.min()),
                "parameter_count": int(rhat.notna().sum()),
                "above_map_rhat_max": int((rhat > self.config.rhat_max).sum()),
            }

        def summarize_arrays(arrays: list[Any]) -> dict[str, Any]:
            """Summarize already-selected scalar posterior elements."""
            rhat_values: list[np.ndarray] = []
            bulk_values: list[np.ndarray] = []
            tail_values: list[np.ndarray] = []
            for values in arrays:
                if values.size == 0:
                    continue
                try:
                    parameter_dims = [
                        dim for dim in values.dims if dim not in {"chain", "draw"}
                    ]
                    if parameter_dims:
                        if len(parameter_dims) == 1:
                            scalar_dim = parameter_dims[0]
                        else:
                            values = values.stack(
                                _diagnostic_scalar=parameter_dims
                            )
                            scalar_dim = "_diagnostic_scalar"
                        sd = np.asarray(
                            values.std(dim=("chain", "draw")), dtype="float64"
                        ).reshape(-1)
                        varying = np.flatnonzero((sd > 0) & np.isfinite(sd))
                        if not varying.size:
                            continue
                        values = values.isel({scalar_dim: varying})
                    else:
                        sd_scalar = float(
                            np.asarray(values.std(dim=("chain", "draw")))
                        )
                        if not np.isfinite(sd_scalar) or sd_scalar <= 0:
                            continue
                    rhat = np.asarray(az.rhat(values), dtype="float64").reshape(-1)
                    bulk = np.asarray(
                        az.ess(values, method="bulk"), dtype="float64"
                    ).reshape(-1)
                    tail = np.asarray(
                        az.ess(values, method="tail"), dtype="float64"
                    ).reshape(-1)
                except Exception:
                    continue
                keep = np.isfinite(rhat) & np.isfinite(bulk) & np.isfinite(tail)
                if np.any(keep):
                    rhat_values.append(rhat[keep])
                    bulk_values.append(bulk[keep])
                    tail_values.append(tail[keep])
            if not rhat_values:
                return empty_summary()
            rhat = np.concatenate(rhat_values)
            bulk = np.concatenate(bulk_values)
            tail = np.concatenate(tail_values)
            return {
                "rhat": float(np.max(rhat)),
                "ess_bulk": float(np.min(bulk)),
                "ess_tail": float(np.min(tail)),
                "parameter_count": int(rhat.size),
                "above_map_rhat_max": int(np.sum(rhat > self.config.rhat_max)),
            }

        def select_vector(name: str, indices: list[int]) -> Any | None:
            if name not in post or not indices:
                return None
            values = post[name]
            return values.isel({values.dims[2]: np.asarray(indices, dtype="int64")})

        core_map_arrays: list[Any] = []
        substance_map_arrays: list[Any] = []
        substance_loading_arrays: list[Any] = []
        substance_correlation_arrays: list[Any] = []
        if core is not None and spec is not None:
            substance_col = (
                core.factor_cols.index("substance")
                if "substance" in core.factor_cols
                else None
            )
            loading_cells = {
                "lam_pos": [(row, col) for row, col, _mu, _sd in spec.pos_cells],
                "lam_cross": [
                    (row, col) for row, col, _mu, _sd in spec.signed_cells
                ],
                "lam_hs": list(spec.hs_cells),
            }
            for name, cells in loading_cells.items():
                core_indices = [
                    index
                    for index, (_row, col) in enumerate(cells)
                    if col != substance_col
                ]
                substance_indices = [
                    index
                    for index, (_row, col) in enumerate(cells)
                    if col == substance_col
                ]
                selected = select_vector(name, core_indices)
                if selected is not None:
                    core_map_arrays.append(selected)
                selected = select_vector(name, substance_indices)
                if selected is not None:
                    substance_map_arrays.append(selected)
                    substance_loading_arrays.append(selected)

            if "sigma" in post:
                core_map_arrays.append(post["sigma"])

            # Gate the scientific correlation coefficients, not an arbitrary
            # chunk of the C-vine's raw partial-correlation vectors.  Only the
            # unique lower triangle is used; fixed G-orthogonal entries have
            # zero posterior SD and are removed by summarize_arrays().
            if "Phi" in post:
                phi = post["Phi"]
                factor_count = len(core.factor_cols)
                core_pairs: list[int] = []
                substance_pairs: list[int] = []
                for row in range(factor_count):
                    for col in range(row):
                        flat_index = row * factor_count + col
                        if substance_col is not None and substance_col in (row, col):
                            substance_pairs.append(flat_index)
                        else:
                            core_pairs.append(flat_index)
                phi_flat = phi.stack(
                    _diagnostic_pair=(phi.dims[-2], phi.dims[-1])
                )
                if core_pairs:
                    core_map_arrays.append(
                        phi_flat.isel(_diagnostic_pair=core_pairs)
                    )
                if substance_pairs:
                    selected_phi = phi_flat.isel(
                        _diagnostic_pair=substance_pairs
                    )
                    substance_map_arrays.append(selected_phi)
                    substance_correlation_arrays.append(selected_phi)

            native_structural = [
                str(name)
                for name in post.data_vars
                if str(name).startswith(("lh_", "lg_", "alpha_", "apsi_"))
            ]
            for name in native_structural:
                is_substance_loading = False
                if name.startswith("lh_") and mixed is not None:
                    item = name[3:]
                    if item in mixed.ng_home:
                        home_col = mixed.e_cols[mixed.ng_home[item]]
                        is_substance_loading = (
                            core.factor_cols[home_col] == "substance"
                        )
                if is_substance_loading:
                    substance_map_arrays.append(post[name])
                    substance_loading_arrays.append(post[name])
                else:
                    core_map_arrays.append(post[name])

            core_map_diag = summarize_arrays(core_map_arrays)
            substance_diag = summarize_arrays(substance_map_arrays)
            substance_loading_diag = summarize_arrays(substance_loading_arrays)
            substance_correlation_diag = summarize_arrays(
                substance_correlation_arrays
            )
            map_diag = summarize_arrays(core_map_arrays + substance_map_arrays)
        else:
            # Backward-compatible fallback for external callers without the
            # stage's item/factor metadata.  No Substance exception is inferred.
            map_names = [
                name
                for name in ["lam_pos", "lam_cross", "lam_hs", "sigma"]
                if name in post
            ]
            map_prefixes = ("Phi_spec_", "lh_", "lg_", "alpha_", "apsi_")
            map_names += [
                str(name)
                for name in post.data_vars
                if str(name).startswith(map_prefixes) and str(name) not in map_names
            ]
            map_diag = summarize_names(map_names)
            core_map_diag = dict(map_diag)
            substance_diag = empty_summary()
            substance_loading_diag = empty_summary()
            substance_correlation_diag = empty_summary()

        nuisance_diag = summarize_names(nuisance_names)

        core_latent_diag = empty_summary()
        substance_latent_diag = empty_summary()
        latent_panel_n = 0
        core_latent_panel_n = 0
        substance_latent_panel_n = 0
        latent_name = "f_e" if "f_e" in post else "z_e" if "z_e" in post else None
        if latent_name is not None and post.sizes.get("chain", 0) > 1:
            latent = post[latent_name]
            patient_dim = latent.dims[2]
            factor_dim = latent.dims[3]
            n_patient = int(latent.sizes[patient_dim])
            panel = np.unique(
                np.linspace(0, n_patient - 1, min(256, n_patient), dtype=int)
            )
            latent_panel = latent.isel({patient_dim: panel})
            if mixed is not None and len(mixed.e_cols) == latent.sizes[factor_dim]:
                explicit_names = [core.factor_cols[col] for col in mixed.e_cols]
                core_indices = [
                    index
                    for index, factor in enumerate(explicit_names)
                    if factor != "substance"
                ]
                substance_indices = [
                    index
                    for index, factor in enumerate(explicit_names)
                    if factor == "substance"
                ]
            else:
                core_indices = list(range(int(latent.sizes[factor_dim])))
                substance_indices = []
            if core_indices:
                core_latent_diag = summarize_arrays(
                    [latent_panel.isel({factor_dim: core_indices})]
                )
                core_latent_panel_n = int(panel.size)
            if substance_indices:
                substance_latent_diag = summarize_arrays(
                    [latent_panel.isel({factor_dim: substance_indices})]
                )
                substance_latent_panel_n = int(panel.size)
            latent_panel_n = int(panel.size)

        latent_diags = [core_latent_diag, substance_latent_diag]
        latent_rhats = [
            value["rhat"] for value in latent_diags if np.isfinite(value["rhat"])
        ]
        latent_bulks = [
            value["ess_bulk"]
            for value in latent_diags
            if np.isfinite(value["ess_bulk"])
        ]
        latent_tails = [
            value["ess_tail"]
            for value in latent_diags
            if np.isfinite(value["ess_tail"])
        ]
        latent_rhat = max(latent_rhats) if latent_rhats else float("nan")
        latent_bulk = min(latent_bulks) if latent_bulks else float("nan")
        latent_tail = min(latent_tails) if latent_tails else float("nan")

        sample_stats = idata.sample_stats
        div = (
            int(np.asarray(sample_stats["diverging"]).sum())
            if "diverging" in sample_stats
            else 0
        )
        divergences_by_chain = (
            np.asarray(sample_stats["diverging"]).sum(axis=1).astype(int).tolist()
            if "diverging" in sample_stats
            else []
        )
        bfmi: list[float] = []
        if "energy" in sample_stats:
            energy = np.asarray(sample_stats["energy"], dtype="float64")
            for chain in energy:
                variance = float(np.var(chain))
                bfmi.append(
                    float(np.mean(np.diff(chain) ** 2) / variance)
                    if variance > 0
                    else float("nan")
                )
        depth_max = None
        depth_cap_fraction = 0.0
        depth_cap_fraction_by_chain: list[float] = []
        if "tree_depth" in sample_stats:
            depth = np.asarray(sample_stats["tree_depth"])
            depth_max = int(np.nanmax(depth))
            depth_cap_fraction = float(
                np.mean(depth >= self.config.max_tree_depth)
            )
            depth_cap_fraction_by_chain = np.mean(
                depth >= self.config.max_tree_depth, axis=1
            ).astype(float).tolist()
        steps_max = None
        steps_cap_fraction = 0.0
        steps_cap_fraction_by_chain: list[float] = []
        if "n_steps" in sample_stats:
            steps = np.asarray(sample_stats["n_steps"])
            steps_max = int(np.nanmax(steps))
            step_cap = 2 ** self.config.max_tree_depth - 1
            steps_cap_fraction = float(np.mean(steps >= step_cap))
            steps_cap_fraction_by_chain = np.mean(
                steps >= step_cap, axis=1
            ).astype(float).tolist()
        acceptance = None
        if "acceptance_rate" in sample_stats:
            rate = np.asarray(sample_stats["acceptance_rate"], dtype="float64")
            acceptance = {
                "mean": float(np.nanmean(rate)),
                "min": float(np.nanmin(rate)),
                "max": float(np.nanmax(rate)),
                "mean_by_chain": np.nanmean(rate, axis=1).astype(float).tolist(),
            }
        congruence_min = float("nan")
        congruence_by_factor: list[float] = []
        salient_sign_disagreements = 0
        core_loading_congruence_min = float("nan")
        substance_loading_congruence = float("nan")
        core_loading_sign_disagreements = 0
        substance_loading_sign_disagreements = 0
        if "Lam" in post and post.sizes.get("chain", 0) > 1:
            lam_chain = np.asarray(post["Lam"].mean("draw"), dtype="float64")
            chain_count, _items, factor_count = lam_chain.shape
            congruence_by_factor = [1.0] * factor_count
            for factor in range(factor_count):
                values: list[float] = []
                for left in range(chain_count):
                    for right in range(left + 1, chain_count):
                        a = lam_chain[left, :, factor]
                        b = lam_chain[right, :, factor]
                        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                        values.append(float(a @ b / denom) if denom > 0 else float("nan"))
                finite_values = [value for value in values if np.isfinite(value)]
                congruence_by_factor[factor] = (
                    float(min(finite_values)) if finite_values else 1.0
                )
            congruence_min = float(np.nanmin(congruence_by_factor))
            salient = np.min(np.abs(lam_chain), axis=0) >= self.config.loading_sign_threshold
            sign_disagreement = np.ptp(np.sign(lam_chain), axis=0) > 0
            disagreements_by_factor = np.sum(
                salient & sign_disagreement, axis=0
            ).astype(int)
            salient_sign_disagreements = int(np.sum(disagreements_by_factor))
            if core is not None and len(core.factor_cols) == factor_count:
                substance_factor = (
                    core.factor_cols.index("substance")
                    if "substance" in core.factor_cols
                    else None
                )
                core_factors = [
                    index
                    for index in range(factor_count)
                    if index != substance_factor
                ]
                if core_factors:
                    core_loading_congruence_min = float(
                        np.nanmin(np.asarray(congruence_by_factor)[core_factors])
                    )
                    core_loading_sign_disagreements = int(
                        np.sum(disagreements_by_factor[core_factors])
                    )
                if substance_factor is not None:
                    substance_loading_congruence = congruence_by_factor[
                        substance_factor
                    ]
                    substance_loading_sign_disagreements = int(
                        disagreements_by_factor[substance_factor]
                    )
            else:
                core_loading_congruence_min = congruence_min
                core_loading_sign_disagreements = salient_sign_disagreements
        rhat_values = [map_diag["rhat"], nuisance_diag["rhat"], latent_rhat]
        bulk_values = [map_diag["ess_bulk"], nuisance_diag["ess_bulk"], latent_bulk]
        tail_values = [map_diag["ess_tail"], nuisance_diag["ess_tail"], latent_tail]
        finite_rhat = [float(v) for v in rhat_values if np.isfinite(v)]
        finite_bulk = [float(v) for v in bulk_values if np.isfinite(v)]
        finite_tail = [float(v) for v in tail_values if np.isfinite(v)]
        return {
            # Aggregate values remain useful for reporting but do not define the
            # tiered pass/fail decision below.
            "rhat": max(finite_rhat) if finite_rhat else float("nan"),
            "ess_bulk": min(finite_bulk) if finite_bulk else float("nan"),
            "ess_tail": min(finite_tail) if finite_tail else float("nan"),
            "map_rhat": map_diag["rhat"],
            "map_ess_bulk": map_diag["ess_bulk"],
            "map_ess_tail": map_diag["ess_tail"],
            "map_parameter_count": map_diag["parameter_count"],
            "core_map_rhat": core_map_diag["rhat"],
            "core_map_ess_bulk": core_map_diag["ess_bulk"],
            "core_map_ess_tail": core_map_diag["ess_tail"],
            "core_map_parameter_count": core_map_diag["parameter_count"],
            "substance_rhat": substance_diag["rhat"],
            "substance_ess_bulk": substance_diag["ess_bulk"],
            "substance_ess_tail": substance_diag["ess_tail"],
            "substance_parameter_count": substance_diag["parameter_count"],
            "substance_loading_rhat": substance_loading_diag["rhat"],
            "substance_loading_ess_bulk": substance_loading_diag["ess_bulk"],
            "substance_loading_ess_tail": substance_loading_diag["ess_tail"],
            "substance_loading_parameter_count": substance_loading_diag[
                "parameter_count"
            ],
            "substance_correlation_rhat": substance_correlation_diag["rhat"],
            "substance_correlation_ess_bulk": substance_correlation_diag[
                "ess_bulk"
            ],
            "substance_correlation_ess_tail": substance_correlation_diag[
                "ess_tail"
            ],
            "substance_correlation_parameter_count": substance_correlation_diag[
                "parameter_count"
            ],
            "nuisance_rhat": nuisance_diag["rhat"],
            "nuisance_ess_bulk": nuisance_diag["ess_bulk"],
            "nuisance_ess_tail": nuisance_diag["ess_tail"],
            "nuisance_parameter_count": nuisance_diag["parameter_count"],
            "nuisance_parameters_above_map_rhat_max": nuisance_diag[
                "above_map_rhat_max"
            ],
            # Backward-compatible names now have their literal intended meaning:
            # structural == map-defining, not map plus alpha/beta nuisance terms.
            "structural_rhat": map_diag["rhat"],
            "structural_ess_bulk": map_diag["ess_bulk"],
            "structural_ess_tail": map_diag["ess_tail"],
            "latent_panel_rhat": latent_rhat,
            "latent_panel_ess_bulk": latent_bulk,
            "latent_panel_ess_tail": latent_tail,
            "latent_panel_patients": latent_panel_n,
            "core_latent_panel_rhat": core_latent_diag["rhat"],
            "core_latent_panel_ess_bulk": core_latent_diag["ess_bulk"],
            "core_latent_panel_ess_tail": core_latent_diag["ess_tail"],
            "core_latent_panel_patients": core_latent_panel_n,
            "substance_latent_panel_rhat": substance_latent_diag["rhat"],
            "substance_latent_panel_ess_bulk": substance_latent_diag["ess_bulk"],
            "substance_latent_panel_ess_tail": substance_latent_diag["ess_tail"],
            "substance_latent_panel_patients": substance_latent_panel_n,
            "bfmi": bfmi,
            "bfmi_min": float(np.nanmin(bfmi)) if bfmi else float("nan"),
            "divergences": div,
            "divergences_by_chain": divergences_by_chain,
            "tree_depth_max": depth_max,
            "tree_depth_cap_fraction": depth_cap_fraction,
            "tree_depth_cap_fraction_by_chain": depth_cap_fraction_by_chain,
            "n_steps_max": steps_max,
            "n_steps_cap_fraction": steps_cap_fraction,
            "n_steps_cap_fraction_by_chain": steps_cap_fraction_by_chain,
            "acceptance_rate": acceptance,
            "loading_congruence_min": congruence_min,
            "loading_congruence_by_factor": congruence_by_factor,
            "salient_loading_sign_disagreements": salient_sign_disagreements,
            "core_loading_congruence_min": core_loading_congruence_min,
            "substance_loading_congruence": substance_loading_congruence,
            "core_loading_sign_disagreements": core_loading_sign_disagreements,
            "substance_loading_sign_disagreements": (
                substance_loading_sign_disagreements
            ),
        }

    @staticmethod
    def _finite(values: tuple[Any, ...]) -> bool:
        return all(value is not None and np.isfinite(value) for value in values)

    def _gate_components(self, diagnostics: dict[str, Any]) -> dict[str, bool]:
        """Evaluate strict-core, provisional-Substance, and global gates."""
        core_map_values = (
            diagnostics.get(
                "core_map_rhat",
                diagnostics.get("map_rhat", diagnostics.get("structural_rhat")),
            ),
            diagnostics.get(
                "core_map_ess_bulk",
                diagnostics.get("map_ess_bulk", diagnostics.get("structural_ess_bulk")),
            ),
            diagnostics.get(
                "core_map_ess_tail",
                diagnostics.get("map_ess_tail", diagnostics.get("structural_ess_tail")),
            ),
        )
        core_map_ok = bool(
            self._finite(core_map_values)
            and core_map_values[0] <= self.config.rhat_max
            and core_map_values[1] >= self.config.ess_min
            and core_map_values[2] >= self.config.ess_min
        )

        nuisance_count = int(diagnostics.get("nuisance_parameter_count", 0))
        nuisance_ok = True
        if nuisance_count:
            nuisance_values = (
                diagnostics.get("nuisance_rhat"),
                diagnostics.get("nuisance_ess_bulk"),
                diagnostics.get("nuisance_ess_tail"),
            )
            nuisance_ok = bool(
                self._finite(nuisance_values)
                and diagnostics["nuisance_rhat"] <= self.config.nuisance_rhat_max
                and diagnostics["nuisance_ess_bulk"] >= self.config.ess_min
                and diagnostics["nuisance_ess_tail"] >= self.config.ess_min
            )

        core_latent_count = int(
            diagnostics.get(
                "core_latent_panel_patients",
                diagnostics.get("latent_panel_patients", 0),
            )
        )
        core_latent_ok = True
        if core_latent_count:
            core_latent_values = (
                diagnostics.get(
                    "core_latent_panel_rhat", diagnostics.get("latent_panel_rhat")
                ),
                diagnostics.get(
                    "core_latent_panel_ess_bulk",
                    diagnostics.get("latent_panel_ess_bulk"),
                ),
                diagnostics.get(
                    "core_latent_panel_ess_tail",
                    diagnostics.get("latent_panel_ess_tail"),
                ),
            )
            core_latent_ok = bool(
                self._finite(core_latent_values)
                and core_latent_values[0] <= self.config.rhat_max
                and core_latent_values[1] >= self.config.ess_min
                and core_latent_values[2] >= self.config.ess_min
            )

        substance_count = int(diagnostics.get("substance_parameter_count", 0))
        substance_values = (
            diagnostics.get("substance_rhat"),
            diagnostics.get("substance_ess_bulk"),
            diagnostics.get("substance_ess_tail"),
        )
        substance_provisional_ok = True
        substance_strict_ok = True
        if substance_count:
            substance_provisional_ok = bool(
                self._finite(substance_values)
                and substance_values[0] <= self.config.substance_rhat_max
                and substance_values[1] >= self.config.substance_ess_min
                and substance_values[2] >= self.config.substance_ess_min
            )
            substance_strict_ok = bool(
                self._finite(substance_values)
                and substance_values[0] <= self.config.rhat_max
                and substance_values[1] >= self.config.ess_min
                and substance_values[2] >= self.config.ess_min
            )

        substance_latent_count = int(
            diagnostics.get("substance_latent_panel_patients", 0)
        )
        if substance_latent_count:
            substance_latent_values = (
                diagnostics.get("substance_latent_panel_rhat"),
                diagnostics.get("substance_latent_panel_ess_bulk"),
                diagnostics.get("substance_latent_panel_ess_tail"),
            )
            substance_provisional_ok = bool(
                substance_provisional_ok
                and self._finite(substance_latent_values)
                and substance_latent_values[0] <= self.config.substance_rhat_max
                and substance_latent_values[1] >= self.config.substance_ess_min
                and substance_latent_values[2] >= self.config.substance_ess_min
            )
            substance_strict_ok = bool(
                substance_strict_ok
                and self._finite(substance_latent_values)
                and substance_latent_values[0] <= self.config.rhat_max
                and substance_latent_values[1] >= self.config.ess_min
                and substance_latent_values[2] >= self.config.ess_min
            )

        core_loading_congruence = diagnostics.get(
            "core_loading_congruence_min",
            diagnostics.get("loading_congruence_min"),
        )
        core_alignment_ok = bool(
            core_loading_congruence is not None
            and np.isfinite(core_loading_congruence)
            and core_loading_congruence >= self.config.loading_congruence_min
            and diagnostics.get(
                "core_loading_sign_disagreements",
                diagnostics.get("salient_loading_sign_disagreements", 1),
            )
            == 0
        )
        substance_loading_count = int(
            diagnostics.get("substance_loading_parameter_count", 0)
        )
        substance_alignment_ok = True
        if substance_loading_count:
            substance_loading_congruence = diagnostics.get(
                "substance_loading_congruence"
            )
            substance_alignment_ok = bool(
                substance_loading_congruence is not None
                and np.isfinite(substance_loading_congruence)
                and substance_loading_congruence
                >= self.config.loading_congruence_min
                and diagnostics.get("substance_loading_sign_disagreements", 1)
                == 0
            )

        bfmi = diagnostics.get("bfmi_min")
        geometry_ok = bool(
            bfmi is not None
            and np.isfinite(bfmi)
            and bfmi >= self.config.bfmi_min
            and diagnostics.get("divergences", 0) == 0
            and diagnostics.get("tree_depth_cap_fraction", 1.0)
            <= self.config.max_depth_fraction
            and max(
                diagnostics.get("tree_depth_cap_fraction_by_chain") or [1.0]
            )
            <= self.config.max_depth_fraction
            and diagnostics.get("n_steps_cap_fraction", 1.0)
            <= self.config.max_depth_fraction
            and max(diagnostics.get("n_steps_cap_fraction_by_chain") or [1.0])
            <= self.config.max_depth_fraction
        )
        core_passed = bool(
            core_map_ok
            and nuisance_ok
            and core_latent_ok
            and core_alignment_ok
            and geometry_ok
        )
        return {
            "core_map": core_map_ok,
            "nuisance": nuisance_ok,
            "core_latent": core_latent_ok,
            "core_alignment": core_alignment_ok,
            "substance_provisional": substance_provisional_ok,
            "substance_strict": substance_strict_ok,
            "substance_alignment": substance_alignment_ok,
            "geometry": geometry_ok,
            "core_passed": core_passed,
            "operational_passed": bool(
                core_passed and substance_provisional_ok and substance_alignment_ok
            ),
            "strict_sampling_passed": bool(
                core_passed and substance_strict_ok and substance_alignment_ok
            ),
        }

    def _passes_gates(self, diagnostics: dict[str, Any]) -> bool:
        """Operational gate: strict core/global diagnostics plus provisional Substance."""
        return self._gate_components(diagnostics)["operational_passed"]


class PatientProjector:
    """Patient coordinate projection and uncertainty summaries.

    Fitting estimates population-level measurement parameters.  Projection asks:
    "Given this fixed measurement map and this patient's observed cells, where is
    the patient on the latent dimensions?"  This is a conditional Gaussian
    calculation for the continuous block.
    """

    def __init__(self, config: MeasurementConfig | None = None):
        self.config = config or MeasurementConfig()

    def conditional_gaussian_scores(
        self,
        core: CoreData,
        posterior,
        *,
        hdi_prob: float = 0.94,
    ) -> dict[str, np.ndarray]:
        """Analytic factor-score posterior at posterior-mean measurement parameters.

        For observed columns O, the Gaussian conditioning formula is:

            f_i | x_iO ~ Normal(
                Phi Lambda_O' Sigma_OO^{-1} (x_iO - mu_iO),
                Phi - Phi Lambda_O' Sigma_OO^{-1} Lambda_O Phi
            )

        The mean is the patient's coordinate; the diagonal of the covariance is
        their score uncertainty.  Patients with few observed home indicators get
        wider SDs and weaker reliability labels.
        """
        Lam = np.asarray(posterior["Lam"].mean(("chain", "draw")).values)
        Phi = np.asarray(posterior["Phi"].mean(("chain", "draw")).values)
        sigma = self.config.psi_floor + np.asarray(posterior["sigma"].mean(("chain", "draw")).values)
        alpha = np.asarray(posterior["alpha"].mean(("chain", "draw")).values) if "alpha" in posterior else np.zeros(core.M.shape[1])
        if "beta" in posterior and core.covariates.shape[1]:
            beta = np.asarray(posterior["beta"].mean(("chain", "draw")).values)
            mu = alpha[None, :] + core.covariates @ beta.T
        else:
            mu = alpha[None, :]
        M = core.M
        mask = np.isfinite(M)
        X = np.nan_to_num(M - mu, nan=0.0)
        patterns, inv = np.unique(mask, axis=0, return_inverse=True)
        F = Lam.shape[1]
        mean = np.full((M.shape[0], F), np.nan)
        sd = np.full((M.shape[0], F), np.nan)
        for p in range(patterns.shape[0]):
            cols = np.flatnonzero(patterns[p])
            rows = np.flatnonzero(inv == p)
            if cols.size == 0:
                # With no observed indicators, the best estimate is the prior:
                # mean zero and covariance Phi.  This is "prior-dominated" by
                # construction, not an imputed score.
                mean[rows] = 0.0
                sd[rows] = np.sqrt(np.clip(np.diag(Phi), 0.0, None))
                continue
            Lam_o = Lam[cols]
            Sigma_o = Lam_o @ Phi @ Lam_o.T + np.diag(sigma[cols] ** 2)
            try:
                Sigma_inv = np.linalg.inv(Sigma_o)
            except np.linalg.LinAlgError:
                Sigma_inv = np.linalg.pinv(Sigma_o)
            B = Phi @ Lam_o.T @ Sigma_inv
            mean[rows] = X[np.ix_(rows, cols)] @ B.T
            cov = Phi - B @ Lam_o @ Phi
            sd[rows] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        z = float(norm.ppf(1 - (1 - hdi_prob) / 2))
        return {"mean": mean, "sd": sd, "hdi_low": mean - z * sd, "hdi_high": mean + z * sd}

    @staticmethod
    def reliability_flags(core: CoreData) -> tuple[np.ndarray, np.ndarray]:
        """Observed home-indicator counts and reliability labels.

        Reliability here is intentionally simple and transparent.  It counts
        observed indicators whose *home* factor is the dimension being scored.
        Cross-loadings can still inform a coordinate, but they do not make a
        dimension "well characterized" for that patient.
        """
        n_obs = np.zeros((core.M.shape[0], len(core.factor_cols)), dtype=int)
        col = {factor: i for i, factor in enumerate(core.factor_cols)}
        obs = np.isfinite(core.M)
        for j, home in enumerate(core.home):
            if home in col:
                n_obs[:, col[home]] += obs[:, j].astype(int)
        tier = np.where(n_obs >= 3, "well", np.where(n_obs >= 1, "partial", "prior-dominated"))
        return n_obs, tier

    def projection_frame(self, core: CoreData, posterior) -> pd.DataFrame:
        """Return per-patient coordinate summary for the continuous block."""
        scores = self.conditional_gaussian_scores(core, posterior)
        n_obs, tier = self.reliability_flags(core)
        frame = pd.DataFrame(index=core.index)
        for c, factor in enumerate(core.factor_cols):
            frame[f"{factor}__mean"] = scores["mean"][:, c]
            frame[f"{factor}__sd"] = scores["sd"][:, c]
            frame[f"{factor}__hdi_low"] = scores["hdi_low"][:, c]
            frame[f"{factor}__hdi_high"] = scores["hdi_high"][:, c]
            frame[f"{factor}__n_obs"] = n_obs[:, c]
            frame[f"{factor}__reliability"] = tier[:, c]
        return frame


class MeasurementVisualizer:
    """Small plotting utilities for notebook use."""

    def __init__(self, config: MeasurementConfig | None = None):
        self.config = config or MeasurementConfig()
        self.config.figure_dir.mkdir(parents=True, exist_ok=True)

    def loading_atlas(self, spec: LoadingSpec, posterior, *, filename: str = "loading_atlas.png") -> Path:
        """Plot posterior mean loadings."""
        Lam = np.asarray(posterior["Lam"].mean(("chain", "draw")).values)
        fig, ax = plt.subplots(figsize=(max(8, Lam.shape[1] * 0.8), max(6, Lam.shape[0] * 0.12)))
        im = ax.imshow(Lam, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(spec.factor_cols)))
        ax.set_xticklabels(spec.factor_cols, rotation=45, ha="right")
        ax.set_yticks([])
        ax.set_title("Posterior loading atlas")
        fig.colorbar(im, ax=ax, label="loading")
        return self._save(fig, filename)

    def phi_heatmap(self, posterior, factors: list[str], *, filename: str = "phi_heatmap.png") -> Path:
        """Plot posterior mean factor correlation matrix."""
        Phi = np.asarray(posterior["Phi"].mean(("chain", "draw")).values)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(Phi, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(factors)))
        ax.set_xticklabels(factors, rotation=45, ha="right")
        ax.set_yticks(range(len(factors)))
        ax.set_yticklabels(factors)
        ax.set_title("Factor correlation matrix")
        fig.colorbar(im, ax=ax, label="correlation")
        return self._save(fig, filename)

    def reliability_bar(self, projection: pd.DataFrame, factors: list[str], *, filename: str = "reliability.png") -> Path:
        """Plot reliability tiers by factor."""
        counts = pd.DataFrame(
            {
                factor: projection[f"{factor}__reliability"].value_counts()
                for factor in factors
                if f"{factor}__reliability" in projection
            }
        ).T.fillna(0)
        counts = counts.reindex(columns=["well", "partial", "prior-dominated"]).fillna(0)
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = np.zeros(len(counts))
        for tier in counts.columns:
            ax.bar(counts.index, counts[tier].to_numpy(), bottom=bottom, label=tier)
            bottom += counts[tier].to_numpy()
        ax.set_title("Patient reliability tiers")
        ax.set_ylabel("patients")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        return self._save(fig, filename)

    def patient_uncertainty(
        self,
        projection: pd.DataFrame,
        patient_index: Any,
        factors: list[str],
        *,
        filename: str = "patient_uncertainty.png",
    ) -> Path:
        """Plot one patient's coordinate mean and interval."""
        row = projection.loc[patient_index]
        means = np.array([row[f"{factor}__mean"] for factor in factors], dtype=float)
        lows = np.array([row[f"{factor}__hdi_low"] for factor in factors], dtype=float)
        highs = np.array([row[f"{factor}__hdi_high"] for factor in factors], dtype=float)
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(factors))
        ax.errorbar(x, means, yerr=[means - lows, highs - means], fmt="o", capsize=3)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(factors, rotation=45, ha="right")
        ax.set_title(f"Patient projection uncertainty: {patient_index}")
        ax.set_ylabel("coordinate")
        return self._save(fig, filename)

    def map_scatter(
        self,
        projection: pd.DataFrame,
        x_factor: str,
        y_factor: str,
        *,
        highlight: dict[str, Any] | None = None,
        filename: str = "map_scatter.png",
    ) -> Path:
        """Two-axis transdiagnostic map: every patient at its posterior-mean
        coordinate, with a few example patients drawn as 94% uncertainty crosses.

        This is the patient-projection demonstration: it shows both *position*
        (where a patient sits on two latent axes) and *uncertainty* (how tightly
        that position is pinned by their observed cells).  ``highlight`` maps a
        label -> patient index; each highlighted patient gets a labelled cross
        whose arms are the 94% HDI on each axis.
        """
        xm = projection[f"{x_factor}__mean"].to_numpy(float)
        ym = projection[f"{y_factor}__mean"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(7.5, 7))
        ax.scatter(xm, ym, s=8, alpha=0.18, color="0.5", linewidths=0, label="all patients")
        ax.axhline(0.0, color="black", lw=0.6)
        ax.axvline(0.0, color="black", lw=0.6)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for k, (label, idx) in enumerate(dict(highlight or {}).items()):
            row = projection.loc[idx]
            cx, cy = float(row[f"{x_factor}__mean"]), float(row[f"{y_factor}__mean"])
            xerr = [[cx - float(row[f"{x_factor}__hdi_low"])], [float(row[f"{x_factor}__hdi_high"]) - cx]]
            yerr = [[cy - float(row[f"{y_factor}__hdi_low"])], [float(row[f"{y_factor}__hdi_high"]) - cy]]
            ax.errorbar(
                cx, cy, xerr=xerr, yerr=yerr, fmt="o", ms=7, capsize=3, lw=2,
                color=colors[k % len(colors)], label=f"{label}: {idx}", zorder=5,
            )
        ax.set_xlabel(f"{x_factor} (coordinate)")
        ax.set_ylabel(f"{y_factor} (coordinate)")
        ax.set_title(f"Patient map: {y_factor} vs {x_factor} (94% HDI crosses)")
        ax.legend(loc="best", fontsize=8)
        return self._save(fig, filename)

    def _save(self, fig, filename: str) -> Path:
        path = self.config.figure_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path


def waic_from_pointwise_loglik(ll: np.ndarray) -> dict[str, float]:
    """WAIC from an [draw, patient] pointwise log-likelihood matrix."""
    S = ll.shape[0]
    lppd_i = logsumexp(ll, axis=0) - np.log(S)
    p_i = ll.var(axis=0, ddof=1)
    elpd_i = lppd_i - p_i
    return {
        "elpd_waic": float(elpd_i.sum()),
        "p_waic": float(p_i.sum()),
        "waic": float(-2.0 * elpd_i.sum()),
    }


def _balanced_idx(cohort: np.ndarray, n_total: int, rng: np.random.Generator) -> np.ndarray:
    cohorts = list(dict.fromkeys(cohort))
    per = max(1, n_total // len(cohorts))
    chosen = []
    for c in cohorts:
        ix = np.flatnonzero(cohort == c)
        chosen.append(rng.choice(ix, size=min(per, len(ix)), replace=False))
    return np.sort(np.concatenate(chosen))


def _standardize_block(block: np.ndarray) -> np.ndarray:
    mu = np.nanmean(block, axis=0)
    sd = np.nanstd(block, axis=0)
    sd[sd <= 0] = 1.0
    return np.nan_to_num((block - mu) / sd)


def _cell_arrays(cells: list[tuple[int, int, float, float]]):
    if not cells:
        z = np.zeros(0)
        return z.astype("int64"), z.astype("int64"), z.astype("float64"), z.astype("float64")
    rows = np.array([c[0] for c in cells], dtype="int64")
    cols = np.array([c[1] for c in cells], dtype="int64")
    mu = np.array([c[2] for c in cells], dtype="float64")
    sd = np.array([c[3] for c in cells], dtype="float64")
    return rows, cols, mu, sd


def _selection(rows: list[int], F: int) -> np.ndarray:
    out = np.zeros((len(rows), F), dtype="float64")
    for k, row in enumerate(rows):
        out[k, row] = 1.0
    return out


def apply_frozen_covariate_design(
    index: pd.Index,
    covariates: pd.DataFrame,
    site: pd.Series | None,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Apply the exact covariate transform learned by ``MeasurementDataset.core``.

    This is the projection-time counterpart of ``_covariate_design``. It uses
    the stored fill values, spline knots, scaling constants, reference levels,
    and final column order; it never re-fits a transform on the target sample.
    """
    names = list(metadata.get("names", []))
    transform = metadata.get("transform", metadata)
    if not names:
        return np.zeros((len(index), 0), dtype="float64")
    cov = covariates.reindex(index)
    columns: dict[str, np.ndarray] = {}
    numeric = transform["numeric"]

    def raw_numeric(name: str) -> tuple[np.ndarray, np.ndarray]:
        raw = (
            pd.to_numeric(cov[name], errors="coerce").to_numpy("float64")
            if name in cov.columns
            else np.full(len(index), np.nan, dtype="float64")
        )
        missing = ~np.isfinite(raw)
        fill = float(numeric[name]["fill"])
        return np.nan_to_num(raw, nan=fill), missing.astype("float64")

    age, age_missing = raw_numeric("age")
    sex, sex_missing = raw_numeric("sex")
    education_name = next(name for name in numeric if name not in {"age", "sex"})
    education, education_missing = raw_numeric(education_name)

    spline = transform["age_spline"]
    degree = int(spline["degree"])
    knots = np.asarray(spline["knot_vector"], dtype="float64")
    # SplineTransformer(extrapolation="constant") evaluates the boundary
    # basis outside the fitted support. Clipping reproduces that behavior.
    age_eval = np.clip(age, knots[degree], knots[-degree - 1])
    age_full = BSpline.design_matrix(
        age_eval, knots, degree, extrapolate=True
    ).toarray()
    age_basis = age_full if bool(spline["include_bias"]) else age_full[:, :-1]
    age_basis = (
        age_basis - np.asarray(spline["center"], dtype="float64")
    ) / np.asarray(spline["scale"], dtype="float64")
    for k in range(age_basis.shape[1]):
        columns[f"age_spline_{k}"] = age_basis[:, k]
        columns[f"age_spline_{k}:sex"] = age_basis[:, k] * sex
    columns["sex"] = sex
    edu_center = float(np.asarray(numeric[education_name]["center"]).ravel()[0])
    edu_scale = float(np.asarray(numeric[education_name]["scale"]).ravel()[0])
    columns[education_name] = (education - edu_center) / edu_scale
    columns["age_missing"] = age_missing
    columns["sex_missing"] = sex_missing
    columns[f"{education_name}_missing"] = education_missing

    site_meta = transform.get("site")
    if site_meta is not None:
        site_values = (
            pd.to_numeric(site.reindex(index), errors="coerce")
            if site is not None
            else pd.Series(np.nan, index=index, dtype="float64")
        )
        site_missing = site_values.isna().to_numpy("float64")
        site_codes = site_values.round().astype("Int64")
        dummies = pd.get_dummies(
            site_codes.astype("object"), prefix="site", dummy_na=False
        )
        for name in site_meta.get("dummy_columns", []):
            columns[name] = (
                dummies[name].to_numpy("float64")
                if name in dummies
                else np.zeros(len(index), dtype="float64")
            )
        columns["site_missing"] = site_missing

    cohort_meta = transform.get("cohort")
    if cohort_meta is not None:
        cohort = pd.Series(index.get_level_values("cohort"), index=index)
        dummies = pd.get_dummies(cohort, prefix="cohort", drop_first=True)
        for name in cohort_meta.get("dummy_columns", []):
            columns[name] = (
                dummies[name].to_numpy("float64")
                if name in dummies
                else np.zeros(len(index), dtype="float64")
            )

    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(f"cannot reconstruct frozen covariate columns: {missing}")
    return np.column_stack([columns[name] for name in names]).astype("float64")


def _config_sig(config: MeasurementConfig) -> dict[str, Any]:
    """Model-affecting config fields that must match for a cached fit to be reused.

    Distinguishes a hard-zero fit from a soft-unlikely fit (and the covariate mode),
    which the stage signature alone does not capture."""
    sig = {
        "model_version": MODEL_VERSION,
        "soft_unlikely": bool(config.soft_unlikely),
        "soft_g_anchor_specific": bool(config.soft_g_anchor_specific),
        "include_covariates": bool(config.include_covariates),
        "covariate_mode": config.covariate_mode if config.include_covariates else "none",
        "covariate_missingness": str(config.covariate_missingness),
        "include_cohort_covariates": bool(config.include_cohort_covariates),
        "psi_floor": float(config.psi_floor),
        "correlation_prior": "LKJ",
        "lkj_eta": float(config.lkj_eta),
        "age_spline_knots": int(config.age_spline_knots),
        "likelihood_mode": str(config.likelihood_mode),
        "cohort_weighted": bool(config.cohort_weighted),
        "orthogonal_factors": sorted(config.orthogonal_factors),
        "equal_home_loading_factors": sorted(config.equal_home_loading_factors),
        "exclude_items": list(config.exclude_items),
        "prior_matrix_sha256": _file_sha256(config.prior_matrix),
        "processed_inputs_sha256": {
            name: _optional_file_sha256(config.processed_dir / name)
            for name in ("baseline_v0.parquet", "covariates_v0.parquet", "site_v0.parquet")
        },
    }
    if config.cross_loading_prior != "hard_zero":
        # Only emitted for the sparse-ESEM variant, so existing hard-zero caches stay valid.
        sig["cross_loading_prior"] = str(config.cross_loading_prior)
        sig["hs_tau0"] = float(config.hs_tau0)
        sig["hs_slab_c"] = float(config.hs_slab_c)
        sig["hs_fixed_tau"] = bool(config.hs_fixed_tau)
        sig["hs_local_df"] = float(config.hs_local_df)
    if config.likelihood_mode == "gaussian_copula":
        sig["copula_min_distinct"] = int(config.copula_min_distinct)
        sig["copula_max_modal_frac"] = float(config.copula_max_modal_frac)
    return sig


def _certification_policy(config: MeasurementConfig) -> dict[str, Any]:
    """Diagnostic thresholds, deliberately separate from the fit cache key."""
    return {
        "map_rhat_max": float(config.rhat_max),
        "nuisance_rhat_max": float(config.nuisance_rhat_max),
        "ess_min": float(config.ess_min),
        "substance_rhat_max": float(config.substance_rhat_max),
        "substance_ess_min": float(config.substance_ess_min),
        "bfmi_min": float(config.bfmi_min),
        "max_depth_fraction": float(config.max_depth_fraction),
        "loading_congruence_min": float(config.loading_congruence_min),
        "loading_sign_threshold": float(config.loading_sign_threshold),
    }


def _cached_model_sig(manifest: dict[str, Any]) -> dict[str, Any]:
    """Read a model-only cache key, including manifests written before gates
    were correctly separated from model-affecting configuration."""
    sig = dict(manifest.get("config_sig") or {})
    sig.pop("certification_gates", None)
    return sig


def _stage_spec(stage: StageDefinition) -> dict[str, Any]:
    """Stable cache-reuse signature for a stage definition."""
    spec = {
        "name": stage.name,
        "factors": list(stage.factors),
        "correlated": stage.correlated,
        "windows": stage.windows,
        "mixed": stage.mixed,
        "explicit_factors": list(stage.explicit_factors),
        "min_cohorts": stage.min_cohorts,
        "n_subsample": stage.n_subsample,
        "balanced": stage.balanced,
        "draws": stage.draws,
        "tune": stage.tune,
        "chains": stage.chains,
        "target_accept": stage.target_accept,
        "seed": stage.seed,
        "specific_cross": bool(stage.specific_cross),
        "cross_sd_scale": float(stage.cross_sd_scale),
        "g_correlated": bool(stage.g_correlated),
        "hurdle_counts": bool(stage.hurdle_counts),
        "enforce_gates": bool(stage.enforce_gates),
    }
    return spec


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_file_sha256(path: Path) -> str | None:
    return _file_sha256(path) if Path(path).exists() else None


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hurdle_nb_logp(y, psi, mu, alpha):
    log_p0 = alpha * (pt.log(alpha) - pt.log(alpha + mu))
    nb_lpmf = (
        pt.gammaln(y + alpha)
        - pt.gammaln(alpha)
        - pt.gammaln(y + 1.0)
        + alpha * (pt.log(alpha) - pt.log(alpha + mu))
        + y * (pt.log(mu) - pt.log(alpha + mu))
    )
    positive = pt.log(psi) + nb_lpmf - pt.log1mexp(log_p0)
    return pt.where(pt.eq(y, 0.0), pt.log1p(-psi), positive)


def dense_pattern_loglik(M: np.ndarray, Lam: np.ndarray, Phi: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Public convenience wrapper for dense observed-cell likelihood checks."""
    return BayesianBifactorESEM.dense_observed_loglik(M, Lam, Phi, sigma)


def solve_observed_gaussian_scores(
    M: np.ndarray,
    Lam: np.ndarray,
    Phi: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standalone conditional Gaussian score helper for toy examples."""
    mask = np.isfinite(M)
    X = np.nan_to_num(M, nan=0.0)
    patterns, inv = np.unique(mask, axis=0, return_inverse=True)
    mean = np.full((M.shape[0], Lam.shape[1]), np.nan)
    sd = np.full_like(mean, np.nan)
    for p in range(patterns.shape[0]):
        cols = np.flatnonzero(patterns[p])
        rows = np.flatnonzero(inv == p)
        if cols.size == 0:
            mean[rows] = 0.0
            sd[rows] = np.sqrt(np.diag(Phi))
            continue
        Lam_o = Lam[cols]
        Sigma_o = Lam_o @ Phi @ Lam_o.T + np.diag(sigma[cols] ** 2)
        chol = np.linalg.cholesky(Sigma_o)
        rhs = Lam_o @ Phi
        solve = solve_triangular(chol, rhs, lower=True)
        B = solve_triangular(chol.T, solve, lower=False).T
        mean[rows] = X[np.ix_(rows, cols)] @ B.T
        cov = Phi - B @ Lam_o @ Phi
        sd[rows] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return mean, sd
