"""OOP orchestration for the variational mixed-likelihood GLLVM atlas engine.

This module is the parallel-engine counterpart to ``measurement_model_oop`` (the NUTS
M1): it keeps the *same scientific measurement contract* (ontology-constrained linear
decoder, positive home loadings, hard-zero forbidden cells, mixed per-item likelihoods,
observed-cell likelihood, patient coordinates with uncertainty) but trains it by
stochastic variational inference (PyTorch, :mod:`face.models.variational.gllvm`).

It **composes** the certified ``MeasurementDataset`` for the data contract and reuses
``LoadingSpec.from_core`` for the loading ontology (so the cell-classification cannot drift
from the canonical engine), then writes the same export schema as the NUTS engine
(``coordinates`` / ``loadings_summary`` / ``phi``).

Defaults target the **8-factor operational map**: metabolic + inflammatory merged into one
``immunometabolic`` factor (prior matrix ``prior_loading_matrix_v3_biomerge_xc.csv``),
substance pinned orthogonal, the 3 earned cross-loadings (ctq37 / psqi11 / psqi17 ->
cognition) freed via ``specific_cross``, and a Gaussian-copula (rank-INT) continuous
likelihood — the same estimand the canonical downstream map uses.

This is an exploration/acceleration arm calibrated against (never replacing) the NUTS
authority.  See ``docs/VGLLVM_MODEL.md``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm  # type: ignore[reportMissingImports]

from face.models.bayesian.measurement_model_oop import (
    CONTINUOUS_FAMILIES,
    DEFAULT_EXPLICIT_FACTORS,
    G_KEY,
    WINDOWS,
    CoreData,
    LoadingSpec,
    MeasurementConfig,
    MeasurementDataset,
    PatientProjector,
    _rank_int,
)
from face.models.variational.gllvm import (
    GLLVMTrainer,
    LoadingOntology,
    TrainingConfig,
    VariationalGLLVM,
)

REPO = Path(__file__).resolve().parents[4]
PROC = MeasurementConfig().processed_dir
BIOMERGE_XC = REPO / "configs" / "prior_loading_matrix_v3_biomerge_xc.csv"
RESULTS = REPO / "results" / "face" / "gllvm_oop"
FIGURES = REPO / "docs" / "figures" / "gllvm_oop"
MODEL_VERSION = "gllvm_vi_2026_06_26_v1"

# The 8-factor operational fit order (Lam/Phi/f_e column order) — matches strata F8_FIT.
F8_FIT = [
    "overall_severity",  # G (index 0, bifactor-orthogonal)
    "cognition",
    "immunometabolic",  # merged metabolic + inflammatory
    "sleep",
    "suicidality",
    "developmental_risk",
    "mania_activation",
    "substance",  # pinned orthogonal (index 7)
]
# Continuous backbone (the warm-start rung): the four continuous-anchored axes.
F_BACKBONE = ["overall_severity", "cognition", "immunometabolic", "sleep"]


@dataclass(frozen=True)
class GLLVMStage:
    """One staged fit target (a rung of the warm-start ladder)."""

    name: str
    factors: list[str]
    windows: bool = False
    correlated: bool = True
    mixed: bool = False
    n_subsample: int | None = None
    balanced: bool = False
    epochs: int = 4000
    lr: float = 1e-2
    warmstart_from: str | None = None


@dataclass(frozen=True)
class GLLVMConfig:
    """Paths + structural switches for the variational GLLVM engine.

    The default object is the **8-factor operational map** under a Gaussian-copula
    (rank-INT) continuous likelihood — the estimand the downstream M2-M5 engines consume.
    """

    processed_dir: Path = PROC
    prior_matrix: Path = BIOMERGE_XC
    output_dir: Path = RESULTS
    figure_dir: Path = FIGURES
    factors: tuple[str, ...] = tuple(F8_FIT)
    explicit_factors: tuple[str, ...] = tuple(DEFAULT_EXPLICIT_FACTORS)
    orthogonal_factors: tuple[str, ...] = ("substance",)
    # Loading ontology
    specific_cross: bool = True  # free the 3 folded cross-loadings in the xc matrix
    cross_sd_scale: float = 0.25
    soft_unlikely: bool = False
    soft_g_anchor_specific: bool = False
    # Likelihood / encoding
    likelihood_mode: str = "gaussian_copula"  # rank-INT continuous; native binary/low-ordinal
    copula_min_distinct: int = 8
    copula_max_modal_frac: float = 0.5
    # Covariates
    include_covariates: bool = True
    covariate_mode: str = "residualize"  # {"residualize", "none"}
    age_spline_knots: int = 4
    # Model / optimization
    psi_floor: float = 0.05
    q_family: str = "per_patient"
    q_rank: int = 0  # 0 = mean-field diagonal q; >0 = low-rank+diagonal q (partially recovers Phi)
    full_cov: bool = False  # full per-patient K×K covariance q (most expressive; closes Phi)
    n_mc_samples: int = 1
    lr: float = 1e-2
    epochs: int = 4000
    batch_size: int | None = None  # None => full batch
    grad_clip_norm: float = 5.0
    early_stop_patience: int | None = 300  # epochs without >rel_tol -ELBO improvement -> stop
    early_stop_rel_tol: float = 5e-5
    phi_penalty_weight: float = 1e-3  # Omega(Phi) off-diagonal L2; 0 to stop shrinking Phi
    sigma_prior_weight: float = 0.0
    count_alpha_prior_weight: float = 0.0
    cutpoint_prior_weight: float = 0.0
    device: str = "cpu"  # {"cpu", "mps"}
    seed: int = 20260605
    # Smoke
    smoke: bool = False

    def measurement_config(self) -> MeasurementConfig:
        """The MeasurementConfig the composed ``MeasurementDataset`` runs under (data
        contract only — likelihood mode, covariates, prior matrix)."""
        return MeasurementConfig(
            processed_dir=self.processed_dir,
            prior_matrix=self.prior_matrix,
            include_covariates=self.include_covariates,
            covariate_mode=self.covariate_mode if self.include_covariates else "none",
            soft_unlikely=self.soft_unlikely,
            soft_g_anchor_specific=self.soft_g_anchor_specific,
            likelihood_mode=self.likelihood_mode,
            copula_min_distinct=self.copula_min_distinct,
            copula_max_modal_frac=self.copula_max_modal_frac,
            age_spline_knots=self.age_spline_knots,
            psi_floor=self.psi_floor,
        )

    def with_smoke_defaults(self) -> GLLVMConfig:
        """Fast wiring variant: tiny epochs, a balanced subsample, single-rung plan.  Not the
        scientific model — it validates imports/masking/copula/ontology/caching/export."""
        return replace(self, smoke=True, epochs=120, lr=2e-2)

    @property
    def stage_plan(self) -> list[GLLVMStage]:
        if self.smoke:
            return [
                GLLVMStage(
                    name="smoke_s8_full",
                    factors=list(self.factors),
                    windows=True,
                    correlated=True,
                    mixed=True,
                    n_subsample=400,
                    balanced=True,
                    epochs=self.epochs,
                    lr=self.lr,
                )
            ]
        return [
            GLLVMStage(
                name="s1_backbone",
                factors=[f for f in F_BACKBONE if f in self.factors],
                windows=False,
                correlated=True,
                mixed=False,
                epochs=max(1, self.epochs // 2),
                lr=self.lr,
            ),
            GLLVMStage(
                name="s8_full",
                factors=list(self.factors),
                windows=True,
                correlated=True,
                mixed=True,
                epochs=self.epochs,
                lr=self.lr,
                warmstart_from="s1_backbone",
            ),
        ]


# --------------------------------------------------------------------------- data
@dataclass
class GLLVMData:
    """Encoded full-item inputs for the variational engine (both modeling blocks)."""

    x: torch.Tensor  # (N, J) encoded, missing filled with 0
    mask: torch.Tensor  # (N, J) bool
    M_raw: np.ndarray  # (N, J) encoded with NaN at missing (for reliability + projector)
    ontology: LoadingOntology
    items: list[str]
    families: list[str]  # GLLVM channel family per item
    likelihood_families: list[str]  # original prior-matrix family per item
    blocks: list[str]  # modeling_block per item
    home: list[str]
    factor_cols: list[str]
    index: pd.Index
    cohort: np.ndarray
    ord_category_maps: dict[int, list[float]]
    item_signs: dict[str, int] = field(default_factory=dict)
    # Gaussian-copula inversion map: item -> (sorted oriented observed values, sorted rank-INT z),
    # for raw-scale synthetic generation (y_oriented = F^-1(Phi(z)) by interpolation).  Exact only
    # when the rank-INT is not covariate-residualized (use covariate_mode="none" for generation).
    copula: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def core_shim(self) -> CoreData:
        """A lightweight CoreData carrying just the fields ``LoadingSpec.from_core`` and
        ``PatientProjector.reliability_flags`` read (items / home / factor_cols / M)."""
        spec_factors = [f for f in self.factor_cols if f != G_KEY]
        return CoreData(
            M=self.M_raw,
            covariates=np.zeros((len(self.index), 0), dtype="float64"),
            covariate_names=[],
            items=self.items,
            home=self.home,
            factor_cols=self.factor_cols,
            spec_factors=spec_factors,
            g_col=self.factor_cols.index(G_KEY),
            cohort=self.cohort,
            index=self.index,
            families={it: f for it, f in zip(self.items, self.likelihood_families, strict=False)},
            signs={},
        )


class GLLVMDataset:
    """Build full-item, family-aware, copula-encoded inputs by composing
    ``MeasurementDataset`` (the data contract) — never re-reading the prior matrix or
    re-implementing the cell rules."""

    def __init__(self, config: GLLVMConfig):
        self.config = config
        self._md = MeasurementDataset(config.measurement_config())

    def build(
        self,
        factors: list[str],
        *,
        windows: bool,
        n_subsample: int | None = None,
        balanced: bool = False,
        seed: int = 20260605,
    ) -> GLLVMData:
        cfg = self.config
        md = self._md
        factor_cols = [G_KEY] + [f for f in factors if f != G_KEY]
        copula = cfg.likelihood_mode == "gaussian_copula"

        baseline = pd.read_parquet(cfg.processed_dir / "baseline_v0.parquet")
        # Full item set: every item whose home factor is active, BOTH modeling blocks.
        items = sorted(
            it
            for it, h in md.home.items()
            if h in factors and it in md.meta.index and it in baseline.columns
        )
        if windows:
            window_items = [
                w
                for w in WINDOWS
                if w in md.meta.index
                and w in baseline.columns
                and any(md._cell_type(w, f) == "plausible_cross" for f in factor_cols)
            ]
            items = sorted(set(items) | set(window_items))

        if n_subsample and n_subsample < len(baseline):
            rng = np.random.default_rng(seed)
            cohort_all = np.asarray(baseline.index.get_level_values("cohort"))
            if balanced:
                from face.models.bayesian.measurement_model_oop import _balanced_idx

                ix = _balanced_idx(cohort_all, n_subsample, rng)
            else:
                ix = np.sort(rng.choice(len(baseline), size=n_subsample, replace=False))
            baseline = baseline.iloc[ix]

        index = baseline.index
        cohort = np.asarray(index.get_level_values("cohort"))
        N = len(baseline)
        J = len(items)

        families: list[str] = []
        like_families: list[str] = []
        blocks: list[str] = []
        homes: list[str] = []
        ord_category_maps: dict[int, list[float]] = {}
        ord_n_cat: dict[int, int] = {}

        M_raw = np.full((N, J), np.nan, dtype="float64")
        gaussian_cols: dict[str, np.ndarray] = {}  # item -> oriented raw values (for residualize)
        gaussian_j: list[int] = []
        item_signs: dict[str, int] = {}

        for j, item in enumerate(items):
            fam = str(md.meta.loc[item, "likelihood_family"])
            sign = int(md.meta.loc[item, "item_sign"])
            item_signs[item] = sign
            block = str(md.meta.loc[item, "modeling_block"])
            like_families.append(fam)
            blocks.append(block)
            homes.append(md.home.get(item, ""))
            values = pd.to_numeric(baseline[item], errors="coerce").to_numpy("float64")
            obs = np.isfinite(values)

            gaussianize = copula and self._gaussianizable(item, baseline, fam)
            if fam in CONTINUOUS_FAMILIES or gaussianize:
                # Gaussian channel: orient by sign, rank-INT over observed cells (monotone-
                # invariant, so no log needed); residualization happens after, in z-space.
                oriented = sign * values
                M_raw[obs, j] = oriented[obs]  # store oriented raw; rank-INT applied below
                gaussian_cols[item] = M_raw[:, j].copy()
                gaussian_j.append(j)
                families.append("gaussian")
            elif fam == "bernoulli":
                v = values.copy()
                if sign < 0:
                    v[obs] = 1.0 - v[obs]
                M_raw[obs, j] = v[obs]
                families.append("bernoulli")
            elif fam == "ordered_logistic":
                v = values.copy()
                uniq = np.sort(np.unique(v[obs]))
                code = {val: i for i, val in enumerate(uniq)}
                coded = np.array([code.get(val, np.nan) for val in v], dtype="float64")
                C = max(2, len(uniq))
                if sign < 0:
                    coded[obs] = (C - 1) - coded[obs]
                M_raw[obs, j] = coded[obs]
                ord_category_maps[j] = [float(x) for x in uniq]
                ord_n_cat[j] = C
                families.append("ordinal")
            elif fam == "neg_binomial":
                v = np.clip(values, 0, None)
                M_raw[obs, j] = v[obs]
                families.append("count")
            else:
                raise ValueError(f"unsupported likelihood family for {item}: {fam}")

        # Rank-INT the gaussian-channel columns over observed cells, then FWL-residualize on
        # covariates (gaussian columns only) — mirrors MeasurementDataset's copula path.
        copula_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if gaussian_j:
            zcols: dict[str, np.ndarray] = {}
            for j in gaussian_j:
                col = M_raw[:, j]  # oriented raw values (pre rank-INT)
                obs = np.isfinite(col)
                z = np.full(col.shape, np.nan)
                if obs.any():
                    zo = _rank_int(col[obs])
                    z[obs] = zo
                    order = np.argsort(col[obs], kind="mergesort")
                    copula_map[items[j]] = (col[obs][order], np.sort(zo))  # oriented<->z, by rank
                M_raw[:, j] = z
                zcols[items[j]] = z
            if cfg.include_covariates and cfg.covariate_mode == "residualize":
                z_df = pd.DataFrame(zcols, index=index)
                z_df = self._residualize(z_df)
                for j in gaussian_j:
                    M_raw[:, j] = z_df[items[j]].to_numpy("float64")

        mask = np.isfinite(M_raw)
        x_filled = np.nan_to_num(M_raw, nan=0.0)
        ontology = self._build_ontology(items, homes, factor_cols, families, ord_n_cat)

        return GLLVMData(
            x=torch.as_tensor(x_filled, dtype=torch.float32),
            mask=torch.as_tensor(mask, dtype=torch.bool),
            M_raw=M_raw,
            ontology=ontology,
            items=items,
            families=families,
            likelihood_families=like_families,
            blocks=blocks,
            home=homes,
            factor_cols=factor_cols,
            index=index,
            cohort=cohort,
            ord_category_maps=ord_category_maps,
            item_signs=item_signs,
            copula=copula_map,
        )

    def _residualize(self, z_df: pd.DataFrame) -> pd.DataFrame:
        residualize = getattr(self._md, "residualize", None)
        if callable(residualize):
            return residualize(z_df)
        # Fallback: reuse the certified FWL helper directly (gaussian-scale columns only).
        return self._md._residualize_on_covariates(z_df)

    def _gaussianizable(self, item: str, baseline: pd.DataFrame, fam: str) -> bool:
        if fam in CONTINUOUS_FAMILIES:
            return True
        return self._md._gaussianizable(item, baseline)

    def _build_ontology(
        self,
        items: list[str],
        homes: list[str],
        factor_cols: list[str],
        families: list[str],
        ord_n_cat: dict[int, int],
    ) -> LoadingOntology:
        """Resolve the loading ontology by calling the certified ``LoadingSpec.from_core``
        over a CoreData shim (anti-drift: one classification source)."""
        cfg = self.config
        shim = CoreData(
            M=np.zeros((0, len(items)), dtype="float64"),
            covariates=np.zeros((0, 0), dtype="float64"),
            covariate_names=[],
            items=items,
            home=homes,
            factor_cols=factor_cols,
            spec_factors=[f for f in factor_cols if f != G_KEY],
            g_col=factor_cols.index(G_KEY),
            cohort=np.zeros(0),
            index=pd.Index([]),
            families={},
            signs={},
        )
        bifactor_g_sd = {f: 0.05 for f in cfg.explicit_factors if f != G_KEY}
        spec = LoadingSpec.from_core(
            shim,
            self._md.matrix,
            windows=True,
            soft_unlikely=cfg.soft_unlikely,
            soft_g_anchor_specific=cfg.soft_g_anchor_specific,
            specific_cross=cfg.specific_cross,
            horseshoe=False,
            cross_sd_scale=cfg.cross_sd_scale,
            bifactor_g_sd=bifactor_g_sd,
        )
        J, K = len(items), len(factor_cols)
        free = np.zeros((J, K), dtype=bool)
        positive = np.zeros((J, K), dtype=bool)
        prior_mean = np.zeros((J, K), dtype=np.float32)
        prior_sd = np.ones((J, K), dtype=np.float32)
        for (j, c, mu, sd) in spec.pos_cells:
            free[j, c] = True
            positive[j, c] = True
            prior_mean[j, c] = mu
            prior_sd[j, c] = max(sd, 1e-3)
        for (j, c, mu, sd) in spec.signed_cells:
            free[j, c] = True
            prior_mean[j, c] = mu
            prior_sd[j, c] = max(sd, 1e-3)
        for (j, c) in spec.hs_cells:  # empty in the operational (no-horseshoe) map
            free[j, c] = True
            prior_mean[j, c] = 0.0
            prior_sd[j, c] = 0.30
        return LoadingOntology(
            free_mask=free,
            positive_mask=positive,
            prior_mean=prior_mean,
            prior_sd=prior_sd,
            item_family=families,
            ord_n_cat=ord_n_cat,
            kind=dict(spec.kind),
        )


# --------------------------------------------------------------------------- runner
def _config_sig(config: GLLVMConfig) -> dict:
    """Structural signature for cache reuse — only fields that change the estimand
    (NOT lr / epochs / device, which are recorded in the manifest instead)."""
    return {
        "prior_matrix": config.prior_matrix.name,
        "factors": list(config.factors),
        "explicit_factors": list(config.explicit_factors),
        "orthogonal_factors": list(config.orthogonal_factors),
        "specific_cross": config.specific_cross,
        "cross_sd_scale": config.cross_sd_scale,
        "soft_unlikely": config.soft_unlikely,
        "soft_g_anchor_specific": config.soft_g_anchor_specific,
        "likelihood_mode": config.likelihood_mode,
        "copula_min_distinct": config.copula_min_distinct,
        "copula_max_modal_frac": config.copula_max_modal_frac,
        "include_covariates": config.include_covariates,
        "covariate_mode": config.covariate_mode if config.include_covariates else "none",
        "psi_floor": config.psi_floor,
        "phi_penalty_weight": config.phi_penalty_weight,
        "q_family": config.q_family,
        "q_rank": config.q_rank,
        "full_cov": config.full_cov,
    }


def _stage_spec(stage: GLLVMStage) -> dict:
    return {
        "name": stage.name,
        "factors": list(stage.factors),
        "windows": stage.windows,
        "correlated": stage.correlated,
        "mixed": stage.mixed,
        "n_subsample": stage.n_subsample,
        "balanced": stage.balanced,
        "warmstart_from": stage.warmstart_from,
    }


class GLLVMRunner:
    """Staged, cached variational fits with name-matched warm-start and a consolidate
    hand-off — mirrors ``StageRunner``'s cache contract (MODEL_VERSION + stage_spec +
    config_sig)."""

    def __init__(self, config: GLLVMConfig | None = None):
        self.config = config or GLLVMConfig()
        self.dataset = GLLVMDataset(self.config)
        self._fits: dict[str, dict] = {}

    def _device(self) -> torch.device:
        want = self.config.device
        if want == "mps" and not torch.backends.mps.is_available():
            print("[gllvm] MPS requested but unavailable; falling back to CPU", flush=True)
            return torch.device("cpu")
        return torch.device(want)

    def _orthogonal_indices(self, factor_cols: list[str]) -> tuple[int, ...]:
        idx = {0}  # G always
        for f in self.config.orthogonal_factors:
            if f in factor_cols:
                idx.add(factor_cols.index(f))
        return tuple(sorted(idx))

    def run_stage(
        self, stage: GLLVMStage, *, overwrite: bool = False, prev: dict | None = None
    ) -> dict:
        out_dir = self.config.output_dir / stage.name
        manifest_path = out_dir / "manifest.json"
        sig = {"model_version": MODEL_VERSION, "stage_spec": _stage_spec(stage),
               "config_sig": _config_sig(self.config)}
        if manifest_path.exists() and not overwrite:
            cached = json.loads(manifest_path.read_text())
            if cached.get("cache_key") == sig:
                print(f"[gllvm] stage={stage.name} cached -> {out_dir}", flush=True)
                fit = self._load_stage(stage, out_dir)
                self._fits[stage.name] = fit
                return fit

        device = self._device()
        data = self.dataset.build(
            stage.factors,
            windows=stage.windows,
            n_subsample=stage.n_subsample,
            balanced=stage.balanced,
            seed=self.config.seed,
        )
        N = len(data.index)
        model = VariationalGLLVM(
            N,
            data.ontology,
            orthogonal_indices=self._orthogonal_indices(data.factor_cols),
            psi_floor=self.config.psi_floor,
            phi_penalty_weight=self.config.phi_penalty_weight,
            sigma_prior_weight=self.config.sigma_prior_weight,
            count_alpha_prior_weight=self.config.count_alpha_prior_weight,
            cutpoint_prior_weight=self.config.cutpoint_prior_weight,
            q_rank=self.config.q_rank,
            full_cov=self.config.full_cov,
            seed=self.config.seed,
        )
        model.initialize_from_data(data.x, data.mask)
        if prev is not None and stage.warmstart_from == prev.get("stage"):
            self._warmstart(model, data, prev)
        model.attach_data(data.x, data.mask)
        model.to(device)

        # Live progress: write the latest training record to progress.json every ~25 epochs so
        # a detached run can be monitored (in addition to the stdout/log trace).
        out_dir.mkdir(parents=True, exist_ok=True)
        progress_path = out_dir / "progress.json"

        def _progress(record: dict) -> None:
            if record["epoch"] % 25 == 0 or record["epoch"] == 1:
                progress_path.write_text(json.dumps(
                    {"stage": stage.name, "epochs_target": stage.epochs, **record}, indent=2))

        t0 = time.time()
        trainer = GLLVMTrainer(
            model,
            TrainingConfig(
                epochs=stage.epochs,
                lr=stage.lr,
                n_mc=self.config.n_mc_samples,
                batch_size=self.config.batch_size,
                grad_clip_norm=self.config.grad_clip_norm,
                seed=self.config.seed,
                early_stop_patience=self.config.early_stop_patience,
                early_stop_rel_tol=self.config.early_stop_rel_tol,
            ),
            progress=_progress,
        )
        history = trainer.fit()
        elapsed = time.time() - t0

        fit = {
            "stage": stage.name,
            "model": model,
            "data": data,
            "history": history,
            "factor_cols": data.factor_cols,
            "stopped_early": trainer.stopped_early,
        }
        self._persist_stage(stage, out_dir, fit, sig, elapsed)
        self._fits[stage.name] = fit
        return fit

    def run_plan(self, *, overwrite: bool = False, stop_after: str | None = None) -> dict:
        prev: dict | None = None
        last: dict | None = None
        for stage in self.config.stage_plan:
            fit = self.run_stage(stage, overwrite=overwrite, prev=prev)
            prev = fit
            last = fit
            if stop_after and stage.name == stop_after:
                break
        return last or {}

    def consolidate(self, fit: dict | None = None) -> Path:
        fit = fit or self._fits.get(self.config.stage_plan[-1].name)
        if fit is None:
            raise RuntimeError("no fit to consolidate; run the plan first")
        out = self.config.output_dir / "consolidate"
        out.mkdir(parents=True, exist_ok=True)
        proj = GLLVMProjector(self.config)
        proj.coordinates_frame(fit).to_parquet(out / "coordinates.parquet")
        proj.loadings_summary(fit).to_csv(out / "loadings_summary.csv", index=False)
        proj.phi_frame(fit).to_csv(out / "phi.csv")
        # Posterior predictive summary (computed while the model is in memory).
        try:
            from face.models.variational import validate as _V

            _V.ppc_from_fit(fit).to_csv(out / "ppc.csv", index=False)
        except Exception as exc:  # PPC is a diagnostic, never block the hand-off
            print(f"[gllvm] ppc skipped: {exc}", flush=True)
        (out / "manifest.json").write_text(
            json.dumps(
                {"model_version": MODEL_VERSION, "canonical_stage": fit["stage"],
                 "factor_cols": fit["factor_cols"], "config_sig": _config_sig(self.config)},
                indent=2,
            )
        )
        print(f"[gllvm] consolidated -> {out}", flush=True)
        return out

    # ---------------------------------------------------------------- internals
    def _warmstart(self, model: VariationalGLLVM, data: GLLVMData, prev: dict) -> None:
        """Copy shared (item, factor) loadings and shared q-coordinate columns from the
        previous rung — protects thin factors from ELBO-flat collapse."""
        prev_model: VariationalGLLVM = prev["model"]
        prev_data: GLLVMData = prev["data"]
        prev_items = {it: j for j, it in enumerate(prev_data.items)}
        prev_factors = {f: c for c, f in enumerate(prev_data.factor_cols)}
        with torch.no_grad():
            pl = prev_model.raw_loading.detach().cpu()
            for j, item in enumerate(data.items):
                if item not in prev_items:
                    continue
                pj = prev_items[item]
                for c, factor in enumerate(data.factor_cols):
                    if factor in prev_factors and bool(data.ontology.free_mask[j, c]):
                        model.raw_loading.data[j, c] = pl[pj, prev_factors[factor]]
            # Patients are row-aligned only when the index matches; copy the shared factor
            # columns of q when the patient sets align (same N and order).
            if prev_data.index.equals(data.index):
                pmu = prev_model.q_mu.weight.detach().cpu()
                plv = prev_model.q_logvar.weight.detach().cpu()
                for c, factor in enumerate(data.factor_cols):
                    if factor in prev_factors:
                        model.q_mu.weight.data[:, c] = pmu[:, prev_factors[factor]]
                        model.q_logvar.weight.data[:, c] = plv[:, prev_factors[factor]]

    def _persist_stage(
        self, stage: GLLVMStage, out_dir: Path, fit: dict, sig: dict, elapsed: float
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        proj = GLLVMProjector(self.config)
        proj.coordinates_frame(fit).to_parquet(out_dir / "coordinates.parquet")
        proj.loadings_summary(fit).to_csv(out_dir / "loadings_summary.csv", index=False)
        proj.phi_frame(fit).to_csv(out_dir / "phi.csv")
        pd.DataFrame(fit["history"]).to_csv(out_dir / "training_history.csv", index=False)
        torch.save(
            {"state_dict": fit["model"].state_dict(), "items": fit["data"].items,
             "factor_cols": fit["data"].factor_cols, "families": fit["data"].families,
             "ord_n_cat": fit["data"].ontology.ord_n_cat},
            out_dir / "model_state.pt",
        )
        history = fit["history"]
        manifest = {
            "cache_key": sig,
            "model_version": MODEL_VERSION,
            "stage": stage.name,
            "elapsed_sec": round(elapsed, 1),
            "N": int(len(fit["data"].index)),
            "J": int(len(fit["data"].items)),
            "factors": list(fit["data"].factor_cols),
            "epochs_run": int(history[-1]["epoch"]) if history else 0,
            "stopped_early": bool(fit.get("stopped_early", False)),
            "final_elbo": float(history[-1]["loss"]) if history else None,
            "final_nll": float(history[-1]["nll"]) if history else None,
            "final_kl": float(history[-1]["kl"]) if history else None,
            "final_grad_norm": float(history[-1].get("grad_norm", float("nan"))) if history else None,
            "optimizer": {"lr": stage.lr, "epochs": stage.epochs,
                          "batch_size": self.config.batch_size, "device": self.config.device,
                          "early_stop_patience": self.config.early_stop_patience},
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _load_stage(self, stage: GLLVMStage, out_dir: Path) -> dict:
        data = self.dataset.build(
            stage.factors,
            windows=stage.windows,
            n_subsample=stage.n_subsample,
            balanced=stage.balanced,
            seed=self.config.seed,
        )
        model = VariationalGLLVM(
            len(data.index),
            data.ontology,
            orthogonal_indices=self._orthogonal_indices(data.factor_cols),
            psi_floor=self.config.psi_floor,
            q_rank=self.config.q_rank,
            full_cov=self.config.full_cov,
            seed=self.config.seed,
        )
        state = torch.load(out_dir / "model_state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        history = pd.read_csv(out_dir / "training_history.csv").to_dict("records")
        return {"stage": stage.name, "model": model, "data": data, "history": history,
                "factor_cols": data.factor_cols}


# --------------------------------------------------------------------------- projector
class GLLVMProjector:
    """Export coordinates / loadings / phi in the canonical (NUTS-engine) schema."""

    def __init__(self, config: GLLVMConfig | None = None):
        self.config = config or GLLVMConfig()

    def coordinates_frame(self, fit: dict, *, hdi_prob: float = 0.94) -> pd.DataFrame:
        model: VariationalGLLVM = fit["model"]
        data: GLLVMData = fit["data"]
        coords = model.coordinates()
        mean, sd = coords["mean"], coords["sd"]
        n_obs, tier = PatientProjector.reliability_flags(data.core_shim())
        z = float(norm.ppf(1 - (1 - hdi_prob) / 2))
        frame = pd.DataFrame(index=data.index)
        for c, factor in enumerate(data.factor_cols):
            frame[f"{factor}__mean"] = mean[:, c]
            frame[f"{factor}__sd"] = sd[:, c]
            frame[f"{factor}__hdi_low"] = mean[:, c] - z * sd[:, c]
            frame[f"{factor}__hdi_high"] = mean[:, c] + z * sd[:, c]
            frame[f"{factor}__n_obs"] = n_obs[:, c]
            frame[f"{factor}__reliability"] = tier[:, c]
        return frame

    def loadings_summary(self, fit: dict) -> pd.DataFrame:
        """One row per interpretable (free) loading cell.  Same columns as the NUTS
        ``export_loadings_summary``; ``ci_*``/``excludes_zero`` are NaN/NA because the VI
        loadings are MAP point estimates (uncertainty deferred to bootstrap / NUTS)."""
        model: VariationalGLLVM = fit["model"]
        data: GLLVMData = fit["data"]
        lam = model.loadings()
        kind = data.ontology.kind
        fidx = {f: i for i, f in enumerate(data.factor_cols)}
        rows: list[dict] = []
        for j, item in enumerate(data.items):
            h = data.home[j]
            fam = data.likelihood_families[j]
            block = data.blocks[j]
            for c, factor in enumerate(data.factor_cols):
                if not bool(data.ontology.free_mask[j, c]):
                    continue
                kk = kind.get((j, c)) or ("primary" if factor == h else
                                         "bifactor_G" if factor == G_KEY else "cross")
                val = float(lam[j, c])
                rows.append(
                    dict(item=item, factor=factor, home=h or "", block=block,
                         likelihood_family=fam, kind=kk, loading=val, abs_loading=abs(val),
                         ci_low=np.nan, ci_high=np.nan, excludes_zero=pd.NA)
                )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["__f"] = df["factor"].map(fidx)
        return df.sort_values(["item", "__f"]).drop(columns="__f").reset_index(drop=True)

    def phi_frame(self, fit: dict) -> pd.DataFrame:
        model: VariationalGLLVM = fit["model"]
        data: GLLVMData = fit["data"]
        return pd.DataFrame(model.phi_matrix(), index=data.factor_cols, columns=data.factor_cols)


# --------------------------------------------------------------------------- visualizer
class GLLVMVisualizer:
    """Small notebook plotting helpers (ELBO trace, Phi heatmap, VI-vs-NUTS overlay)."""

    def __init__(self, config: GLLVMConfig | None = None):
        self.config = config or GLLVMConfig()
        self.config.figure_dir.mkdir(parents=True, exist_ok=True)

    def elbo_trace(self, fit: dict, name: str = "elbo_trace") -> str:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]

        hist = pd.DataFrame(fit["history"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist["epoch"], hist["loss"], label="-ELBO")
        ax.plot(hist["epoch"], hist["nll"], label="NLL", alpha=0.6)
        ax.plot(hist["epoch"], hist["kl"], label="KL", alpha=0.6)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.set_title(f"GLLVM {fit['stage']} training")
        path = self.config.figure_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)

    def training_curves(self, history, *, title: str = "GLLVM training", name: str = "training_curves") -> str:
        """Multi-panel SVI diagnostic trace: -ELBO + components (log-y), and the gradient
        norm.  ``history`` is a fit dict, a list of records, or a DataFrame."""
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]

        if isinstance(history, dict) and "history" in history:
            hist = pd.DataFrame(history["history"])
        elif isinstance(history, pd.DataFrame):
            hist = history
        else:
            hist = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ax = axes[0]
        ax.plot(hist["epoch"], hist["loss"], color="C3", lw=2, label="-ELBO")
        ax.set_xlabel("epoch"); ax.set_ylabel("-ELBO"); ax.set_title("Negative ELBO"); ax.legend()
        ax = axes[1]
        for col, c in (("nll", "C0"), ("kl", "C1"), ("penalty", "C2")):
            if col in hist:
                ax.plot(hist["epoch"], hist[col].clip(lower=1e-6), color=c, label=col, alpha=0.85)
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("component (log)")
        ax.set_title("ELBO components"); ax.legend()
        ax = axes[2]
        if "grad_norm" in hist:
            ax.plot(hist["epoch"], hist["grad_norm"], color="C4")
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("|grad| (log)")
        ax.set_title("Gradient norm")
        fig.suptitle(title)
        path = self.config.figure_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)

    def phi_heatmap(self, fit: dict, name: str = "phi_heatmap") -> str:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]

        phi = GLLVMProjector(self.config).phi_frame(fit)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(phi.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(phi)))
        ax.set_yticks(range(len(phi)))
        ax.set_xticklabels(phi.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(phi.index, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("GLLVM Phi")
        path = self.config.figure_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return str(path)
