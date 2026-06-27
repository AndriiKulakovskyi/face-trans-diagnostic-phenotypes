"""OOP soft-region stratification engine on the Gaussian-copula measurement map (M2, reworked).

This is a clean, config-first OOP engine that reworks the FACE M2 stratification as a **continuum with
soft operational regions** built on the Gaussian-copula map (the cohort-weighted full-N **8-dimension**
map — the immunometabolic merge + 3 earned cross-loadings — at
``results/face/oop_measurement/copula/weighted_8d``). It is a parallel engine beside the
imperative ``scripts/20-26`` + ``src/face/strata/{mixture,structure,archetypes,validation}.py``, exactly as
``src/face/models/bayesian/measurement_model_oop.py`` lives beside the certified ``continuous_core``.

Design stance (deliberate):
  * The 8-dim space is a **continuum, not a clustering problem.** We do not look for natural kinds; we
    define **operational regions with soft transition boundaries** (probabilistic membership, no hard edges)
    and then ask whether those regions are **useful** (here: a baseline/internal question — temporal
    persistence and prognosis are deferred to the later M3/M4 reruns).
  * All validated math is **reused, not reimplemented** — the OOP classes wrap the proven array-in/array-out
    kernels in ``src/face/strata/{mixture,structure,archetypes,validation}.py`` and ``scoring.py``. This
    module is orchestration + the copula-coordinate handoff + the operational-region framing + visualization.
  * **No imputation** (uncertainty propagates; unobserved cells, e.g. substance=DR, inherit a wide posterior
    SD and self-down-weight in the measurement-error mixture). **Diagnosis is validation-only**, never a
    model input. UMAP/PCA are **visualization-only**, never a clustering input.

Layers (mirroring ``measurement_model_oop``):
  * ``StrataConfig``      — frozen config + ``with_*()`` factories + ``_config_sig`` cache key.
  * ``StrataStage``       — one stage recipe of the deterministic plan.
  * ``CoordinateSet``     — the per-patient (X, S, draws) coordinate container + ``arm()`` slicing.
  * ``StrataData``        — build (``prepare``) + load the copula coordinate set.
  * ``StructureGate``     — cluster-vs-continuum discovery gate (wraps ``structure.py``).
  * ``SoftRegionModel``   — soft operational regions: measurement-error mixture (wraps ``mixture.py``).
  * ``ArchetypeModel``    — continuum co-view: archetypal analysis (wraps ``archetypes.py``).
  * ``UsefulnessValidator``— internal usefulness battery (wraps ``validation.py``).
  * ``StrataRunner``      — staged, cached orchestration (deterministic; no MCMC warm-start).
  * ``StrataProjector``   — per-patient membership frame (the M3-compatible hand-off).
  * ``StrataVisualizer``  — figures (structure panel, region/archetype profiles, soft-boundary map, ...).

Unlike ``measurement_model_oop`` the stages are deterministic numpy/EM (not MCMC), so ``StrataRunner``
caches ``.npz``/``.parquet`` artifacts (not ``idata.nc``) with the same ``model_version`` + ``stage_spec`` +
``config_sig`` reuse guard, and there is **no warm-start** (the one intentional simplification).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
COPULA_MAP = REPO / "results" / "face" / "oop_measurement" / "copula" / "weighted_8d"
RESULTS = REPO / "results" / "face" / "strata_oop"
FIGURES = REPO / "docs" / "figures" / "strata_oop"
REPORTS = REPO / "reports"
MODEL_VERSION = "strata_oop_2026_06_26_v2_8factor"

G_KEY = "overall_severity"
MAP_STAGE = "hs_s5_merged_xc"          # the weighted full-N 8-factor operational fit (M1 the coordinates read)
FOLDED_MATRIX = REPO / "configs" / "prior_loading_matrix_v3_biomerge_xc.csv"
# Fit order: the exact factor list the 8-factor map was fit with — must match the idata's Lam/Phi/f_e columns.
F8_FIT = ["overall_severity", "cognition", "immunometabolic", "sleep", "suicidality",
          "developmental_risk", "mania_activation", "substance"]
# Canonical 8-dim presentation order. The immunometabolic merge folds the two continuous biology factors
# (metabolic + inflammatory) into one; everything else is unchanged.
CANON = ["overall_severity", "cognition", "immunometabolic", "sleep", "mania_activation",
         "suicidality", "developmental_risk", "substance"]
SPECIFICS = [f for f in CANON if f != G_KEY]          # Arm B = the G-residualized "pure profile" view
# Reliability split: which axes are scored from the continuous block vs the explicit (f_e) block under the
# copula vertical. immunometabolic is continuous (merge of the two continuous biology factors); mania_activation
# stays continuous; only suicidality / developmental_risk / substance remain explicit-anchored (Ke=4 with G,
# unchanged by the merge).
CONT_AXES = ["overall_severity", "cognition", "immunometabolic", "sleep", "mania_activation"]
EXPL_AXES = ["suicidality", "developmental_risk", "substance"]


@dataclass(frozen=True)
class StrataStage:
    """One rung of the deterministic strata plan (analogue of ``StageDefinition``)."""
    name: str
    kind: str                                  # dispatch key: coordinates|structure|regions|archetypes|usefulness|consolidate
    seed: int = 20260621
    # coordinate prep
    n_coord_draws: int = 200                   # joint posterior draws kept per patient
    # region (XD mixture) knobs
    K: int | None = None                       # tessellation granularity; None => operational-K selection
    K_sweep: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    K_family: tuple[int, ...] = (2, 3, 4)      # nested K-family exported in the hand-off as conventions; the
    #                                            OPERATIVE K is deferred to M4/M5 incremental validity (not a
    #                                            cache key — the consolidate stage always rebuilds, and the
    #                                            family fits are cheap post-hoc projections of the same cloud)
    # archetype knobs
    A: int | None = None                       # None => scree knee (operational choice)
    A_sweep: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    n_draw: int = 40                           # posterior draws projected for membership uncertainty
    arms: tuple[str, ...] = ("A", "B")         # A = all-9 (full phenotype), B = specifics-only (G-residualized)
    # structure-gate knobs
    n_uncertainty_draws: int = 20


@dataclass(frozen=True)
class StrataConfig:
    """Central configuration: paths + model-affecting switches (the latter enter ``_config_sig``)."""
    map_dir: Path = COPULA_MAP                 # the certified copula fit consumed
    output_dir: Path = RESULTS
    figure_dir: Path = FIGURES
    hdi_prob: float = 0.94
    seed: int = 20260621
    # --- model-affecting (gate cache reuse) ---
    coord_source: str = "copula_weighted_8d"   # provenance label of the map (8-factor immunometabolic merge)
    use_full_Si: bool = False                  # diagonal S (default) vs full [N,8,8] per-patient covariance
    region_reg: float = 1e-4                   # XD covariance regularizer
    knee_gain: float = 0.02                    # archetype scree-knee threshold
    confidence_tiers: tuple[float, float] = (0.5, 0.8)   # boundary (<.5) / soft / core (>=.8) on max membership
    smoke: bool = False

    # ---- convenience factories (mirroring MeasurementConfig.with_*) ----
    def with_full_si(self) -> StrataConfig:
        """Use the full per-patient covariance S_i [N,9,9] (the faithfulness arm) instead of diagonal."""
        return replace(self, use_full_Si=True)

    def with_smoke_defaults(self) -> StrataConfig:
        """Fast wiring config: tiny sweeps and few draws (validates the path, not the science)."""
        return replace(self, smoke=True)

    @property
    def coords_dir(self) -> Path:
        return self.output_dir / "coordinates"

    @property
    def stage_plan(self) -> list[StrataStage]:
        """The five-stage continuum plan: coordinates -> structure -> regions -> archetypes ->
        usefulness -> consolidate (the lead operational view is the soft tessellation; archetypes are the
        continuum co-view)."""
        if self.smoke:
            return [
                StrataStage("coordinates", "coordinates", n_coord_draws=40),
                StrataStage("structure", "structure", n_uncertainty_draws=4),
                StrataStage("regions", "regions", K_sweep=(2, 3, 4)),
                StrataStage("archetypes", "archetypes", A_sweep=(2, 3, 4), n_draw=8),
                StrataStage("usefulness", "usefulness"),
                StrataStage("consolidate", "consolidate"),
            ]
        return [
            StrataStage("coordinates", "coordinates", n_coord_draws=200),
            StrataStage("structure", "structure", n_uncertainty_draws=20),
            StrataStage("regions", "regions"),
            StrataStage("archetypes", "archetypes", n_draw=40),
            StrataStage("usefulness", "usefulness"),
            StrataStage("consolidate", "consolidate"),
        ]


def _config_sig(config: StrataConfig) -> dict:
    """Model-affecting fields only — guards cache reuse so a diagonal-S run never reuses a full-S artifact."""
    return {
        "coord_source": str(config.coord_source),
        "use_full_Si": bool(config.use_full_Si),
        "region_reg": float(config.region_reg),
        "knee_gain": float(config.knee_gain),
        "confidence_tiers": list(config.confidence_tiers),
        "hdi_prob": float(config.hdi_prob),
        "smoke": bool(config.smoke),
    }


def _stage_spec(stage: StrataStage) -> dict:
    """The stage recipe that, with model_version + config_sig, keys the cache."""
    return {"name": stage.name, "kind": stage.kind, "seed": stage.seed,
            "n_coord_draws": stage.n_coord_draws, "K": stage.K, "K_sweep": list(stage.K_sweep),
            "A": stage.A, "A_sweep": list(stage.A_sweep), "n_draw": stage.n_draw,
            "arms": list(stage.arms), "n_uncertainty_draws": stage.n_uncertainty_draws}


def _normalized_entropy(resp: np.ndarray) -> np.ndarray:
    """Per-row normalized membership entropy H_i = -Sum_k r_ik log r_ik / log K, in [0, 1].
    0 = a single dominant region (core), 1 = maximally on-the-fence (a boundary patient)."""
    r = np.clip(resp, 1e-12, 1.0)
    K = r.shape[1]
    if K <= 1:
        return np.zeros(r.shape[0])
    return (-(r * np.log(r)).sum(1)) / np.log(K)


def _confidence_tier(resp: np.ndarray, tiers: tuple[float, float]) -> np.ndarray:
    """Per-patient assignment-confidence tier from the max membership: this is what makes the soft
    transition boundaries operational. ``boundary`` (max < lo) = the patient lives between regions;
    ``soft`` (lo <= max < hi); ``core`` (max >= hi) = a confident dominant region."""
    lo, hi = tiers
    mx = resp.max(1)
    return np.where(mx >= hi, "core", np.where(mx >= lo, "soft", "boundary"))


@dataclass
class CoordinateSet:
    """Per-patient coordinates on the copula map + propagated uncertainty (analogue of ``CoreData``)."""
    X: np.ndarray                              # [N, 8] posterior-mean coords (latent z-scale; NO re-standardize)
    S: np.ndarray                              # [N, 8] diagonal sd**2  OR  [N, 8, 8] full cov if use_full_Si
    draws: np.ndarray                          # [n_draw, N, 8] coherent joint posterior draws
    dims: list[str]                            # == CANON
    index: pd.MultiIndex                       # (cohort, patient_id)
    n_obs: np.ndarray                          # [N, 8] observed home-indicator counts (reliability)
    reliability: np.ndarray                    # [N, 8] tiers {well, partial, prior-dominated}
    validation: pd.DataFrame                   # cohort/arm/age/sex/education/site -- VALIDATION ONLY

    def arm(self, which: str = "A") -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Return (X, S, col-indices) for Arm A (all-9) or Arm B (specifics-only, G-residualized).
        ``draws`` are sliced by the returned col-indices at the call site (kept whole here)."""
        if which == "A":
            cols = list(range(len(self.dims)))
        elif which == "B":
            cols = [self.dims.index(f) for f in SPECIFICS]
        else:
            raise ValueError(f"arm must be 'A' or 'B', got {which!r}")
        Xa = self.X[:, cols]
        Sa = self.S[:, cols] if self.S.ndim == 2 else self.S[np.ix_(range(self.S.shape[0]), cols, cols)]
        return Xa, Sa, cols


class StrataData:
    """Build (``prepare``) and load the per-patient copula coordinate set.

    The coordinate set is produced ONCE from the certified copula fit and cached; downstream stages read it.
    Because the cohort-weighted copula fit was run at full N, its posterior already contains the per-patient
    explicit latents ``f_e`` for [overall_severity, suicidality, developmental_risk, substance] — so unlike
    the native M2.0 we do NOT re-run an expensive full-N projection. We read ``f_e`` from the posterior and
    condition the marginalized specifics ``f_m | f_e`` (reusing ``scoring.coherent_joint_coords``), giving one
    coherent 8-dim posterior draw per sample (the no-imputation, uncertainty-propagating contract).
    """

    def __init__(self, config: StrataConfig | None = None):
        self.config = config or StrataConfig()

    # -- the coordinate artifacts (mirror the legacy results/face/m2 contract, in our own subdir) --
    @property
    def _full(self) -> Path:
        return self.config.coords_dir / "coordinates_full.parquet"

    @property
    def _draws(self) -> Path:
        return self.config.coords_dir / "coordinates_draws.npz"

    @property
    def _cov(self) -> Path:
        return self.config.coords_dir / "coordinates_cov.npz"

    @property
    def _vt(self) -> Path:
        return self.config.coords_dir / "validation_table.parquet"

    def prepare(self, *, n_coord_draws: int = 200, overwrite: bool = False) -> CoordinateSet:
        """Score all 9,013 patients on all 8 axes from the copula fit; cache the coordinate artifacts.

        Reads the explicit latents ``f_e`` directly from the copula posterior (already full-N), conditions
        the marginalized specifics ``f_m | f_e`` under the shared Phi (``coherent_joint_coords``), and assigns
        reliability tiers (continuous axes from observed home-indicator counts; explicit axes from
        ``explicit_nobs``). No re-fit, no imputation. Reads the 8-factor operational map (immunometabolic
        merge + the 3 earned cross-loadings) at ``MAP_STAGE`` under the folded prior matrix.
        """
        if self._full.exists() and not overwrite:
            return self.load()

        import arviz as az  # noqa: PLC0415
        from scipy.stats import norm  # noqa: PLC0415

        from face.models.bayesian.measurement_model_oop import (  # noqa: PLC0415
            DEFAULT_EXPLICIT_FACTORS,
            MeasurementConfig,
            MeasurementDataset,
            PatientProjector,
        )
        from face.strata.scoring import coherent_joint_coords, explicit_nobs  # noqa: PLC0415

        self.config.coords_dir.mkdir(parents=True, exist_ok=True)
        # 1) rebuild the copula full-N MixedData (same call the weighted 8-factor fit used -> row-aligned to f_e)
        mcfg = MeasurementConfig(likelihood_mode="gaussian_copula", cohort_weighted=True,
                                 prior_matrix=FOLDED_MATRIX, output_dir=self.config.map_dir)
        dataset = MeasurementDataset(mcfg)
        mp = dataset.mixed(F8_FIT, explicit_factors=DEFAULT_EXPLICIT_FACTORS, min_cohorts=2,
                           balanced=False, n_subsample=None)
        idata = az.from_netcdf(str(self.config.map_dir / MAP_STAGE / "idata.nc"))
        manifest = json.loads((self.config.map_dir / MAP_STAGE / "manifest.json").read_text())
        base_index = mp.base.index

        # 2) explicit latents f_e straight from the posterior (already full-N) -> coherent_joint_coords only
        #    needs the draws + a diag dict (it recomputes mean/sd/cov from the assembled coords).
        fe = np.asarray(idata.posterior["f_e"].values)            # [chain, draw, N, Ke]
        Se = fe.shape[0] * fe.shape[1]
        fe = fe.reshape((Se,) + fe.shape[2:])                     # [Se, N, Ke], mp.e_cols order
        diag = {"source": "copula_posterior_f_e", "fit_rhat": manifest.get("diagnostics", {}).get("rhat"),
                "n_draws": int(min(n_coord_draws, Se))}
        ch = coherent_joint_coords(mp, idata, projection={"draws": fe, "diag": diag}, n_draws=n_coord_draws)

        # 3) reliability: continuous axes from the continuous home-indicator counts; explicit axes from f_e
        nobs_c, tier_c = PatientProjector.reliability_flags(mp.base)   # [N, 8] over mp.base.factor_cols
        cidx = {f: i for i, f in enumerate(mp.base.factor_cols)}
        en = explicit_nobs(mp)                                    # n_obs over [G, suic, dev, substance]
        en_df = pd.DataFrame(en["n_obs"], index=base_index, columns=en["fcols"])

        # 4) assemble the per-patient frame in CANON order
        cmap = {f: ch["cols"].index(f) for f in CANON}
        z = float(norm.ppf(1 - (1 - self.config.hdi_prob) / 2))
        N = ch["mean"].shape[0]
        full = pd.DataFrame(index=base_index)
        X = np.full((N, len(CANON)), np.nan)
        sd = np.full((N, len(CANON)), np.nan)
        draws = np.full((ch["draws"].shape[0], N, len(CANON)), np.nan, dtype="float32")
        for di, f in enumerate(CANON):
            ci = cmap[f]
            m, s = ch["mean"][:, ci], ch["sd"][:, ci]
            if f in EXPL_AXES:
                n = en_df[f].to_numpy()
                rel = np.where(n >= 3, "well", np.where(n >= 1, "partial", "prior-dominated"))
            else:
                n, rel = nobs_c[:, cidx[f]], tier_c[:, cidx[f]]
            X[:, di], sd[:, di] = m, s
            draws[:, :, di] = ch["draws"][:, :, ci]
            full[f"{f}__mean"] = np.round(m, 3)
            full[f"{f}__sd"] = np.round(s, 3)
            full[f"{f}__hdi_lo"] = np.round(m - z * s, 3)
            full[f"{f}__hdi_hi"] = np.round(m + z * s, 3)
            full[f"{f}__n_obs"] = n
            full[f"{f}__reliability"] = rel
        order = [cmap[f] for f in CANON]
        cov = ch["cov"][np.ix_(range(N), order, order)].astype("float32")   # per-patient S_i in CANON order

        # 5) validation table (validation-only; never a clustering input)
        vt = self._validation_table(base_index)

        # 6) cache
        cohort = np.asarray(base_index.get_level_values("cohort"))
        pid = np.asarray(base_index.get_level_values("patient_id"))
        full.reset_index().to_parquet(self._full)
        np.savez_compressed(self._draws, draws=draws, dims=np.array(CANON), cohort=cohort, patient_id=pid)
        np.savez_compressed(self._cov, cov=cov, dims=np.array(CANON), cohort=cohort, patient_id=pid)
        vt.reset_index().to_parquet(self._vt)
        (self.config.coords_dir / "manifest.json").write_text(json.dumps({
            "model_version": MODEL_VERSION, "stage": "coordinates", "config_sig": _config_sig(self.config),
            "N": int(N), "n_coord_draws": int(draws.shape[0]), "fit_rhat": diag["fit_rhat"]}, indent=2))
        return self.load()

    def _validation_table(self, base_index) -> pd.DataFrame:
        """cohort / arm (DSM-5 subtype) / age / sex / education / site — from the unified dictionary df."""
        from face.data import build_unified_dataframe  # noqa: PLC0415
        w = build_unified_dataframe("data", "data/face-common-vars.xlsx", ["READY", "PARTIAL"], format="wide")
        w = w.assign(cohort=w["cohort"].astype(str).str.lower(), patient_id=w["usubjid_patients"].astype(str))
        keep = {"age_V0": "age", "sex_V0": "sex", "education_years_V0": "education_years",
                "siteid_city_V0": "siteid_city", "arm": "arm"}
        return w.set_index(["cohort", "patient_id"])[list(keep)].rename(columns=keep).reindex(base_index)

    def load(self) -> CoordinateSet:
        """Read the cached coordinate artifacts into a ``CoordinateSet``."""
        if not self._full.exists():
            raise FileNotFoundError(f"coordinates not prepared — run StrataData.prepare() ({self._full})")
        full = pd.read_parquet(self._full).set_index(["cohort", "patient_id"])
        dz = np.load(self._draws, allow_pickle=True)
        X = np.column_stack([full[f"{f}__mean"].to_numpy() for f in CANON])
        sd = np.column_stack([full[f"{f}__sd"].to_numpy() for f in CANON])
        n_obs = np.column_stack([full[f"{f}__n_obs"].to_numpy() for f in CANON])
        reliability = np.column_stack([full[f"{f}__reliability"].to_numpy() for f in CANON])
        if self.config.use_full_Si:
            S = np.load(self._cov, allow_pickle=True)["cov"]
        else:
            S = sd ** 2
        vt = pd.read_parquet(self._vt).set_index(["cohort", "patient_id"]).reindex(full.index)
        return CoordinateSet(X=X, S=S, draws=dz["draws"], dims=list(CANON), index=full.index,
                             n_obs=n_obs, reliability=reliability, validation=vt)


# ----------------------------------------------------------------------------------------------------------
# Structure-discovery gate  (cluster vs continuum)  -- wraps src/face/strata/structure.py
# ----------------------------------------------------------------------------------------------------------
class StructureGate:
    """Characterize the shape of the coordinate cloud BEFORE defining regions: clustered vs continuum vs
    branched. The honest null is 'continuum' (regions are then a soft overlay, not natural kinds). All tests
    are thin wrappers over ``structure.py``; the gate runs on both arms and is uncertainty-aware over draws.
    """

    def __init__(self, config: StrataConfig | None = None):
        self.config = config or StrataConfig()

    def battery(self, coords: CoordinateSet, *, arm: str = "A") -> dict:
        """Run the full discovery battery and synthesize the verdict for one arm."""
        from face.strata.structure import (  # noqa: PLC0415
            dip_test,
            gap_statistic,
            gmm_bic_sweep,
            hdbscan_summary,
            hopkins,
            mapper_graph,
            silhouette_sweep,
            verdict,
        )
        X, _, cols = coords.arm(arm)
        dims = [coords.dims[c] for c in cols]
        seed = self.config.seed
        diag = {
            "hopkins": hopkins(X, seed=seed),
            "dip": dip_test(X, dims),
            "gmm_bic": gmm_bic_sweep(X, seed=seed),
            "silhouette": silhouette_sweep(X, seed=seed),
            "gap": gap_statistic(X, seed=seed),
            "hdbscan": hdbscan_summary(X),
        }
        # Mapper lens = the G coordinate (severity) for arm A; PC1 for arm B (no G column there)
        lens = X[:, dims.index(G_KEY)] if G_KEY in dims else X[:, 0]
        G, node_members = mapper_graph(X, lens, seed=seed)
        import networkx as nx  # noqa: PLC0415
        diag["mapper"] = {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
                          "n_components": nx.number_connected_components(G) if G.number_of_nodes() else 0}
        return {"arm": arm, "diagnostics": diag, "verdict": verdict(diag)}

    def uncertainty_stability(self, coords: CoordinateSet, *, arm: str = "A", n_draw: int = 20) -> dict:
        """Re-run the cluster tendency + optimal-K over posterior draws: is the verdict stable when each
        patient's coordinate is sampled from its posterior rather than fixed at the mean?"""
        from face.strata.structure import uncertainty_sweep  # noqa: PLC0415
        _, _, cols = coords.arm(arm)
        return uncertainty_sweep(coords.draws, cols, n_draw=n_draw, seed=self.config.seed)

    def null_comparison(self, coords: CoordinateSet, *, arm: str = "A", n_null: int = 10,
                        Ks=range(2, 7)) -> dict:
        """Compare the cloud's clustering metrics to a SINGLE-GAUSSIAN 'structureless continuum' null with the
        same mean/covariance. This is the decisive separation test: Hopkins and GMM-BIC tendency signals fire
        on the null too (a Gaussian is non-uniform and, here, non-Gaussian shape aside, GMM over-segments), so
        only the **silhouette** (separation) discriminates clusters from a continuum. If the real silhouette is
        within the null band (z < ~2), the apparent structure is what a unimodal continuum already produces ->
        no discrete clusters; the GMM-BIC gain then reflects only the cloud's non-Gaussian shape (its skew /
        archetype corners), which the archetypes — not clusters — describe."""
        from face.strata.structure import gmm_bic_sweep, hopkins, silhouette_sweep  # noqa: PLC0415
        X, _, _ = coords.arm(arm)
        rng = np.random.default_rng(self.config.seed)
        mu, cov = X.mean(0), np.cov(X.T)

        def _m(Z):
            return {"hopkins": float(hopkins(Z, seed=0)),
                    "best_silhouette": float(silhouette_sweep(Z, Ks, seed=0)["peak"]),
                    "gmm_bic_gain": float(max(0.0, gmm_bic_sweep(Z, range(1, max(Ks) + 2), seed=0).get("gain_over_k1", 0.0)))}

        real = _m(X)
        null = [_m(rng.multivariate_normal(mu, cov, size=len(X))) for _ in range(n_null)]
        out = {"real": real, "null_mean": {}, "null_sd": {}, "z": {}}
        for k in real:
            v = np.array([n[k] for n in null])
            out["null_mean"][k], out["null_sd"][k] = float(v.mean()), float(v.std())
            out["z"][k] = float((real[k] - v.mean()) / (v.std() + 1e-9))
        out["verdict"] = ("separation indistinguishable from a structureless continuum"
                          if out["z"]["best_silhouette"] < 2.0 else "separation exceeds the continuum null")
        return out


# ----------------------------------------------------------------------------------------------------------
# Soft operational regions  (measurement-error mixture)  -- wraps src/face/strata/mixture.py
# ----------------------------------------------------------------------------------------------------------
@dataclass
class RegionFit:
    """A soft tessellation: the lead operational-region view. ``resp`` are the soft transition memberships."""
    K: int
    arm: str
    cols: list[int]
    dims: list[str]
    pi: np.ndarray                             # [K] mixing weights
    mu: np.ndarray                             # [K, D] deconvolved (noise-free) region centroids
    V: np.ndarray                              # [K, D, D] region covariances
    resp: np.ndarray                           # [N, K] soft responsibilities (region membership)
    entropy: np.ndarray                        # [N] normalized membership entropy (0=core, 1=boundary)
    tier: np.ndarray                           # [N] core/soft/boundary confidence tier
    names: list[str]                           # [K] data-driven region labels
    bic: float
    map_label: np.ndarray = field(default=None)   # [N] argmax region


class SoftRegionModel:
    """Define soft operational regions via the measurement-error (Extreme-Deconvolution) mixture, which
    propagates each patient's per-coordinate uncertainty ``S_i``. The responsibilities are the *soft
    transition boundaries*; the deconvolved ``mu_k`` are noise-free region centroids. Wraps ``mixture.py``.
    """

    def __init__(self, config: StrataConfig | None = None):
        self.config = config or StrataConfig()

    def bic_sweep(self, coords: CoordinateSet, *, arm: str = "A", Ks=(2, 3, 4, 5, 6, 7, 8)) -> dict:
        from face.strata.mixture import bic_sweep  # noqa: PLC0415
        X, S, _ = coords.arm(arm)
        return bic_sweep(X, S, Ks=Ks, seed=self.config.seed)

    def fit(self, coords: CoordinateSet, K: int, *, arm: str = "A") -> RegionFit:
        from face.strata.archetypes import name_archetypes  # noqa: PLC0415
        from face.strata.mixture import xd_em  # noqa: PLC0415
        X, S, cols = coords.arm(arm)
        dims = [coords.dims[c] for c in cols]
        fit = xd_em(X, S, K, reg=self.config.region_reg, seed=self.config.seed)
        resp = fit["resp"]
        return RegionFit(
            K=K, arm=arm, cols=cols, dims=dims, pi=fit["pi"], mu=fit["mu"], V=fit["V"], resp=resp,
            entropy=_normalized_entropy(resp), tier=_confidence_tier(resp, self.config.confidence_tiers),
            names=name_archetypes(fit["mu"], dims), bic=fit["bic"], map_label=resp.argmax(1))


# ----------------------------------------------------------------------------------------------------------
# Archetypes  (continuum co-view)  -- wraps src/face/strata/archetypes.py
# ----------------------------------------------------------------------------------------------------------
@dataclass
class ArchetypeFit:
    """The continuum-honest co-view: each patient is a convex blend of A extreme phenotypes (no hard edges)."""
    A: int
    arm: str
    cols: list[int]
    dims: list[str]
    Z: np.ndarray                              # [A, D] archetype profiles (the corners)
    W: np.ndarray                              # [N, A] simplex weights (membership)
    W_sd: np.ndarray                           # [N, A] membership uncertainty (over posterior draws)
    names: list[str]                           # [A] data-driven labels
    entropy: np.ndarray                        # [N] normalized blend entropy
    tier: np.ndarray                           # [N] core/soft/boundary
    explained_variance: float
    stability: dict                            # cross-seed reproducibility
    map_label: np.ndarray = field(default=None)   # [N] dominant archetype


class ArchetypeModel:
    """Archetypal analysis: the continuum-honest representation (extreme phenotypes + simplex blends).
    Wraps ``archetypes.py``; propagates membership uncertainty by projecting posterior draws onto the fixed
    anchors. Runs on either arm."""

    def __init__(self, config: StrataConfig | None = None):
        self.config = config or StrataConfig()

    def select_A(self, coords: CoordinateSet, *, arm: str = "A", As=(2, 3, 4, 5, 6, 7, 8)) -> dict:
        from face.strata.archetypes import select_A  # noqa: PLC0415
        X, _, _ = coords.arm(arm)
        return select_A(X, As=As, seed=self.config.seed, knee_gain=self.config.knee_gain)

    def select_A_operational(self, coords: CoordinateSet, *, arm: str = "A", As=(2, 3, 4, 5, 6, 7, 8),
                             stability_min: float = 0.8) -> dict:
        """Choose A as an OPERATIONAL granularity (the archetype analogue of ``choose_K_operational``):
        the EV scree always rewards more corners, but on the copula coordinates the high-A archetypes chase
        the wide explicit-axis noise and become seed-unstable. So pick the LARGEST A whose cross-seed
        archetype reproducibility (min Tucker congruence) stays >= ``stability_min`` — a stable, communicable
        set of extreme phenotypes, not the most corners the EV will admit."""
        from face.strata.archetypes import explained_variance, fit_aa, stability  # noqa: PLC0415
        X, _, _ = coords.arm(arm)
        rows = []
        for A in As:
            _, _, _, rss = fit_aa(X, A, seed=self.config.seed, n_init=2)
            rows.append({"A": int(A), "explained_variance": float(explained_variance(X, rss)),
                         "stability": float(stability(X, A, seeds=(0, 1, 2))["min_tucker_congruence"])})
        stable = [r for r in rows if r["stability"] >= stability_min]
        chosen = max(stable, key=lambda r: r["A"]) if stable else max(rows, key=lambda r: r["stability"])
        return {"sweep": rows, "chosen_A": int(chosen["A"]),
                "rationale": (f"largest A with cross-seed stability >= {stability_min}" if stable
                              else "no A met the stability floor; fell back to the most-stable A")}

    def fit(self, coords: CoordinateSet, A: int, *, arm: str = "A", n_draw: int = 40) -> ArchetypeFit:
        from face.strata.archetypes import (  # noqa: PLC0415
            explained_variance,
            fit_aa,
            name_archetypes,
            project_draws,
            stability,
        )
        X, _, cols = coords.arm(arm)
        dims = [coords.dims[c] for c in cols]
        _, Z, W, rss = fit_aa(X, A, seed=self.config.seed, n_init=4)
        unc = project_draws(Z, coords.draws, cols, n_draw=n_draw, seed=self.config.seed)
        return ArchetypeFit(
            A=A, arm=arm, cols=cols, dims=dims, Z=Z, W=W, W_sd=unc["sd"],
            names=name_archetypes(Z, dims), entropy=_normalized_entropy(W),
            tier=_confidence_tier(W, self.config.confidence_tiers),
            explained_variance=explained_variance(X, rss),
            stability=stability(X, A, seeds=(0, 1, 2)), map_label=W.argmax(1))

    def anchor_uncertainty(self, coords: CoordinateSet, A: int, *, arm: str = "A", n_draw: int = 40) -> dict:
        from face.strata.archetypes import archetype_location_uncertainty  # noqa: PLC0415
        X, _, cols = coords.arm(arm)
        return archetype_location_uncertainty(X, coords.draws, cols, A, n_draw=n_draw,
                                              hdi_prob=self.config.hdi_prob, seed=self.config.seed)


# ----------------------------------------------------------------------------------------------------------
# Internal usefulness battery  -- wraps src/face/strata/validation.py
# ----------------------------------------------------------------------------------------------------------
class UsefulnessValidator:
    """Answer 'are the soft regions useful?' as an INTERNAL/baseline question (temporal persistence and
    prognosis are deferred to the M3/M4 reruns). Five criteria, each with a PASS/CONDITIONAL/FAIL gate:
    assignment (are patients assignable?), not-just-severity (Q2), transdiagnostic (Q3), stable/not-artefact
    (Q4), tighter-than-DSM-5. Diagnosis (cohort / DSM-5 ``arm``) enters only as a validation label."""

    def __init__(self, config: StrataConfig | None = None):
        self.config = config or StrataConfig()

    def assignment(self, fit: RegionFit | ArchetypeFit) -> dict:
        from face.strata.validation import assignment_usefulness  # noqa: PLC0415
        resp = fit.resp if isinstance(fit, RegionFit) else fit.W
        return assignment_usefulness(resp)

    def not_just_severity(self, coords: CoordinateSet, labels: np.ndarray) -> dict:
        """Q2: do the regions encode specific biology beyond the G severity ladder?"""
        from face.strata.validation import eta_squared  # noqa: PLC0415
        eta = eta_squared(labels, coords.X)
        gi = coords.dims.index(G_KEY)
        spec = float(np.mean([eta[i] for i in range(len(coords.dims)) if i != gi]))
        return {"eta_per_axis": {coords.dims[i]: float(eta[i]) for i in range(len(coords.dims))},
                "eta_G": float(eta[gi]), "mean_eta_specifics": spec,
                "gate": "PASS" if spec > eta[gi] else "CONDITIONAL"}

    def transdiagnostic(self, coords: CoordinateSet, labels: np.ndarray) -> dict:
        """Q3: is the partition independent of diagnosis (cohort & DSM-5 subtype)? Want ARI ~ 0."""
        from face.strata.validation import ari, cramers_v  # noqa: PLC0415
        cohort = np.asarray(coords.index.get_level_values("cohort"))
        arm = coords.validation["arm"].astype(str).to_numpy()
        ac, ad = ari(labels, cohort), ari(labels, arm)
        return {"ari_cohort": ac, "ari_dsm5": ad,
                "cramers_v_cohort": cramers_v(labels, cohort), "cramers_v_dsm5": cramers_v(labels, arm),
                "gate": "PASS" if max(abs(ac), abs(ad)) < 0.05 else ("FAIL" if max(abs(ac), abs(ad)) > 0.15 else "CONDITIONAL")}

    def stable_not_artifact(self, coords: CoordinateSet, labels: np.ndarray, K: int, *, n_perm: int = 30) -> dict:
        """Q4: reproducible across seeds AND not predictable from the coverage (missingness) pattern."""
        from face.strata.validation import coverage_artifact, tess_seed_stability  # noqa: PLC0415
        X, S, _ = coords.arm("A")
        stab = tess_seed_stability(X, S, K)
        cov = coverage_artifact(coords.n_obs, np.asarray(labels), seed=self.config.seed, n_perm=n_perm)
        gate = "PASS" if (stab["mean_ari"] >= 0.8 and cov["perm_p_value"] > 0.05) else (
            "FAIL" if stab["mean_ari"] < 0.6 else "CONDITIONAL")
        return {"seed_ari": stab, "coverage_artifact": cov, "gate": gate}

    def tighter_than_dsm5(self, coords: CoordinateSet, K: int) -> dict:
        """The 'better description' test: does a free K-region tessellation fit the cloud better (lower XD
        BIC, higher mean eta^2) than the DSM-5 partition? Descriptive, not a clinical-superiority claim."""
        from face.strata.mixture import xd_em, xd_fixed_labels  # noqa: PLC0415
        from face.strata.validation import eta_squared  # noqa: PLC0415
        X, S, _ = coords.arm("A")
        arm = coords.validation["arm"].astype(str).to_numpy()
        free = xd_em(X, S, K, reg=self.config.region_reg, seed=self.config.seed)
        dsm = xd_fixed_labels(X, S, arm)
        eta_free = float(eta_squared(free["resp"].argmax(1), X).mean())
        eta_dsm = float(eta_squared(arm, X).mean())
        return {"free_bic": free["bic"], "dsm5_bic": dsm["bic"], "free_K": K, "dsm5_K": dsm["K"],
                "mean_eta_free": eta_free, "mean_eta_dsm5": eta_dsm,
                "gate": "PASS" if (free["bic"] < dsm["bic"] and eta_free > eta_dsm) else "CONDITIONAL"}

    def battery(self, coords: CoordinateSet, region_fit: RegionFit, *, n_perm: int = 30) -> dict:
        """Run the full internal battery on the lead soft-tessellation MAP partition; return per-criterion
        results + an overall verdict (worst-of, with named blockers)."""
        labels = region_fit.map_label
        out = {
            "assignment": self.assignment(region_fit),
            "not_just_severity": self.not_just_severity(coords, labels),
            "transdiagnostic": self.transdiagnostic(coords, labels),
            "stable_not_artifact": self.stable_not_artifact(coords, labels, region_fit.K, n_perm=n_perm),
            "tighter_than_dsm5": self.tighter_than_dsm5(coords, region_fit.K),
        }
        gates = {k: v["gate"] for k, v in out.items()}
        rank = {"FAIL": 0, "CONDITIONAL": 1, "PASS": 2}
        worst = min(gates.values(), key=lambda g: rank[g])
        out["summary"] = {"gates": gates, "overall": worst,
                          "blockers": [k for k, g in gates.items() if g != "PASS"]}
        return out


# ----------------------------------------------------------------------------------------------------------
# Staged orchestration + caching  (deterministic; no MCMC warm-start)
# ----------------------------------------------------------------------------------------------------------
def _jsonify(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _save_payload(out: Path, payload: dict) -> None:
    """Persist a stage result: ndarray values -> arrays.npz, everything else -> data.json."""
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in payload.items() if isinstance(v, np.ndarray)}
    rest = {k: v for k, v in payload.items() if not isinstance(v, np.ndarray)}
    if arrays:
        np.savez_compressed(out / "arrays.npz", **arrays)
    (out / "data.json").write_text(json.dumps(rest, indent=2, default=_jsonify))


def _load_payload(out: Path) -> dict:
    d = json.loads((out / "data.json").read_text())
    if (out / "arrays.npz").exists():
        z = np.load(out / "arrays.npz", allow_pickle=True)
        d.update({k: z[k] for k in z.files})
    return d


def _region_payload(rf: RegionFit) -> dict:
    return {"kind": "regions", "K": rf.K, "arm": rf.arm, "cols": rf.cols, "dims": rf.dims, "names": rf.names,
            "bic": rf.bic, "pi": rf.pi, "mu": rf.mu, "V": rf.V, "resp": rf.resp, "entropy": rf.entropy,
            "tier": rf.tier, "map_label": rf.map_label}


def _region_from_payload(d: dict) -> RegionFit:
    return RegionFit(K=int(d["K"]), arm=str(d["arm"]), cols=list(np.asarray(d["cols"]).tolist()),
                     dims=list(d["dims"]), pi=np.asarray(d["pi"]), mu=np.asarray(d["mu"]),
                     V=np.asarray(d["V"]), resp=np.asarray(d["resp"]), entropy=np.asarray(d["entropy"]),
                     tier=np.asarray(d["tier"]), names=list(d["names"]), bic=float(d["bic"]),
                     map_label=np.asarray(d["map_label"]))


def _arch_payload(af: ArchetypeFit) -> dict:
    return {"kind": "archetypes", "A": af.A, "arm": af.arm, "cols": af.cols, "dims": af.dims,
            "names": af.names, "explained_variance": af.explained_variance, "stability": af.stability,
            "Z": af.Z, "W": af.W, "W_sd": af.W_sd, "entropy": af.entropy, "tier": af.tier,
            "map_label": af.map_label}


def _arch_from_payload(d: dict) -> ArchetypeFit:
    return ArchetypeFit(A=int(d["A"]), arm=str(d["arm"]), cols=list(np.asarray(d["cols"]).tolist()),
                        dims=list(d["dims"]), Z=np.asarray(d["Z"]), W=np.asarray(d["W"]),
                        W_sd=np.asarray(d["W_sd"]), names=list(d["names"]),
                        entropy=np.asarray(d["entropy"]), tier=np.asarray(d["tier"]),
                        explained_variance=float(d["explained_variance"]), stability=dict(d["stability"]),
                        map_label=np.asarray(d["map_label"]))


class StrataRunner:
    """Walk the deterministic strata plan, caching each stage to ``output_dir/<stage>/`` and reusing the
    cache when ``model_version`` + ``stage_spec`` + ``config_sig`` all match. Unlike the MCMC measurement
    runner there is no warm-start; cross-stage dependency is by reading the prior stage's cached artifact.
    The accumulated in-memory ``state`` carries the live objects (coords / region/archetype fits) through a
    single ``run_plan`` call so figures + consolidation can use them directly."""

    def __init__(self, config: StrataConfig | None = None):
        self.config = config or StrataConfig()
        self.data = StrataData(self.config)
        self.gate = StructureGate(self.config)
        self.regions = SoftRegionModel(self.config)
        self.arch = ArchetypeModel(self.config)
        self.validator = UsefulnessValidator(self.config)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _cache_ok(self, out: Path, stage: StrataStage) -> bool:
        mf = out / "manifest.json"
        if not mf.exists():
            return False
        m = json.loads(mf.read_text())
        return (m.get("model_version") == MODEL_VERSION and m.get("stage_spec") == _stage_spec(stage)
                and m.get("config_sig") == _config_sig(self.config))

    def _write_manifest(self, out: Path, stage: StrataStage, summary: dict, N: int) -> None:
        (out / "manifest.json").write_text(json.dumps({
            "model_version": MODEL_VERSION, "stage": stage.name, "stage_spec": _stage_spec(stage),
            "config_sig": _config_sig(self.config), "N": int(N), "summary": summary}, indent=2))

    def run_stage(self, stage: StrataStage, state: dict, *, overwrite: bool = False, n_perm: int = 30) -> dict:
        """Run (or load from cache) one stage; update + return ``state``."""
        out = self.config.output_dir / stage.name
        cached = self._cache_ok(out, stage) and not overwrite
        if stage.kind == "coordinates":
            coords = self.data.prepare(n_coord_draws=stage.n_coord_draws, overwrite=overwrite)
            state["coords"] = coords
            self._write_manifest(out, stage, {"N": int(coords.X.shape[0])}, coords.X.shape[0])
            return state

        coords: CoordinateSet = state["coords"]
        if stage.kind == "structure":
            if cached:
                state["structure"] = _load_payload(out)
            else:
                res = {arm: self.gate.battery(coords, arm=arm) for arm in stage.arms}
                res["uncertainty_A"] = self.gate.uncertainty_stability(coords, arm="A",
                                                                       n_draw=stage.n_uncertainty_draws)
                # serialize the single-Gaussian falsification null (the decisive separation test) so the
                # real-vs-null silhouette / z / GMM-gain are traceable artifacts, not plotting-time constants.
                res["falsification_null"] = self.gate.null_comparison(coords, arm="A")
                _save_payload(out, {"verdict_A": res["A"]["verdict"], "verdict_B": res["B"]["verdict"],
                                    "diagnostics_A": res["A"]["diagnostics"],
                                    "diagnostics_B": res["B"]["diagnostics"],
                                    "uncertainty_A": res["uncertainty_A"],
                                    "falsification_null": res["falsification_null"]})
                state["structure"] = _load_payload(out)
            self._write_manifest(out, stage, {"verdict_A": state["structure"]["verdict_A"]["label"]},
                                 coords.X.shape[0])
            return state

        if stage.kind == "regions":
            if cached:
                d = _load_payload(out)
                state["region_A"] = _region_from_payload(_load_payload(out / "A"))
                state["region_B"] = _region_from_payload(_load_payload(out / "B"))
                state["choose_K"] = d["choose_K"]
            else:
                from face.strata.validation import choose_K_operational  # noqa: PLC0415
                XA, SA, _ = coords.arm("A")
                ck = choose_K_operational(XA, SA, Ks=stage.K_sweep, seeds=(1, 2, 3), seed=self.config.seed)
                K = stage.K or ck["chosen_K"]
                ra, rb = self.regions.fit(coords, K, arm="A"), self.regions.fit(coords, K, arm="B")
                _save_payload(out / "A", _region_payload(ra))
                _save_payload(out / "B", _region_payload(rb))
                _save_payload(out, {"choose_K": ck, "K": K})
                state["region_A"], state["region_B"], state["choose_K"] = ra, rb, ck
            self._write_manifest(out, stage, {"K": state["region_A"].K,
                                              "chosen_K": state["choose_K"]["chosen_K"]}, coords.X.shape[0])
            return state

        if stage.kind == "archetypes":
            if cached:
                state["arch_A"] = _arch_from_payload(_load_payload(out / "A"))
                state["arch_B"] = _arch_from_payload(_load_payload(out / "B"))
                state["select_A"] = _load_payload(out)["select_A"]
            else:
                sel = self.arch.select_A_operational(coords, arm="A", As=stage.A_sweep)
                A = stage.A or sel["chosen_A"]
                aa, ab = (self.arch.fit(coords, A, arm="A", n_draw=stage.n_draw),
                          self.arch.fit(coords, A, arm="B", n_draw=stage.n_draw))
                _save_payload(out / "A", _arch_payload(aa))
                _save_payload(out / "B", _arch_payload(ab))
                _save_payload(out, {"select_A": sel, "A": A})
                state["arch_A"], state["arch_B"], state["select_A"] = aa, ab, sel
            self._write_manifest(out, stage, {"A": state["arch_A"].A,
                                              "ev": state["arch_A"].explained_variance}, coords.X.shape[0])
            return state

        if stage.kind == "usefulness":
            if cached:
                state["usefulness"] = _load_payload(out)
            else:
                uv = self.validator.battery(coords, state["region_A"], n_perm=n_perm)
                _save_payload(out, uv)
                state["usefulness"] = uv
            self._write_manifest(out, stage, {"overall": state["usefulness"]["summary"]["overall"]},
                                 coords.X.shape[0])
            return state

        if stage.kind == "consolidate":
            proj = StrataProjector(self.config)
            # nested K-family overlay (conventions; operative K deferred to M4/M5) — cheap XD refits of the
            # same cloud, exported alongside the operational tessellation and the continuous/archetype views.
            family = [self.regions.fit(coords, k, arm="A") for k in stage.K_family]
            sweep = (state.get("choose_K") or {}).get("sweep")
            arch_B = state.get("arch_B")
            frame = proj.membership_frame(coords, state["arch_A"], state["region_A"],
                                          region_family=family, arch_fit_B=arch_B)
            out.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(out / "patient_strata.parquet")
            menu = proj.k_family_menu(coords, family, choose_K_sweep=sweep)
            menu.to_csv(out / "k_family_menu.csv", index=False)
            # frozen archetype anchors (native archetype_profiles.csv schema) for M4 armB_block + M3 projection
            proj.archetype_profiles(state["arch_A"], arch_B).to_csv(out / "archetype_profiles.csv", index=False)
            state["patient_strata"], state["k_family_menu"] = frame, menu
            self._write_manifest(out, stage, {"rows": len(frame), "cols": list(frame.columns),
                                              "operational_K": int(state["region_A"].K),
                                              "K_family": list(stage.K_family),
                                              "operative_K": "deferred to M4/M5 incremental validity"},
                                 len(frame))
            return state

        raise ValueError(f"unknown stage kind: {stage.kind}")

    def run_plan(self, *, stop_after: str | None = None, overwrite: bool = False, n_perm: int = 30) -> dict:
        """Run the configured plan in order, returning the accumulated state. ``stop_after`` halts after the
        named stage (e.g. 'structure' for the continuum discussion gate)."""
        state: dict = {}
        for stage in self.config.stage_plan:
            state = self.run_stage(stage, state, overwrite=overwrite, n_perm=n_perm)
            if stop_after and stage.name == stop_after:
                break
        return state

    def load_state(self) -> dict:
        """Reconstruct the accumulated state from cached stage artifacts (no recompute) — for the figures
        script and the notebook's load mode. Skips any stage that has not been run yet."""
        out = self.config.output_dir
        state: dict = {"coords": self.data.load()}
        if (out / "structure" / "data.json").exists():
            state["structure"] = _load_payload(out / "structure")
        if (out / "regions" / "A" / "data.json").exists():
            state["region_A"] = _region_from_payload(_load_payload(out / "regions" / "A"))
            state["region_B"] = _region_from_payload(_load_payload(out / "regions" / "B"))
            state["choose_K"] = _load_payload(out / "regions")["choose_K"]
        if (out / "archetypes" / "A" / "data.json").exists():
            state["arch_A"] = _arch_from_payload(_load_payload(out / "archetypes" / "A"))
            state["arch_B"] = _arch_from_payload(_load_payload(out / "archetypes" / "B"))
            state["select_A"] = _load_payload(out / "archetypes")["select_A"]
        if (out / "usefulness" / "data.json").exists():
            state["usefulness"] = _load_payload(out / "usefulness")
        ps = out / "consolidate" / "patient_strata.parquet"
        if ps.exists():
            state["patient_strata"] = pd.read_parquet(ps)
        return state


# ----------------------------------------------------------------------------------------------------------
# Per-patient membership frame (the M3-compatible hand-off) + out-of-sample projection
# ----------------------------------------------------------------------------------------------------------
class StrataProjector:
    """Assemble the per-patient membership frame consumed downstream (M3/M4), with soft memberships,
    entropy, confidence tiers and boundary flags. Column contract: keyed (cohort, patient_id); ``arch_*`` /
    ``tess_*`` prefixes (auto-joined by ``prognosis.frame``); ``arm`` (validation-only).

    On a continuum K is a granularity *convention*, not a discovered kind-count, so the hand-off does not
    privilege a single K: alongside the operational ``tess_*`` tessellation (the smallest confidently
    assignable + stable K) it carries (i) the load-bearing continuous coordinates + draws (exported by
    ``StrataData``) and the soft archetype simplex (``arch_*``), and (ii) a nested **K-family** overlay
    (``tessfam_k{K}_*``, prefix chosen so it never matches the ``tess_`` selector and pollutes the
    operational contract). The *operative* granularity — which K, if any, adds clinical value — is a
    downstream question answered by M4/M5 incremental predictive/treatment validity, not by an internal
    parsimony rule here (``k_family_menu`` is the decision menu they consume)."""

    def __init__(self, config: StrataConfig | None = None):
        self.config = config or StrataConfig()

    def membership_frame(self, coords: CoordinateSet, arch_fit: ArchetypeFit,
                         region_fit: RegionFit,
                         region_family: list[RegionFit] | None = None,
                         arch_fit_B: ArchetypeFit | None = None) -> pd.DataFrame:
        df = pd.DataFrame(index=coords.index)
        for a in range(arch_fit.A):
            df[f"arch_w{a}"] = np.round(arch_fit.W[:, a], 4)
            df[f"arch_w{a}_sd"] = np.round(arch_fit.W_sd[:, a], 4)
        df["arch_dominant"] = arch_fit.map_label.astype(int)
        df["arch_dominant_name"] = [arch_fit.names[k] for k in arch_fit.map_label]
        df["arch_entropy"] = np.round(arch_fit.entropy, 4)
        df["arch_confidence_tier"] = arch_fit.tier
        df["arch_boundary"] = (arch_fit.tier == "boundary").astype(int)
        # Arm-B (G-residualized) archetype weights — the clean ⊥G phenotype view M4 uses as `+archetypesB`.
        if arch_fit_B is not None:
            for a in range(arch_fit_B.A):
                df[f"archB_w{a}"] = np.round(arch_fit_B.W[:, a], 4)
            df["archB_dominant"] = arch_fit_B.map_label.astype(int)
            df["archB_entropy"] = np.round(arch_fit_B.entropy, 4)
        # operational tessellation — the M3/M4 ``tess_`` contract (the chosen operational K)
        for k in range(region_fit.K):
            df[f"tess_r{k}"] = np.round(region_fit.resp[:, k], 4)
        df["tess_MAP"] = region_fit.map_label.astype(int)
        df["tess_MAP_name"] = [region_fit.names[k] for k in region_fit.map_label]
        df["tess_entropy"] = np.round(region_fit.entropy, 4)
        df["tess_confidence_tier"] = region_fit.tier
        df["tess_boundary"] = (region_fit.tier == "boundary").astype(int)
        # nested K-family overlay (conventions; operative K deferred to M4/M5). The ``tessfam_`` prefix does
        # NOT match prognosis.frame's ``tess_`` selector, so the family never leaks into the operational view.
        for rf in (region_family or []):
            for k in range(rf.K):
                df[f"tessfam_k{rf.K}_r{k}"] = np.round(rf.resp[:, k], 4)
            df[f"tessfam_k{rf.K}_MAP"] = rf.map_label.astype(int)
            df[f"tessfam_k{rf.K}_entropy"] = np.round(rf.entropy, 4)
            df[f"tessfam_k{rf.K}_tier"] = rf.tier
        df["arm"] = coords.validation["arm"].astype(str).to_numpy()
        return df.reset_index()

    def k_family_menu(self, coords: CoordinateSet, region_family: list[RegionFit],
                      choose_K_sweep: list[dict] | None = None) -> pd.DataFrame:
        """The decision menu for the nested K-family: per-K assignment confidence + entropy, cross-seed
        stability (reused from the operational K-sweep), and the per-axis variance explained (eta^2 — what
        each K actually splits on). It exists so M4/M5 can choose the operative granularity by *external*
        (predictive/treatment) validity rather than by an internal parsimony tiebreak on a flat BIC basin."""
        from face.strata.validation import assignment_usefulness, eta_squared  # noqa: PLC0415
        sweep = {int(r["K"]): r for r in (choose_K_sweep or [])}
        gi = coords.dims.index(G_KEY)
        rows = []
        for rf in region_family:
            au = assignment_usefulness(rf.resp)
            eta = eta_squared(rf.map_label, coords.X)
            spec = float(np.mean([eta[i] for i in range(len(coords.dims)) if i != gi]))
            s = sweep.get(rf.K, {})
            row = {"K": int(rf.K), "bic": float(s.get("bic", np.nan)),
                   "confident_dominant_frac": round(au["confident_dominant_frac"], 4),
                   "median_norm_entropy": round(au["median_norm_entropy"], 4),
                   "seed_ari": float(s.get("seed_ari", np.nan)),
                   "mean_eta_specifics": round(spec, 4),
                   "eta_overall_severity": round(float(eta[gi]), 4)}
            for i, d in enumerate(coords.dims):
                row[f"eta_{d}"] = round(float(eta[i]), 4)
            rows.append(row)
        return pd.DataFrame(rows)

    def archetype_profiles(self, arch_fit_A: ArchetypeFit,
                           arch_fit_B: ArchetypeFit | None = None) -> pd.DataFrame:
        """The frozen archetype profile anchors, in the native M2 ``archetype_profiles.csv`` schema so the
        downstream M4 ``reference.armB_block`` (Arm-B projection) and the M3 V1/V2 projection can consume them
        without re-fitting: one row per archetype per arm, columns = ``arm``/``archetype``/``name`` + one per
        CANON axis. Arm ``A_all9`` carries all 8 axes; arm ``B_specifics`` carries the 7 ⊥G specifics (its
        ``overall_severity`` is left NaN — armB_block only reads the specifics). NB: the ``A_all9`` arm label
        is a historical schema key kept stable across M2→M3→M4 (the map is now 8-dim, not 9)."""
        rows = []
        for arm_label, af in (("A_all9", arch_fit_A), ("B_specifics", arch_fit_B)):
            if af is None:
                continue
            for a in range(af.A):
                row = {"arm": arm_label, "archetype": a, "name": af.names[a]}
                row.update({ax: np.nan for ax in CANON})
                for j, dim in enumerate(af.dims):
                    row[dim] = float(af.Z[a, j])
                rows.append(row)
        return pd.DataFrame(rows, columns=["arm", "archetype", "name", *CANON])

    def project_new(self, X_new: np.ndarray, arch_fit: ArchetypeFit, region_fit: RegionFit,
                    *, S_new: np.ndarray | None = None) -> pd.DataFrame:
        """Out-of-sample membership: archetype simplex weights (project onto fixed anchors) + region
        responsibilities under the fixed deconvolved (mu, V). ``S_new`` is the per-patient measurement
        error (zeros if unknown)."""
        from face.strata.archetypes import project_to_Z  # noqa: PLC0415
        from face.strata.mixture import _estep_k  # noqa: PLC0415
        Wa = project_to_Z(X_new[:, arch_fit.cols], arch_fit.Z)
        Xr = X_new[:, region_fit.cols]
        Sr = np.zeros_like(Xr) if S_new is None else S_new[:, region_fit.cols]
        logr = np.column_stack([np.log(region_fit.pi[k] + 1e-300)
                                + _estep_k(Xr, Sr, region_fit.mu[k], region_fit.V[k])[0]
                                for k in range(region_fit.K)])
        m = logr.max(1, keepdims=True)
        resp = np.exp(logr - m)
        resp /= resp.sum(1, keepdims=True)
        out = pd.DataFrame({"arch_dominant": Wa.argmax(1), "tess_MAP": resp.argmax(1)})
        for a in range(arch_fit.A):
            out[f"arch_w{a}"] = Wa[:, a]
        for k in range(region_fit.K):
            out[f"tess_r{k}"] = resp[:, k]
        return out


# ----------------------------------------------------------------------------------------------------------
# Figures  (UMAP/PCA are VISUALIZATION-ONLY -- never a clustering input)
# ----------------------------------------------------------------------------------------------------------
class StrataVisualizer:
    """Standard strata figures. The soft-boundary map is the centrepiece: it *shows* the soft transition
    boundaries (points shaded by membership entropy) rather than asserting them. Embeddings (PCA/UMAP) are
    used only to draw, never to fit."""

    def __init__(self, config: StrataConfig | None = None):
        self.config = config or StrataConfig()
        self.config.figure_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, filename: str) -> Path:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        path = self.config.figure_dir / filename
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path

    def _mpl(self):
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        return plt

    def structure_panel(self, structure_payload: dict, filename: str = "structure_panel.png") -> Path:
        """The discovery verdict + the six clustered-vs-continuum signals (arm A)."""
        plt = self._mpl()
        v = structure_payload["verdict_A"]
        ev = v["evidence"]
        sigs = {"hopkins>0.75": ev["hopkins"] > 0.75, "silhouette>0.25": ev["silhouette_peak"] > 0.25,
                "interior BIC-K": not ev["gmm_monotone"], "gap K>1": ev["gap_k_opt"] > 1,
                "PC1 multimodal": ev["dip_pc1_p"] < 0.05, "HDBSCAN clusters": ev["hdbscan_n"] >= 2}
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.barh(list(sigs), [1 if s else 0 for s in sigs.values()],
                color=["#cf6679" if s else "#3b6fb6" for s in sigs.values()])
        ax.set_xlim(0, 1); ax.set_xticks([0, 1]); ax.set_xticklabels(["no", "yes"])
        ax.set_title(f"Structure-discovery gate (arm A): {v['label'].upper()}  "
                     f"({v['clustered_score']}/{v['n_signals']} clustered signals)")
        return self._save(fig, filename)

    def region_profiles(self, region_fit: RegionFit, filename: str = "region_profiles.png") -> Path:
        """Heatmap of the deconvolved (noise-free) region centroids: which axis defines each region."""
        return self._profile_heatmap(region_fit.mu, region_fit.dims, region_fit.names,
                                     f"Soft regions (K={region_fit.K}, arm {region_fit.arm}) — centroids", filename)

    def archetype_profiles(self, arch_fit: ArchetypeFit, filename: str = "archetype_profiles.png") -> Path:
        """Heatmap of the archetype (extreme-phenotype) profiles."""
        return self._profile_heatmap(arch_fit.Z, arch_fit.dims, arch_fit.names,
                                     f"Archetypes (A={arch_fit.A}, arm {arch_fit.arm}) — profiles", filename)

    def _profile_heatmap(self, M, dims, names, title, filename) -> Path:
        plt = self._mpl()
        fig, ax = plt.subplots(figsize=(1.1 * len(dims) + 2, 0.6 * len(M) + 1.5))
        vmax = float(np.abs(M).max()) or 1.0
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(M)))
        ax.set_yticklabels([f"{i}: {n}" for i, n in enumerate(names)], fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.7, label="z (burden)")
        return self._save(fig, filename)

    def boundary_map(self, coords: CoordinateSet, region_fit: RegionFit, x_factor: str | None = None,
                     y_factor: str | None = None, filename: str = "boundary_map.png") -> Path:
        """The soft-transition-boundary figure: 2-axis map, points shaded by membership entropy (dark = a
        boundary patient living between regions), region centroids marked. When the axes are not given, pick
        the two axes the regions most separate on (largest between-centroid spread) so the figure is always
        informative about what the regions actually split on."""
        plt = self._mpl()
        if x_factor is None or y_factor is None:
            spread = region_fit.mu.max(0) - region_fit.mu.min(0)          # per arm-axis between-centroid range
            top = np.argsort(-spread)[:2]
            x_factor = region_fit.dims[int(top[0])]
            y_factor = region_fit.dims[int(top[1])] if len(top) > 1 else region_fit.dims[int(top[0])]
        xi, yi = coords.dims.index(x_factor), coords.dims.index(y_factor)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        sc = ax.scatter(coords.X[:, xi], coords.X[:, yi], c=region_fit.entropy, cmap="magma_r",
                        s=6, alpha=0.5, vmin=0, vmax=1)
        # region centroids (in the full-arm space; index the two plotted axes via cols)
        if xi in region_fit.cols and yi in region_fit.cols:
            cx, cy = region_fit.cols.index(xi), region_fit.cols.index(yi)
            ax.scatter(region_fit.mu[:, cx], region_fit.mu[:, cy], c="#1b7837", s=200, marker="X",
                       edgecolor="white", linewidth=1.5, zorder=5)
            for k, (mx, my) in enumerate(zip(region_fit.mu[:, cx], region_fit.mu[:, cy])):
                ax.annotate(str(k), (mx, my), fontsize=11, fontweight="bold", color="#1b7837")
        ax.set_xlabel(x_factor + (" (G)" if x_factor == G_KEY else "")); ax.set_ylabel(y_factor)
        ax.set_title(f"Soft operational regions on the continuum (K={region_fit.K})\n"
                     "shading = membership entropy (dark = boundary patient)")
        fig.colorbar(sc, ax=ax, label="normalized membership entropy")
        return self._save(fig, filename)

    def confidence_bars(self, region_fit: RegionFit, filename: str = "confidence_bars.png") -> Path:
        """Per-region core/soft/boundary assignment-confidence tier counts."""
        plt = self._mpl()
        tiers = ["core", "soft", "boundary"]
        colors = {"core": "#1b7837", "soft": "#7fbf7b", "boundary": "#cf6679"}
        counts = {t: [int(((region_fit.map_label == k) & (region_fit.tier == t)).sum())
                      for k in range(region_fit.K)] for t in tiers}
        fig, ax = plt.subplots(figsize=(1.2 * region_fit.K + 2, 4))
        btm = np.zeros(region_fit.K)
        for t in tiers:
            ax.bar(range(region_fit.K), counts[t], bottom=btm, label=t, color=colors[t])
            btm += np.array(counts[t])
        ax.set_xticks(range(region_fit.K))
        ax.set_xticklabels([f"{k}\n{region_fit.names[k]}" for k in range(region_fit.K)], fontsize=7)
        ax.set_ylabel("patients"); ax.legend(); ax.set_title("Assignment confidence per region")
        return self._save(fig, filename)

    def embedding(self, coords: CoordinateSet, region_fit: RegionFit, filename: str = "embedding.png") -> Path:
        """2-D PCA embedding (VISUALIZATION-ONLY) colored by region / cohort / DSM-5 — to eyeball that the
        regions are a soft overlay and that cohort/diagnosis do not separate."""
        from sklearn.decomposition import PCA  # noqa: PLC0415
        plt = self._mpl()
        Z = PCA(2, random_state=self.config.seed).fit_transform(coords.X)
        cohort = np.asarray(coords.index.get_level_values("cohort"))
        arm = coords.validation["arm"].astype(str).to_numpy()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for ax, lab, title in [(axes[0], region_fit.map_label, "soft region (MAP)"),
                               (axes[1], cohort, "cohort"), (axes[2], arm, "DSM-5 arm")]:
            for u in pd.unique(lab):
                m = lab == u
                ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.4, label=str(u))
            ax.set_title(f"PCA (viz-only) — {title}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            if len(pd.unique(lab)) <= 8:
                ax.legend(fontsize=7, markerscale=2)
        fig.tight_layout()
        return self._save(fig, filename)
