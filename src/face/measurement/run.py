"""M1 canonical fit — the transdiagnostic dimensional map (the load-bearing milestone).

Encodes the ONE canonical recipe that produces the 8-factor operational map M2-M5 consume, replacing the
tangle of `run_horseshoe_map.py --weighted --fold --salvage` notebook invocations (and their stale
`results/face/oop_measurement/...` paths). Two phases, each warm-starting the next:

  Phase 1 — balanced anchor (N=2000, cohort-balanced): the horseshoe immunometabolic-merge ladder
    hs_s1_merged (G,cognition,immunometabolic,sleep) -> hs_s3_merged (+developmental,mania) ->
    hs_s5_merged_xc (full 8-factor MIXED, +suicidality+substance, hard-zero + 3 earned cross-loadings).
    Converges cleanly and anchors the substance loadings.  -> results/m1_measurement/_balanced/

  Phase 2 — cohort-weighted salvage (full N=9,013): re-fit ONLY the mixed stage with substance pinned
    ORTHOGONAL (its cross-factor correlations are non-identifiable), warm-started from the Phase-1 mixed
    fit.  This is the canonical estimand.  -> results/m1_measurement/primary/{idata.nc, manifest.json}

Config of record (oracle-matched): gaussian-copula (rank-INT) likelihood, residualized covariates,
regularized-horseshoe off-home cross-loadings folded to the 3 earned cells, seed 20260605,
mixed stage draws=1500 tune=2000 chains=4 target_accept=0.95.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import replace

from face.config import paths, registry
from face.measurement.engine import (
    DEFAULT_EXPLICIT_FACTORS,
    MeasurementConfig,
    StageDefinition,
    StageRunner,
)

SEED = 20260605
F1 = ["overall_severity", "cognition", "immunometabolic", "sleep"]
F3 = F1 + ["developmental_risk", "mania_activation"]
F8 = ["overall_severity", "cognition", "immunometabolic", "sleep",
      "suicidality", "developmental_risk", "mania_activation", "substance"]

RESULTS = paths.results("m1")                       # results/m1_measurement
MERGED_MATRIX = registry.path("loading_matrix.immunometabolic")            # 8-factor biology merge
CROSSLOAD_MATRIX = registry.path("loading_matrix.immunometabolic_crossload")  # merge + 3 earned cross-loads


def _budgets(smoke: bool):
    """(continuous draws/tune, mixed draws/tune, chains) — tiny for the smoke wiring check."""
    return (40, 40, 40, 40, 2) if smoke else (1000, 1000, 1500, 2000, 4)


def run_m1(*, mode: str = "production", seed: int = SEED, overwrite: bool = False,
           log=print) -> dict:
    """Run the canonical two-phase M1 fit. ``mode`` in {smoke, production}.

    Returns the final manifest dict. The canonical map lands at ``results/m1_measurement/primary/``.
    """
    smoke = mode == "smoke"
    dc, dt, mc, mt, ch = _budgets(smoke)
    # balanced (Phase 1) vs full-N cohort-weighted (Phase 2) sampling knobs
    bal = dict(balanced=True, n_subsample=40 if smoke else 2000)
    full = dict(balanced=False, n_subsample=None)
    cont_bal = dict(correlated=True, windows=True, mixed=False, seed=seed, **bal)
    mixed_common = dict(correlated=True, windows=True, mixed=True,
                        explicit_factors=list(DEFAULT_EXPLICIT_FACTORS), min_cohorts=2, seed=seed)

    # ---- stage definitions (shared factor lists / budgets) ----
    s1 = StageDefinition("hs_s1_merged", F1, draws=dc, tune=dt, chains=ch, target_accept=0.95, **cont_bal)
    s3 = StageDefinition("hs_s3_merged", F3, draws=dc, tune=dt, chains=ch, target_accept=0.95, **cont_bal)
    s5_bal = StageDefinition("hs_s5_merged_xc", F8, draws=mc, tune=mt, chains=ch, target_accept=0.95,
                             specific_cross=True, cross_sd_scale=1.0, **mixed_common, **bal)

    # ================= Phase 1 — balanced anchor (horseshoe merge ladder) =================
    bal_cfg = replace(MeasurementConfig().with_gaussian_copula(),
                      prior_matrix=CROSSLOAD_MATRIX, cohort_weighted=False,
                      output_dir=RESULTS / "_balanced", figure_dir=RESULTS / "_balanced")
    runner = StageRunner(bal_cfg)
    log(f"[m1] Phase 1 — balanced anchor ({'smoke' if smoke else 'N=2000'}); factors={F8}")
    prev = None
    for st in (s1, s3, s5_bal):
        t0 = time.time()
        _idata, man = runner.run_stage(st, overwrite=overwrite, prev_stage=prev)
        log(f"[m1]   {st.name}: diagnostics={man.get('diagnostics', {})} elapsed={round(time.time()-t0)}s")
        prev = st
    bal_src = bal_cfg.output_dir / s5_bal.name / "idata.nc"

    # ================= Phase 2 — cohort-weighted salvage (substance orthogonal) =================
    final_cfg = replace(MeasurementConfig().with_gaussian_copula(),
                        prior_matrix=CROSSLOAD_MATRIX, cohort_weighted=(not smoke),
                        output_dir=RESULTS, figure_dir=paths.figures("m1")).with_substance_orthogonal()
    # seed the warm-start dir from the converged balanced mixed fit
    seed_dir = RESULTS / "bal_seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(bal_src, seed_dir / "idata.nc")
    bal_seed = StageDefinition("bal_seed", F8, draws=mc, tune=mt, chains=ch, target_accept=0.95,
                               specific_cross=True, cross_sd_scale=1.0, **mixed_common, **full)
    primary = StageDefinition("primary", F8, draws=mc, tune=mt, chains=ch, target_accept=0.95,
                              specific_cross=True, cross_sd_scale=1.0, **mixed_common, **full)
    log(f"[m1] Phase 2 — {'smoke' if smoke else 'full-N cohort-weighted'} salvage; "
        f"substance orthogonal {final_cfg.orthogonal_factors}; warm-start from balanced")
    t0 = time.time()
    _idata, man = StageRunner(final_cfg).run_stage(primary, overwrite=True, prev_stage=bal_seed)
    log(f"[m1]   primary: diagnostics={man.get('diagnostics', {})} elapsed={round(time.time()-t0)}s")
    log(f"[m1] DONE -> {RESULTS / 'primary' / 'idata.nc'}")
    return man
