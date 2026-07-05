"""M3 canonical fit — temporal coherence of the map + strata (V0 defines, follow-up validates).

Encodes the ONE canonical recipe, replacing its retired notebook driver. Scores
V1/V2 under the FIXED copula M1/M2 — never re-discovers the 8-dim map or the A=5 archetypes on later
visits — and walks the deterministic plan:

    invariance (G1) -> panel (V1/V2 scoring + A=5 membership) -> attrition (G6 IPW)
        -> trait_state (G3 ICC) -> persistence (G4) -> consolidate (M4 hand-off)

Prior-milestone objects consumed, all frozen: M1 map at results/m1_measurement/primary/idata.nc; M2
coords + frozen archetype profiles at results/m2_strata/{coordinates,consolidate}; the V0 standardization
spec + the V1/V2 baselines + V0 covariates under data/processed/. Config of record: A=5, seed 20260622,
projection draws=500 tune=600 chains=2, n_keep_draws=200 (smoke: draws=80 tune=80, n_keep=40).
"""
from __future__ import annotations

import time
from dataclasses import replace

from face.config import paths
from face.temporal.engine import TemporalConfig, TemporalRunner

SEED = 20260622                                     # the M3 config-of-record seed (scorer/projection/G3/G4)
RESULTS = paths.results("m3")                       # results/m3_temporal


def run_m3(*, mode: str = "production", seed: int = SEED, overwrite: bool = False,
           log=print) -> dict:
    """Run the canonical M3 temporal-coherence plan. ``mode`` in {smoke, production}.

    Returns a summary dict (per-stage headline metrics). The M4 hand-off lands at
    ``results/m3_temporal/consolidate/patient_panel.parquet``.
    """
    smoke = mode == "smoke"
    config = TemporalConfig(output_dir=RESULTS, seed=seed)
    if smoke:
        config = replace(config.with_smoke_defaults(), seed=seed)

    runner = TemporalRunner(config)
    log(f"[m3] {'smoke' if smoke else 'production'} temporal-coherence; A={config.A} "
        f"seed={config.seed}; scoring V1/V2 under the FIXED copula M1/M2")
    log(f"[m3]   map={config.map_dir}  strata={config.strata_dir}  out={config.output_dir}")

    state: dict = {}
    for stage in runner.PLAN:                        # invariance -> panel -> attrition -> trait_state ...
        t0 = time.time()
        state = runner.run_stage(stage, state, overwrite=overwrite)
        log(f"[m3]   {stage.name}: elapsed={round(time.time() - t0)}s")

    summary: dict = {"mode": mode, "seed": config.seed, "A": config.A}
    panel = state.get("patient_panel", state.get("panel"))
    if panel is not None:
        summary["panel_rows"] = int(len(panel))
    ts = state.get("trait_state")
    if ts is not None:
        summary["g3_trait_state"] = ts[["axis", "icc", "verdict"]].to_dict("records")
    log(f"[m3] DONE -> {RESULTS / 'consolidate' / 'patient_panel.parquet'}")
    return summary
