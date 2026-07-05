"""M5 canonical fit — treatment-response heterogeneity on the fixed M1-M4 objects (bounds-and-defends).

Encodes the ONE canonical recipe that produces the M5 treatment verdict the article consumes, replacing its retired notebook driver. Consumes the FIXED M4 frame (results/m4_prognosis/frame/analysis_frame.parquet)
+ the harmonized drug-class exposures (raw TRAITEMENTS) + response endpoints, and walks the causal arc:

  exposures -> frame -> propensity (overlap gate) -> moderation (treat x map-axis EIV/fixed interaction
  + E-value, over the durable trio AND the A=5 archetypes) -> confounder (does the M4 carrier survive
  treatment adjustment?) -> tolerability -> heterogeneity -> atlas -> consolidate.

Config of record (oracle-matched): horizon V2, moderation_reps (durable, archetypes), seed 20260605,
draws=700 tune=700 chains=4. Observational-TAU: the map is prognostic + descriptive, not prescriptive.
The canonical verdict table lands at results/m5_treatment/consolidate/treatment_summary.csv.
"""
from __future__ import annotations

import time
from dataclasses import replace

from face.config import paths
from face.treatment.engine import TreatmentConfig, TreatmentRunner

SEED = 20260605


def _budgets(smoke: bool):
    """(draws, tune, chains) — tiny for the smoke wiring check, else the reported MCMC budget."""
    return (120, 120, 2) if smoke else (700, 700, 4)


def run_m5(*, mode: str = "production", seed: int = SEED, overwrite: bool = False,
           log=print) -> dict:
    """Run the canonical M5 treatment-moderation plan. ``mode`` in {smoke, production}.

    Consumes the fixed M4 frame at ``results/m4_prognosis/frame/`` (run ``face fit m4`` first). Returns a
    summary dict; the verdict table lands at ``results/m5_treatment/consolidate/treatment_summary.csv``.
    """
    smoke = mode == "smoke"
    draws, tune, chains = _budgets(smoke)
    results = paths.results("m5")                    # results/m5_treatment
    cfg = replace(TreatmentConfig(), seed=seed, draws=draws, tune=tune, chains=chains, smoke=smoke,
                  output_dir=results, figure_dir=paths.figures("m5"))
    runner = TreatmentRunner(cfg)
    log(f"[m5] treatment moderation ({'smoke' if smoke else 'production'}); "
        f"reps={list(cfg.moderation_reps)} horizon={cfg.horizon} seed={seed}")
    m4_frame = cfg.prognosis_dir / "frame" / "analysis_frame.parquet"
    if not m4_frame.exists():
        raise FileNotFoundError(f"[m5] missing M4 hand-off {m4_frame}; run `face fit m4` before m5")

    state: dict = {}
    for stage in cfg.stage_plan:
        t0 = time.time()
        state = runner.run_stage(stage, state, overwrite=overwrite)
        n = None
        obj = state.get(stage.name)
        if obj is not None and hasattr(obj, "__len__"):
            n = int(len(obj))
        log(f"[m5]   {stage.name}: rows={n} elapsed={round(time.time() - t0)}s")

    summary: dict = {"mode": mode, "seed": seed, "output_dir": str(results)}
    summ = state.get("treatment_summary")
    if summ is not None and len(summ):
        summary["verdict_rows"] = int(len(summ))
    log(f"[m5] DONE -> {results / 'consolidate' / 'treatment_summary.csv'}")
    return summary
