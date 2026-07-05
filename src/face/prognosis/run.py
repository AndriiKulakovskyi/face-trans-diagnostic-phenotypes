"""M4 canonical fit — prognosis: does the durable map *predict* beyond DSM-5 + severity?

Encodes the ONE canonical recipe (the deterministic 8-stage plan the copula M4 engine walks), replacing its retired notebook driver. The engine is a pure *consumer* of the fixed upstream objects — it never
re-discovers or re-scores:

  M2 (results/m2_strata)  — baseline coordinates (mean/sd/reliability) + the A=5 archetype simplex +
                            the nested K-family tessellation + covariates.
  M3 (results/m3_temporal/attrition) — strata-independent attrition IPW weights.
  outcomes (data/processed/baseline_v{0,1,2}.parquet + configs/prognosis_outcomes.yaml) — native-scale,
                            NaN-honest, never imputed.

Plan (each stage cached on MODEL_VERSION + stage_spec + config_sig): frame -> reference ->
incremental (the operative-K selector) -> transdiagnostic -> endpoints -> clinical_value -> robustness ->
consolidate (the M5 hand-off). Config of record (oracle-matched): horizon V2, encodings {+durable,
+archetypesA, +archetypesB, +tessfamily, +specifics8}, seed 20260610, draws=800 tune=1000 chains=4.
"""
from __future__ import annotations

import time
from dataclasses import replace

from face.prognosis.engine import PrognosisConfig, PrognosisRunner

SEED = 20260610


def _budgets(smoke: bool):
    """(draws, tune, chains, target_accept) — tiny for the smoke wiring check, else the reported plan."""
    return (120, 120, 2, 0.9) if smoke else (800, 1000, 4, 0.95)


def run_m4(*, mode: str = "production", seed: int = SEED, overwrite: bool = False,
           log=print) -> dict:
    """Run the canonical M4 prognosis plan. ``mode`` in {smoke, production}.

    Consumes the FIXED M2 coords/archetypes (``results/m2_strata``) + M3 IPW
    (``results/m3_temporal/attrition``) + outcomes. Returns a summary dict; the M5 hand-off lands at
    ``results/m4_prognosis/consolidate/`` and the archetype atlas at ``results/m4_prognosis/endpoints/``.
    """
    smoke = mode == "smoke"
    draws, tune, chains, target_accept = _budgets(smoke)
    config = replace(PrognosisConfig(), seed=seed, draws=draws, tune=tune, chains=chains,
                     target_accept=target_accept, smoke=smoke)
    runner = PrognosisRunner(config)
    log(f"[m4] mode={'smoke' if smoke else 'production'} (draws={draws} tune={tune} chains={chains}); "
        f"horizon={config.horizon} encodings={list(config.encodings)}")
    log(f"[m4]   inputs: M2={config.strata_dir}  M3-IPW={config.ipw_dir}  outcomes={config.config_path}")

    state: dict = {}
    for stage in config.stage_plan:
        t0 = time.time()
        state = runner.run_stage(stage, state, overwrite=overwrite)
        note = ""
        if stage.kind == "frame" and "frame" in state:
            note = f"rows={len(state['frame'])}"
        elif stage.kind == "incremental" and "operative_k" in state:
            note = f"operative_K={state['operative_k'].get('operative_K')}"
        elif stage.kind == "consolidate" and "prognosis_summary" in state:
            note = f"summary_rows={len(state['prognosis_summary'])}"
        log(f"[m4]   {stage.name}: {note} elapsed={round(time.time() - t0)}s")

    operative = state.get("operative_k", {})
    summary = {
        "mode": mode, "seed": seed, "output_dir": str(config.output_dir),
        "frame_rows": int(len(state["frame"])) if "frame" in state else None,
        "operative_K": operative.get("operative_K"),
        "operative_verdict": operative.get("verdict"),
        "handoff": {
            "patient_risk": str(config.output_dir / "consolidate" / "prognosis_patient_risk.parquet"),
            "prognosis_summary": str(config.output_dir / "consolidate" / "prognosis_summary.csv"),
            "archetype_atlas": str(config.output_dir / "endpoints" / "archetype_atlas.csv"),
        },
    }
    log(f"[m4] DONE -> {config.output_dir / 'consolidate' / 'prognosis_summary.csv'}")
    return summary
