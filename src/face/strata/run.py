"""M2 canonical fit — soft-region stratification on the M1 transdiagnostic map.

Encodes the ONE canonical recipe that turns the fixed M1 primary map into the operational strata
hand-off M3-M5 consume, replacing the ``run_strata_model_oop.py`` notebook invocation. The engine is
deterministic numpy/EM (no MCMC, no warm-start): a six-stage plan cached under ``results/m2_strata/<stage>/``.

Plan of record: coordinates (score 9,013 patients on the 8 axes from the M1 copula posterior) ->
structure (continuum gate + falsification null) -> regions (operational-K XD mixture, arms A/B) ->
archetypes (operational-A archetypal analysis, arms A/B; the reported A=5 simplex) -> usefulness
(internal 5-criterion battery) -> consolidate (per-patient membership frame + nested K-family overlay +
frozen archetype anchors). The map is a CONTINUUM; the operative K is deferred to M4/M5 incremental
validity. Consumes results/m1_measurement/primary; writes results/m2_strata/{coordinates/,
consolidate/{patient_strata.parquet, k_family_menu.csv, archetype_profiles.csv}}.
"""
from __future__ import annotations

import time

from face.config import paths
from face.strata.engine import StrataConfig, StrataRunner

SEED = 20260605  # CLI-uniform signature seed; the strata engine's science seed of record is 20260621
RESULTS = paths.results("m2")       # results/m2_strata
FIGURES = paths.figures("m2")       # docs/figures/m2_strata


def _base_config(smoke: bool) -> StrataConfig:
    """Base config: smoke flips the plan to tiny sweeps/draws (a wiring check, not science)."""
    cfg = StrataConfig(output_dir=RESULTS, figure_dir=FIGURES)
    return cfg.with_smoke_defaults() if smoke else cfg


def run_m2(*, mode: str = "production", seed: int = SEED, overwrite: bool = False,
           log=print) -> dict:
    """Run the canonical six-stage M2 strata fit. ``mode`` in {smoke, production}.

    Deterministic (numpy/EM); each stage caches under ``results/m2_strata/<stage>/`` (``consolidate``
    always rebuilds). Reads the fixed M1 primary map at ``results/m1_measurement/primary/``. Returns a
    summary dict; the operational hand-off lands at ``results/m2_strata/consolidate/``.
    """
    smoke = mode == "smoke"
    config = _base_config(smoke)
    runner = StrataRunner(config)
    log(f"[m2] mode={'smoke' if smoke else 'production'} map={config.map_dir} out={config.output_dir}")

    state: dict = {}
    for stage in config.stage_plan:
        t0 = time.time()
        state = runner.run_stage(stage, state, overwrite=overwrite, n_perm=30)
        log(f"[m2]   {stage.name}: elapsed={round(time.time() - t0)}s")

    coords = state["coords"]
    region = state["region_A"]
    arch = state["arch_A"]
    frame = state["patient_strata"]
    menu = state["k_family_menu"]
    summary = {
        "milestone": "m2_strata", "mode": mode,
        "map_dir": str(config.map_dir), "output_dir": str(config.output_dir),
        "N": int(coords.X.shape[0]),
        "structure_verdict_A": state["structure"]["verdict_A"]["label"],
        "operational_K": int(region.K),
        "archetypes_A": int(arch.A),
        "patient_strata_rows": int(len(frame)),
        "k_family_Ks": [int(k) for k in menu["K"]],
        "handoff": {
            "patient_strata": str(config.output_dir / "consolidate" / "patient_strata.parquet"),
            "k_family_menu": str(config.output_dir / "consolidate" / "k_family_menu.csv"),
            "archetype_profiles": str(config.output_dir / "consolidate" / "archetype_profiles.csv"),
        },
    }
    log(f"[m2] DONE -> {config.output_dir / 'consolidate' / 'patient_strata.parquet'} "
        f"(K={region.K}, A={arch.A}, N={summary['N']})")
    return summary
