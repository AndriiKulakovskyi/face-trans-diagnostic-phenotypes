"""Canonical corrected M1 fit for the eight-factor transdiagnostic map.

The production path implements the Methods contract: rank-INT Gaussian block,
native Bernoulli/ordered-logistic/negative-binomial block, joint item-level
covariates, exhaustive routing, dynamic explicit-factor closure, an LKJ prior on
the free specific-factor correlation block, an explicit local-independence
de-duplication profile, and a full-N cohort-balanced generalized posterior.
Smoke, diagnostic, and production artifacts are isolated.
"""
from __future__ import annotations

import json
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
F_SUBSTANCE = F3 + ["substance"]
F8 = [
    "overall_severity",
    "cognition",
    "immunometabolic",
    "sleep",
    "suicidality",
    "developmental_risk",
    "mania_activation",
    "substance",
]

RESULTS = paths.results("m1")
CORRECTED_ROOT = RESULTS / "corrected_v6_correlated_substance"
CROSSLOAD_MATRIX = registry.path("loading_matrix.immunometabolic_crossload")
CONTINUATION_RETRY_MAP_RHAT_MAX = 1.015
CONTINUATION_RETRY_NUISANCE_RHAT_MAX = 1.025
CONTINUATION_RETRY_ESS_FRACTION = 0.75
CONTINUATION_ADVANCE_CAP_FRACTION_MAX = 0.10

# Deletion-only handling of severe within-instrument/assay local dependence.
# Each retained counterpart remains in the model: CVLT total recall, WAIS coding,
# CTQ/PSQI/FAST component scores, ordinal CTQ minimization/denial, BMI and waist,
# supine HR/BP, WBC, LDL, and ALT.  Totals or deterministic recodes are excluded
# when their components are retained.  Cholesterol is excluded while LDL is
# retained as the more clinically actionable lipid measure; assay provenance must
# still be checked before this choice is described as removal of a mathematically
# derived duplicate.  See docs/M1_LOCAL_INDEPENDENCE_EXCLUSIONS.md.
LOCAL_INDEPENDENCE_EXCLUSIONS = (
    "cvlt_short_delay_free_recall",
    "cvlt_long_delay_free_recall",
    "wais_ivt_index",
    "ctq39",
    "ctq41",
    "psqi",
    "fast",
    "weight",
    "hrstanding",
    "eghrmn",
    "sysbpstanding",
    "diabpstanding",
    "neut",
    "chol",
    "ast_lbstresc",
)

# The current baseline export does not provide valid alcohol/cannabis lifetime
# disorder flags. BP's multi-select children omit most parent-positive rows; SZ
# V0 is constant negative despite positive diagnostic branches. These are data-
# validity exclusions, not local-independence deletions. See the aggregate audit
# produced by scripts/00_audit_substance_harmonization.py.
DATA_QUALITY_EXCLUSIONS = (
    "suoccur_alcool",
    "suoccur_cannabis",
)
PRIMARY_EXCLUSIONS = (*LOCAL_INDEPENDENCE_EXCLUSIONS, *DATA_QUALITY_EXCLUSIONS)

# With the two native SUD indicators quarantined, substance is measured by the
# Gaussianized nicotine indicators and is marginalized. Keep this assertion in
# sync with MeasurementDataset.mixed()'s dynamic native-routing closure.
PRIMARY_EXPLICIT_FACTORS = [
    factor for factor in DEFAULT_EXPLICIT_FACTORS if factor != "substance"
]


def _eligible_continuation_retry(runner: StageRunner, manifest: dict) -> bool:
    """Return whether a continuous stage merits one longer Monte Carlo retry.

    Small R-hat or ESS misses with otherwise healthy HMC geometry indicate that
    more retained draws may be sufficient.  The bounded screen prevents retries
    for multimodality or geometry failures.  It does not weaken certification:
    the retry must satisfy the original tiered thresholds.
    """
    diagnostics = manifest.get("diagnostics", {})
    map_rhat_key = "core_map_rhat" if "core_map_rhat" in diagnostics else "map_rhat"
    map_bulk_key = (
        "core_map_ess_bulk" if "core_map_ess_bulk" in diagnostics else "map_ess_bulk"
    )
    map_tail_key = (
        "core_map_ess_tail" if "core_map_ess_tail" in diagnostics else "map_ess_tail"
    )
    required = (
        map_rhat_key,
        "nuisance_rhat",
        map_bulk_key,
        map_tail_key,
        "nuisance_ess_bulk",
        "nuisance_ess_tail",
    )
    if any(diagnostics.get(key) is None for key in required):
        return False
    if diagnostics[map_rhat_key] > CONTINUATION_RETRY_MAP_RHAT_MAX:
        return False
    if diagnostics["nuisance_rhat"] > CONTINUATION_RETRY_NUISANCE_RHAT_MAX:
        return False
    retry_ess_min = CONTINUATION_RETRY_ESS_FRACTION * runner.config.ess_min
    if any(diagnostics[key] < retry_ess_min for key in required[2:]):
        return False
    counterfactual = dict(
        diagnostics,
        map_rhat=runner.config.rhat_max,
        core_map_rhat=runner.config.rhat_max,
        nuisance_rhat=runner.config.nuisance_rhat_max,
        map_ess_bulk=runner.config.ess_min,
        map_ess_tail=runner.config.ess_min,
        core_map_ess_bulk=runner.config.ess_min,
        core_map_ess_tail=runner.config.ess_min,
        nuisance_ess_bulk=runner.config.ess_min,
        nuisance_ess_tail=runner.config.ess_min,
    )
    return runner._passes_gates(counterfactual)


def _eligible_continuation_advance(runner: StageRunner, manifest: dict) -> bool:
    """Allow a nonterminal rung to warm-start the next model with a warning.

    This is not certification.  It applies only when every inferential and
    chain-alignment gate passes after neutralizing tree/step-cap diagnostics, the
    aggregate cap fraction is at most 10%, and the excess is isolated to at most
    one chain.  Widespread or severe saturation therefore remains fail-closed.
    """
    diagnostics = manifest.get("diagnostics", {})
    depth_by_chain = diagnostics.get("tree_depth_cap_fraction_by_chain")
    steps_by_chain = diagnostics.get("n_steps_cap_fraction_by_chain")
    depth_fraction = diagnostics.get("tree_depth_cap_fraction")
    steps_fraction = diagnostics.get("n_steps_cap_fraction")
    if any(
        value is None
        for value in (
            depth_by_chain,
            steps_by_chain,
            depth_fraction,
            steps_fraction,
        )
    ):
        return False
    cap = runner.config.max_depth_fraction
    if not (
        depth_fraction > cap
        or steps_fraction > cap
        or any(value > cap for value in depth_by_chain)
        or any(value > cap for value in steps_by_chain)
    ):
        return False
    if max(depth_fraction, steps_fraction) > CONTINUATION_ADVANCE_CAP_FRACTION_MAX:
        return False
    if sum(value > cap for value in depth_by_chain) > 1:
        return False
    if sum(value > cap for value in steps_by_chain) > 1:
        return False
    counterfactual = dict(
        diagnostics,
        tree_depth_cap_fraction=0.0,
        tree_depth_cap_fraction_by_chain=[0.0] * len(depth_by_chain),
        n_steps_cap_fraction=0.0,
        n_steps_cap_fraction_by_chain=[0.0] * len(steps_by_chain),
    )
    return runner._passes_gates(counterfactual)


def _recipe(mode: str) -> tuple[dict, dict, int]:
    if mode == "smoke":
        return (
            {"draws": 20, "tune": 30, "chains": 2},
            {"draws": 20, "tune": 40, "chains": 2, "n_subsample": 60},
            7,
        )
    if mode == "diagnostic":
        return (
            {"draws": 500, "tune": 800, "chains": 4},
            # The N=600 screen admitted two finite-sample cognition basins even
            # though their full-data likelihoods were decisively separated.
            # Use the same balanced N=2,000 support as the continuation fit so
            # the diagnostic tests the model that will actually initialize the
            # full-N posterior rather than a noisier surrogate estimand.
            {"draws": 500, "tune": 1000, "chains": 4, "n_subsample": 2000},
            10,
        )
    if mode == "production":
        return (
            {"draws": 1000, "tune": 1000, "chains": 4},
            {"draws": 1500, "tune": 2000, "chains": 4, "n_subsample": 2000},
            10,
        )
    raise ValueError(f"unknown M1 mode {mode!r}")


def run_m1(
    *,
    mode: str = "production",
    seed: int = SEED,
    overwrite: bool = False,
    run_correlated_g: bool = False,
    log=print,
) -> dict:
    """Run one isolated M1 ladder and return its terminal manifest.

    Production runs the full-N primary after a balanced continuation fit.  The
    optional correlated-G sensitivity is permitted only after the primary passes
    its fail-closed certification gate.
    """
    cont_budget, mixed_budget, max_tree_depth = _recipe(mode)
    output_dir = CORRECTED_ROOT / mode
    base_config = (
        MeasurementConfig()
        .with_gaussian_copula()
        .with_equal_home_loadings("substance")
        .with_excluded_items(*PRIMARY_EXCLUSIONS)
    )
    config = replace(
        base_config,
        prior_matrix=CROSSLOAD_MATRIX,
        covariate_mode="in_likelihood",
        covariate_missingness="mean_indicator",
        cohort_weighted=True,
        max_tree_depth=max_tree_depth,
        output_dir=output_dir,
        figure_dir=paths.figures("m1") / "corrected_v6_correlated_substance" / mode,
    )
    runner = StageRunner(config)

    balanced = {
        "balanced": True,
        "n_subsample": mixed_budget["n_subsample"],
    }
    mixed_common = {
        "correlated": True,
        "windows": True,
        "mixed": True,
        "explicit_factors": list(PRIMARY_EXPLICIT_FACTORS),
        "min_cohorts": 2,
        "seed": seed,
        "target_accept": 0.95,
        "specific_cross": True,
        "cross_sd_scale": 1.0,
        "hurdle_counts": False,
    }
    simple_core = StageDefinition(
        "simple_core",
        F1,
        correlated=False,
        windows=False,
        target_accept=0.95,
        seed=seed,
        balanced=True,
        n_subsample=mixed_budget["n_subsample"],
        enforce_gates=mode != "smoke",
        **cont_budget,
    )
    s1 = StageDefinition(
        "continuous_s1",
        F1,
        correlated=True,
        windows=True,
        target_accept=0.90,
        seed=seed,
        balanced=True,
        n_subsample=mixed_budget["n_subsample"],
        enforce_gates=mode != "smoke",
        **cont_budget,
    )
    s3 = StageDefinition(
        "continuous_s3",
        F3,
        correlated=True,
        windows=True,
        target_accept=0.90,
        seed=seed,
        balanced=True,
        n_subsample=mixed_budget["n_subsample"],
        enforce_gates=mode != "smoke",
        **cont_budget,
    )
    continuous_substance = StageDefinition(
        "continuous_substance",
        F_SUBSTANCE,
        correlated=True,
        windows=True,
        target_accept=0.95,
        seed=seed,
        balanced=True,
        n_subsample=mixed_budget["n_subsample"],
        specific_cross=True,
        cross_sd_scale=1.0,
        enforce_gates=mode != "smoke",
        **cont_budget,
    )
    mixed_balanced = StageDefinition(
        "mixed_balanced",
        F8,
        **mixed_common,
        **balanced,
        draws=mixed_budget["draws"],
        tune=mixed_budget["tune"],
        chains=mixed_budget["chains"],
        enforce_gates=mode != "smoke",
    )

    log(
        f"[m1] corrected-v6-correlated-substance mode={mode} output={output_dir} "
        f"covariates=in_likelihood count=negative_binomial "
        f"specific_covariance=full_lkj substance_loading=tau_equivalent "
        f"excluded={list(PRIMARY_EXCLUSIONS)}"
    )
    def run_continuation(stage: StageDefinition, prev_stage: StageDefinition | None):
        """Run a continuous rung and, when justified, one longer named retry."""
        started = time.time()
        try:
            _idata, payload = runner.run_stage(
                stage, overwrite=overwrite, prev_stage=prev_stage
            )
            log(
                f"[m1] {stage.name} complete in {time.time() - started:.0f}s; "
                f"diagnostics={payload.get('diagnostics')}"
            )
            return stage, payload
        except RuntimeError as failure:
            stage_failure = failure

        manifest_path = output_dir / stage.name / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {}
        )
        if mode == "diagnostic" and _eligible_continuation_retry(runner, manifest):
            retry = replace(
                stage,
                name=f"{stage.name}_retry",
                draws=max(1000, stage.draws * 2),
                tune=max(1000, stage.tune),
                seed=stage.seed + 101,
            )
            diagnostics = manifest["diagnostics"]
            min_ess = min(
                diagnostics["map_ess_bulk"],
                diagnostics["map_ess_tail"],
                diagnostics["nuisance_ess_bulk"],
                diagnostics["nuisance_ess_tail"],
            )
            log(
                f"[m1] {stage.name} had bounded Monte Carlo shortfalls "
                f"(map_rhat={diagnostics['map_rhat']:.6f}, "
                f"nuisance_rhat={diagnostics['nuisance_rhat']:.6f}, "
                f"min_ess={min_ess:.1f}); running "
                f"{retry.name} with draws={retry.draws}, tune={retry.tune}"
            )
            started = time.time()
            _idata, payload = runner.run_stage(
                retry, overwrite=overwrite, prev_stage=stage
            )
            log(
                f"[m1] {retry.name} complete in {time.time() - started:.0f}s; "
                f"diagnostics={payload.get('diagnostics')}"
            )
            return retry, payload

        if _eligible_continuation_advance(runner, manifest):
            diagnostics = manifest["diagnostics"]
            manifest["continuation_decision"] = {
                "accepted": True,
                "certified": False,
                "reason": "isolated_maximum_tree_depth_efficiency_warning",
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tree_depth_cap_fraction": diagnostics["tree_depth_cap_fraction"],
                "tree_depth_cap_fraction_by_chain": diagnostics[
                    "tree_depth_cap_fraction_by_chain"
                ],
                "n_steps_cap_fraction": diagnostics["n_steps_cap_fraction"],
                "n_steps_cap_fraction_by_chain": diagnostics[
                    "n_steps_cap_fraction_by_chain"
                ],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            log(
                f"[m1] ADVANCE_WITH_WARNING {stage.name}: certification remains "
                f"failed; continuation accepted for isolated tree-depth "
                f"inefficiency (depth={diagnostics['tree_depth_cap_fraction']:.3f}, "
                f"steps={diagnostics['n_steps_cap_fraction']:.3f})"
            )
            return stage, manifest

        raise stage_failure

    prev = None
    terminal = None
    for stage in (simple_core, s1, s3, continuous_substance):
        prev, terminal = run_continuation(stage, prev)

    started = time.time()
    _idata, terminal = runner.run_stage(
        mixed_balanced, overwrite=overwrite, prev_stage=prev
    )
    log(
        f"[m1] {mixed_balanced.name} complete in {time.time() - started:.0f}s; "
        f"diagnostics={terminal.get('diagnostics')}"
    )
    prev = mixed_balanced

    if mode != "production":
        return terminal or {}

    primary = StageDefinition(
        "primary",
        F8,
        **mixed_common,
        balanced=False,
        n_subsample=None,
        draws=mixed_budget["draws"],
        tune=mixed_budget["tune"],
        chains=mixed_budget["chains"],
        enforce_gates=True,
    )
    _idata, terminal = runner.run_stage(
        primary, overwrite=overwrite, prev_stage=mixed_balanced
    )
    log(f"[m1] primary certification={terminal['certification']}")

    if run_correlated_g:
        correlated_g = replace(
            primary,
            name="sensitivity_correlated_g",
            g_correlated=True,
            enforce_gates=True,
        )
        _idata, terminal = runner.run_stage(
            correlated_g, overwrite=overwrite, prev_stage=primary
        )
        log(f"[m1] correlated-G certification={terminal['certification']}")
    return terminal
