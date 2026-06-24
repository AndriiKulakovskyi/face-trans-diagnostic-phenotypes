"""M4 outcome registry + endpoints — the outcome config (`load_outcome_config`), native-scale outcome
extraction (`extract_outcomes`), and the derived remission / recovery / deterioration endpoints
(`derive_endpoints`). Consumed by the copula OOP M4 (`prognosis_model_oop.py`), which assembles its own
analysis frame from the `strata_oop` hand-off. Outcomes are read on the **native clinical scale**
(`data/processed/baseline_v{0,1,2}.parquet`, NaN = missing) — never standardized, never imputed; a cohort
that does not collect an outcome stays NaN. (The native-panel assembler `build_analysis_frame` /
`predictor_draw_tensor` was removed with the native M4 pipeline, 2026-06-24.) Methods: docs/PROGNOSIS_MODEL.md.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[3]
_PROC = _REPO / "data" / "processed"

# Outcome-modeling likelihoods M4 supports (the GLM head, not the M1 indicator family). CGI-S is a
# 1-7 scale modeled gaussian (as M1 treats it) with an ordinal robustness variant at stage 43.
_FAMILIES = {"gaussian", "ordinal", "bernoulli"}
_COHORTS = {"bp", "sz", "dr"}
_DIRECTIONS = {"higher_better", "lower_better"}
# Which baseline-severity term enters the R2 reference rung for this outcome (see PROGNOSIS_MODEL):
#   G              — the error-aware latent severity coordinate (overall_severity)
#   baseline_cgi   — the manifest clinician severity rating cgi01 at V0
#   baseline_outcome — the outcome's own V0 value already enters at R3y; R2 then uses G to avoid
#                      double-counting (used when the outcome IS a severity/functioning scale).
_SEV_ANCHORS = {"G", "baseline_cgi", "baseline_outcome"}


@dataclass(frozen=True)
class OutcomeSpec:
    """One prognostic outcome from `configs/m4_outcomes.yaml`. `source_var` is the harmonized
    canonical name; `family` drives the GLM likelihood; `cohort_scope` is the set of cohorts that
    collect it (others -> NaN, never imputed); `severity_anchor` selects the R2 baseline-severity term;
    the optional thresholds define the binary remission/response arms; `role` is primary|secondary."""

    name: str
    label: str
    source_var: str
    family: str
    direction: str
    cohort_scope: tuple[str, ...]
    severity_anchor: str
    role: str
    remission_threshold: dict | None = None
    response_threshold: dict | None = None


@dataclass(frozen=True)
class OutcomeConfig:
    """The parsed M4 outcome registry: `meta` (horizons, seed, hdi_prob) + the validated specs."""

    meta: dict
    outcomes: list[OutcomeSpec]

    def primary(self) -> list[OutcomeSpec]:
        return [o for o in self.outcomes if o.role == "primary"]

    def by_name(self, name: str) -> OutcomeSpec:
        for o in self.outcomes:
            if o.name == name:
                return o
        raise KeyError(name)


def load_outcome_config(path: str | Path, *, available_vars=None) -> OutcomeConfig:
    """Parse + validate the M4 outcome registry. Validates family / cohort_scope / direction /
    severity_anchor. If `available_vars` is given (the harmonized variable set), any outcome whose
    `source_var` is absent is **warned-and-skipped** (e.g. `mars`, which is not in the panel)."""
    d = yaml.safe_load(Path(path).read_text())
    meta = dict(d.get("meta", {}))
    avail = None if available_vars is None else set(available_vars)
    specs: list[OutcomeSpec] = []
    for name, o in (d.get("outcomes") or {}).items():
        fam = str(o["family"])
        if fam not in _FAMILIES:
            raise ValueError(f"outcome {name!r}: family {fam!r} not in {sorted(_FAMILIES)}")
        if str(o["direction"]) not in _DIRECTIONS:
            raise ValueError(f"outcome {name!r}: direction must be one of {sorted(_DIRECTIONS)}")
        scope = tuple(str(c).lower() for c in o["cohort_scope"])
        bad = set(scope) - _COHORTS
        if bad:
            raise ValueError(f"outcome {name!r}: unknown cohort(s) {sorted(bad)}")
        anchor = str(o.get("severity_anchor", "G"))
        if anchor not in _SEV_ANCHORS:
            raise ValueError(f"outcome {name!r}: severity_anchor {anchor!r} not in {sorted(_SEV_ANCHORS)}")
        src = str(o["source_var"])
        if avail is not None and src not in avail:
            warnings.warn(f"M4 outcome {name!r}: source_var {src!r} absent from harmonized vars — skipping", stacklevel=2)
            continue
        specs.append(
            OutcomeSpec(
                name=name,
                label=str(o.get("label", name)),
                source_var=src,
                family=fam,
                direction=str(o["direction"]),
                cohort_scope=scope,
                severity_anchor=anchor,
                role=str(o.get("role", "secondary")),
                remission_threshold=o.get("remission_threshold"),
                response_threshold=o.get("response_threshold"),
            )
        )
    return OutcomeConfig(meta=meta, outcomes=specs)


# --------------------------------------------------------------------------------------------------
# Stage 41 — the analysis frame: join the fixed M3 panel (baseline coords + measurement error) to the
# outcomes (native scale, NaN-honest), the three map representations, the reference covariates, and
# the M3 attrition weights. One row per V0-roster patient. Pure helpers are unit-tested in tests/m4.
# --------------------------------------------------------------------------------------------------

def _threshold(y: pd.Series, spec: dict) -> pd.Series:
    """Binary remission endpoint from a level threshold, e.g. {">=": 71} or {"<=": 2}. NaN preserved."""
    (op, thr), = spec.items()
    if op == ">=":
        out = (y >= thr).astype(float)
    elif op == "<=":
        out = (y <= thr).astype(float)
    else:
        raise ValueError(f"remission threshold op {op!r} not in {{'>=','<='}}")
    return out.where(y.notna())


def _response(y0: pd.Series, yt: pd.Series, spec: dict) -> pd.Series:
    """Binary response endpoint from a baseline→horizon drop, e.g. {"drop>=": 2} (absolute) or
    {"pct_drop>=": 0.5} (proportional). NaN where either endpoint is missing."""
    (op, thr), = spec.items()
    valid = y0.notna() & yt.notna()
    if op == "drop>=":
        out = ((y0 - yt) >= thr).astype(float)
    elif op == "pct_drop>=":
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = (y0 - yt) / y0.replace(0, np.nan)
        out = (frac >= thr).astype(float)
        valid = valid & y0.ne(0)
    else:
        raise ValueError(f"response threshold op {op!r} not in {{'drop>=','pct_drop>='}}")
    return out.where(valid)


def derive_endpoints(frame: pd.DataFrame, specs, *, horizon: str = "V2") -> pd.DataFrame:
    """Add `{name}__remission_{H}` / `{name}__response_{H}` binary columns from the configured
    thresholds, in place on a copy. Requires `{name}__V0` and `{name}__{H}` columns to exist."""
    out = frame.copy()
    for o in specs:
        y0c, ytc = f"{o.name}__V0", f"{o.name}__{horizon}"
        if ytc not in out.columns:
            continue
        if o.remission_threshold:
            out[f"{o.name}__remission_{horizon}"] = _threshold(out[ytc], o.remission_threshold)
        if o.response_threshold and y0c in out.columns:
            out[f"{o.name}__response_{horizon}"] = _response(out[y0c], out[ytc], o.response_threshold)
    return out


def extract_outcomes(specs, *, visits=("V0", "V1", "V2"), proc_dir: str | Path = _PROC) -> pd.DataFrame:
    """Native-scale outcomes pivoted to (cohort, patient_id) × `{name}__{visit}`. Out-of-`cohort_scope`
    patients -> NaN (never imputed); an outcome absent from a visit table -> all-NaN column."""
    proc_dir = Path(proc_dir)
    tables = {v: pd.read_parquet(proc_dir / f"baseline_{v.lower()}.parquet") for v in visits}
    cols: dict[str, pd.Series] = {}
    for o in specs:
        for v in visits:
            df = tables[v]
            if o.source_var in df.columns:
                s = pd.to_numeric(df[o.source_var], errors="coerce")
            else:
                s = pd.Series(np.nan, index=df.index)
            coh = s.index.get_level_values("cohort")
            cols[f"{o.name}__{v}"] = s.where(coh.isin(list(o.cohort_scope)))
    res = pd.concat(cols, axis=1)
    res.index = res.index.set_names(["cohort", "patient_id"])
    return res

