"""M4.0 — outcome-registry parsing/validation (pure; no sampling, no data)."""
from __future__ import annotations

from pathlib import Path

import pytest

from face.prognosis.frame import OutcomeConfig, load_outcome_config

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "m4_outcomes.yaml"


def _write(tmp_path, body: str) -> Path:
    p = tmp_path / "outcomes.yaml"
    p.write_text(body)
    return p


def test_real_config_parses_and_locks_primaries():
    cfg = load_outcome_config(CONFIG)
    assert isinstance(cfg, OutcomeConfig)
    assert {o.name for o in cfg.primary()} == {"egf", "cgi_s"}      # PI-locked 2026-06-10
    egf = cfg.by_name("egf")
    assert egf.source_var == "egf" and egf.cohort_scope == ("bp", "sz", "dr")
    assert egf.remission_threshold == {">=": 71}
    # SZ-absent secondaries are correctly scoped BP/DR
    assert cfg.by_name("fast").cohort_scope == ("bp", "dr")
    assert cfg.meta["primary_horizon"] == "V2" and cfg.meta["secondary_horizon"] == "V1"


def test_warn_and_skip_missing_source_var(tmp_path):
    body = (
        "meta: {primary_horizon: V2}\n"
        "outcomes:\n"
        "  mars: {label: m, source_var: mars, family: gaussian, direction: lower_better,\n"
        "         cohort_scope: [bp], severity_anchor: G, role: secondary}\n"
        "  egf: {label: e, source_var: egf, family: gaussian, direction: higher_better,\n"
        "        cohort_scope: [bp, sz, dr], severity_anchor: baseline_outcome, role: primary}\n"
    )
    p = _write(tmp_path, body)
    with pytest.warns(UserWarning, match="mars"):
        cfg = load_outcome_config(p, available_vars={"egf", "cgi01"})
    assert {o.name for o in cfg.outcomes} == {"egf"}                # mars dropped, egf kept


def test_bad_family_raises(tmp_path):
    body = ("outcomes:\n  x: {source_var: x, family: poisson, direction: lower_better,\n"
            "       cohort_scope: [bp], severity_anchor: G}\n")
    with pytest.raises(ValueError, match="family"):
        load_outcome_config(_write(tmp_path, body))


def test_bad_cohort_raises(tmp_path):
    body = ("outcomes:\n  x: {source_var: x, family: gaussian, direction: lower_better,\n"
            "       cohort_scope: [bp, xx], severity_anchor: G}\n")
    with pytest.raises(ValueError, match="cohort"):
        load_outcome_config(_write(tmp_path, body))


def test_bad_direction_and_anchor_raise(tmp_path):
    bad_dir = ("outcomes:\n  x: {source_var: x, family: gaussian, direction: up,\n"
               "       cohort_scope: [bp], severity_anchor: G}\n")
    with pytest.raises(ValueError, match="direction"):
        load_outcome_config(_write(tmp_path, bad_dir))
    bad_anchor = ("outcomes:\n  x: {source_var: x, family: gaussian, direction: lower_better,\n"
                  "       cohort_scope: [bp], severity_anchor: nonsense}\n")
    with pytest.raises(ValueError, match="severity_anchor"):
        load_outcome_config(_write(tmp_path, bad_anchor))
