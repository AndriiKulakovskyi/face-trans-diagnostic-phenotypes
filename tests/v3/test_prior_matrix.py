"""V3 config + prior-matrix contract tests.

These enforce the load-bearing invariants of the config-first measurement layer:
  * the soft-prior matrix is COMPLETE (full item x factor grid, regenerable from configs);
  * every modeled item carries exactly one valid likelihood family and >=1 home cell;
  * the general factor is bifactor-identified (anchors load on G only);
  * diagnosis / covariates / outcomes are NEVER modeled as latent indicators (no leakage).

Run: pytest tests/v3 -q   (pythonpath=src is set in pyproject).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
MATRIX = CONFIGS / "prior_loading_matrix_v3.csv"

VALID_FAMILIES = {"gaussian", "lognormal", "student_t", "ordered_logistic",
                  "bernoulli", "neg_binomial", "poisson"}
CONTINUOUS = {"gaussian", "lognormal", "student_t"}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    assert MATRIX.exists(), "prior_loading_matrix_v3.csv missing — run v3.priors.build_matrix"
    return list(csv.DictReader(MATRIX.open()))


@pytest.fixture(scope="module")
def dims() -> dict:
    return yaml.safe_load((CONFIGS / "dimensions.yaml").read_text())


@pytest.fixture(scope="module")
def priors() -> dict:
    return yaml.safe_load((CONFIGS / "priors.yaml").read_text())


# ---------------------------------------------------------------- completeness / shape
def test_matrix_is_full_grid(rows):
    items = {r["item"] for r in rows}
    factors = {r["factor"] for r in rows}
    assert len(rows) == len(items) * len(factors), "matrix is not a complete item x factor grid"


def test_every_item_has_one_likelihood(rows):
    fam_by_item: dict[str, set] = {}
    for r in rows:
        fam_by_item.setdefault(r["item"], set()).add(r["likelihood_family"])
    bad = {it: f for it, f in fam_by_item.items() if len(f) != 1}
    assert not bad, f"items with inconsistent/missing likelihood: {bad}"


def test_likelihood_families_valid(rows):
    fams = {r["likelihood_family"] for r in rows}
    assert fams <= VALID_FAMILIES, f"unknown likelihood family present: {fams - VALID_FAMILIES}"


def test_modeling_block_consistent(rows):
    for r in rows:
        exp = "continuous" if r["likelihood_family"] in CONTINUOUS else "explicit"
        assert r["modeling_block"] == exp, f"{r['item']}: block != family routing"


def test_every_item_has_home_cell(rows):
    """Every modeled item must anchor SOMEWHERE (primary or g_anchor) — no orphan items."""
    home = {r["item"] for r in rows if r["prior_type"] in ("primary", "g_anchor")}
    allitems = {r["item"] for r in rows}
    orphans = allitems - home
    assert not orphans, f"items with no primary/g_anchor home cell: {orphans}"


# ----------------------------------------------------------- general-factor identification
def test_g_anchors_orthogonal_to_specifics(rows, dims):
    """G anchors load on G (g_anchor) and ~0 on every specific (g_anchor_on_specific)."""
    g_key = dims["general_factor"]["key"]
    specifics = {f["key"] for f in dims["factors"] if f.get("model_factor")}
    by_item: dict[str, dict[str, str]] = {}
    for r in rows:
        by_item.setdefault(r["item"], {})[r["factor"]] = r["prior_type"]
    g_anchor_items = [it for it, fc in by_item.items() if fc.get(g_key) == "g_anchor"]
    assert g_anchor_items, "no G-anchor items found"
    for it in g_anchor_items:
        for sp in specifics:
            assert by_item[it][sp] == "g_anchor_on_specific", \
                f"G-anchor {it} not held ~0 on specific {sp}"


def test_g_anchor_specific_is_near_zero(rows, priors):
    sd = float(priors["tiers"]["g_anchor_on_specific"]["sd"])
    assert sd <= 0.01, "g_anchor_on_specific prior is not a near-hard zero"
    for r in rows:
        if r["prior_type"] == "g_anchor_on_specific":
            assert abs(float(r["prior_sd"]) - sd) < 1e-9


# ----------------------------------------------------------------- no leakage / diagnosis
def test_no_diagnosis_or_covariate_as_indicator(rows, dims):
    """cohort/arm/site/age/sex and declared covariates are NEVER latent indicators."""
    items = {r["item"] for r in rows}
    forbidden = set(dims.get("covariates", [])) | set(dims.get("validation_labels", []))
    forbidden |= {"cohort", "arm", "siteid_city", "visit", "visitnum", "usubjid_patients"}
    leak = items & forbidden
    assert not leak, f"covariate/diagnosis leaked into indicators: {leak}"


def test_covariate_only_items_excluded(rows, dims):
    """Items tagged covariate_only in any factor must not be modeled indicators."""
    items = {r["item"] for r in rows}
    cov_only: set[str] = set()
    for f in dims["factors"]:
        cov_only |= set(f.get("covariate_only", []) or [])
    leak = items & cov_only
    assert not leak, f"covariate_only items leaked into indicators: {leak}"


# -------------------------------------------------------------------- prior tier values
def test_prior_tiers_match_config(rows, priors):
    tiers = priors["tiers"]
    for r in rows:
        t = tiers[r["prior_type"]]
        assert abs(float(r["prior_mean"]) - t["mean"]) < 1e-9
        assert abs(float(r["prior_sd"]) - t["sd"]) < 1e-9


def test_primary_cells_are_sign_anchored(rows):
    for r in rows:
        if r["prior_type"] in ("primary", "g_anchor"):
            assert r["sign_constraint"] == "positive", \
                f"{r['item']}->{r['factor']} home loading must be sign-anchored"
