"""Unit tests for trans_diag.domains (item → domain aggregation).

Covers: instrument-stem grouping, masked mean with a min-items threshold (no
imputation), biology composite sign orientation, and that one multi-item
instrument collapses to a single balanced dimension.

Run:  pytest tests/test_domains.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trans_diag.domains import (
    BIOLOGY_COMPOSITES,
    DOMAIN_SECTIONS,
    build_domain_scores,
    instrument_stem,
)
from trans_diag.variable import Variable


def _var(name, section):
    return Variable(
        canonical_name=name, bp_csv_col=f"{name}_bp", sz_csv_col=f"{name}_sz",
        dr_csv_col=f"{name}_dr", dtype="float", unit_or_value_set="",
        cluster_readiness="READY", clinical_rationale="", rule="",
        section=section, label=name, findings="",
    )


def test_instrument_stem():
    assert instrument_stem("isf01a") == "isf"
    assert instrument_stem("cssrs3") == "cssrs"
    assert instrument_stem("psqi11") == "psqi"
    assert instrument_stem("madrs") == "madrs"
    # no trailing item number → stays its own single-item domain
    assert instrument_stem("agedebut_cigarettes_lt") == "agedebut_cigarettes_lt"


def test_multi_item_instrument_collapses_to_one_domain():
    rng = np.random.default_rng(0)
    n = 50
    X = pd.DataFrame({
        "madrs1": rng.normal(0, 1, n),
        "madrs2": rng.normal(0, 1, n),
        "madrs3": rng.normal(0, 1, n),
    })
    variables = [_var(c, "AUTO-QUESTIONNAIRES") for c in X.columns]
    scores, meta = build_domain_scores(X, variables)
    assert list(scores.columns) == ["madrs"]          # 3 items → 1 domain
    assert meta.loc["madrs", "n_items"] == 3
    assert meta.loc["madrs", "kind"] == "symptom"


def test_min_items_threshold_masks_underobserved():
    # 4-item instrument; a patient observing only 1/4 (<50%) → NaN
    X = pd.DataFrame({
        "foo1": [1.0, 2.0, 3.0, 4.0, np.nan],
        "foo2": [1.0, 2.0, 3.0, 4.0, np.nan],
        "foo3": [1.0, 2.0, 3.0, 4.0, np.nan],
        "foo4": [1.0, 2.0, 3.0, 4.0, 9.0],   # last patient: only this item
    })
    variables = [_var(c, "SUICIDE") for c in X.columns]
    scores, _ = build_domain_scores(X, variables, min_items_frac=0.5)
    assert np.isnan(scores["foo"].iloc[4])           # 1/4 observed → masked
    assert scores["foo"].iloc[:4].notna().all()


def test_biology_composite_sign_orientation():
    # metabolic-style composite: bmi (+1), hdl (-1, lower = worse)
    rng = np.random.default_rng(1)
    n = 40
    bmi = rng.normal(25, 4, n)
    hdl = rng.normal(1.3, 0.3, n)
    X = pd.DataFrame({"bmi": bmi, "hdl": hdl})
    variables = [_var("bmi", "CONSTANTES ET ECG"), _var("hdl", "BILAN BIOLOGIQUE")]
    biology = {"metabolic": [("bmi", +1), ("hdl", -1)]}
    scores, meta = build_domain_scores(X, variables, symptom_sections=set(),
                                       biology=biology)
    assert list(scores.columns) == ["metabolic"]
    assert meta.loc["metabolic", "kind"] == "biology"
    # high BMI + low HDL → high (pathological) score; correlation with bmi>0, hdl<0
    assert np.corrcoef(scores["metabolic"], bmi)[0, 1] > 0
    assert np.corrcoef(scores["metabolic"], hdl)[0, 1] < 0


def test_constants_reference_biology_composites():
    # the curated composites reference the canonicals we expect
    assert "metabolic_syndrome" in BIOLOGY_COMPOSITES
    members = dict(BIOLOGY_COMPOSITES["metabolic_syndrome"])
    assert members["hdl"] == -1            # HDL is protective → reversed
    assert members["bmi"] == +1
    assert "BILAN BIOLOGIQUE" in DOMAIN_SECTIONS
    assert "AUTO-QUESTIONNAIRES" in DOMAIN_SECTIONS
