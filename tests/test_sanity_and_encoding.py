"""Unit tests for the v2-driven loading layer: sanity bounds + new encodings.

These cover the capabilities added so the loader honours
``data/face-common-vars-v2.xlsx`` — all on **synthetic** inputs (no real CSVs),
matching the rest of the unit suite:

  - ``Variable`` carries optional ``sanity_min``/``sanity_max`` (default None)
  - ``_apply_sanity_bounds`` nulls out-of-range cells (never clips, never imputes)
    and reports the count; it is a no-op without bounds or on non-numeric dtypes
  - ms→s normalization for ECG intervals (qt/rr/qtc): value > 5 ⇒ ÷1000
  - SZ education text tokens ("BAC", "CAP") parsed to the BP/DR ordinal
  - ``derive_siteid_city`` prefers the fondacode network code over raw SITEID,
    resolving the SZ "16 vs 19" mislabel

Run:  pytest tests/test_sanity_and_encoding.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trans_diag.loader import _apply_sanity_bounds, _harmonize_series
from trans_diag.rules import (
    RULES,
    derive_siteid_city,
)
from trans_diag.variable import Variable


def _var(name, dtype, *, smin=None, smax=None):
    return Variable(
        canonical_name=name,
        bp_csv_col=f"{name}_bp",
        sz_csv_col=f"{name}_sz",
        dr_csv_col=f"{name}_dr",
        dtype=dtype,
        unit_or_value_set="",
        cluster_readiness="READY",
        clinical_rationale="r",
        rule="",
        section="BILAN BIOLOGIQUE",
        label=name,
        findings="",
        sanity_min=smin,
        sanity_max=smax,
    )


# ---------------------------------------------------------------------------
# Variable carries optional sanity fields
# ---------------------------------------------------------------------------

def test_variable_sanity_fields_default_none():
    v = _var("x", "float")
    assert v.sanity_min is None and v.sanity_max is None


def test_variable_sanity_fields_set():
    v = _var("gluc", "float", smin=0.3, smax=200.0)
    assert v.sanity_min == 0.3 and v.sanity_max == 200.0


# ---------------------------------------------------------------------------
# Sanity bounds: null out-of-range, never clip, never impute
# ---------------------------------------------------------------------------

def test_sanity_nulls_out_of_range():
    s = pd.Series([0.3, 5.0, 200.0, 488.0, -1.0])  # 488 (mg/dL slip) & -1 out
    v = _var("gluc", "float", smin=0.3, smax=200.0)
    out, n = _apply_sanity_bounds(s, v)
    assert n == 2
    assert out.tolist()[:3] == [0.3, 5.0, 200.0]          # in-range untouched
    assert pd.isna(out.iloc[3]) and pd.isna(out.iloc[4])  # out-of-range -> NA


def test_sanity_does_not_clip():
    # an out-of-range value becomes NA, NOT the boundary value (no winsorizing)
    s = pd.Series([5000.0])
    v = _var("alt", "float", smin=1.0, smax=20000.0)
    out, n = _apply_sanity_bounds(s, v)
    assert n == 0 and out.iloc[0] == 5000.0  # within [1,20000]: kept exactly
    v2 = _var("alt", "float", smin=1.0, smax=100.0)
    out2, n2 = _apply_sanity_bounds(s, v2)
    assert n2 == 1 and pd.isna(out2.iloc[0])  # outside: NA, not clipped to 100


def test_sanity_noop_without_bounds():
    s = pd.Series([1.0, 1e9, -5.0])
    out, n = _apply_sanity_bounds(s, _var("x", "float"))
    assert n == 0 and out.tolist() == s.tolist()


def test_sanity_noop_on_non_numeric():
    s = pd.Series(["a", "b"], dtype="string")
    out, n = _apply_sanity_bounds(s, _var("lbl", "string", smin=0, smax=1))
    assert n == 0 and out.tolist() == ["a", "b"]


def test_sanity_one_sided_bound():
    s = pd.Series([-1.0, 0.0, 10.0])
    out, n = _apply_sanity_bounds(s, _var("count", "float", smin=0.0))
    assert n == 1 and pd.isna(out.iloc[0]) and out.iloc[2] == 10.0


def test_sanity_preserves_existing_na():
    s = pd.Series([np.nan, 5.0, 999.0])
    out, n = _apply_sanity_bounds(s, _var("x", "float", smin=0, smax=10))
    assert n == 1 and pd.isna(out.iloc[0]) and pd.isna(out.iloc[2])


# ---------------------------------------------------------------------------
# ms→s normalization (qt / rr / qtc)
# ---------------------------------------------------------------------------

def test_ms_to_seconds_mixed_units():
    for canon in ("qt", "rr", "qtc"):
        rule = RULES[canon]
        s = pd.Series([0.40, 360.0, 0.36, np.nan])  # seconds & ms mixed
        out = rule(s, "DR")
        assert out.iloc[0] == 0.40            # already seconds, untouched
        assert abs(out.iloc[1] - 0.360) < 1e-9  # 360 ms -> 0.360 s
        assert out.iloc[2] == 0.36
        assert pd.isna(out.iloc[3])


def test_v2_only_rule_skipped_for_v1():
    # When use_v2_rules=False the qt rule is bypassed (identity_cast), so a
    # millisecond value is NOT divided — preserving legacy v1 behaviour.
    v = Variable("qt", "qt_bp", "qt_sz", "qt_dr", "float", "", "READY",
                 "r", "", "CONSTANTES ET ECG", "qt", "")
    s = pd.Series([360.0, 0.4])
    v1 = _harmonize_series(v, s, "DR", use_v2_rules=False)
    v2 = _harmonize_series(v, s, "DR", use_v2_rules=True)
    assert v1.iloc[0] == 360.0          # untouched under v1
    assert abs(v2.iloc[0] - 0.36) < 1e-9  # converted under v2


# ---------------------------------------------------------------------------
# Education: SZ text tokens → ordinal
# ---------------------------------------------------------------------------

def test_edulevel_parses_text_tokens():
    rule = RULES["edulevel"]
    s = pd.Series(["BAC", "14", "CAP", "Doctorat", "nonsense", np.nan])
    out = rule(s, "SZ")
    assert out.iloc[0] == 12.0   # BAC
    assert out.iloc[1] == 14.0   # numeric passthrough
    assert out.iloc[2] == 13.0   # CAP
    assert out.iloc[3] == 20.0   # Doctorat
    assert pd.isna(out.iloc[4]) and pd.isna(out.iloc[5])


# ---------------------------------------------------------------------------
# siteid_city via fondacode (resolves the SZ 16/19 mislabel)
# ---------------------------------------------------------------------------

def test_siteid_city_prefers_fondacode():
    siteid = pd.Series([16, 16, 1])                       # raw, partly mislabelled
    fonda = pd.Series(["1902098181", "1902098181", "102523071"])  # network 19,19,1
    out = derive_siteid_city(siteid, fonda, "SZ")
    assert out.tolist() == [19.0, 19.0, 1.0]   # fondacode wins over raw 16


def test_siteid_city_falls_back_to_raw_when_fondacode_missing():
    siteid = pd.Series([7, 3])
    fonda = pd.Series([np.nan, "bad"])  # unparseable -> fall back to raw siteid
    out = derive_siteid_city(siteid, fonda, "DR")
    assert out.tolist() == [7.0, 3.0]


# ---------------------------------------------------------------------------
# SUICIDE attempt-detail items: BP text → code, DR numeric passthrough
# ---------------------------------------------------------------------------

def test_suicide_method_rules_encode_bp_text():
    # ltsg05 (overdose lethality): BP French labels → Beck ordinal; "Mort" = 6.
    rule = RULES["ltsg05"]
    s = pd.Series([
        "Pas de conséquences médicales ou de traitement, ou minime",
        "Mort",
        "unmapped label",
        np.nan,
    ])
    out = rule(s, "BP")
    assert out.iloc[0] == 0
    assert out.iloc[1] == 6
    assert pd.isna(out.iloc[2]) and pd.isna(out.iloc[3])


def test_suicide_method_rules_pass_dr_codes_through():
    # DR is already numeric; the rule must leave its codes intact.
    rule = RULES["ltsg05"]
    out = rule(pd.Series([1.0, 3.0, 4.0]), "DR")
    assert out.dropna().tolist() == [1, 3, 4]


def test_suicide_method_rules_are_v2_gated():
    from trans_diag.loader import _V2_ONLY_RULES
    for canon in ("ltsv02", "ltsv04", "ltsv05", "ltsv06",
                  "ltsg03", "ltsg05", "ltsg06"):
        assert canon in _V2_ONLY_RULES  # never alters the v1 pipeline
