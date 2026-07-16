"""Skip-logic decoding (gate=No -> structural 0) for the ISF suicide module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face.data.rules import RULES
from face.data.skip_logic import SkipRule, decode_skip_logic


def test_fills_structural_zero_only_where_gate_no_and_missing():
    df = pd.DataFrame({
        "isf05": [0.0, 1.0, np.nan, 0.0],   # No, Yes, unknown, No
        "isf07": [np.nan, np.nan, np.nan, 5.0],
    })
    out, _ = decode_skip_logic(df, rules=[SkipRule("isf05", ("isf07",))])
    vals = out["isf07"].tolist()
    assert vals[0] == 0.0          # gate No + missing  -> recovered 0
    assert np.isnan(vals[1])       # gate Yes + missing -> genuinely missing
    assert np.isnan(vals[2])       # gate unknown       -> not inferred
    assert vals[3] == 5.0          # existing value     -> never overwritten


def test_does_not_overwrite_inconsistent_gate_no_but_count_positive():
    df = pd.DataFrame({"isf05": [0.0], "isf07": [3.0]})
    out, _ = decode_skip_logic(df, rules=[SkipRule("isf05", ("isf07",))])
    assert out["isf07"].tolist()[0] == 3.0


def test_cascade_isf05_to_isf08a_via_isf08():
    # row0: never attempted -> isf08 set No -> isf08a 0
    # row1: attempted, no violent (isf08 No) -> isf08a 0
    # row2: attempted, violent (isf08 Yes), count missing -> stays NaN
    df = pd.DataFrame({
        "isf05": [0.0, 1.0, 1.0],
        "isf08": [np.nan, 0.0, 1.0],
        "isf08a": [np.nan, np.nan, np.nan],
    })
    out, _ = decode_skip_logic(df)  # default SUICIDE_SKIP_RULES (ordered)
    assert out["isf08"].tolist()[0] == 0.0     # isf05=No cascaded into isf08
    assert out["isf08a"].tolist()[0] == 0.0
    assert out["isf08a"].tolist()[1] == 0.0
    assert np.isnan(out["isf08a"].tolist()[2])  # real attempter, count unrecorded


def test_never_smoker_recovers_zero_pack_years_without_zeroing_fagerstrom():
    df = pd.DataFrame({
        # 1=never, 2=former, 3=current, missing=unknown.
        "suncf_cigarettes_lt": [1.0, 2.0, 3.0, np.nan, 1.0],
        "sudose_cigarettes_lt": [np.nan, np.nan, np.nan, np.nan, 4.0],
        "fagers": [np.nan, np.nan, np.nan, np.nan, np.nan],
    })

    out, report = decode_skip_logic(df)

    assert out["sudose_cigarettes_lt"].tolist()[0] == 0.0
    assert np.isnan(out["sudose_cigarettes_lt"].tolist()[1])
    assert np.isnan(out["sudose_cigarettes_lt"].tolist()[2])
    assert np.isnan(out["sudose_cigarettes_lt"].tolist()[3])
    assert out["sudose_cigarettes_lt"].tolist()[4] == 4.0
    assert out["fagers"].isna().all()
    assert {
        "gate": "suncf_cigarettes_lt",
        "dependent": "sudose_cigarettes_lt",
        "n_filled": 1,
    } in report


def test_report_counts_filled_cells():
    df = pd.DataFrame({"isf05": [0.0, 0.0, 1.0], "isf07": [np.nan, np.nan, np.nan]})
    _, report = decode_skip_logic(df, rules=[SkipRule("isf05", ("isf07",))])
    assert report == [{"gate": "isf05", "dependent": "isf07", "n_filled": 2}]


def test_missing_columns_is_noop():
    df = pd.DataFrame({"foo": [1.0, 2.0]})
    out, report = decode_skip_logic(df)
    assert out.equals(df)
    assert report == []


def test_not_inplace_by_default():
    df = pd.DataFrame({"isf05": [0.0], "isf07": [np.nan]})
    out, _ = decode_skip_logic(df, rules=[SkipRule("isf05", ("isf07",))])
    assert np.isnan(df["isf07"].tolist()[0])   # original untouched
    assert out["isf07"].tolist()[0] == 0.0


def test_isf07_inequality_rule_decodes_ceiling_and_comma():
    fn = RULES["isf07"]
    out = fn(pd.Series([">10", "3", "3,5", "abc", None, 4]), "SZ").tolist()
    assert out[0] == 10.0      # '>10' ceiling token -> 10
    assert out[1] == 3.0
    assert out[2] == 3.5       # decimal comma
    assert np.isnan(out[3])    # junk -> NaN
    assert np.isnan(out[4])    # None -> NaN
    assert out[5] == 4.0


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/bipolar.csv").exists(),
    reason="confidential cohort CSVs absent (clean clone)",
)
def test_integration_recovers_isf07_coverage():
    from face.data import (
        build_unified_dataframe,
        load_variables,
        to_harmonized_dataset,
    )

    variables = load_variables("data/face-common-vars.xlsx")
    df = build_unified_dataframe(
        "data", "data/face-common-vars.xlsx",
        readiness=["READY", "PARTIAL"], format="long",
    )
    ds_on = to_harmonized_dataset(df, variables, visit="V0", apply_skip_logic=True)
    ds_off = to_harmonized_dataset(df, variables, visit="V0", apply_skip_logic=False)
    if "isf07" not in ds_on.X.columns:
        pytest.skip("isf07 not in feature matrix")
    cov_on = ds_on.X["isf07"].notna().mean()
    cov_off = ds_off.X["isf07"].notna().mean()
    assert cov_on > cov_off          # decoding strictly recovers data
    assert cov_on > 0.6              # ~0.72-0.92 per cohort, pooled well above raw ~0.3


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/bipolar.csv").exists(),
    reason="confidential cohort CSVs absent (clean clone)",
)
def test_integration_recovers_never_smoker_pack_years_only():
    from face.data import (
        build_unified_dataframe,
        load_variables,
        to_harmonized_dataset,
    )

    variables = load_variables("data/face-common-vars.xlsx")
    df = build_unified_dataframe(
        "data", "data/face-common-vars.xlsx",
        readiness=["READY", "PARTIAL"], format="long",
    )
    ds_on = to_harmonized_dataset(df, variables, visit="V0", apply_skip_logic=True)
    ds_off = to_harmonized_dataset(df, variables, visit="V0", apply_skip_logic=False)

    status = ds_off.X["suncf_cigarettes_lt"]
    pack_off = ds_off.X["sudose_cigarettes_lt"]
    pack_on = ds_on.X["sudose_cigarettes_lt"]
    recovered = status.eq(1) & pack_off.isna()

    assert int(recovered.sum()) > 3000
    assert pack_on.loc[recovered].eq(0).all()
    assert pack_on.loc[~recovered].equals(pack_off.loc[~recovered])
    assert ds_on.X["fagers"].equals(ds_off.X["fagers"])
