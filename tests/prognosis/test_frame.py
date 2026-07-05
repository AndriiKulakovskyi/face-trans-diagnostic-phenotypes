"""M4.1 — analysis-frame helpers (pure; synthetic fixtures, no real artifacts)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from face.prognosis.frame import (
    OutcomeSpec,
    _response,
    _threshold,
    derive_endpoints,
    extract_outcomes,
)


def _spec(name, src, scope, family="gaussian", direction="lower_better", **kw):
    return OutcomeSpec(name=name, label=name, source_var=src, family=family, direction=direction,
                       cohort_scope=tuple(scope), severity_anchor="G", role="secondary", **kw)


# ---- threshold / response endpoint logic ----
def test_threshold_ge_and_le_preserve_nan():
    y = pd.Series([80.0, 60.0, np.nan])
    assert _threshold(y, {">=": 71}).tolist()[:2] == [1.0, 0.0]
    assert np.isnan(_threshold(y, {">=": 71}).iloc[2])
    assert _threshold(pd.Series([1.0, 3.0]), {"<=": 2}).tolist() == [1.0, 0.0]


def test_response_absolute_and_pct_drop():
    y0 = pd.Series([6.0, 4.0, 10.0, np.nan])
    yt = pd.Series([3.0, 4.0, 4.0, 2.0])
    assert _response(y0, yt, {"drop>=": 2}).tolist()[:3] == [1.0, 0.0, 1.0]
    assert np.isnan(_response(y0, yt, {"drop>=": 2}).iloc[3])          # y0 missing -> NaN
    # pct drop: (6-3)/6=.5 yes ; (4-4)/4=0 no ; (10-4)/10=.6 yes
    assert _response(y0, yt, {"pct_drop>=": 0.5}).tolist()[:3] == [1.0, 0.0, 1.0]


def test_derive_endpoints_adds_expected_columns():
    frame = pd.DataFrame({"egf__V0": [50.0, 80.0], "egf__V2": [72.0, 60.0],
                          "cgi_s__V0": [5.0, 4.0], "cgi_s__V2": [2.0, 4.0]})
    specs = [_spec("egf", "egf", ["bp"], direction="higher_better", remission_threshold={">=": 71}),
             _spec("cgi_s", "cgi01", ["bp"], remission_threshold={"<=": 2},
                   response_threshold={"drop>=": 2})]
    out = derive_endpoints(frame, specs, horizon="V2")
    assert out["egf__remission_V2"].tolist() == [1.0, 0.0]            # 72>=71, 60<71
    assert out["cgi_s__remission_V2"].tolist() == [1.0, 0.0]          # 2<=2, 4>2
    assert out["cgi_s__response_V2"].tolist() == [1.0, 0.0]           # 5-2=3>=2, 4-4=0<2


# ---- cohort-scope masking (the no-imputation invariant) ----
def test_extract_outcomes_masks_out_of_scope_cohort(tmp_path):
    idx = pd.MultiIndex.from_tuples([("bp", "1"), ("dr", "2")], names=["cohort", "patient_id"])
    pd.DataFrame({"fast": [10.0, 20.0]}, index=idx).to_parquet(tmp_path / "baseline_v0.parquet")
    out = extract_outcomes([_spec("fast", "fast", ["bp"])], visits=("V0",), proc_dir=tmp_path)
    assert out.loc[("bp", "1"), "fast__V0"] == 10.0
    assert np.isnan(out.loc[("dr", "2"), "fast__V0"])                 # DR out of scope -> NaN, not imputed


def test_extract_outcomes_absent_var_is_all_nan(tmp_path):
    idx = pd.MultiIndex.from_tuples([("bp", "1")], names=["cohort", "patient_id"])
    pd.DataFrame({"egf": [55.0]}, index=idx).to_parquet(tmp_path / "baseline_v0.parquet")
    out = extract_outcomes([_spec("ghost", "not_a_var", ["bp"])], visits=("V0",), proc_dir=tmp_path)
    assert out["ghost__V0"].isna().all()


