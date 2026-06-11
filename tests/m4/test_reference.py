"""M4.2 — reference design builder (pure; synthetic frame, no sampling)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face.prognosis.frame import OutcomeSpec
from face.prognosis.reference import (design_for_rung, modeling_frame, outcome_vector,
                                      severity_column, site_index)

EGF = OutcomeSpec(name="egf", label="egf", source_var="egf", family="gaussian",
                  direction="higher_better", cohort_scope=("bp", "sz", "dr"),
                  severity_anchor="baseline_outcome", role="primary")
CGI = OutcomeSpec(name="cgi_s", label="cgi", source_var="cgi01", family="gaussian",
                  direction="lower_better", cohort_scope=("bp", "sz", "dr"),
                  severity_anchor="G", role="primary")


def _frame(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "egf__V0": rng.normal(50, 10, n), "egf__V2": rng.normal(60, 10, n),
        "cgi_s__V0": rng.integers(1, 8, n).astype(float),
        "overall_severity__mean": rng.normal(0, 1, n),
        "age": rng.normal(40, 10, n), "sex": rng.integers(0, 2, n).astype(float),
        "siteid_city": rng.integers(0, 3, n).astype(float),
        "arm": rng.choice(["BP1", "SZ", "MDD"], n),
    })


def test_severity_column_routes_by_anchor():
    assert severity_column(EGF, cgi_baseline_col="cgi_s__V0") == "cgi_s__V0"
    assert severity_column(CGI, cgi_baseline_col="cgi_s__V0") == "overall_severity__mean"


def test_rungs_are_strictly_nested():
    sub = _frame()
    cols = {}
    for rung in ("R0", "R1", "R2", "R3y"):
        X, names = design_for_rung(sub, EGF, rung, severity_col="cgi_s__V0", horizon="V2")
        assert X.shape == (len(sub), len(names))
        cols[rung] = names
    assert cols["R0"] == ["age", "sex"]
    assert set(cols["R0"]).issubset(cols["R1"])               # nesting
    assert set(cols["R1"]).issubset(cols["R2"])
    assert set(cols["R2"]).issubset(cols["R3y"])
    assert any(c.startswith("arm_") for c in cols["R1"])      # diagnosis enters at R1
    assert "sev::cgi_s__V0" in cols["R2"]                     # severity enters at R2
    assert "egf__V0" in cols["R3y"]                           # baseline outcome enters at R3y


def test_modeling_frame_drops_incomplete_rows_no_imputation():
    sub = _frame()
    sub.loc[sub.index[0], "egf__V2"] = np.nan
    sub.loc[sub.index[1], "age"] = np.nan
    out = modeling_frame(sub, EGF, horizon="V2", severity_col="cgi_s__V0")
    assert len(out) == len(sub) - 2                           # both incomplete rows dropped, none filled


def test_outcome_vector_standardizes_gaussian():
    sub = _frame()
    y, fam, n_cat = outcome_vector(sub, EGF, horizon="V2")
    assert fam == "gaussian" and n_cat is None
    assert abs(y.mean()) < 1e-9 and abs(y.std() - 1.0) < 1e-9


def test_site_index_is_contiguous():
    sub = _frame()
    idx, n = site_index(sub)
    assert n == sub["siteid_city"].nunique()
    assert set(np.unique(idx)) == set(range(n))
