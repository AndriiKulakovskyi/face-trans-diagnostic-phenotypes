"""M4.5 — clinical endpoint construction (pure; synthetic frame, no imputation)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from face.prognosis.endpoints import build_endpoints, wilson_ci


def _frame():
    return pd.DataFrame({
        "egf__V0": [50.0, 80.0, 40.0, np.nan], "egf__V1": [55.0, 75.0, 50.0, 60.0],
        "egf__V2": [75.0, 60.0, 45.0, 70.0],
        "cgi_s__V0": [4.0, 2.0, 5.0, 3.0], "cgi_s__V1": [3.0, 2.0, 5.0, 4.0],
        "cgi_s__V2": [2.0, 4.0, 5.0, 3.0],
    })


def _col(out, name):
    return out[f"ep_{name}"].tolist()


def _eq(a, b):  # NaN-aware list compare
    return all((np.isnan(x) and np.isnan(y)) or x == y for x, y in zip(a, b))


def test_endpoint_logic_and_nan_propagation():
    out = build_endpoints(_frame())
    assert _eq(_col(out, "egf_remission"), [1.0, 0.0, 0.0, np.nan])         # V2>=71; row3 V0 missing
    assert _eq(_col(out, "egf_recovery"), [1.0, np.nan, 0.0, np.nan])       # only among baseline-impaired (V0<61)
    assert _eq(_col(out, "egf_deterioration"), [0.0, 1.0, 0.0, np.nan])     # V2 <= V0-10
    assert _eq(_col(out, "egf_sustained_impair"), [0.0, 0.0, 1.0, np.nan])  # <61 at V1 & V2 (needs all 3)
    assert _eq(_col(out, "cgi_remission"), [1.0, 0.0, 0.0, 0.0])            # CGI-S<=2
    assert _eq(_col(out, "cgi_relapse"), [0.0, 1.0, 0.0, 0.0])              # CGI-S rise >= 2
    assert _eq(_col(out, "cgi_sustained_severe"), [0.0, 0.0, 1.0, 0.0])     # >=4 at V1 & V2


def test_wilson_ci_brackets_the_rate():
    lo, hi = wilson_ci(5, 10)
    assert 0 <= lo < 0.5 < hi <= 1
    assert np.isnan(wilson_ci(0, 0)[0])                                     # empty -> NaN, no crash
    lo0, hi0 = wilson_ci(0, 20)                                             # zero events -> lo 0, hi small+
    assert lo0 == 0.0 and 0 < hi0 < 0.25
