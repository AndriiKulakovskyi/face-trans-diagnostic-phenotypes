"""M4.4 — head-to-head + transdiagnostic helpers (pure; no sampling)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from face.prognosis.frame import OutcomeSpec
from face.prognosis.reference import arm_block, foundation_design
from face.prognosis.transdiagnostic import dominance_verdict, interaction_block

EGF = OutcomeSpec(name="egf", label="egf", source_var="egf", family="gaussian",
                  direction="higher_better", cohort_scope=("bp", "sz", "dr"),
                  severity_anchor="baseline_outcome", role="primary")


def _frame(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "egf__V0": rng.normal(50, 10, n), "egf__V2": rng.normal(60, 10, n),
        "cgi_s__V0": rng.integers(1, 8, n).astype(float),
        "overall_severity__mean": rng.normal(0, 1, n),
        "age": rng.normal(40, 10, n), "sex": rng.integers(0, 2, n).astype(float),
        "siteid_city": rng.integers(0, 3, n).astype(float),
        "arm": rng.choice(["BP1", "SZ", "MDD"], n), "cohort": rng.choice(["bp", "sz", "dr"], n),
    })


def test_interaction_block_drops_reference_cohort():
    map_X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])      # N=3, P=2
    out = interaction_block(map_X, [0, 1, 2], n_cohorts=3)       # -> [3, P*(3-1)=4]
    assert out.shape == (3, 4)
    assert (out[0] == 0).all()                                  # cohort 0 = reference -> all-zero
    np.testing.assert_array_equal(out[1], [3.0, 4.0, 0.0, 0.0])  # sz block active on row 1
    np.testing.assert_array_equal(out[2], [0.0, 0.0, 5.0, 6.0])  # dr block active on row 2


def _h2h(mbd, dbm):
    return pd.DataFrame({"contrast": ["map beyond DSM-5 (B−A)", "DSM-5 beyond map (B−C)"],
                         "verdict": [mbd, dbm]})


def test_dominance_verdict_reads_the_asymmetry():
    assert dominance_verdict(_h2h("predictive", "ambiguous")) == "map-dominates"
    assert dominance_verdict(_h2h("ambiguous", "predictive")) == "dsm5-dominates"
    assert dominance_verdict(_h2h("predictive", "predictive")) == "co-informative"
    assert dominance_verdict(_h2h("ambiguous", "ambiguous")) == "neither"


def test_foundation_excludes_arm_and_map():
    sub = _frame()
    Xd, nd = foundation_design(sub, EGF, severity_col="cgi_s__V0", horizon="V2")
    assert nd == ["age", "sex", "sev::cgi_s__V0", "egf__V0"]     # no arm, no map
    assert Xd.shape == (len(sub), 4)


def test_arm_block_is_dummy_encoded():
    sub = _frame()
    Xa, na = arm_block(sub)
    assert all(c.startswith("arm_") for c in na)
    assert Xa.shape == (len(sub), len(na)) and len(na) == sub["arm"].nunique() - 1
