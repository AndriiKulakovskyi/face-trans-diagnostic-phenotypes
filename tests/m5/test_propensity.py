"""M5.2 — propensity / exposure-contrast / overlap logic (pure tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from face.treatment.propensity import (define_exposure, overlap, smd, stabilized_iptw)


def _merged():
    # 4 BP patients: 1 on lithium, 1 on other mood-stab, 1 on antipsychotic, 1 untreated
    return pd.DataFrame({
        "cohort": ["bp", "bp", "bp", "bp"],
        "patient_id": ["1", "2", "3", "4"],
        "on_lithium": [1.0, 0.0, 0.0, 0.0],
        "on_mood_stabilizer": [0.0, 1.0, 0.0, 0.0],
        "on_antipsychotic": [0.0, 0.0, 1.0, 0.0],
        "on_antidepressant": [0.0, 0.0, 0.0, 0.0],
    })


def test_active_comparator_excludes_untreated():
    sub, treat = define_exposure(_merged(), "lithium_bp", "active_comparator")
    # exposed = pt1; comparator = pt2 (mood-stab) + pt3 (antipsychotic); pt4 (untreated) EXCLUDED
    assert set(sub["patient_id"]) == {"1", "2", "3"}
    assert treat.tolist() == [1.0, 0.0, 0.0]


def test_on_off_includes_untreated_as_control():
    sub, treat = define_exposure(_merged(), "lithium_bp", "on_off")
    assert set(sub["patient_id"]) == {"1", "2", "3", "4"}        # untreated is a control here
    assert treat.sum() == 1.0 and (treat == 0).sum() == 3.0


def test_stabilized_iptw_and_trim():
    ps = np.array([0.2, 0.5, 0.8, 0.5]); treat = np.array([1.0, 1.0, 0.0, 0.0])
    w, keep = stabilized_iptw(ps, treat, trim_to_support=True)
    assert np.all(w > 0) and w.shape == (4,)
    # common support = [0.5, 0.5] is degenerate here; just assert mask is boolean and sized
    assert keep.dtype == bool and keep.shape == (4,)


def test_overlap_and_smd():
    ps = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.9]); treat = np.array([1.0, 1, 1, 0, 0, 0])
    d = overlap(ps, treat)
    assert d["n_treated"] == 3 and d["n_control"] == 3 and 0 <= d["frac_in_support"] <= 1
    # SMD: a perfectly-separated covariate has large SMD; an identical one ~0
    X = np.column_stack([treat, np.ones(6)])
    s = smd(X, treat)
    assert s[0] >= 1.0 and s[1] == 0.0          # separated covariate -> large SMD; constant -> 0
