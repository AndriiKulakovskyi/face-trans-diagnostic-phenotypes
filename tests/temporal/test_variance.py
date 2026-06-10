"""M3 G3 — trait/state pattern grouping + verdict logic (pure; no sampling)."""
from __future__ import annotations

import pandas as pd

from face.temporal.variance import _verdict, patient_patterns, raw_icc


def _panel():
    rows = [("A::1", "V0", 1.0, 0.1), ("A::1", "V1", 1.1, 0.1), ("A::1", "V2", 0.9, 0.1),  # 3-visit
            ("A::2", "V0", -1.0, 0.1), ("A::2", "V1", -1.2, 0.1),                           # V0,V1
            ("A::3", "V0", 0.0, 0.5),                                                       # V0 only
            ("A::4", "V0", 0.5, 0.2), ("A::4", "V2", 0.4, 0.2)]                             # V0,V2 (skip V1)
    df = pd.DataFrame(rows, columns=["patient_uid", "visit", "ax__mean", "ax__sd"])
    df["n_visits"] = df.groupby("patient_uid")["visit"].transform("count")
    return df


def test_patient_patterns_groups_by_visit_set():
    pat = patient_patterns(_panel())
    assert set(pat) == {(0, 1, 2), (0, 1), (0,), (0, 2)}
    assert pat[(0, 1, 2)].shape == (1, 3)          # one patient, three visits
    assert pat[(0, 1)].shape == (1, 2)
    assert pat[(0,)].shape == (1, 1)
    assert list(pat[(0, 1, 2)][0]) == [0, 1, 2]    # row indices of patient A::1, t-sorted


def test_verdict_bands():
    assert _verdict(0.70, 0.85) == "trait"          # CI clears 0.5, mean ≥ 0.6
    assert _verdict(0.10, 0.30) == "state"          # CI under 0.5, mean ≤ 0.4
    assert _verdict(0.40, 0.70) == "mixed"          # straddles 0.5


def test_raw_icc_in_unit_interval():
    r = raw_icc(_panel(), ["ax"], min_visits=2)
    assert "ax" in r and 0.0 <= r["ax"] <= 1.0
