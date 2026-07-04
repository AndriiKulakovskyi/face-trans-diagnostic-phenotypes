"""DR MARS reverse-coding harmonization — the data-layer fix.

DR's raw MARS-10 total is summed with the opposite item polarity (mirror image of BP/SZ); the rule
reflects DR (10 − x) onto the common 0–10 "higher = better adherence" scale and leaves BP/SZ unchanged.
"""
from __future__ import annotations

import pandas as pd

from face.data.rules import harmonize_mars


def test_dr_mars_is_reflected():
    s = pd.Series([0, 2, 5, 10], dtype="float64")
    assert harmonize_mars(s, "DR").tolist() == [10.0, 8.0, 5.0, 0.0]   # 10 − x


def test_bp_and_sz_mars_pass_through():
    s = pd.Series([0, 2, 8, 10], dtype="float64")
    assert harmonize_mars(s, "BP").tolist() == [0.0, 2.0, 8.0, 10.0]
    assert harmonize_mars(s, "SZ").tolist() == [0.0, 2.0, 8.0, 10.0]


def test_mars_coerces_non_numeric_to_nan():
    out = harmonize_mars(pd.Series(["bad", "3", None]), "BP")
    assert pd.isna(out.iloc[0]) and out.iloc[1] == 3.0 and pd.isna(out.iloc[2])


def test_reflection_preserves_0_10_range():
    s = pd.Series(range(11), dtype="float64")        # 0..10
    out = harmonize_mars(s, "DR")
    assert out.min() == 0.0 and out.max() == 10.0     # stays in-bounds, no clipping needed
