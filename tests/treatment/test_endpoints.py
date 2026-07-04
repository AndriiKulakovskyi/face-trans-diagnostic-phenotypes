"""M5.0 — treatment-response endpoint construction (pure; CGI codings, 0=not-assessed -> NaN)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from face.treatment.endpoints import build_endpoints, load_m5_config

REPO = Path(__file__).resolve().parents[2]


def _frame():
    return pd.DataFrame({
        "cgi02": [1, 2, 3, 4, 0, 7],     # CGI-I: 1/2 responder; 3/4/7 no; 0 = not assessed -> NaN
        "cgi03a": [1, 2, 3, 4, 0, 1],    # therapeutic effect: 1/2 good; 0 -> NaN
        "cgi03b": [1, 2, 3, 4, 1, 4],    # side-effects: >=3 significant
        "cgi01": [5, 3, 4, 6, 2, 5],     # CGI-S (for resistance)
        "mars": [3, 8, 5, 10, 6, 2],     # adherence 0-10
    })


def _eq(a, b):
    return all((np.isnan(x) and np.isnan(y)) or x == y for x, y in zip(a, b, strict=False))


def test_endpoint_logic_and_not_assessed_to_nan():
    out = build_endpoints(_frame(), mars_low=5, resistance_cgis=4)
    assert _eq(out["ep_response"].tolist(), [1, 1, 0, 0, np.nan, 0])          # cgi02 in {1,2}; 0->NaN
    assert _eq(out["ep_therapeutic_effect"].tolist(), [1, 1, 0, 0, np.nan, 1])
    assert _eq(out["ep_side_effects"].tolist(), [0, 0, 1, 1, 0, 1])           # cgi03b >= 3
    # resistance = CGI-S>=4 AND not-improved(cgi02>=3); row4 cgi02=0->NaN -> resistance NaN
    assert _eq(out["ep_resistance"].tolist(), [0, 0, 1, 1, np.nan, 1])
    assert _eq(out["ep_low_adherence"].tolist(), [1, 0, 1, 0, 0, 1])          # mars <= 5


def test_threshold_is_configurable():
    out = build_endpoints(_frame(), mars_low=8, resistance_cgis=5)
    assert out["ep_low_adherence"].tolist()[:2] == [1.0, 1.0]                 # mars<=8 now flags 8 too
    # resistance now needs CGI-S>=5: row2 (cgi01=4) drops to 0
    assert out["ep_resistance"].iloc[2] == 0.0


def test_real_config_parses():
    cfg = load_m5_config(REPO / "configs" / "treatment_outcomes.yaml")
    assert cfg["meta"]["primary_horizon"] == "V2"
    assert set(cfg["endpoints"]) == {"response", "therapeutic_effect", "resistance",
                                     "side_effects", "low_adherence"}
    assert "cgi02" in cfg["signals"]
