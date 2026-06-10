"""M4.2 — nested model comparison (verdict bands pure; one tiny ΔELPD integration)."""
from __future__ import annotations

import numpy as np

from face.prognosis.compare import delta_elpd, incremental_verdict
from face.prognosis.glm import fit_glm


def test_incremental_verdict_bands():
    assert incremental_verdict(10.0, 2.0) == "predictive"        # 10 - 4 > 0
    assert incremental_verdict(-10.0, 2.0) == "not-predictive"   # -10 + 4 < 0
    assert incremental_verdict(1.0, 3.0) == "ambiguous"          # straddles 0
    assert incremental_verdict(5.0, 0.0) == "ambiguous"          # degenerate SE


def test_delta_elpd_prefers_the_model_with_the_real_predictor():
    rng = np.random.default_rng(7)
    n = 200
    xtrue = rng.normal(size=n)
    xnoise = rng.normal(size=n)
    y = 1.0 * xtrue + rng.normal(0, 0.5, n)
    y = (y - y.mean()) / y.std()
    fit_kw = dict(family="gaussian", draws=250, tune=250, chains=2, seed=0)
    fits = {
        "noise_only": fit_glm(y, xnoise[:, None], **fit_kw),     # reference (no real signal)
        "with_true": fit_glm(y, np.column_stack([xnoise, xtrue]), **fit_kw),
    }
    df = delta_elpd(fits, reference="noise_only").set_index("model")
    assert df.loc["noise_only", "d_elpd_vs_ref"] == 0.0          # reference vs itself
    assert df.loc["with_true", "d_elpd_vs_ref"] > 0              # the real predictor improves ELPD
    assert df.loc["with_true", "verdict"] == "predictive"
