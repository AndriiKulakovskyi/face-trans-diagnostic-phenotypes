"""Golden tests for the OOP prognosis engine (synthetic — no copula artifacts / heavy fits needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from face.prognosis.compare import delta_elpd
from face.prognosis.engine import (
    IncrementalValidator,
    PrognosisConfig,
    _config_sig,
    arch_cols,
    archB_cols,
    drop_one,
    encoding_block,
    expand_encodings,
    tess_family,
)
from face.prognosis.glm import fit_glm


def test_config_sig_and_smoke_factory():
    base = PrognosisConfig()
    assert _config_sig(base) == _config_sig(PrognosisConfig())             # stable
    sm = base.with_smoke_defaults()
    assert sm.smoke and not base.smoke
    assert sm.chains == 2 and sm.draws < base.draws
    assert _config_sig(sm) != _config_sig(base)                           # smoke changes the key


def test_dynamic_encoding_discovery():
    """The one generalization vs the native engine: discover A and the K-family from the hand-off columns."""
    df = pd.DataFrame(columns=[
        "arch_w0", "arch_w1", "arch_w2", "arch_w3", "arch_w0_sd", "arch_dominant",
        "archB_w0", "archB_w1", "archB_w2", "archB_w3",
        "tess_r0", "tess_r1",
        "tessfam_k2_r0", "tessfam_k2_r1",
        "tessfam_k3_r0", "tessfam_k3_r1", "tessfam_k3_r2",
        "tessfam_k4_r0", "tessfam_k4_r1", "tessfam_k4_r2", "tessfam_k4_r3"])
    assert arch_cols(df) == ["arch_w0", "arch_w1", "arch_w2", "arch_w3"]   # A=4, no _sd, ordered
    assert archB_cols(df) == ["archB_w0", "archB_w1", "archB_w2", "archB_w3"]
    fam = tess_family(df)
    assert set(fam) == {2, 3, 4}
    assert fam[3] == ["tessfam_k3_r0", "tessfam_k3_r1", "tessfam_k3_r2"]   # ordered within K
    assert drop_one(["a", "b", "c"]) == ["a", "b"]                         # simplex drop-one reference


def test_encoding_block_and_expand():
    """The shared encoding builder (used by both incremental + robustness) yields drop-one fixed blocks and
    expands the K-family; the same helper guarantees robustness stresses the identical block scored."""
    n = 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.random(n) for c in
                       ["arch_w0", "arch_w1", "arch_w2", "arch_w3",
                        "tessfam_k3_r0", "tessfam_k3_r1", "tessfam_k3_r2"]})
    assert expand_encodings(("+archetypesA", "+tessfamily"), df) == ["+archetypesA", "+tess_k3"]
    kw, extra = encoding_block(df, "+archetypesA", df, profiles_path="/nonexistent.csv")
    assert kw == {} and extra.shape == (n, 3)                          # A=4 -> drop-one -> 3 columns
    kw3, extra3 = encoding_block(df, "+tess_k3", df, profiles_path="/nonexistent.csv")
    assert extra3.shape == (n, 2)                                      # K=3 -> drop-one -> 2 columns
    assert encoding_block(df, "+tess_k9", df, profiles_path="/nonexistent.csv") is None   # absent K


def _cmp(rows):
    return pd.DataFrame(rows, columns=["outcome", "model", "d_elpd_vs_ref", "se_d_elpd", "verdict"])


def test_operative_k_picks_predictive_best_tessellation():
    comp = _cmp([
        ("egf", "+durable", 3.0, 2.0, "ambiguous"),
        ("egf", "+tess_k2", 1.0, 2.0, "ambiguous"),
        ("egf", "+tess_k3", 9.0, 3.0, "predictive"),
        ("egf", "+tess_k4", 6.0, 3.0, "predictive"),
    ])
    op = IncrementalValidator.operative_k(comp)
    assert op["operative_K"] == 3
    assert "operative K = 3" in op["verdict"]
    assert op["family_K"] == [2, 3, 4]


def test_operative_k_defers_to_continuum_when_continuous_wins():
    comp = _cmp([
        ("egf", "+durable", 12.0, 3.0, "predictive"),
        ("egf", "+archetypesA", 8.0, 3.0, "predictive"),
        ("egf", "+tess_k2", 1.0, 2.0, "ambiguous"),
        ("egf", "+tess_k3", 2.0, 2.0, "ambiguous"),
    ])
    op = IncrementalValidator.operative_k(comp)
    assert op["operative_K"] is None
    assert "no hard K" in op["verdict"]


def test_glm_wrap_and_delta_elpd_smoke():
    """The kernels the engine wraps run and compare: a tiny gaussian GLM + paired ΔELPD with a verdict."""
    rng = np.random.default_rng(0)
    N = 120
    x = rng.normal(size=(N, 2))
    y = 0.7 * x[:, 0] + rng.normal(scale=0.5, size=N)
    y = (y - y.mean()) / y.std()
    f0 = fit_glm(y, x[:, :1], family="gaussian", draws=60, tune=60, chains=1, seed=0)
    f1 = fit_glm(y, x, family="gaussian", draws=60, tune=60, chains=1, seed=0)
    d = delta_elpd({"R0": f0, "R1": f1}, reference="R0")
    assert {"model", "d_elpd_vs_ref", "verdict"} <= set(d.columns)
    assert set(d["model"]) == {"R0", "R1"}
    assert {"idata", "coef", "rhat", "divergences"} <= set(f1)         # the wrapped fit dict is intact
    assert isinstance(f1["divergences"], int)
