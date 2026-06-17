"""Golden test (issue P7-03): the measurement engine recovers planted structure on synthetic FACE data.

Generates a synthetic FACE-like dataset with KNOWN loadings (biology near-⟂G by construction), points the
engine at it via the in-process ``PROC`` override, fits the marginalized S1 model, and checks that (a) the
home loadings are recovered and (b) the planted biology-⟂-G ordering re-emerges (metabolic/inflammatory
carry less G than cognition/sleep). This is the end-to-end "external user reproduces on synthetic data"
check the confidential cohort data otherwise blocks. Small sampler settings keep it CI-fast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymc")


def test_engine_recovers_synthetic_loadings(tmp_path):
    import pymc as pm

    import face.models.bayesian.continuous_core as cc
    from synthetic.generate_face_like import generate

    outdir, truth = generate(n=900, seed=0, out=tmp_path)
    cc.PROC = Path(outdir)                                 # point the engine at synthetic data
    try:
        prep = cc.prepare(cc.S1_FACTORS)                  # S1: bifactor, Φ = I
        assert prep.items == truth["items"]
        with cc.build_marginalized(prep):
            idata = pm.sample(draws=150, tune=250, chains=2, nuts_sampler="numpyro",
                              random_seed=0, progressbar=False, idata_kwargs={"log_likelihood": False})
    finally:
        cc.PROC = Path(cc.REPO) / "data" / "processed"     # restore for other tests

    Lam_hat = np.asarray(idata.posterior["Lam"].mean(("chain", "draw")).values)   # [J, F]
    Lam_true = np.asarray(truth["Lam_true"])
    fcol = {f: i for i, f in enumerate(truth["factor_cols"])}

    home_hat = np.array([Lam_hat[j, fcol[truth["home"][it]]] for j, it in enumerate(prep.items)])
    home_true = np.array([Lam_true[j, fcol[truth["home"][it]]] for j, it in enumerate(truth["items"])])
    r = float(np.corrcoef(home_hat, home_true)[0, 1])
    assert r > 0.8, f"home-loading recovery corr = {r:.3f}"

    gcol = fcol[cc.G_KEY]
    bio = [j for j, it in enumerate(prep.items) if truth["home"][it] in ("metabolic", "inflammatory")]
    cog = [j for j, it in enumerate(prep.items) if truth["home"][it] in ("cognition", "sleep")]
    assert np.mean(np.abs(Lam_hat[bio, gcol])) < np.mean(np.abs(Lam_hat[cog, gcol])), \
        "planted biology-⟂-G ordering not recovered"
