"""EIV-GLM uncertainty arm — the faithful H3 test.

P1 fed per-patient sd to XGBoost as a *feature*. The honest test is the **errors-in-variables Bayesian GLM**
(`face.prognosis.glm.fit_glm`): the latent true coordinate is inferred with the known M1 per-patient sd plugged
in, so wide-posterior coordinates self-down-weight and the slope is attenuation-corrected. We compare, by
LOO-ΔELPD, three nested logistic models on the binary endpoint:

    ref       DSM-5 + latent-G severity + baseline GAF
    lat_mu    ref + the 9 coordinate means (fixed effects)
    lat_eiv   ref + the 9 coordinates as EIV (means + per-patient sd)

H3 is supported iff ``lat_eiv`` beats ``lat_mu``. (P2a predicts a near-null — the recovery gap is item-level
compression, not measurement noise — so this doubles as a falsification of the noise explanation.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import CANON, SEED, data


def _ref_std(E: pd.DataFrame) -> np.ndarray:
    arm = E["arm"] if "arm" in E.columns else pd.Series("na", index=E.index)
    dx = pd.get_dummies(arm.astype("category"), dummy_na=True).to_numpy("float64")
    cont = E[["G_mean", "egf__V0"]].to_numpy("float64")
    cont = np.nan_to_num((cont - np.nanmean(cont, 0)) / (np.nanstd(cont, 0) + 1e-9))
    return np.hstack([dx, cont])


def _std_coords(E: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Standardise the 9 coordinate means to unit variance; scale the sds by the same factor (so the EIV
    measurement scale matches the Normal(0,1) prior idiom)."""
    means = E[[f"{a}__mean" for a in CANON]].to_numpy("float64")
    sds = E[[f"{a}__sd" for a in CANON]].to_numpy("float64")
    mu, sd = means.mean(0), means.std(0) + 1e-9
    return (means - mu) / sd, sds / sd


def eiv_uncertainty(*, target: str = "egf_recovery", horizon: str = "V2", scope: str = "pooled",
                    seed: int = SEED, draws: int = 600, tune: int = 800, chains: int = 4) -> dict:
    from face.prognosis.compare import delta_elpd
    from face.prognosis.glm import fit_glm

    cohorts = None if scope == "pooled" else ("bp", "dr")
    frame = data.assemble(cohorts=cohorts)
    E = frame[data.eligible(frame, target, horizon)]
    y = E[f"ep_{target}__{horizon}"].to_numpy("int64")
    ref = _ref_std(E)
    cmu, csd = _std_coords(E)

    kw = dict(family="bernoulli", draws=draws, tune=tune, chains=chains, seed=seed, target_accept=0.95)
    fits = {
        "ref": fit_glm(y, ref, **kw),
        "lat_mu": fit_glm(y, np.hstack([ref, cmu]), **kw),
        "lat_eiv": fit_glm(y, ref, eiv_obs=cmu, eiv_sd=csd, **kw),
    }
    vs_ref = delta_elpd(fits, reference="ref")
    eiv_vs_mu = delta_elpd({"lat_mu": fits["lat_mu"], "lat_eiv": fits["lat_eiv"]}, reference="lat_mu")
    diag = {k: {"rhat": v.get("rhat"), "div": v.get("divergences")} for k, v in fits.items()}
    return {"target": target, "horizon": horizon, "scope": scope, "n": int(len(y)), "events": int(y.sum()),
            "vs_ref": vs_ref, "eiv_vs_mu": eiv_vs_mu, "diag": diag}
