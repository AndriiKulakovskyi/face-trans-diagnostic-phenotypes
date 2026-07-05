"""Recovery-gap diagnostic — *what does raw carry that the 9-dim map drops?*

P1 found raw beats the latent map by ~0.04 AUC on recovery. This asks why: fit ``REF + RAW + LAT-A`` on the
recovery endpoint, compute TreeSHAP (built into XGBoost), and read off (a) how the predictive mass splits
across the three blocks, and (b) the top raw indicators — annotated with their M1 ``home_factor`` so we can
tell whether they are *within-factor* (the map compressed them away — a fixable lossiness) or *off-map*
(no home factor / cross-loaders — a genuine limit of the 9-dim summary).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import SEED, data, models


def _named_design(E: pd.DataFrame, rawE: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    arm = E["arm"] if "arm" in E.columns else pd.Series("na", index=E.index)
    ref = pd.get_dummies(arm.astype("category"), dummy_na=True).add_prefix("ref:dx=")
    ref["ref:G_mean"] = E["G_mean"].to_numpy()
    ref["ref:egf_V0"] = E["egf__V0"].to_numpy()
    lat = E[data.latent_blocks()["LAT-A"]]
    parts = [ref.reset_index(drop=True), rawE.reset_index(drop=True), lat.reset_index(drop=True)]
    X = pd.concat(parts, axis=1)
    block = np.array(["ref"] * ref.shape[1] + ["raw"] * rawE.shape[1] + ["latent"] * lat.shape[1])
    return X, block


def recovery_gap_shap(*, scope: str = "pooled", horizon: str = "V2", target: str = "egf_recovery",
                      seed: int = SEED, topk: int = 25) -> dict:
    cohorts = None if scope == "pooled" else ("bp", "dr")
    frame = data.assemble(cohorts=cohorts)
    el = data.eligible(frame, target, horizon)
    E = frame[el]
    rawE = data.load_raw().reindex(E.index)
    y = E[f"ep_{target}__{horizon}"].to_numpy("int64")
    X, block = _named_design(E, rawE)

    import xgboost as xgb
    clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **models.xgb_params(seed))
    clf.fit(X.to_numpy("float32"), y)
    dm = xgb.DMatrix(X.to_numpy("float32"))
    shap = clf.get_booster().predict(dm, pred_contribs=True)[:, :-1]      # drop bias column
    imp = np.abs(shap).mean(axis=0)

    df = pd.DataFrame({"feature": X.columns, "block": block, "mean_abs_shap": imp})
    meta = pd.read_parquet(data.PROC / "indicator_metadata.parquet").set_index("item")
    df["home_factor"] = df["feature"].map(meta["home_factor"]).where(df["block"] == "raw")
    df["family"] = df["feature"].map(meta["likelihood_family"]).where(df["block"] == "raw")
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    mass = df.groupby("block")["mean_abs_shap"].sum()
    mass = (mass / mass.sum()).round(3)
    raw_top = df[df.block == "raw"].head(topk).copy()
    # within-factor (belongs to one of the 9 modelled axes) vs off-map (no home factor / cross-loader window)
    raw_top["on_map"] = raw_top["home_factor"].notna()
    return {"table": df, "mass": mass, "raw_top": raw_top, "n": int(len(y)), "events": int(y.sum())}
