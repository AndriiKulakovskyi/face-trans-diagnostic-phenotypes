"""Nested model comparison — does adding a block improve *held-out* prediction (LOO-ELPD)?

The incremental-validity verdict for M4. Per model we compute the leave-one-out ELPD and, against a
named reference (the bar, R3y), the **paired** ELPD difference with the SE of that difference — the
standard "is the gain bigger than its uncertainty?" test. `incremental_verdict` applies the project's
"CI excludes the null" idiom (gain − 2·SE > 0 → predictive).
"""
from __future__ import annotations

import numpy as np


def _pointwise(idata, var_name: str):
    """LOO ELPD object + pointwise elpd_i + max Pareto-k."""
    import arviz as az

    lo = az.loo(idata, var_name=var_name, pointwise=True)
    # arviz>=1.1 renamed ELPDData fields: elpd (was elpd_loo), elpd_i (was loo_i), p (was p_loo).
    elpd = float(getattr(lo, "elpd", getattr(lo, "elpd_loo", np.nan)))
    p_loo = float(getattr(lo, "p", getattr(lo, "p_loo", np.nan)))
    elpd_i = np.asarray(getattr(lo, "elpd_i", getattr(lo, "loo_i", None))).ravel()
    pk = getattr(lo, "pareto_k", None)
    pk = np.asarray(pk).ravel() if pk is not None else np.array([np.nan])
    return {"elpd": elpd, "se": float(lo.se), "p_loo": p_loo,
            "elpd_i": elpd_i, "max_pareto_k": float(np.nanmax(pk))}


def incremental_verdict(d_elpd: float, se_diff: float, *, k: float = 2.0) -> str:
    """`predictive` if d_elpd − k·SE > 0, `not-predictive` if d_elpd + k·SE < 0, else `ambiguous`."""
    if not np.isfinite(se_diff) or se_diff == 0:
        return "ambiguous"
    if d_elpd - k * se_diff > 0:
        return "predictive"
    if d_elpd + k * se_diff < 0:
        return "not-predictive"
    return "ambiguous"


def delta_elpd(fits: dict, reference: str, *, var_name: str = "y", k: float = 2.0):
    """Per model: LOO ELPD, and the paired ΔELPD vs `reference` with the SE of the difference. `fits` is
    `{model_name: fit_result}` (each a `glm.fit_glm` dict carrying `idata`). Returns a tidy DataFrame
    sorted by ΔELPD; the reference row has ΔELPD = 0."""
    import pandas as pd

    ref = _pointwise(fits[reference]["idata"], var_name)
    rows = []
    for name, f in fits.items():
        lo = _pointwise(f["idata"], var_name)
        d_i = lo["elpd_i"] - ref["elpd_i"]
        d = float(d_i.sum())
        se = float(np.sqrt(len(d_i)) * d_i.std(ddof=1)) if len(d_i) > 1 else float("nan")
        rows.append({
            "model": name, "elpd_loo": round(lo["elpd"], 2), "se_elpd": round(lo["se"], 2),
            "p_loo": round(lo["p_loo"], 1), "d_elpd_vs_ref": round(d, 2),
            "se_d_elpd": round(se, 2), "verdict": incremental_verdict(d, se, k=k),
            "max_pareto_k": round(lo["max_pareto_k"], 2),
            "rhat": round(f.get("rhat", float("nan")), 3), "divergences": int(f.get("divergences", 0)),
        })
    df = pd.DataFrame(rows).sort_values("d_elpd_vs_ref", ascending=False).reset_index(drop=True)
    return df


def coefficient_table(fit, names=None):
    """The fixed-effect posterior summary from a fit; optionally map `beta[i]` -> predictor names."""
    coef = fit["coef"].copy()
    if names is not None:
        mapping = {f"beta[{i}]": nm for i, nm in enumerate(names)}
        coef["term"] = coef["term"].map(lambda t: mapping.get(t, t))
    return coef
