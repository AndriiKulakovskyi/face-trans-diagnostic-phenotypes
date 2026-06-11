"""M4.4 — head-to-head vs DSM-5 + transdiagnostic homogeneity (the G5 test M3 deferred).

Does the transdiagnostic map beat the DSM-5 diagnosis at *predicting the future*, and does its edge
hold *within* each cohort (not just pooled)? The **dominance** test fits four nested models on a
shared foundation (nuisance + baseline outcome + severity):

  D  foundation        A  D + DSM-5 arm        C  D + map        B  D + DSM-5 + map

and reads the asymmetry: the map **dominates** if it adds a lot beyond DSM-5 (B−A) while DSM-5 adds
~nothing beyond the map (B−C). **Transdiagnostic homogeneity** is a cohort×map interaction ELPD test —
a null interaction means the map predicts the same way in every cohort (not a diagnosis proxy).

"Better than DSM-5" is defined on outcome ELPD, never on agreement with DSM-5 (the house rule).
"""
from __future__ import annotations

import numpy as np

from face.prognosis.compare import delta_elpd

# the four dominance contrasts, as (label, model, reference)
_CONTRASTS = (
    ("DSM-5 beyond foundation (A−D)", "A", "D"),
    ("map beyond foundation (C−D)", "C", "D"),
    ("map beyond DSM-5 (B−A)", "B", "A"),
    ("DSM-5 beyond map (B−C)", "B", "C"),
)


def head_to_head(fits: dict, *, var_name: str = "y"):
    """The four dominance contrasts from `fits = {D, A, C, B}` (each a `glm.fit_glm` result). Returns a
    tidy DataFrame: contrast, ΔELPD, SE, verdict."""
    import pandas as pd

    rows = []
    for label, model, ref in _CONTRASTS:
        d = delta_elpd({ref: fits[ref], model: fits[model]}, reference=ref, var_name=var_name)
        r = d[d.model == model].iloc[0]
        rows.append({"contrast": label, "d_elpd": float(r.d_elpd_vs_ref), "se": float(r.se_d_elpd),
                     "verdict": r.verdict})
    return pd.DataFrame(rows)


def dominance_verdict(h2h) -> str:
    """Read the asymmetry: `map-dominates` if map-beyond-DSM5 is predictive and DSM5-beyond-map is not;
    `co-informative` if both add; `dsm5-dominates` if the reverse; else `neither`."""
    g = h2h.set_index("contrast")["verdict"]
    mbd = g.get("map beyond DSM-5 (B−A)")
    dbm = g.get("DSM-5 beyond map (B−C)")
    if mbd == "predictive" and dbm != "predictive":
        return "map-dominates"
    if dbm == "predictive" and mbd != "predictive":
        return "dsm5-dominates"
    if mbd == "predictive" and dbm == "predictive":
        return "co-informative"
    return "neither"


def interaction_block(map_X, cohort_codes, n_cohorts: int):
    """cohort×map interaction columns (reference cohort 0 dropped). `map_X` [N, P]; `cohort_codes` [N]
    in 0..n_cohorts-1. Returns [N, P·(n_cohorts−1)] — the homogeneity test's added block."""
    map_X = np.asarray(map_X, dtype="float64")
    n = map_X.shape[0]
    cohort_codes = np.asarray(cohort_codes)
    blocks = []
    for c in range(1, n_cohorts):
        ind = (cohort_codes == c).astype("float64")[:, None]
        blocks.append(map_X * ind)
    return np.column_stack(blocks) if blocks else np.empty((n, 0))
