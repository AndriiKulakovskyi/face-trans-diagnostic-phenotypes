"""M5.2 — treatment propensity + overlap (the identification gate before any moderation).

Confounding by indication is the dominant threat (TREATMENT_MODEL §4, §7): treatment is prescribed on
presentation, not randomized. Before estimating *any* effect we must show the exposed and the (active-
comparator) control groups **overlap** on the confounders — otherwise the effect, and a fortiori its
moderation by the map, is unidentifiable (a *channeled* treatment like clozapine, reserved for the
resistant, may simply have no comparable controls). This module:
  - defines the exposure contrast (active-comparator primary; on/off sensitivity),
  - fits `P(treat | confounders)` where confounders = severity (CGI-S + error-corrected G) + diagnosis
    (arm) + demographics + **the map coordinates** (a confounder of prescription AND the moderator —
    it enters here for exchangeability, and its treat×axis interaction is the M5.2 estimand),
  - reports overlap / common support + covariate balance (SMD) before vs after stabilized IPTW.
No imputation: a patient with no exposure record is dropped from the contrast, never coerced to control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from face.prognosis import CANON  # 9-dim order + the durable trio

# question -> {cohort, exposed flag, active-comparator flags, label}
QUESTIONS = {
    "lithium_bp": {"cohort": "bp", "exposed": "on_lithium",
                   "comparator": ["on_mood_stabilizer", "on_antipsychotic"],
                   "label": "lithium vs other maintenance (BP)"},
    "clozapine_sz": {"cohort": "sz", "exposed": "on_clozapine", "comparator": ["on_antipsychotic"],
                     "label": "clozapine vs other antipsychotic (SZ)"},
    "antipsychotic_bp": {"cohort": "bp", "exposed": "on_antipsychotic",
                         "comparator": ["on_mood_stabilizer", "on_antidepressant"],
                         "label": "antipsychotic vs other maintenance (BP)"},
}
CONF_COVARS = ("age", "sex", "cgi_s__V0", "overall_severity__mean")  # + arm dummies + the 9 map coords


def define_exposure(merged: pd.DataFrame, question: str, mode: str = "active_comparator"):
    """Return (sub-frame, treat 0/1 array) for `question` under `mode`
    ('active_comparator' = exposed vs other-active; 'on_off' = exposed vs everyone-not-exposed)."""
    spec = QUESTIONS[question]
    sub = merged[merged["cohort"] == spec["cohort"]].copy()
    exposed = sub[spec["exposed"]] == 1
    if mode == "active_comparator":
        comp_any = sub[spec["comparator"]].eq(1).any(axis=1)
        comparator = comp_any & (sub[spec["exposed"]] == 0)        # on another active drug, not the index one
        mask = exposed | comparator
    elif mode == "on_off":
        mask = sub[spec["exposed"]].notna()                         # off = recorded-not-exposed (no imputation)
    else:
        raise ValueError(mode)
    sub = sub[mask].copy()
    treat = (sub[spec["exposed"]] == 1).to_numpy(dtype=float)
    return sub, treat


def confounder_matrix(sub: pd.DataFrame):
    """Confounder design for the propensity model: demographics + severity + DSM-5 arm + the 9 map
    coordinates (the map as a prescription confounder). Returns (X standardized, column names, row mask)."""
    parts, names = [], []
    for c in CONF_COVARS:
        parts.append(pd.to_numeric(sub[c], errors="coerce").to_numpy()[:, None]); names.append(c)
    arm = pd.get_dummies(sub["arm"].astype("string"), prefix="arm", dummy_na=False).astype(float)
    parts.append(arm.to_numpy()); names += list(arm.columns)
    coords = sub[[f"{ax}__mean" for ax in CANON]].apply(pd.to_numeric, errors="coerce").to_numpy()
    parts.append(coords); names += [f"{ax}__mean" for ax in CANON]
    X = np.column_stack(parts)
    row_ok = np.isfinite(X).all(axis=1)
    Xs = StandardScaler().fit_transform(X[row_ok])
    return Xs, names, row_ok


def propensity_score(X, treat, *, seed: int = 20260611):
    """In-sample P(treat=1 | X) from an L2 logistic balancing model (PS = a balancing score)."""
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(X, treat)
    return lr.predict_proba(X)[:, 1]


def overlap(ps, treat):
    """Common-support diagnostics for the propensity score by treatment arm."""
    ps, treat = np.asarray(ps), np.asarray(treat)
    p1, p0 = ps[treat == 1], ps[treat == 0]
    lo, hi = max(p1.min(), p0.min()), min(p1.max(), p0.max())
    in_support = (ps >= lo) & (ps <= hi)
    return {"n_treated": int(treat.sum()), "n_control": int((treat == 0).sum()),
            "ps_treated_median": float(np.median(p1)), "ps_control_median": float(np.median(p0)),
            "common_lo": float(lo), "common_hi": float(hi),
            "frac_in_support": float(in_support.mean()), "n_out_of_support": int((~in_support).sum())}


def stabilized_iptw(ps, treat, *, clip=(0.01, 0.99), trim_to_support=True):
    """Stabilized IPTW; PS clipped off 0/1, weights returned with a common-support mask."""
    ps, treat = np.asarray(ps), np.asarray(treat)
    psc = np.clip(ps, *clip)
    pt = treat.mean()
    w = np.where(treat == 1, pt / psc, (1 - pt) / (1 - psc))
    keep = np.ones_like(treat, dtype=bool)
    if trim_to_support:
        p1, p0 = ps[treat == 1], ps[treat == 0]
        lo, hi = max(p1.min(), p0.min()), min(p1.max(), p0.max())
        keep = (ps >= lo) & (ps <= hi)
    return w, keep


def smd(X, treat, w=None):
    """Per-covariate standardized mean difference (|treated − control| / pooled sd); w = optional IPTW."""
    X, treat = np.asarray(X, float), np.asarray(treat)
    w = np.ones_like(treat, float) if w is None else np.asarray(w, float)
    out = []
    for j in range(X.shape[1]):
        xj = X[:, j]
        m1 = np.average(xj[treat == 1], weights=w[treat == 1]); m0 = np.average(xj[treat == 0], weights=w[treat == 0])
        v1, v0 = xj[treat == 1].var(), xj[treat == 0].var()
        sd = np.sqrt((v1 + v0) / 2) or 1.0
        out.append(abs(m1 - m0) / sd)
    return np.asarray(out)
