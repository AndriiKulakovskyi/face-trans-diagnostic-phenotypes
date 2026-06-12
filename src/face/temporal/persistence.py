"""G4 — stratum persistence + the spine-vs-corner geometric test (docs/TEMPORAL_MODEL.md §6).

The geometric route to the §1.4 prediction: does the patient slide along the **severity spine** while the
**biology corner stays**? Operationalized per patient, uncertainty-aware (a move counts only if it clears
measurement error, the G0 reliable-change rule):

  * `reliable_change_rate` — per axis, the fraction of patients whose V0→Vk coordinate change is reliable
    (|Δ|/SE ≥ 1.96, SE² = σ²_s+σ²_t). The geometric analogue of G3's state — and the G3⟷G4 synthesis lever.
  * `spine_corner` — decompose Δx into the spine (severity, 1-dof χ² reliable-change) vs the corner
    subspace (multi-dof χ²): does the spine move while the corner holds?  Reports the full 8-specific corner
    and the cleaner **biology corner** (metabolic/inflammatory/cognition, the licensed trait axes).
  * `membership_persistence` — soft transition matrix + dominant-archetype agreement + weight cosine on the
    **Arm-B (G-residualized)** archetypes: does corner *identity* persist independent of severity?

Memberships are soft (a continuum), so persistence is read in coordinates/weights, not hard labels — central
patients churn argmax by geometry, not instability. Restricted to patients present at both endpoints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPECIFICS = ["cognition", "metabolic", "inflammatory", "sleep", "mania_activation",
             "suicidality", "developmental_risk", "substance"]
BIOLOGY_CORNER = ["metabolic", "inflammatory", "cognition"]      # the licensed trait axes (cleanest §1.4 corner)
SPINE = "overall_severity"
Z = 1.96                                                          # 94%/95% reliable-change threshold


def _interval(panel: pd.DataFrame, s: str = "V0", t: str = "V2"):
    """The (visit-s, visit-t) rows for patients present at BOTH, aligned on patient_uid."""
    a = panel[panel.visit == s].set_index("patient_uid")
    b = panel[panel.visit == t].set_index("patient_uid")
    both = a.index.intersection(b.index)
    return a.loc[both], b.loc[both]


def _delta_se(a, b, ax):
    d = b[f"{ax}__mean"].to_numpy() - a[f"{ax}__mean"].to_numpy()
    se = np.sqrt(a[f"{ax}__sd"].to_numpy() ** 2 + b[f"{ax}__sd"].to_numpy() ** 2)
    return d, se


def reliable_change_rate(panel, axes, *, s="V0", t="V2") -> pd.DataFrame:
    """Per-axis fraction of patients with a reliable V0→Vk change (the geometric state signal)."""
    a, b = _interval(panel, s, t)
    rows = []
    for ax in axes:
        d, se = _delta_se(a, b, ax)
        rci = d / se
        rows.append(dict(axis=ax, n=len(d), frac_reliable=round(float(np.mean(np.abs(rci) >= Z)), 3),
                         frac_decrease=round(float(np.mean(rci <= -Z)), 3),
                         frac_increase=round(float(np.mean(rci >= Z)), 3),
                         mean_abs_delta=round(float(np.mean(np.abs(d))), 3)))
    return pd.DataFrame(rows)


def spine_corner(panel, *, s="V0", t="V2", alpha=0.94) -> dict:
    """Spine-vs-corner reliable-change decomposition. Spine = severity (χ²₁); corner = the specifics
    (χ² with that many dof). Returns rates for the full 8-corner and the 3-axis biology corner + the 2×2."""
    from scipy.stats import chi2
    a, b = _interval(panel, s, t)

    def d2(axset):
        out = np.zeros(len(a))
        for ax in axset:
            d, se = _delta_se(a, b, ax)
            out = out + (d / se) ** 2
        return out
    spine = d2([SPINE]) > chi2.ppf(alpha, 1)
    corner = d2(SPECIFICS) > chi2.ppf(alpha, len(SPECIFICS))
    bio = d2(BIOLOGY_CORNER) > chi2.ppf(alpha, len(BIOLOGY_CORNER))
    return dict(n=int(len(a)), spine_rate=float(spine.mean()), corner_rate=float(corner.mean()),
                bio_corner_rate=float(bio.mean()),
                spine_not_bio=float((spine & ~bio).mean()),       # the §1.4 cell: spine moves, biology holds
                bio_not_spine=float((bio & ~spine).mean()),
                both=float((spine & bio).mean()), neither=float((~spine & ~bio).mean()))


def membership_persistence(panel, *, arm="archB", A=8, s="V0", t="V2") -> dict:
    """Soft transition matrix (row-normalized), dominant-archetype agreement, Cohen's κ, and the
    weight-vector cosine distribution for the chosen archetype arm (Arm B = G-residualized = primary)."""
    from sklearn.metrics import cohen_kappa_score
    a, b = _interval(panel, s, t)
    wcols = [f"{arm}_w{k}" for k in range(A)]
    Ra, Rb = a[wcols].to_numpy(), b[wcols].to_numpy()
    T = Ra.T @ Rb                                                  # [A, A] soft transitions (Σ_i r_k(s) r_l(t))
    T = T / np.clip(T.sum(1, keepdims=True), 1e-12, None)
    da, db = a[f"{arm}_dominant"].to_numpy(), b[f"{arm}_dominant"].to_numpy()
    cos = (Ra * Rb).sum(1) / np.clip(np.linalg.norm(Ra, axis=1) * np.linalg.norm(Rb, axis=1), 1e-12, None)
    return dict(n=int(len(a)), transition=T, dominant_agree=float((da == db).mean()),
                kappa=float(cohen_kappa_score(da, db)), cos_median=float(np.median(cos)),
                cos_q10=float(np.quantile(cos, 0.10)), cos=cos)


def trajectory_types(panel, axis=SPINE) -> dict:
    """Per-patient trajectory class on `axis` for 3-visit patients (coarse with 3 points): stable (no
    reliable leg), drifting (monotone reliable), oscillating (reliable legs of opposite sign)."""
    p = panel[panel.n_visits >= 3]
    a0 = p[p.visit == "V0"].set_index("patient_uid")
    a1 = p[p.visit == "V1"].set_index("patient_uid")
    a2 = p[p.visit == "V2"].set_index("patient_uid")
    idx = a0.index.intersection(a1.index).intersection(a2.index)
    a0, a1, a2 = a0.loc[idx], a1.loc[idx], a2.loc[idx]

    def leg(x, y):
        d = y[f"{axis}__mean"].to_numpy() - x[f"{axis}__mean"].to_numpy()
        se = np.sqrt(x[f"{axis}__sd"].to_numpy() ** 2 + y[f"{axis}__sd"].to_numpy() ** 2)
        return d / se
    r1, r2 = leg(a0, a1), leg(a1, a2)
    s1, s2 = (np.abs(r1) >= Z), (np.abs(r2) >= Z)
    stable = ~s1 & ~s2
    osc = s1 & s2 & (np.sign(r1) != np.sign(r2))
    drift = (s1 | s2) & ~osc
    return dict(n=int(len(idx)), stable=float(stable.mean()), drifting=float(drift.mean()),
                oscillating=float(osc.mean()))
