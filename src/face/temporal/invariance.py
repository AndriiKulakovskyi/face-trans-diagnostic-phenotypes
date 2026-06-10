"""G1 — longitudinal measurement invariance (docs/TEMPORAL_MODEL.md §4).

The temporal analogue of the cross-cohort §6 (`scripts/06_invariance.py`): refit the simple-structure
backbone PER VISIT (cohort→visit), **z-scored in-sample** (Tucker congruence is scale-invariant, so the
frozen V0 spec is deliberately NOT used here — this tests loading *shape*, not level), and compare the
primary loadings V1/V2 vs V0. Metric invariance = Tucker φ of the primary loadings per factor; configural =
the same backbone certifies at each visit. The scored panel (stage 34) is licensed as patient-*change* only
on axes that clear metric invariance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from face.confirm import corr_no_g_prep
from face.models.bayesian.continuous_core import S1_FACTORS, prepare
from face.runner import quick_diag, sample_marginalized

PHI_GOOD, PHI_OK = 0.95, 0.85
MIN_OBS = 30                       # an item is "testable" at a visit if observed in ≥ this many patients


def tucker_phi(a, b) -> float:
    """Tucker's congruence coefficient between two loading vectors (scale-invariant)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum()) / den if den > 0 else float("nan")


def fit_visit_backbone(visit: str, *, factors: list[str] | None = None, n_total: int = 1800,
                       seed: int = 20260605, draws: int = 500, tune: int = 600, chains: int = 2,
                       target_accept: float = 0.9, label: str = "", step: str = "") -> tuple[dict, dict]:
    """Fit the per-visit simple-structure backbone (cohort-balanced subsample, in-sample z-scored).

    ``factors`` selects the backbone (default S1; S3A adds the continuous-anchored developmental_risk).
    Returns ``(rec, diag)`` where ``rec`` maps item → ``(home_factor, primary_loading, n_obs)`` (the
    invariance comparison object) and ``diag`` is the convergence record. Reads ``baseline_{visit}.parquet``.
    """
    base = prepare(factors or S1_FACTORS, correlated=True, windows=False, visit=visit,
                   balanced=True, n_subsample=n_total, seed=seed)
    prep = corr_no_g_prep(base)                       # simple structure (drop bifactor-G cross-loadings)
    idata = sample_marginalized(prep, draws=draws, tune=tune, chains=chains,
                                target_accept=target_accept, seed=seed, label=label, step=step)
    dg = quick_diag(idata)
    Lam = idata.posterior["Lam"].mean(("chain", "draw")).values        # [J, F]
    nobs = (~np.isnan(prep.M)).sum(0)
    home_col = {f: prep.factor_cols.index(f) for f in prep.factor_cols}
    rec = {it: (prep.home[j], float(Lam[j, home_col[prep.home[j]]]), int(nobs[j]))
           for j, it in enumerate(prep.items) if prep.home[j]}
    diag = dict(fit=f"{visit} s{seed}", rhat=round(float(dg["rhat"]), 3), ess=round(float(dg["ess"])),
                div=int(dg["div"]),
                converged=bool(dg["rhat"] <= 1.05 and dg["ess"] >= 100 and dg["div"] == 0))
    return rec, diag


def congruence_over_visits(fits: dict, factors, visits, seeds, reference: str = "V0",
                           converged: set | None = None) -> pd.DataFrame:
    """Tucker φ of the primary loadings at each follow-up visit vs the reference, per factor (averaged over
    seeds; items observed ≥MIN_OBS at both visits). ``fits`` is keyed ``(visit, seed)`` → rec. If
    ``converged`` (a set of ``(visit, seed)`` keys) is given, non-converged fits are excluded — a φ must
    rest on a fit that passed the convergence gate, never a degenerate one."""
    rows = []
    for v in visits:
        if v == reference:
            continue
        for f in factors:
            phis, nit = [], 0
            for s in seeds:
                if converged is not None and ((reference, s) not in converged or (v, s) not in converged):
                    continue
                ref, rec = fits.get((reference, s)), fits.get((v, s))
                if ref is None or rec is None:
                    continue
                common = [it for it in rec if it in ref and rec[it][0] == f and ref[it][0] == f
                          and rec[it][2] >= MIN_OBS and ref[it][2] >= MIN_OBS]
                if len(common) >= 2:
                    a = [ref[it][1] for it in common]
                    b = [rec[it][1] for it in common]
                    phis.append(tucker_phi(a, b)); nit = len(common)
            if phis:
                rows.append(dict(factor=f, visit=v, n_items=nit,
                                 phi_mean=round(float(np.mean(phis)), 3),
                                 phi_min=round(float(np.min(phis)), 3)))
    return pd.DataFrame(rows)


def axis_license(cong: pd.DataFrame) -> pd.DataFrame:
    """Per-factor temporal-invariance verdict from the worst follow-up φ: invariant (φ≥0.95) / partial
    (≥0.85) / non-invariant. This is the license stage 34 attaches to the panel."""
    rows = []
    for f, g in cong.groupby("factor"):
        mn = float(g["phi_mean"].min())
        verdict = "invariant" if mn >= PHI_GOOD else ("partial" if mn >= PHI_OK else "non-invariant")
        rows.append(dict(axis=f, min_phi=round(mn, 3), license=verdict))
    return pd.DataFrame(rows).sort_values("axis").reset_index(drop=True)
