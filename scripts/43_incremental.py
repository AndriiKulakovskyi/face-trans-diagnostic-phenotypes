#!/usr/bin/env python3
"""43 — M4.3 the headline: does the map add value beyond diagnosis + severity?

On top of the R3y bar (diagnosis + severity + baseline outcome) we add, per primary outcome, each of
the three map representations and the ceiling, and ask whether held-out prediction improves (ΔELPD vs
R3y) and whether the durable-biology coefficients clear their uncertainty band:

  +durable      R3y + EIV(cognition, metabolic, inflammatory)   ← representation 1 (continuous, headline)
  +archetypes   R3y + the 8 archetype memberships               ← representation 2 (deployable)
  +tessellation R3y + the 4-region tessellation                 ← representation 3 (coarse)
  +specifics8   R3y + EIV(all 8 ⊥G specific axes)               ← ceiling

The durable coordinates enter as errors-in-variables (the M1 per-patient SD plugged) so their effects
are attenuation-corrected and poorly-measured patients self-down-weight. Q1 = ΔELPD vs R3y; the
in-sample read is the durable β 94% interval excluding 0. Q2 (egf) re-fits the durable model under the
**error-aware G** severity as well as manifest CGI-S — the effect must survive both. Methods:
docs/PROGNOSIS_MODEL.md (M4.3).

    python3 scripts/43_incremental.py [--smoke]

Writes results/face/m4/{incremental_comparison.csv, coef_durable_{outcome}.csv} and
docs/figures/{43_added_value.png, 43_calibration.png}, reports/43_incremental.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpyro
import pandas as pd

numpyro.set_host_device_count(4)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.prognosis import DURABLE  # noqa: E402
from face.prognosis.compare import delta_elpd  # noqa: E402
from face.prognosis.frame import load_outcome_config  # noqa: E402
from face.prognosis.glm import fit_glm  # noqa: E402
from face.prognosis.reference import (ARCH_COLS, SPECIFICS, TESS_COLS, coord_eiv_block,  # noqa: E402
                                      design_for_rung, fixed_block, modeling_frame,
                                      outcome_vector, severity_column, site_index)

CONFIG = REPO / "configs" / "m4_outcomes.yaml"
M4 = REPO / "results" / "face" / "m4"
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
CGI_BASELINE = "cgi_s__V0"
MODELS = ("R3y", "+durable", "+archetypes", "+tessellation", "+specifics8")


def _fit_all(sub, spec, *, sev_col, horizon, fit_kw):
    """Fit the R3y bar + the four map representations on the same sample. Returns (fits, X_r3y, eiv)."""
    y, fam, n_cat = outcome_vector(sub, spec, horizon=horizon)
    grp, ng = site_index(sub)
    Xr, _ = design_for_rung(sub, spec, "R3y", severity_col=sev_col, horizon=horizon)
    dob, dsd, _ = coord_eiv_block(sub, DURABLE)
    sob, ssd, _ = coord_eiv_block(sub, SPECIFICS)
    arch, _ = fixed_block(sub, ARCH_COLS)
    tess, _ = fixed_block(sub, TESS_COLS)
    base = dict(family=fam, group=grp, n_groups=ng, n_cat=n_cat, **fit_kw)
    fits = {}
    for name, kw, X in [
        ("R3y", {}, Xr),
        ("+durable", dict(eiv_obs=dob, eiv_sd=dsd), Xr),
        ("+archetypes", {}, np.column_stack([Xr, arch])),
        ("+tessellation", {}, np.column_stack([Xr, tess])),
        ("+specifics8", dict(eiv_obs=sob, eiv_sd=ssd), Xr),
    ]:
        print(f"    [{spec.name}/{name}] N={len(sub)} fitting ...", flush=True)
        fits[name] = fit_glm(y, X, **base, **kw)
    return fits, Xr, (dob, dsd), (y, fam)


def _durable_coef(fit, *, severity_label, outcome):
    c = fit["coef"]
    rows = c[c.term.str.startswith("beta_eiv")].reset_index(drop=True)
    rows = rows.assign(outcome=outcome, axis=list(DURABLE), severity=severity_label)
    return rows[["outcome", "axis", "severity", "mean", "sd", "eti_lo", "eti_hi", "p_direction"]]


def _predict_mean(fit, X, eiv_obs=None):
    """Posterior-mean linear predictor (site RI ~0 omitted) — for the calibration scatter."""
    post = fit["idata"].posterior
    eta = float(post["alpha"].mean().values) + np.zeros(X.shape[0])
    if "beta" in post and X.shape[1]:
        eta = eta + X @ np.asarray(post["beta"].mean(dim=("chain", "draw")).values)
    if eiv_obs is not None and "beta_eiv" in post:
        eta = eta + eiv_obs @ np.asarray(post["beta_eiv"].mean(dim=("chain", "draw")).values)
    return eta


def main(smoke: bool = False) -> None:
    M4.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    cfg = load_outcome_config(CONFIG)
    horizon = cfg.meta.get("primary_horizon", "V2")
    seed = int(cfg.meta.get("seed", 20260610))
    fit_kw = (dict(draws=150, tune=150, chains=2, seed=seed) if smoke
              else dict(draws=800, tune=800, chains=4, seed=seed))

    frame = pd.read_parquet(M4 / "analysis_frame.parquet")
    comparisons, durable_coefs, calib = [], [], {}
    for spec in cfg.primary():
        sev = severity_column(spec, cgi_baseline_col=CGI_BASELINE)
        sub = modeling_frame(frame, spec, horizon=horizon, severity_col=sev)
        print(f"  [{spec.name}] incremental models (severity={sev}) ...", flush=True)
        fits, Xr, (dob, dsd), (y, fam) = _fit_all(sub, spec, sev_col=sev, horizon=horizon, fit_kw=fit_kw)
        cmp = delta_elpd(fits, reference="R3y").assign(outcome=spec.name, n=len(sub))
        comparisons.append(cmp)
        durable_coefs.append(_durable_coef(fits["+durable"], severity_label=sev, outcome=spec.name))
        calib[spec.name] = (y, _predict_mean(fits["+durable"], Xr, dob), fam)

        # Q2 — the durable effect must also survive the *alternate* (error-aware G) severity, unless
        # that severity is already the baseline outcome (then the two operationalizations coincide).
        alt = "overall_severity__mean" if sev != "overall_severity__mean" else CGI_BASELINE
        if alt != f"{spec.name}__V0" and alt != sev:
            subq = modeling_frame(frame, spec, horizon=horizon, severity_col=alt)
            yq, famq, ncq = outcome_vector(subq, spec, horizon=horizon)
            gq, ngq = site_index(subq)
            Xq, _ = design_for_rung(subq, spec, "R3y", severity_col=alt, horizon=horizon)
            ob, sd, _ = coord_eiv_block(subq, DURABLE)
            print(f"    [{spec.name}/+durable@alt-severity={alt}] fitting (Q2) ...", flush=True)
            fq = fit_glm(yq, Xq, family=famq, group=gq, n_groups=ngq, n_cat=ncq,
                         eiv_obs=ob, eiv_sd=sd, **fit_kw)
            durable_coefs.append(_durable_coef(fq, severity_label=alt, outcome=spec.name))

    comp = pd.concat(comparisons, ignore_index=True)
    coef = pd.concat(durable_coefs, ignore_index=True)
    comp.to_csv(M4 / "incremental_comparison.csv", index=False)
    coef.to_csv(M4 / "coef_durable.csv", index=False)
    _fig_added_value(comp, coef, cfg)
    _fig_calibration(calib, cfg)
    _report(cfg, comp, coef, horizon, smoke)


def _fig_added_value(comp, coef, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outcomes = [o.name for o in cfg.primary()]
    reps = [m for m in MODELS if m != "R3y"]
    fig, ax = plt.subplots(2, len(outcomes), figsize=(6.2 * len(outcomes), 8.4), squeeze=False)
    for j, name in enumerate(outcomes):
        sub = comp[comp.outcome == name].set_index("model").reindex(MODELS)
        a = ax[0][j]
        vals = sub.loc[reps, "d_elpd_vs_ref"].values
        ses = sub.loc[reps, "se_d_elpd"].values
        colors = ["#2c7fb8" if (v - 2 * s) > 0 else ("#888" if (v + 2 * s) > 0 else "#d73027")
                  for v, s in zip(vals, ses)]
        a.bar(range(len(reps)), vals, color=colors)
        a.errorbar(range(len(reps)), vals, yerr=2 * ses, fmt="none", ecolor="#222", capsize=3)
        a.axhline(0, color="k", lw=0.8)
        a.set_xticks(range(len(reps)))
        a.set_xticklabels(reps, rotation=20, ha="right", fontsize=9)
        a.set_title(f"{name}: added value vs R3y (N={int(sub['n'].iloc[0])})")
        a.set_ylabel("ΔELPD vs bar (±2·SE, ↑ better)")
        a.grid(axis="y", alpha=0.3)
        # forest of durable betas under the configured severity
        b = ax[1][j]
        cfg_sev = ("cgi_s__V0" if name == "egf" else "overall_severity__mean")
        cc = coef[(coef.outcome == name) & (coef.severity == cfg_sev)]
        cc = cc.set_index("axis").reindex(list(DURABLE))
        ypos = range(len(DURABLE))
        b.errorbar(cc["mean"].values, list(ypos),
                   xerr=[cc["mean"].values - cc["eti_lo"].values, cc["eti_hi"].values - cc["mean"].values],
                   fmt="o", color="#2c7fb8", capsize=3)
        b.axvline(0, color="k", lw=0.8)
        b.set_yticks(list(ypos))
        b.set_yticklabels(list(DURABLE))
        b.set_title(f"{name}: durable-axis effect (EIV, 94% HDI)")
        b.set_xlabel("standardized β on the z-scored outcome")
        b.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "43_added_value.png", dpi=130)
    plt.close(fig)


def _fig_calibration(calib, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outcomes = [o.name for o in cfg.primary()]
    fig, ax = plt.subplots(1, len(outcomes), figsize=(5.6 * len(outcomes), 5), squeeze=False)
    for j, name in enumerate(outcomes):
        y, pred, fam = calib[name]
        a = ax[0][j]
        a.scatter(pred, y, s=6, alpha=0.25, color="#2c7fb8")
        lim = [min(pred.min(), y.min()), max(pred.max(), y.max())]
        a.plot(lim, lim, "k--", lw=1)
        r2 = 1 - np.var(y - pred) / np.var(y)
        a.set_title(f"{name}: +durable fit (in-sample R² = {r2:.2f})")
        a.set_xlabel("predicted (posterior mean)")
        a.set_ylabel("observed (z-scored)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "43_calibration.png", dpi=130)
    plt.close(fig)


def _verdict_row(sub, model):
    r = sub.loc[model]
    return f"{r['d_elpd_vs_ref']:+.1f} ± {r['se_d_elpd']:.1f} ({r['verdict']})"


def _report(cfg, comp, coef, horizon, smoke):
    md = [
        "# 43 — M4.3 incremental validity: does the map beat diagnosis + severity?", "",
        ("> ⚠️ SMOKE run (tiny draws) — indicative only.\n" if smoke else ""),
        "Each map representation added on top of the R3y bar; **ΔELPD vs R3y** is the held-out "
        "added value (Q1), and the durable-axis **β 94% HDI excluding 0** is the in-sample read. The "
        "durable coordinates enter as errors-in-variables (M1 SD propagated). A small/ambiguous ΔELPD "
        "with a credibly non-zero coefficient is an honest, reportable result — the biology adds a real "
        "but modest signal against a strong autoregressive baseline.", "",
    ]
    for name in [o.name for o in cfg.primary()]:
        sub = comp[comp.outcome == name].set_index("model").reindex(MODELS)
        n = int(sub["n"].iloc[0])
        md += [f"## {name}  (N = {n})", "",
               sub.loc[[m for m in MODELS if m != "R3y"],
                       ["elpd_loo", "d_elpd_vs_ref", "se_d_elpd", "verdict", "max_pareto_k", "rhat"]]
               .to_markdown(), "",
               "Durable-axis effects (standardized β on the z-scored outcome; EIV, 94% HDI):", ""]
        cfg_sev = "cgi_s__V0" if name == "egf" else "overall_severity__mean"
        cc = coef[(coef.outcome == name) & (coef.severity == cfg_sev)].set_index("axis").reindex(list(DURABLE))
        cc = cc[["mean", "eti_lo", "eti_hi", "p_direction"]]
        md += [cc.to_markdown(), ""]
        alt = coef[(coef.outcome == name) & (coef.severity != cfg_sev) & (coef.axis.isin(DURABLE))]
        if name == "egf" and len(alt):
            md += ["Q2 — same axes under the **error-aware G** severity (must survive both):", "",
                   alt[alt.severity == "overall_severity__mean"].set_index("axis")
                   .reindex(list(DURABLE))[["mean", "eti_lo", "eti_hi", "p_direction"]].to_markdown(), ""]
    md += [
        "## Read", "",
        "- **Representations compared**: continuous durable coords vs the 8 archetypes vs the 4-region "
        "tessellation vs the 8-specifics ceiling — which carries predictive value, and whether the "
        "deployable archetypes retain it.",
        "- **Q2**: a durable effect is only credited if its HDI excludes 0 under *both* the manifest "
        "CGI-S and the error-aware G severity (egf; for cgi_s the two coincide since CGI-S is the "
        "baseline outcome).",
        "- Held-out ΔELPD is the honest performance metric; the calibration scatter (in-sample R²) and "
        "the added-value bars are in `docs/figures/43_{added_value,calibration}.png`.", "",
        "## Decision for the gate",
        "Confirm which representations clear Q1/Q2 per outcome before the transdiagnostic / "
        "head-to-head-vs-DSM-5 stage (44) and the robustness sweep (46).", "",
        "Artifacts: `results/face/m4/{incremental_comparison.csv, coef_durable.csv}` · "
        "`docs/figures/43_{added_value,calibration}.png`.",
    ]
    (REPORTS / "43_incremental.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
