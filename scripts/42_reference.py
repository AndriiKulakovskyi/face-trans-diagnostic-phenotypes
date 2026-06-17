#!/usr/bin/env python3
"""42 — M4.2 the reference (null) models: the diagnosis + severity + baseline-outcome bar.

Fits the strictly-nested ladder per primary outcome on the complete-case modelling sample, so the
held-out ELPD of each rung is comparable:

  R0 nuisance (age + sex + site RI) -> R1 + DSM-5 arm -> R2 + severity -> R3y + baseline outcome

This establishes how much a clinician's standard information (diagnosis + severity + where the patient
is today) already predicts the 2-year outcome — the bar the transdiagnostic map must beat in stage 43.
Predictors are point estimates here (the error-aware G-severity + the durable-coordinate EIV blocks
enter at stage 43). Fit unweighted; IPW is the stage-46 attrition sensitivity. Methods:
docs/PROGNOSIS_MODEL.md (M4.2).

    python3 scripts/42_reference.py [--smoke]

Writes results/face/m4/{elpd_reference.csv, coef_reference_{outcome}.csv}, reports/42_reference.md,
docs/figures/42_reference.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpyro
import pandas as pd

numpyro.set_host_device_count(4)                      # parallel chains on the Mac's cores

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.prognosis.compare import coefficient_table, delta_elpd  # noqa: E402
from face.prognosis.frame import load_outcome_config  # noqa: E402
from face.prognosis.glm import fit_glm  # noqa: E402
from face.prognosis.reference import (  # noqa: E402
    RUNGS,
    design_for_rung,
    modeling_frame,
    outcome_vector,
    severity_column,
    site_index,
)

CONFIG = REPO / "configs" / "m4_outcomes.yaml"
M4 = REPO / "results" / "face" / "m4"
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
CGI_BASELINE = "cgi_s__V0"                            # the manifest baseline-CGI-S column in the frame


def fit_ladder(frame, spec, *, horizon, fit_kw) -> dict:
    sev = severity_column(spec, cgi_baseline_col=CGI_BASELINE)
    sub = modeling_frame(frame, spec, horizon=horizon, severity_col=sev)
    y, fam, n_cat = outcome_vector(sub, spec, horizon=horizon)
    grp, ng = site_index(sub)
    fits, designs = {}, {}
    for rung in RUNGS:
        X, names = design_for_rung(sub, spec, rung, severity_col=sev, horizon=horizon)
        print(f"    [{spec.name}/{rung}] N={len(sub)} P={len(names)} fitting ...", flush=True)
        fits[rung] = fit_glm(y, X, family=fam, group=grp, n_groups=ng, n_cat=n_cat, **fit_kw)
        designs[rung] = names
    cmp = delta_elpd(fits, reference="R0").assign(outcome=spec.name, n=len(sub), severity=sev)
    return {"sub_n": len(sub), "fits": fits, "designs": designs, "cmp": cmp, "sev": sev}


def main(smoke: bool = False) -> None:
    M4.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    cfg = load_outcome_config(CONFIG)
    horizon = cfg.meta.get("primary_horizon", "V2")
    seed = int(cfg.meta.get("seed", 20260610))
    fit_kw = (dict(draws=150, tune=150, chains=2, seed=seed) if smoke
              else dict(draws=1000, tune=1000, chains=4, seed=seed))
    frame = pd.read_parquet(M4 / "analysis_frame.parquet")

    cmps, coefs = [], {}
    for spec in cfg.primary():
        print(f"  [{spec.name}] reference ladder ...", flush=True)
        res = fit_ladder(frame, spec, horizon=horizon, fit_kw=fit_kw)
        cmps.append(res["cmp"])
        coef = coefficient_table(res["fits"]["R3y"], names=res["designs"]["R3y"])
        coef.insert(0, "outcome", spec.name)
        coef.to_csv(M4 / f"coef_reference_{spec.name}.csv", index=False)
        coefs[spec.name] = coef

    comp = pd.concat(cmps, ignore_index=True)
    comp.to_csv(M4 / "elpd_reference.csv", index=False)
    _figure(comp)
    _report(cfg, comp, coefs, horizon, smoke)


def _report(cfg, comp, coefs, horizon, smoke):
    md = [
        "# 42 — M4.2 reference models (the diagnosis + severity bar)", "",
        ("> ⚠️ SMOKE run (tiny draws) — numbers indicative only.\n" if smoke else ""),
        "The strictly-nested ladder per primary outcome (R0 nuisance → R1 +diagnosis → R2 +severity → "
        f"R3y +baseline outcome), fit on the complete-case V0→{horizon} sample. ΔELPD is vs **R0** "
        "(how much each clinician-available block improves held-out prediction). **R3y is the bar** the "
        "transdiagnostic map must beat in stage 43.", "",
    ]
    for name in [o.name for o in cfg.primary()]:
        sub = comp[comp.outcome == name].set_index("model").reindex(RUNGS)
        n = int(sub["n"].iloc[0])
        md += [f"## {name}  (N = {n}, severity = {sub['severity'].iloc[0]})", "",
               sub[["elpd_loo", "se_elpd", "d_elpd_vs_ref", "se_d_elpd", "verdict",
                    "max_pareto_k", "rhat"]].to_markdown(), ""]
        # R3y coefficient highlights (standardized; |mean| largest first, excluding intercept)
        coef = coefs[name]
        coef = coef[coef.term != "alpha"].reindex(coef[coef.term != "alpha"]["mean"].abs()
                                                  .sort_values(ascending=False).index)
        top = coef.head(6)[["term", "mean", "eti_lo", "eti_hi", "p_direction"]]
        md += ["R3y standardized coefficients (top |effect|; outcome z-scored):", "",
               top.to_markdown(index=False), ""]
    md += [
        "## Read", "",
        "- The ladder shows the **bar**: how well diagnosis + severity + baseline value already predict "
        f"the {horizon} outcome. A large R0→R3y ΔELPD that saturates by R3y means the autoregressive "
        "baseline carries most of the signal — exactly why stage 43 must beat **R3y**, not R0.",
        "- For **cgi_s**, severity = the G coordinate (CGI-S itself is the baseline outcome at R3y); for "
        "**egf**, severity = baseline CGI-S. Both point estimates here; the error-aware G enters at stage 43.",
        "- Convergence (max R-hat, Pareto-k) reported per rung; any rung breaching the gate is re-fit "
        "before stage 43 builds on it.", "",
        "## Decision for the gate",
        "Confirm the reference ladders converged and the R3y bar is established per primary outcome, "
        "before adding the durable-coordinate / strata blocks (stage 43).", "",
        "Artifacts: `results/face/m4/{elpd_reference.csv, coef_reference_*.csv}` · "
        "`docs/figures/42_reference.png`.",
    ]
    (REPORTS / "42_reference.md").write_text("\n".join(md))
    print("\n".join(md))


def _figure(comp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outcomes = list(dict.fromkeys(comp["outcome"]))
    fig, ax = plt.subplots(1, len(outcomes), figsize=(6 * len(outcomes), 4.5), squeeze=False)
    for j, name in enumerate(outcomes):
        sub = comp[comp.outcome == name].set_index("model").reindex(RUNGS)
        a = ax[0][j]
        a.bar(range(len(RUNGS)), sub["d_elpd_vs_ref"].values, color="#4575b4")
        a.errorbar(range(len(RUNGS)), sub["d_elpd_vs_ref"].values, yerr=sub["se_d_elpd"].values,
                   fmt="none", ecolor="#333", capsize=3)
        a.set_xticks(range(len(RUNGS)))
        a.set_xticklabels(RUNGS)
        a.axhline(0, color="k", lw=0.8)
        a.set_title(f"{name}: ΔELPD vs R0 (N={int(sub['n'].iloc[0])})")
        a.set_ylabel("ΔELPD (held-out, ↑ better)")
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "42_reference.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
