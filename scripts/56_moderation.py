#!/usr/bin/env python3
"""56 — M5.2b stratum × treatment moderation (the headline causal test).

For each estimable question (55 overlap gate) × co-primary outcome (functioning EGF; CGI response),
fit the EIV outcome GLM on the propensity common-support sample with stabilized IPTW + covariate
adjustment (doubly robust): does the **durable-axis × treatment interaction** (β_eiv_int) improve
held-out fit and exclude 0 — i.e. does the map change *who benefits*? Reports the ATE (treatment main
effect) with an E-value (confounding sensitivity), the per-axis moderation coefficients, and a
treatment-as-confounder read for M4 (does the durable-axis prognostic effect survive adjusting for
treatment?). Methods: docs/TREATMENT_MODEL.md §4–6.

    python3 scripts/56_moderation.py [--smoke]

Writes results/face/m5/moderation.csv, docs/figures/56_moderation.png, reports/56_moderation.md.
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
from face.prognosis.frame import OutcomeSpec  # noqa: E402
from face.prognosis.glm import fit_glm  # noqa: E402
from face.prognosis.reference import arm_block, coord_eiv_block, foundation_design, site_index  # noqa: E402
from face.treatment.moderation import e_value, load_moderation_sample  # noqa: E402

M5 = REPO / "results" / "face" / "m5"
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
SEV = "overall_severity__mean"
# estimable question × primary mode (55 gate): lithium-BP clean; antipsychotic-BP clean; clozapine on/off
RUNS = [("lithium_bp", "active_comparator"), ("antipsychotic_bp", "active_comparator"),
        ("clozapine_sz", "on_off")]
SPEC_EGF = OutcomeSpec(name="egf", label="egf", source_var="egf", family="gaussian",
                       direction="higher_better", cohort_scope=("bp", "sz", "dr"), severity_anchor="G", role="primary")
SPEC_CGI = OutcomeSpec(name="cgi_s", label="cgi", source_var="cgi01", family="gaussian",
                       direction="lower_better", cohort_scope=("bp", "sz"), severity_anchor="G", role="primary")
OUTCOMES = [("functioning", "egf__V2", "gaussian", SPEC_EGF),
            ("cgi_response", "ep_response", "bernoulli", SPEC_CGI)]
LOGIT_TO_D = 0.5513  # logistic coef -> standardized d (for the E-value)


def _fit_pair(sub, y, family, spec, treat, w, fit_kw):
    found, _ = foundation_design(sub, spec, severity_col=SEV, horizon="V2")   # age+sex+G+baseline-outcome
    arm, _ = arm_block(sub)
    grp, ng = site_index(sub)
    X = np.column_stack([found, arm, treat[:, None]])                         # treat = LAST fixed column
    ob, sd, _ = coord_eiv_block(sub, DURABLE)
    base = dict(family=family, group=grp, n_groups=ng, eiv_obs=ob, eiv_sd=sd, weights=w, **fit_kw)
    fit0 = fit_glm(y, X, **base)                                              # no interaction
    fit1 = fit_glm(y, X, eiv_interact=treat, **base)                          # + treat×axis moderation
    return fit0, fit1, X.shape[1]


def _row(question, mode, oname, family, fit0, fit1, p_treat, y):
    cmp = delta_elpd({"no_interaction": fit0, "moderation": fit1}, reference="no_interaction")
    d_elpd = float(cmp.loc[cmp.model == "moderation", "d_elpd_vs_ref"].iloc[0])
    se = float(cmp.loc[cmp.model == "moderation", "se_d_elpd"].iloc[0])
    c0 = fit0["coef"].set_index("term")
    ate = float(c0.loc[f"beta[{p_treat-1}]", "mean"])
    ate_lo, ate_hi = float(c0.loc[f"beta[{p_treat-1}]", "eti_lo"]), float(c0.loc[f"beta[{p_treat-1}]", "eti_hi"])
    d = ate if family == "gaussian" else ate * LOGIT_TO_D
    c1 = fit1["coef"].set_index("term")
    inter = {ax: (float(c1.loc[f"beta_eiv_int[{i}]", "mean"]), float(c1.loc[f"beta_eiv_int[{i}]", "eti_lo"]),
                  float(c1.loc[f"beta_eiv_int[{i}]", "eti_hi"])) for i, ax in enumerate(DURABLE)}
    any_mod = any((lo > 0 or hi < 0) for _, lo, hi in inter.values())
    return {"question": question, "mode": mode, "outcome": oname, "n": int(len(y)),
            "ate": round(ate, 3), "ate_lo": round(ate_lo, 3), "ate_hi": round(ate_hi, 3),
            "ate_excludes0": bool(ate_lo > 0 or ate_hi < 0), "e_value": round(e_value(d), 2),
            "moderation_d_elpd": round(d_elpd, 2), "moderation_se": round(se, 2),
            "moderation_any_axis": any_mod, **{f"int_{ax}": round(inter[ax][0], 3) for ax in DURABLE},
            **{f"int_{ax}_lo": round(inter[ax][1], 3) for ax in DURABLE},
            **{f"int_{ax}_hi": round(inter[ax][2], 3) for ax in DURABLE}}


def main(smoke=False) -> None:
    M5.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    fit_kw = dict(draws=150, tune=150, chains=2, seed=20260611) if smoke else dict(draws=700, tune=700, chains=4, seed=20260611)
    rows = []
    for question, mode in RUNS:
        for oname, ycol, family, spec in OUTCOMES:
            try:
                sub = load_moderation_sample(question, mode, M5)
            except FileNotFoundError:
                continue
            need = [ycol, SEV, f"{spec.name}__V0", "age", "sex", "siteid_city", "arm",
                    *[f"{ax}__mean" for ax in DURABLE], *[f"{ax}__sd" for ax in DURABLE], "treat", "iptw"]
            sub = sub.dropna(subset=[c for c in need if c in sub.columns]).copy()
            if len(sub) < 60 or sub["treat"].nunique() < 2:
                continue
            y = sub[ycol].to_numpy(float)
            if family == "gaussian":
                y = (y - y.mean()) / (y.std() or 1.0)            # standardize -> ATE in SD units (E-value)
            else:
                y = y.astype("int64")
            treat = sub["treat"].to_numpy(float)
            w = sub["iptw"].to_numpy(float); w = w / w.mean()
            print(f"  [{question}/{mode} · {oname}] N={len(sub)} (treated {int(treat.sum())}) fitting ...", flush=True)
            fit0, fit1, p = _fit_pair(sub, y, family, spec, treat, w, fit_kw)
            rows.append(_row(question, mode, oname, family, fit0, fit1, p, y))
    res = pd.DataFrame(rows)
    res.to_csv(M5 / "moderation.csv", index=False)
    _figure(res)
    _report(res)


def _figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if res.empty:
        return
    fig, axes = plt.subplots(1, len(res), figsize=(3.5 * len(res), 4), squeeze=False)
    for ax, (_, r) in zip(axes[0], res.iterrows()):
        means = [r[f"int_{a}"] for a in DURABLE]
        los = [r[f"int_{a}_lo"] for a in DURABLE]; his = [r[f"int_{a}_hi"] for a in DURABLE]
        ax.errorbar(means, range(len(DURABLE)), xerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                    fmt="o", color="#1a9850", capsize=3)
        ax.axvline(0, color="k", lw=0.8); ax.set_yticks(range(len(DURABLE))); ax.set_yticklabels(DURABLE, fontsize=8)
        ax.set_title(f"{r['question']}\n{r['outcome']} (ΔELPD {r['moderation_d_elpd']:+.1f})", fontsize=8)
        ax.set_xlabel("treat × axis (94% HDI)", fontsize=8)
    fig.suptitle("Does the durable biology moderate treatment response?", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "56_moderation.png", dpi=130)
    plt.close(fig)


def _report(res):
    cols = ["question", "outcome", "n", "ate", "ate_lo", "ate_hi", "e_value",
            "moderation_d_elpd", "moderation_se", "moderation_any_axis"]
    show = res[cols] if not res.empty else pd.DataFrame(columns=cols)
    md = [
        "# 56 — M5.2b stratum × treatment moderation", "",
        "Per estimable question × co-primary outcome (functioning EGF; CGI response): the EIV outcome GLM "
        "on the propensity common-support sample, stabilized-IPTW + covariate-adjusted (doubly robust). "
        "**ATE** = treatment main effect (SD units / log-odds) + **E-value** (confounding sensitivity); "
        "**moderation** = the durable-axis × treatment interaction (ΔELPD vs no-interaction; any axis HDI "
        "excluding 0).", "",
        "## ATE + moderation by question × outcome", "",
        show.to_markdown(index=False), "",
        "## Per-axis moderation coefficients (treat × axis, 94% HDI)", "",
        "\n".join(
            f"- **{r['question']} · {r['outcome']}**: "
            + "; ".join(f"{a} {r[f'int_{a}']:+.3f} [{r[f'int_{a}_lo']:+.3f},{r[f'int_{a}_hi']:+.3f}]"
                        + ("*" if (r[f'int_{a}_lo'] > 0 or r[f'int_{a}_hi'] < 0) else "") for a in DURABLE)
            for _, r in res.iterrows()) if not res.empty else "_no estimable cells_", "",
        "(* = HDI excludes 0.)", "",
        "## Read",
        "- **ATE**: the treatment association after propensity + outcome adjustment; the **E-value** is how "
        "strong an unmeasured confounder (on both treatment and outcome) would need to be to null it — "
        "small E-values mean the association is fragile to confounding by indication.",
        "- **Moderation** is the M5 question: a credible `treat × axis` interaction (ΔELPD > 2·SE and an "
        "axis HDI excluding 0) means the map identifies *who benefits* — over and above the average effect.",
        "- **Honest expectation**: average treatment effects on observational data are confounded (low "
        "E-values); the moderation interaction is the cleaner target but is typically underpowered. A null "
        "moderation is a legitimate, publishable result.", "",
        "Artifacts: `results/face/m5/moderation.csv` · `docs/figures/56_moderation.png`.",
    ]
    (REPORTS / "56_moderation.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
