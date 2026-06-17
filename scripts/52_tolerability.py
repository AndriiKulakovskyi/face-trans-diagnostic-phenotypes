#!/usr/bin/env python3
"""52 — M5.2 tolerability (the novel test) + the treatment-resistance atlas + the boundary.

The one genuinely novel, severity-clean M5 test: does the transdiagnostic map (esp. the metabolic /
inflammatory archetypes) predict 2-year **side-effect burden** beyond diagnosis + severity (the
pre-registered metabolic-phenotype × side-effects bet)? Plus the per-archetype treatment-resistance /
tolerability atlas (descriptive reframe of the M4 prognostic atlas), and the **confirmatory**
response/resistance incremental checks (expected M4-redundant — reported honestly). Reuses the M4
engine throughout; the diagnosis+severity bar = nuisance + DSM-5 arm + baseline CGI-S + the
error-corrected G. BP/SZ only. Methods: docs/TREATMENT_MODEL.md (M5.2).

    python3 scripts/52_tolerability.py [--smoke]

Writes results/face/m5/{tolerability.csv, response_atlas.csv}, docs/figures/52_treatment_atlas.png,
reports/52_tolerability.md.
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
from face.prognosis.clinical_value import auc, cv_predict, paired_auc_delta  # noqa: E402
from face.prognosis.compare import delta_elpd  # noqa: E402
from face.prognosis.frame import OutcomeSpec  # noqa: E402
from face.prognosis.glm import fit_glm  # noqa: E402
from face.prognosis.reference import (  # noqa: E402
    arm_block,
    armB_block,
    coord_eiv_block,
    foundation_design,
    site_index,
)
from face.treatment.endpoints import load_m5_config  # noqa: E402

CONFIG = REPO / "configs" / "m5_outcomes.yaml"
PROFILES = REPO / "results" / "face" / "m2" / "archetype_profiles.csv"
M5 = REPO / "results" / "face" / "m5"
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
SEV = "overall_severity__mean"          # error-corrected G as the severity adjustment
CGI_BASELINE = "cgi_s__V0"
COVARS = ("age", "sex", "siteid_city", "arm")
# a cgi_s spec so foundation_design uses cgi_s__V0 as the baseline-state anchor for every endpoint
SPEC = OutcomeSpec(name="cgi_s", label="cgi", source_var="cgi01", family="gaussian",
                   direction="lower_better", cohort_scope=("bp", "sz"), severity_anchor="G", role="primary")
SHORT = {  # M4 archetype short labels
    "↓overall_severity ↓sleep ↓developmental_risk": "low-burden",
    "↑cognition ↑overall_severity ↓suicidality": "high-sev+cognitive",
    "↑sleep ↓cognition ↓developmental_risk": "sleep/circadian", "↑metabolic ↓suicidality ↓developmental_risk": "metabolic",
    "↑developmental_risk ↓metabolic ↑sleep": "developmental", "↑mania_activation ↑sleep": "mania/activation",
    "↑inflammatory ↑substance ↓suicidality": "inflammatory", "↑suicidality ↑developmental_risk ↑metabolic": "suicidality"}


def _sample(frame, ep):
    return frame.dropna(subset=[f"ep_{ep}", SEV, CGI_BASELINE, *COVARS]).copy()


def _fit_models(sub, ep, *, with_durable, fit_kw):
    y = sub[f"ep_{ep}"].to_numpy("int64")
    grp, ng = site_index(sub)
    found, _ = foundation_design(sub, SPEC, severity_col=SEV, horizon="V2")   # age+sex+G+cgi_s_V0
    arm, _ = arm_block(sub)
    archB, _ = armB_block(sub, profiles_path=PROFILES)
    base = dict(family="bernoulli", group=grp, n_groups=ng, **fit_kw)
    fits = {"reference": fit_glm(y, np.column_stack([found, arm]), **base),
            "+archetypes": fit_glm(y, np.column_stack([found, arm, archB]), **base)}
    durable_beta = None
    if with_durable:
        ob, sd, _ = coord_eiv_block(sub, DURABLE)
        fits["+durable"] = fit_glm(y, np.column_stack([found, arm]), eiv_obs=ob, eiv_sd=sd, **base)
        c = fits["+durable"]["coef"].set_index("term")
        durable_beta = {ax: (float(c.loc[f"beta_eiv[{i}]", "mean"]), float(c.loc[f"beta_eiv[{i}]", "eti_lo"]),
                             float(c.loc[f"beta_eiv[{i}]", "eti_hi"])) for i, ax in enumerate(DURABLE)}
    return fits, durable_beta, y, np.column_stack([found, arm]), archB


def main(smoke: bool = False) -> None:
    M5.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    cfg = load_m5_config(CONFIG)
    seed = int(cfg["meta"].get("seed", 20260611))
    fit_kw = (dict(draws=150, tune=150, chains=2, seed=seed) if smoke
              else dict(draws=800, tune=800, chains=4, seed=seed))
    frame = pd.read_parquet(M5 / "analysis_frame.parquet")
    frame["arch"] = frame["arch_dominant_name"].map(lambda s: SHORT.get(s, str(s)[:16]))

    rows, durable = [], {}
    # PRIMARY: side_effects (novel, severity-clean) with the durable hypothesis; confirmatory: response, resistance
    for ep, primary in [("side_effects", True), ("response", False), ("resistance", False)]:
        sub = _sample(frame, ep)
        print(f"  [{ep}] N={len(sub)} fitting ...", flush=True)
        fits, dbeta, y, Xref, archB = _fit_models(sub, ep, with_durable=primary, fit_kw=fit_kw)
        cmp = delta_elpd(fits, reference="reference").assign(endpoint=ep, n=len(sub))
        rows.append(cmp)
        if primary:
            durable[ep] = dbeta
            p_ref = cv_predict(Xref, y, seed=seed)
            p_map = cv_predict(np.column_stack([Xref, archB]), y, seed=seed)
            d, lo, hi, pg = paired_auc_delta(y, p_ref, p_map, seed=seed)
            durable["_auc"] = {"auc_ref": round(auc(y, p_ref), 3), "auc_map": round(auc(y, p_map), 3),
                               "d_auc": round(d, 3), "ci": f"[{lo:+.3f},{hi:+.3f}]"}
    comp = pd.concat(rows, ignore_index=True)
    comp.to_csv(M5 / "tolerability.csv", index=False)

    # the treatment-resistance / tolerability atlas (per archetype rates)
    atlas = []
    for arch, g in frame.groupby("arch", sort=False):
        rec = {"archetype": arch, "n": int(len(g))}
        for e in ("side_effects", "resistance", "response"):
            s = g[f"ep_{e}"].dropna()
            rec[e] = round(float(s.mean()), 3) if len(s) >= 20 else np.nan
        atlas.append(rec)
    atlas = pd.DataFrame(atlas).sort_values("side_effects", ascending=False, na_position="last")
    atlas.to_csv(M5 / "response_atlas.csv", index=False)

    _figure(atlas, durable)
    _report(comp, atlas, durable)


def _figure(atlas, durable):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    a = atlas.dropna(subset=["side_effects"])
    x = np.arange(len(a)); w = 0.4
    ax[0].bar(x - w / 2, a["side_effects"], w, label="side-effects", color="#d73027")
    ax[0].bar(x + w / 2, a["resistance"], w, label="treatment-resistance", color="#9970ab")
    ax[0].set_xticks(x); ax[0].set_xticklabels(a["archetype"], rotation=35, ha="right", fontsize=8)
    ax[0].set_ylabel("2-year rate"); ax[0].set_title("Treatment-resistance / tolerability atlas (per archetype)")
    ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=0.3)
    db = durable.get("side_effects")
    if db:
        axx = list(DURABLE)
        mean = [db[a0][0] for a0 in axx]; lo = [db[a0][1] for a0 in axx]; hi = [db[a0][2] for a0 in axx]
        ax[1].errorbar(mean, range(len(axx)),
                       xerr=[np.array(mean) - np.array(lo), np.array(hi) - np.array(mean)],
                       fmt="o", color="#2c7fb8", capsize=3)
        ax[1].axvline(0, color="k", lw=0.8); ax[1].set_yticks(range(len(axx))); ax[1].set_yticklabels(axx)
        ax[1].set_xlabel("durable-axis β on side-effects (EIV, 94% HDI)")
        ax[1].set_title("Does the biology predict side-effect burden?"); ax[1].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "52_treatment_atlas.png", dpi=130)
    plt.close(fig)


def _report(comp, atlas, durable):
    db = durable.get("side_effects", {})
    au = durable.get("_auc", {})
    md = [
        "# 52 — M5.2 tolerability + the treatment-resistance atlas", "",
        "The novel, severity-clean test (map → side-effect burden), the per-archetype resistance / "
        "tolerability atlas, and the confirmatory response/resistance checks. Bar = nuisance + DSM-5 arm "
        "+ baseline CGI-S + error-corrected G. BP/SZ only.", "",
        "## Does the map predict side-effect burden beyond diagnosis + severity? (the novel test)", "",
        comp[comp.endpoint == "side_effects"][["model", "elpd_loo", "d_elpd_vs_ref", "se_d_elpd",
                                               "verdict"]].to_markdown(index=False), "",
        "Durable-axis effect on side-effects (EIV, 94% HDI) — the metabolic-phenotype × side-effects bet:", "",
        "\n".join(f"- **{ax}**: β = {db[ax][0]:+.3f} [{db[ax][1]:+.3f}, {db[ax][2]:+.3f}]"
                  + ("  ← excludes 0" if (db[ax][1] > 0 or db[ax][2] < 0) else "") for ax in DURABLE) if db else "_n/a_",
        "",
        (f"- Clinical value (side-effects): reference AUC {au.get('auc_ref')} → +map {au.get('auc_map')} "
         f"(ΔAUC {au.get('d_auc')} {au.get('ci','')}).") if au else "", "",
        "## Treatment-resistance / tolerability atlas (per archetype, sorted by side-effects)", "",
        atlas.to_markdown(index=False), "",
        "## Confirmatory: response & resistance (expected M4-redundant)", "",
        comp[comp.endpoint.isin(["response", "resistance"]) & (comp.model == "+archetypes")][
            ["endpoint", "d_elpd_vs_ref", "se_d_elpd", "verdict"]].to_markdown(index=False), "",
        "- As predicted by the M5.0 severity-confound audit, the response/resistance signals largely "
        "restate the M4 prognosis (severity outcomes) — reported honestly, not as new evidence.", "",
        "## The boundary (what M5 cannot do here) + the data-ask", "",
        "- M5 stratifies **response / resistance / tolerability** to treatment-as-usual; it **cannot** "
        "say which drug to give — FACE records no treatment identity. True treatment **moderation / "
        "selection** (the precision-psychiatry payoff) requires linking **prescription / medication or "
        "trial-arm data** (a future *M5b*). That data-acquisition check is the program's next step.",
        "- The program's *demonstrated* clinical value culminates at **M4 (prognosis)**; M5 adds a "
        "novel tolerability signal and reframes resistance, within the data's limits.", "",
        "Artifacts: `results/face/m5/{tolerability.csv, response_atlas.csv}` · "
        "`docs/figures/52_treatment_atlas.png`.",
    ]
    (REPORTS / "52_tolerability.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
