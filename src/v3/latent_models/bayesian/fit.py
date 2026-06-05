"""V3 Bayesian engine — staged fit driver.

Loads the stage's item set, builds the config-first model, samples (NUTS, cores=1),
certifies (R-hat / divergences / ESS / Heywood), extracts loadings + Phi + posterior
factor scores, and writes an aggregate stage report. Warm-starts from the previous
stage's posterior means when available.

Outputs (results/v3/bayesian/stage{S}/): loadings.csv, phi.csv, factor_scores.csv,
diagnostics.json, stage_report.md, idata.nc  (per-patient artifacts gitignored).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from v3.latent_models.bayesian.data import load_model_data            # noqa: E402
from v3.latent_models.bayesian.init import efa_initvals               # noqa: E402
from v3.latent_models.bayesian.model import build_model, load_cell_priors  # noqa: E402

CFG = yaml.safe_load((REPO / "configs" / "bayesian_model.yaml").read_text())
OUT = REPO / "results" / "v3" / "bayesian"
SEED = CFG.get("seed", 20260605)


def stage_item_set(spec: dict) -> list[str] | None:
    """Continuous + explicit item set for a stage (None = full pools / all items)."""
    if spec.get("full_pools"):
        return None
    cont: list[str] = []
    for items in CFG["core_continuous"].values():
        cont += items
    if spec.get("include_g"):
        cont += CFG["g_anchors_continuous"]
    if spec.get("add_severity_anchors"):
        cont += CFG.get("g_anchors_severity", [])
    expl: list[str] = []
    for g in spec.get("explicit", []) or []:
        expl += CFG["explicit"].get(g, [])
    return cont + expl


def specific_order(spec: dict) -> list[str]:
    order = list(CFG["specific_factors"])
    if spec.get("full_pools"):
        order += CFG.get("scale_factors", [])
    return order


def run_stage(stage: int, smoke: bool = False, n_per_cohort: int = 500,
              draws: int | None = None, tune: int | None = None,
              chains: int | None = None) -> dict:
    spec = dict(CFG["stages"][stage]); spec["general_factor"] = CFG["general_factor"]
    samp = CFG["sampler"]
    draws = draws or samp["draws"]; tune = tune or samp["tune"]; chains = chains or samp["chains"]
    mg = spec.get("min_group", samp["min_group"])
    if smoke:
        n_per_cohort, draws, tune, chains = 150, 60, 120, 2

    import arviz as az
    import pymc as pm

    keep = stage_item_set(spec)
    data = load_model_data(n_per_cohort=n_per_cohort, min_group=mg, sleep="objective",
                           items_keep=keep)
    cell_priors = load_cell_priors()
    model, meta = build_model(data, spec, cell_priors, specific_order(spec),
                              psi_floor=samp["psi_floor"], lkj_eta=2.0)

    fc = meta["factor_cols"]
    print(f"\n=== Stage {stage}: {spec['name']} ===")
    print(f"N={data.M.shape[0]} cont-J={data.M.shape[1]} patterns={len(data.patterns)} "
          f"(drop {data.n_drop}) | factors={fc} | explicit-factors={meta['expl_factors']} | "
          f"expl={len(meta['expl_items'])} items")

    # jitter+adapt_diag is the robust init for the per-pattern marginalized geometry
    # (ADVI destabilizes it -> NaN); ADVI is reserved for the Stage-6 scale path.
    # For Stage>=1 (bifactor G / cross-loadings) seed loadings from an EFA warm-start.
    initvals = None
    if spec.get("include_g") or spec.get("cross_loadings"):
        try:
            iv = efa_initvals(data, meta, spec, cell_priors, len(CFG["specific_factors"]))
            initvals = iv or None
            if initvals:
                print(f"warm-start: EFA initvals for {list(initvals)}")
        except Exception as e:
            print(f"warm-start skipped ({e}); falling back to jitter")
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                          target_accept=spec.get("target_accept", samp["target_accept"]),
                          random_seed=SEED, progressbar=False, init="jitter+adapt_diag",
                          initvals=initvals, idata_kwargs={"log_likelihood": False})

    res = _diagnose_and_write(stage, spec, data, meta, idata, az, smoke)
    return res


def _diagnose_and_write(stage, spec, data, meta, idata, az, smoke) -> dict:
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    post = idata.posterior
    fc = meta["factor_cols"]

    # loadings from the Lam deterministic (active cells only)
    Lam = post["Lam"].mean(("chain", "draw")).values            # [J, F]
    active = {}
    for (j, c) in meta["pos_cells"] + meta["sgn_cells"]:
        active[(j, c)] = float(Lam[j, c])
    rows = []
    for (j, c), v in active.items():
        rows.append({"indicator": data.cont_items[j], "factor": fc[c], "loading": round(v, 3),
                     "kind": "primary" if (j, c) in set(map(tuple, meta["pos_cells"])) else "cross"})
    load = pd.DataFrame(rows).sort_values(["factor", "kind", "indicator"])

    # Phi
    Phi = post["Phi"].mean(("chain", "draw")).values
    phi = pd.DataFrame(Phi, index=fc, columns=fc)

    # diagnostics
    vnames = [v for v in ["lam_pos", "lam_cross", "psi_raw", "Phi_spec"] if v in post]
    vnames += [f"D_{f}" for f in meta.get("expl_factors", []) if f"D_{f}" in post]
    vnames += [f"lam_{it}" for it in meta.get("expl_items", []) if f"lam_{it}" in post]
    summ = az.summary(idata, var_names=vnames)
    rc = "r_hat" if "r_hat" in summ.columns else "rhat"
    ec = next((c for c in summ.columns if c.startswith("ess")), None)
    rhat_max = float(pd.to_numeric(summ[rc], errors="coerce").max())
    ess_min = float(pd.to_numeric(summ[ec], errors="coerce").min()) if ec else float("nan")
    heywood = bool((load["loading"].abs() > CFG["gates"]["heywood_loading_cap"]).any())
    g = CFG["gates"]
    certified = (round(rhat_max, 3) <= g["rhat_max"] and div <= g["divergences_max"]
                 and (np.isnan(ess_min) or ess_min >= g["ess_min"]) and not heywood)

    # posterior factor scores (Thomson, continuous block) per pattern
    Mv = data.M; N, J = Mv.shape; F = len(fc)
    psi_m = (CFG["sampler"]["psi_floor"] + post["psi_raw"].mean(("chain", "draw")).values)
    nu_m = post["nu"].mean(("chain", "draw")).values
    Sig = Lam @ Phi @ Lam.T + np.diag(psi_m)
    Fsc = np.full((N, F), np.nan)
    for o, rws in data.patterns.items():
        oi = list(o)
        B = Phi @ Lam[oi].T @ np.linalg.pinv(Sig[np.ix_(oi, oi)])
        Fsc[rws] = (Mv[np.ix_(rws, oi)] - nu_m[oi]) @ B.T
    sc = pd.DataFrame(Fsc, columns=[f"F_{f}" for f in fc], index=data.index)
    sc.insert(0, "cohort", data.cohort)
    # Thomson-score correlation: a DIFFERENT estimand from the model Phi parameter
    # (score corrs are inflated/blended by the regression scoring). We report BOTH so
    # the principled latent correlation (model Phi) and the score-based number are never
    # conflated. (Stage 0: model Phi sleep×aff 0.40 vs score-corr 0.54 — both reproduced.)
    score_corr = pd.DataFrame(Fsc, columns=fc).corr()

    # explicit standalone factors (suicidality/substance): posterior latent means + loadings
    # + POST-HOC correlation with the continuous factor scores (the certified `04` read-out).
    expl_corr, expl_load_rows = None, []
    for f in meta.get("expl_factors", []):
        if f"D_{f}" in post:
            sc[f"D_{f}"] = post[f"D_{f}"].mean(("chain", "draw")).values
    for it in meta.get("expl_items", []):
        if f"lam_{it}" in post:
            expl_load_rows.append({"indicator": it, "factor": data.expl_home[it],
                                   "loading": round(float(post[f"lam_{it}"].mean()), 3),
                                   "kind": "explicit"})
    if meta.get("expl_factors"):
        cols_all = [f"F_{f}" for f in fc] + [f"D_{f}" for f in meta["expl_factors"] if f"D_{f}" in post]
        expl_corr = sc[cols_all].corr()

    # G survival report (Stage >= 1)
    g_report = {}
    if spec.get("include_g") and CFG["general_factor"] in fc:
        gcol = fc.index(CFG["general_factor"])
        g_load = load[(load.factor == CFG["general_factor"])]
        g_report = {
            "n_g_loadings": int(len(g_load)),
            "mean_abs_g_loading": round(float(g_load["loading"].abs().mean()), 3) if len(g_load) else 0.0,
            "g_anchor_loadings": g_load[g_load.kind == "primary"]["loading"].round(2).tolist(),
            "specific_on_g_loadings": g_load[g_load.kind == "cross"]["loading"].round(2).tolist(),
        }

    out = OUT / f"stage{stage}{'_smoke' if smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    load_all = pd.concat([load, pd.DataFrame(expl_load_rows)], ignore_index=True) if expl_load_rows else load
    load_all.to_csv(out / "loadings.csv", index=False)
    phi.round(3).to_csv(out / "phi.csv")
    if expl_corr is not None:
        expl_corr.round(3).to_csv(out / "phi_explicit.csv")
    score_corr.round(3).to_csv(out / "phi_scores.csv")
    sc.round(3).to_csv(out / "factor_scores.csv")
    diag = {"stage": stage, "name": spec["name"], "N": int(N), "cont_J": int(J),
            "patterns": len(data.patterns), "dropped": int(data.n_drop),
            "factors": fc, "rhat_max": round(rhat_max, 4), "ess_min": round(ess_min, 1),
            "divergences": div, "heywood": heywood, "certified": certified, "g": g_report}
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    try:
        idata.to_netcdf(str(out / "idata.nc"))
    except Exception:
        pass

    md = [f"# V3 Bayesian measurement model — Stage {stage}: {spec['name']}", "",
          f"_{spec.get('question','')}_", "",
          f"N={N}, cont-J={J}, {len(data.patterns)} patterns (dropped {data.n_drop}), "
          f"V0, cohort-balanced, no imputation.", "",
          f"## Certification — {'**CERTIFIED**' if certified else '**NOT certified — provisional**'}",
          f"- max R-hat **{rhat_max:.3f}** · min ESS {ess_min:.0f} · divergences {div} · "
          f"Heywood {heywood}", ""]
    if g_report:
        md += ["## General factor (G) survival",
               f"- G anchor loadings: {g_report['g_anchor_loadings']}",
               f"- specific-item loadings on G (bifactor): {g_report['specific_on_g_loadings']}",
               f"- mean |G loading| = **{g_report['mean_abs_g_loading']}** over {g_report['n_g_loadings']} cells",
               ""]
    md += ["## Factor correlation matrix — model Φ (latent parameter, the principled estimand)",
           phi.round(2).to_markdown(), "",
           f"- mean |off-diagonal| = **{np.abs(Phi[np.triu_indices(F,1)]).mean():.2f}**", "",
           "## Thomson-score correlation (for comparability; inflated vs model Φ)",
           score_corr.round(2).to_markdown(), "",
           "_Model Φ is the latent factor correlation; score correlations blend factors via "
           "the regression scoring and run higher. Report model Φ as canonical._", "",
           "## Loadings", load.to_markdown(index=False), ""]
    if expl_corr is not None:
        expl_names = [c for c in expl_corr.columns if c.startswith("D_")]
        md += ["## Explicit standalone factors (suicidality / substance) — post-hoc score correlation",
               "_Mixed-likelihood (Bernoulli/neg-binomial) latents; their relationship to the "
               "continuous dimensions is read post-hoc from scores (the certified `04` design)._",
               expl_corr.round(2).to_markdown(), "",
               "### Explicit indicator loadings",
               pd.DataFrame(expl_load_rows).to_markdown(index=False) if expl_load_rows else "_none_", ""]
        for dn in expl_names:
            row = expl_corr.loc[dn, [c for c in expl_corr.columns if c.startswith("F_")]]
            md.append(f"- **{dn[2:]}** vs dimensions: " + ", ".join(f"{k[2:]} {v:+.2f}" for k, v in row.items()))
        md.append("")
    md += ["Artifacts: loadings.csv · phi.csv · phi_scores.csv"
           + (" · phi_explicit.csv" if expl_corr is not None else "")
           + " · factor_scores.csv · diagnostics.json · idata.nc"]
    (out / "stage_report.md").write_text("\n".join(md))

    print(f"R-hat {rhat_max:.3f} · ESS {ess_min:.0f} · div {div} · Heywood {heywood} · "
          f"{'CERTIFIED' if certified else 'NOT CERTIFIED'}")
    if g_report:
        print(f"G: anchors {g_report['g_anchor_loadings']} | specifics-on-G {g_report['specific_on_g_loadings']}")
    if expl_corr is not None:
        for dn in [c for c in expl_corr.columns if c.startswith("D_")]:
            row = expl_corr.loc[dn, [c for c in expl_corr.columns if c.startswith("F_")]]
            print(f"{dn}: " + ", ".join(f"{k[2:]} {v:+.2f}" for k, v in row.items()))
    print("Phi:\n", phi.round(2).to_string())
    print("wrote:", out.relative_to(REPO))
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-per-cohort", type=int, default=500)
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--tune", type=int, default=None)
    ap.add_argument("--chains", type=int, default=None)
    args = ap.parse_args()
    run_stage(args.stage, smoke=args.smoke, n_per_cohort=args.n_per_cohort,
              draws=args.draws, tune=args.tune, chains=args.chains)


if __name__ == "__main__":
    main()
