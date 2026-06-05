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
          f"(drop {data.n_drop}) | factors={fc} | z={meta['z_factors']} | "
          f"expl={len(meta['expl_items'])} items")

    # jitter+adapt_diag is the robust init for the per-pattern marginalized geometry
    # (ADVI destabilizes it -> NaN); ADVI is reserved for the Stage-6 scale path.
    init = "jitter+adapt_diag"
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                          target_accept=spec.get("target_accept", samp["target_accept"]),
                          random_seed=SEED, progressbar=False, init=init,
                          idata_kwargs={"log_likelihood": False})

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
    summ = az.summary(idata, var_names=vnames)
    rc = "r_hat" if "r_hat" in summ.columns else "rhat"
    ec = next((c for c in summ.columns if c.startswith("ess")), None)
    rhat_max = float(pd.to_numeric(summ[rc], errors="coerce").max())
    ess_min = float(pd.to_numeric(summ[ec], errors="coerce").min()) if ec else float("nan")
    heywood = bool((load["loading"].abs() > CFG["gates"]["heywood_loading_cap"]).any())
    g = CFG["gates"]
    certified = (rhat_max < g["rhat_max"] and div <= g["divergences_max"]
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
    load.to_csv(out / "loadings.csv", index=False)
    phi.round(3).to_csv(out / "phi.csv")
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
    md += ["## Factor correlation matrix", phi.round(2).to_markdown(), "",
           f"- mean |off-diagonal| = **{np.abs(Phi[np.triu_indices(F,1)]).mean():.2f}**", "",
           "## Loadings", load.to_markdown(index=False), "",
           "Artifacts: loadings.csv · phi.csv · factor_scores.csv · diagnostics.json · idata.nc"]
    (out / "stage_report.md").write_text("\n".join(md))

    print(f"R-hat {rhat_max:.3f} · ESS {ess_min:.0f} · div {div} · Heywood {heywood} · "
          f"{'CERTIFIED' if certified else 'NOT CERTIFIED'}")
    if g_report:
        print(f"G: anchors {g_report['g_anchor_loadings']} | specifics-on-G {g_report['specific_on_g_loadings']}")
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
