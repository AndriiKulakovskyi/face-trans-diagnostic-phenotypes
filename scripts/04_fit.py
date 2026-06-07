#!/usr/bin/env python3
"""04 — fit a measurement-model stage (continuous core, full-N V0).

    python3 scripts/04_fit.py --stage 1            # S1: bifactor, simple structure, Phi=I
    python3 scripts/04_fit.py --stage 2            # S2: ESEM cross-loadings + windows + Phi
    python3 scripts/04_fit.py --stage 2 --subsample 800   # quick smoke
    python3 scripts/04_fit.py --stage 2 --explicit        # explicit-latent triangulation

Reads data/processed/baseline_v0.parquet + configs/prior_loading_matrix_v3.csv (via the engine
src/face/models/bayesian/continuous_core), fits the marginalized (Woodbury) bifactor/ESEM continuous
core, certifies (R-hat / ESS / divergences / Heywood), and writes:
  reports/04_stage{S}_report.md      certification + loadings + Phi summary (shareable aggregate)
  reports/04_stage{S}_loadings.csv   per-(item, factor) posterior loading (aggregate)
  reports/04_stage{S}_phi.csv        inter-dimension correlation matrix (S2+)
  results/face/stage{S}/{factor_scores.csv, diagnostics.json, idata.nc}   (per-patient -> gitignored)
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from face.models.bayesian.continuous_core import (  # noqa: E402
    S1_FACTORS, S3_FACTORS, build_marginalized, build_model, prepare, thomson_scores,
    warmstart_initvals)

REPORTS = REPO / "reports"
GATES = dict(rhat=1.01, ess=400.0, div=0, heywood=2.5)
PSI_FLOOR = 0.05

# Stage flags: S1 = simple structure + Phi=I; S2 = inter-dimension Phi + MADRS/QIDS/STAI window
# cross-loadings. The specific<->specific (metabolic<->inflammatory) cross-loadings are NOT freed:
# they are rotationally aliased with Phi_{metab,inflam} (not separately identifiable, and the ridge
# made full-N intractable), so Phi carries that association. Warm-start from the certified S1 basin
# (continuation, §4.2); target_accept 0.95 (the S1-proven step size; the geometry no longer needs 0.99).
STAGE_FLAGS = {
    1: dict(correlated=False, windows=False, specific_cross=False),
    2: dict(correlated=True, windows=True, specific_cross=False, window_sd_scale=1.0),
    # S3a: add suicidality + developmental_risk via their continuous anchors (marginalized,
    # 7-factor Φ). The non-Gaussian indicators are S3b (explicit-latent block, separate path).
    3: dict(correlated=True, windows=True, specific_cross=False, window_sd_scale=1.0),
}
STAGE_FACTORS = {1: S1_FACTORS, 2: S1_FACTORS, 3: S3_FACTORS}
# ta 0.9 is enough: the marginalized posterior has NO funnel (latents integrated out) -> ~21
# leapfrogs/iter, 0 divergences; 0.95+ just shrinks steps and slows the fit with no quality gain.
STAGE_TARGET_ACCEPT = {1: 0.95, 2: 0.90, 3: 0.90}


def run(stage: int = 1, subsample: int | None = None, marginalized: bool = True,
        gpu: bool = False, draws: int = 1000, tune: int = 1000, chains: int = 4) -> dict:
    import arviz as az
    import pymc as pm

    flags = STAGE_FLAGS.get(stage, STAGE_FLAGS[1])
    factors = STAGE_FACTORS.get(stage, S1_FACTORS)
    prep = prepare(factors, n_subsample=subsample, **flags)
    N, J = prep.M.shape
    mode = "marginalized (Woodbury)" if marginalized else "explicit-latent"
    n_cross = sum(1 for v in prep.kind.values() if v == "cross")
    n_win = sum(1 for v in prep.kind.values() if v == "window")
    print(f"Stage {stage} [{mode}]: N={N} J={J} F={len(prep.factor_cols)} factors={prep.factor_cols}\n"
          f"  pos cells {len(prep.pos_cells)} · signed cells {len(prep.sgn_cells)} "
          f"(cross {n_cross} · window {n_win}) · Phi {'LKJ' if prep.correlated else 'I'} · "
          f"obs cells {int((~np.isnan(prep.M)).sum()):,}")
    model = build_marginalized(prep) if marginalized else build_model(prep)
    target_accept = STAGE_TARGET_ACCEPT.get(stage, 0.95)
    initvals = None                                              # continuation warm-start (§4.2)
    if stage >= 2:
        prev = stage - 1
        prev_prep = prepare(STAGE_FACTORS.get(prev, S1_FACTORS), **STAGE_FLAGS.get(prev, STAGE_FLAGS[1]))
        initvals = warmstart_initvals(prep, from_stage=prev, from_items=prev_prep.items)
        if initvals:
            print(f"  warm-start: S{prev} posterior -> {sorted(initvals)} (chains start in the prior-stage basin)")
    use_numpyro = marginalized or gpu          # marginalized needs the JAX backend (batched F×F linalg)

    def _sample(iv):
        with model:
            if use_numpyro:
                return pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept,
                                 random_seed=20260605, nuts_sampler="numpyro", initvals=iv,
                                 idata_kwargs={"log_likelihood": False})
            return pm.sample(draws=draws, tune=tune, chains=chains, cores=1, target_accept=target_accept,
                             random_seed=20260605, progressbar=False, init="jitter+adapt_diag",
                             initvals=iv, idata_kwargs={"log_likelihood": False})

    try:
        idata = _sample(initvals)
    except Exception as e:                      # bad warm-start point (e.g. NaN init) -> jitter fallback
        if initvals is None:
            raise
        print(f"  warm-start init failed ({type(e).__name__}); falling back to jitter init")
        idata = _sample(None)
    return _diagnose_write(stage, prep, idata, az, marginalized)


def _loadings_frame(prep, post) -> pd.DataFrame:
    Lam = post["Lam"].mean(("chain", "draw")).values                  # [J, F]
    rows = []
    for (j, c), kind in prep.kind.items():
        rows.append(dict(item=prep.items[j], factor=prep.factor_cols[c],
                         loading=round(float(Lam[j, c]), 3), kind=kind, home=prep.home[j]))
    return pd.DataFrame(rows)


def _diagnose_write(stage, prep, idata, az, marginalized=True) -> dict:
    out = REPO / "results" / "face" / f"stage{stage}"
    out.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    post = idata.posterior
    spec = prep.spec_factors
    load = _loadings_frame(prep, post)
    load.to_csv(REPORTS / f"04_stage{stage}_loadings.csv", index=False)

    # Phi (inter-dimension correlation among specifics; G held orthogonal)
    Phi = post["Phi"].mean(("chain", "draw")).values if "Phi" in post else np.eye(len(prep.factor_cols))
    phi_df = pd.DataFrame(Phi, index=prep.factor_cols, columns=prep.factor_cols)
    if prep.correlated:
        phi_df.round(3).to_csv(REPORTS / f"04_stage{stage}_phi.csv")
    si = [prep.factor_cols.index(f) for f in spec]
    offdiag = Phi[np.ix_(si, si)][np.triu_indices(len(si), 1)] if len(si) >= 2 else np.array([])

    summ = az.summary(idata, var_names=[v for v in ["lam_pos", "lam_cross", "sigma", "Phi_spec"]
                                        if v in post])
    # LKJCorr returns a full matrix whose diagonal is a constant 1 (sd 0) -> NaN R-hat / 0 ESS;
    # drop zero-variance rows so the gate reflects only sampled parameters.
    if "sd" in summ.columns:
        summ = summ[pd.to_numeric(summ["sd"], errors="coerce") > 0]
    rc = "r_hat" if "r_hat" in summ.columns else "rhat"
    ec = next((c for c in summ.columns if c.startswith("ess")), None)
    rhat = float(pd.to_numeric(summ[rc], errors="coerce").max())
    ess = float(pd.to_numeric(summ[ec], errors="coerce").min()) if ec else float("nan")
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    heywood = bool((load.loading.abs() > GATES["heywood"]).any())
    cert = (round(rhat, 3) <= GATES["rhat"] and (np.isnan(ess) or ess >= GATES["ess"])
            and div <= GATES["div"] and not heywood)
    diag = dict(stage=stage, N=int(prep.M.shape[0]), J=int(prep.M.shape[1]),
                factors=prep.factor_cols, correlated=prep.correlated,
                n_specific_cross_cells=int((load.kind == "cross").sum()),
                n_window_cells=int((load.kind == "window").sum()),
                rhat_max=round(rhat, 4), ess_min=round(ess, 1), divergences=div,
                heywood=heywood, mean_abs_phi_offdiag=round(float(np.abs(offdiag).mean()), 3)
                if offdiag.size else 0.0, certified=cert)
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))

    # per-patient factor scores: explicit-latent reads them off the posterior; the marginalized
    # model integrates the latents out -> post-hoc Thomson scores (provisional checkpoint read-out).
    if "G" in post:
        sc = pd.DataFrame({"cohort": prep.cohort, "G": post["G"].mean(("chain", "draw")).values})
        Dm = post["D"].mean(("chain", "draw")).values
        for i, f in enumerate(spec):
            sc[f] = Dm[:, i]
        sc.to_csv(out / "factor_scores.csv", index=False)
    elif marginalized:
        Lam = post["Lam"].mean(("chain", "draw")).values
        psi = (PSI_FLOOR + post["sigma"].mean(("chain", "draw")).values) ** 2
        Fsc = thomson_scores(prep, Lam, Phi, psi)
        sc = pd.DataFrame(Fsc, columns=prep.factor_cols)
        sc.insert(0, "cohort", prep.cohort)
        sc.round(3).to_csv(out / "factor_scores.csv", index=False)
    try:
        idata.to_netcdf(str(out / "idata.nc"))
    except Exception:
        pass

    md = _report_md(stage, prep, load, phi_df, offdiag, diag, marginalized)
    (REPORTS / f"04_stage{stage}_report.md").write_text("\n".join(md))
    print("\n".join(md[:7]))
    print(f"\nwrote reports/04_stage{stage}_report.md + loadings.csv"
          f"{' + phi.csv' if prep.correlated else ''} ; results/face/stage{stage}/")
    return diag


def _report_md(stage, prep, load, phi_df, offdiag, diag, marginalized) -> list[str]:
    spec = prep.spec_factors
    gA = load[load.kind == "g_anchor"].sort_values("loading", ascending=False)
    eng = "Marginalized (Woodbury)" if marginalized else "Explicit-latent"
    title = "continuous-core bifactor" if not prep.correlated else "continuous core + Φ + windows (ESEM)"
    nsx = diag["n_specific_cross_cells"]
    md = [f"# Stage {stage} — {title} (full-N V0)", "",
          f"N={diag['N']:,} · J={diag['J']} · factors={diag['factors']}. {eng}, "
          "observed-cell Gaussian likelihood, no imputation."
          + ("" if not prep.correlated else
             f" Φ ~ LKJ(2) over specifics (G ⊥ specifics); {diag['n_window_cells']} MADRS/QIDS/STAI "
             f"window cross-loadings; {nsx} specific↔specific cross-loadings"
             + (" (metabolic↔inflammatory association carried by Φ)." if nsx == 0
                else " free (ridge-guarded sensitivity arm).")), "",
          f"## Certification — {'**CERTIFIED**' if diag['certified'] else '**NOT certified — provisional**'}",
          f"- max R-hat **{diag['rhat_max']:.3f}** · min ESS {diag['ess_min']:.0f} · "
          f"divergences {diag['divergences']} · Heywood {diag['heywood']} "
          f"(gates: R-hat≤{GATES['rhat']}, ESS≥{int(GATES['ess'])}, div=0)", "",
          "## G (functional burden) — anchor loadings", gA[["item", "loading"]].to_markdown(index=False), "",
          "## Specific factors — mean primary home loading",
          load[load.kind == "primary"].groupby("factor")["loading"].mean().round(2).to_markdown(), "",
          "## Bifactor — mean |loading on G| of each specific factor's items (G ⊥ biology check)",
          (load[load.kind == "bifactor_G"].groupby("home")["loading"]
           .apply(lambda s: round(float(s.abs().mean()), 2)).to_markdown()), ""]
    if prep.correlated:
        md += ["## Inter-dimension correlations Φ (specific block; G orthogonal by construction)",
               phi_df.loc[spec, spec].round(2).to_markdown(),
               f"\n- mean |off-diagonal| = **{np.abs(offdiag).mean():.2f}**" if offdiag.size else "", ""]
        win = load[load.kind == "window"]
        if len(win):
            md += ["## MADRS / QIDS / STAI windows — where they land (signed cross-loadings)",
                   win.sort_values(["item", "factor"])[["item", "factor", "loading"]].to_markdown(index=False), ""]
        cr = load[load.kind == "cross"]
        if len(cr):
            cr = cr.reindex(cr.loading.abs().sort_values(ascending=False).index)
            md += ["## Specific↔specific cross-loadings (ridge-guarded sensitivity arm, |loading| desc)",
                   cr[["item", "home", "factor", "loading"]].head(15).to_markdown(index=False), ""]
        else:
            md += ["_Specific↔specific (metabolic↔inflammatory) cross-loadings not freed: "
                   "rotationally aliased with Φ and not separately identifiable — Φ carries that "
                   "association (see Φ above)._", ""]
    md += [f"Artifacts: `reports/04_stage{stage}_loadings.csv`"
           + (f" · `04_stage{stage}_phi.csv`" if prep.correlated else "")
           + f" · `results/face/stage{stage}/` (per-patient, gitignored)."]
    return md


# ---------------------------------- S3b (mixed-likelihood) ----------------------------------
def run_mixed(subsample: int | None = None, draws: int = 600, tune: int = 1000,
              chains: int = 4, label: str = "stage3b") -> dict:
    import arviz as az
    import pymc as pm
    from face.models.bayesian.continuous_core import build_mixed, prepare_mixed  # noqa: E402

    mp = prepare_mixed(n_subsample=subsample)
    base = mp.base
    e_names = [base.factor_cols[c] for c in mp.e_cols]
    n_ng = len(mp.bin_items) + len(mp.ord_items) + len(mp.cnt_items)
    print(f"Stage 3b [mixed-likelihood]: N={base.M.shape[0]} cont-J={base.M.shape[1]} F={len(base.factor_cols)}\n"
          f"  explicit f_e={e_names} · marginalized={[base.factor_cols[c] for c in mp.m_cols]}\n"
          f"  non-Gaussian: {len(mp.bin_items)} binary · {len(mp.ord_items)} ordinal · "
          f"{len(mp.cnt_items)} count ({n_ng} total) · obs cont cells {int((~np.isnan(base.M)).sum()):,}")
    model = build_mixed(mp)
    # warm-start the continuous loadings from the certified S3a (stage 3)
    initvals = None
    prev = prepare(S3_FACTORS, correlated=True, windows=True)
    initvals = warmstart_initvals(base, from_stage=3, from_items=prev.items)
    if initvals:
        print(f"  warm-start: S3a posterior -> {sorted(initvals)}")

    def _samp(iv):
        with model:
            return pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.95,
                             random_seed=20260605, nuts_sampler="numpyro", initvals=iv,
                             idata_kwargs={"log_likelihood": False})
    try:
        idata = _samp(initvals)
    except Exception as e:
        print(f"  warm-start failed ({type(e).__name__}); jitter fallback")
        idata = _samp(None)
    return _diagnose_write_mixed(mp, idata, az, label)


def _diagnose_write_mixed(mp, idata, az, label="stage3b") -> dict:
    out = REPO / "results" / "face" / label
    out.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    base, post, fc = mp.base, idata.posterior, mp.base.factor_cols
    e_names = [fc[c] for c in mp.e_cols]
    ng_items = mp.bin_items + mp.ord_items + mp.cnt_items

    # loadings: continuous (from Lam) + non-Gaussian (lh_/lg_)
    Lam = post["Lam"].mean(("chain", "draw")).values
    rows = [dict(item=base.items[j], factor=fc[c], loading=round(float(Lam[j, c]), 3),
                 kind=base.kind[(j, c)], home=base.home[j], block="continuous") for (j, c) in base.kind]
    for it in ng_items:
        hn = e_names[mp.ng_home[it]]
        if f"lh_{it}" in post:
            rows.append(dict(item=it, factor=hn, loading=round(float(post[f"lh_{it}"].mean()), 3),
                             kind="primary", home=hn, block="non-gaussian"))
        if f"lg_{it}" in post:
            rows.append(dict(item=it, factor="overall_severity", loading=round(float(post[f"lg_{it}"].mean()), 3),
                             kind="bifactor_G", home=hn, block="non-gaussian"))
    load = pd.DataFrame(rows)
    load.to_csv(REPORTS / f"04_{label}_loadings.csv", index=False)

    Phi = post["Phi"].mean(("chain", "draw")).values
    phi_df = pd.DataFrame(Phi, index=fc, columns=fc)
    phi_df.round(3).to_csv(REPORTS / f"04_{label}_phi.csv")
    spec = [f for f in fc if f != "overall_severity"]
    si = [fc.index(f) for f in spec]
    offdiag = Phi[np.ix_(si, si)][np.triu_indices(len(si), 1)]

    # certification over STRUCTURAL params (loadings, Phi, non-Gaussian) — z_e latents reported separately
    vnames = [v for v in ["lam_pos", "lam_cross", "sigma", "Phi_spec"] if v in post]
    vnames += [f"lh_{it}" for it in ng_items if f"lh_{it}" in post]
    vnames += [f"lg_{it}" for it in ng_items if f"lg_{it}" in post]
    summ = az.summary(idata, var_names=vnames)
    if "sd" in summ.columns:
        summ = summ[pd.to_numeric(summ["sd"], errors="coerce") > 0]
    rc = "r_hat" if "r_hat" in summ.columns else "rhat"
    ec = next((c for c in summ.columns if c.startswith("ess")), None)
    rhat = float(pd.to_numeric(summ[rc], errors="coerce").max())
    ess = float(pd.to_numeric(summ[ec], errors="coerce").min()) if ec else float("nan")
    ze = az.summary(idata, var_names=["z_e"])                          # per-patient latent mixing
    ze_ess = float(pd.to_numeric(ze[ec], errors="coerce").min()) if ec in ze.columns else float("nan")
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    heywood = bool((load.loading.abs() > GATES["heywood"]).any())
    cert = (round(rhat, 3) <= GATES["rhat"] and (np.isnan(ess) or ess >= GATES["ess"])
            and div <= GATES["div"] and not heywood)
    diag = dict(stage="3b", N=int(base.M.shape[0]), cont_J=int(base.M.shape[1]),
                explicit_factors=e_names, n_nongaussian=len(ng_items), factors=fc,
                rhat_max=round(rhat, 4), ess_min=round(ess, 1), z_e_ess_min=round(ze_ess, 1),
                divergences=div, heywood=heywood,
                mean_abs_phi_offdiag=round(float(np.abs(offdiag).mean()), 3), certified=cert)
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))

    if "f_e" in post:                                                 # explicit factor scores (G, suic, dev)
        fe = post["f_e"].mean(("chain", "draw")).values
        sc = pd.DataFrame(fe, columns=[f"F_{n}" for n in e_names])
        sc.insert(0, "cohort", base.cohort)
        sc.round(3).to_csv(out / "factor_scores.csv", index=False)
    try:
        idata.to_netcdf(str(out / "idata.nc"))
    except Exception:
        pass

    ng = load[load.block == "non-gaussian"]
    md = ["# Stage 3b — mixed-likelihood suicidality + developmental (full-N V0)", "",
          f"N={diag['N']:,} · continuous J={diag['cont_J']} + {diag['n_nongaussian']} non-Gaussian "
          f"(binary/ordinal/count). Explicit latents f_e={e_names}; the continuous specifics are "
          "marginalized and coupled to f_e through the shared Φ (conditional decomposition). "
          "Observed-cell likelihoods, no imputation.", "",
          f"## Certification — {'**CERTIFIED**' if cert else '**NOT certified — provisional**'}",
          f"- max R-hat **{rhat:.3f}** · min ESS {ess:.0f} (structural) · z_e latent min ESS {ze_ess:.0f} · "
          f"divergences {div} · Heywood {heywood} (gates: R-hat≤{GATES['rhat']}, ESS≥{int(GATES['ess'])}, div=0)", "",
          "## Inter-dimension correlations Φ (specific block; G orthogonal)",
          phi_df.loc[spec, spec].round(2).to_markdown(),
          f"\n- mean |off-diagonal| = **{np.abs(offdiag).mean():.2f}**", "",
          "## Suicidality factor — where its indicators load (home loading · G bifactor)"]
    for it in mp.bin_items + mp.ord_items + mp.cnt_items:
        if mp.ng_home[it] == 1:
            h = ng[(ng.item == it) & (ng.kind == "primary")]["loading"]
            g = ng[(ng.item == it) & (ng.kind == "bifactor_G")]["loading"]
            md.append(f"- {it}: home {h.iloc[0] if len(h) else float('nan'):+.2f} · G {g.iloc[0] if len(g) else float('nan'):+.2f}")
    md += ["", "## Developmental-risk factor — non-Gaussian indicators (home · G)"]
    for it in mp.bin_items + mp.ord_items + mp.cnt_items:
        if mp.ng_home[it] == 2:
            h = ng[(ng.item == it) & (ng.kind == "primary")]["loading"]
            g = ng[(ng.item == it) & (ng.kind == "bifactor_G")]["loading"]
            md.append(f"- {it}: home {h.iloc[0] if len(h) else float('nan'):+.2f} · G {g.iloc[0] if len(g) else float('nan'):+.2f}")
    md += ["", f"Artifacts: `reports/04_{label}_loadings.csv` · `04_{label}_phi.csv` · "
           f"`results/face/{label}/` (per-patient, gitignored)."]
    (REPORTS / f"04_{label}_report.md").write_text("\n".join(md))
    print("\n".join(md[:7]))
    print(f"\nwrote reports/04_{label}_report.md + loadings.csv + phi.csv ; results/face/{label}/")
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--mixed", action="store_true",
                    help="S3b: add the non-Gaussian suicidality/developmental block (explicit f_e)")
    ap.add_argument("--explicit", action="store_true",
                    help="explicit-latent engine (default: marginalized Woodbury via NumPyro)")
    ap.add_argument("--gpu", action="store_true", help="route NUTS through NumPyro/JAX (CUDA box)")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    a = ap.parse_args()
    if a.mixed:
        run_mixed(a.subsample, draws=a.draws, tune=a.tune, chains=a.chains)
    else:
        run(a.stage, a.subsample, marginalized=not a.explicit, gpu=a.gpu,
            draws=a.draws, tune=a.tune, chains=a.chains)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--explicit", action="store_true",
                    help="explicit-latent engine (default: marginalized Woodbury via NumPyro)")
    ap.add_argument("--gpu", action="store_true", help="route NUTS through NumPyro/JAX (CUDA box)")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    a = ap.parse_args()
    run(a.stage, a.subsample, marginalized=not a.explicit, gpu=a.gpu,
        draws=a.draws, tune=a.tune, chains=a.chains)


if __name__ == "__main__":
    main()
