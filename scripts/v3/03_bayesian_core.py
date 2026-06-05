"""V3 Phase F — Bayesian sparse bifactor (PRIMARY discovery engine), core prototype.

Patient-level, observed-data likelihood (NO imputation): the likelihood is evaluated only on each
patient's OBSERVED cells (a long (patient, indicator, value) table); missing cells never appear.
This proves the engine on the well-measured 3-cohort CONTINUOUS core from the eligibility audit
(V3-1) with the missingness read-out from the atlas (V3-2):

    eta_ij = nu_j + lamG_j * G_i + lamS_j * D_{f(j),i}      x_ij ~ Normal(eta_ij, sigma_j)

  • G              general clinical/biomedical burden (bifactor general factor)
  • D_f            specifics: cognition · metabolic · inflammatory · sleep  (orthogonal to G and each other)
  • loadings       HalfNormal (indicators oriented so higher = more burden -> positive loadings;
                   this is the sparse/identified prototype; signed loadings + ESEM cross-loadings are next)
  • cognition MNAR arm   cog_observed_i ~ Bernoulli(logit a + b * G_i)   (tests V3-2: burden -> non-completion)

Prototype = NUTS on a cohort-stratified subsample with full diagnostics (R-hat / divergences / ESS),
per docs/V3_PLAN.md F7 ("fit a smaller core first, then scale up"). Scale-up to full N via ADVI/NumPyro.

Outputs (aggregate only): results/v3/bayesian/{core_model.md, loadings.csv, factor_scores.csv}
Run:   python3 scripts/v3/03_bayesian_core.py [--smoke] [--n-per-cohort 500] [--draws 500]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset  # noqa: E402

OUT = ROOT / "results" / "v3" / "bayesian"
SEED = 20260605

# indicator -> (factor, orient, log)  ; orient -1 negates so the result is "higher = more burden"
SPEC = {
    # cognition (good scores negated so higher = worse cognition; TMT-B already higher=worse)
    "cvlt_total_recall": ("cognition", -1, False), "wais_code_std": ("cognition", -1, False),
    "wais_digitspan_std": ("cognition", -1, False), "verbal_fluency_semantic": ("cognition", -1, False),
    "tmt_b_time_sec": ("cognition", +1, True),
    # metabolic (HDL negated; lipids/glucose log)
    "bmi": ("metabolic", +1, False), "wstcir": ("metabolic", +1, False), "trig": ("metabolic", +1, True),
    "gluc": ("metabolic", +1, True), "hdl": ("metabolic", -1, False), "sysbpsupine": ("metabolic", +1, False),
    # inflammatory (skewed counts log)
    "crp": ("inflammatory", +1, True), "wbc": ("inflammatory", +1, True), "neut": ("inflammatory", +1, True),
    "plat": ("inflammatory", +1, True), "mono_lbstresc": ("inflammatory", +1, True),
    # sleep — PSQI 3-cohort core (ESS/CSM dropped: BP/DR-only circadian extension; V3-5 fix)
    "psqi": ("sleep", +1, False), "psqi11": ("sleep", +1, False),
    "psqi13": ("sleep", +1, False), "psqi15": ("sleep", +1, False),
}
FACTORS = ["cognition", "metabolic", "inflammatory", "sleep"]


def prep(args):
    variables = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    df = build_unified_dataframe("data", str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL"], format="long")
    ds = to_harmonized_dataset(df, variables, visit="V0", normalize=False, apply_skip_logic=True)
    X = ds.X
    inds = [c for c in SPEC if c in X.columns]
    cohort = pd.Series(X.index.get_level_values("cohort"), index=X.index)

    # transform + orient + standardize (on observed support)
    M = pd.DataFrame(index=X.index)
    for c in inds:
        _, orient, dolog = SPEC[c]
        v = pd.to_numeric(X[c], errors="coerce").astype(float)
        if dolog:
            v = np.log1p(v - np.nanmin(v) + 1e-6) if np.nanmin(v) <= 0 else np.log(v)
        v = orient * v
        sd = v.std()
        M[c] = (v - v.mean()) / sd if sd and sd > 0 else np.nan

    # cohort-BALANCED subsample (BP>>SZ>>DR: equal n per cohort removes BP dominance — the
    # alternative to a 1/n_cohort weight). select='complete' takes the n MOST-COMPLETE patients
    # per cohort (denser observed matrix -> better identification); 'random' takes a random n.
    rng = np.random.default_rng(SEED)
    keep = M.notna().any(axis=1)
    M, coh = M[keep], cohort[keep]
    obs_count = M.notna().sum(axis=1).values
    if not args.full:
        idx = []
        for c in ["bp", "sz", "dr"]:
            pool = np.where(coh.values == c)[0]
            take = min(args.n_per_cohort, len(pool))
            if args.select == "complete":
                order = pool[np.argsort(-obs_count[pool], kind="stable")]  # densest first
                idx.extend(order[:take])
            else:
                idx.extend(rng.choice(pool, size=take, replace=False))
        idx = np.sort(np.array(idx))
        M, coh = M.iloc[idx], coh.iloc[idx]

    N, J = M.shape
    facidx = np.array([FACTORS.index(SPEC[c][0]) for c in M.columns])
    cog_cols = [c for c in M.columns if SPEC[c][0] == "cognition"]
    cog_obs = M[cog_cols].notna().any(axis=1).astype(int).values
    Mv = M.values
    ii, jj = np.where(~np.isnan(Mv))
    vv = Mv[ii, jj]
    return M, coh, dict(N=N, J=J, facidx=facidx, ii=ii, jj=jj, vv=vv, cog_obs=cog_obs,
                        names=list(M.columns), n_obs=len(vv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true", help="use all patients (slow)")
    ap.add_argument("--n-per-cohort", type=int, default=500)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--tune", type=int, default=500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--ta", type=float, default=0.95, help="target_accept")
    ap.add_argument("--min-group", type=int, default=10,
                    help="marg model: keep only observed-patterns with >= this many patients (drops the "
                         "rare-pattern tail to bound the #Cholesky ops -> tractable NUTS gradients)")
    ap.add_argument("--select", choices=["complete", "random"], default="complete",
                    help="cohort-BALANCED subsample (BP>>SZ>>DR): 'complete' = the n most-complete "
                         "patients per cohort (denser + balanced); 'random' = random n per cohort")
    ap.add_argument("--model", choices=["marg", "corr", "bifactor"], default="marg",
                    help="marg = MARGINALIZED correlated factors (factors integrated out -> no latent "
                         "funnel; yields Phi; the clean/identified model); corr = explicit-latent "
                         "correlated factors (divergent); bifactor = G + specifics (weakly identified)")
    args = ap.parse_args()
    if args.smoke:
        args.n_per_cohort, args.draws, args.tune, args.chains = 120, 60, 60, 2

    import arviz as az
    import pymc as pm

    M, coh, d = prep(args)
    print(f"N={d['N']} patients · J={d['J']} indicators · {d['n_obs']:,} observed cells "
          f"({d['n_obs']/(d['N']*d['J']):.0%} dense) · cohorts={dict(coh.value_counts())}")

    import pytensor.tensor as pt
    facidx, ii, jj, vv = d["facidx"], d["ii"], d["jj"], d["vv"]
    F, names, J = len(FACTORS), d["names"], d["J"]
    Mv = M.to_numpy()
    onehot = np.zeros((J, F)); onehot[np.arange(J), facidx] = 1.0
    # observed-pattern groups for the marginalized likelihood (each patient's observed indicator set)
    patterns = {}
    for i in range(Mv.shape[0]):
        o = tuple(np.flatnonzero(~np.isnan(Mv[i])))
        if o:
            patterns.setdefault(o, []).append(i)
    if args.model == "marg":
        kept = {o: r for o, r in patterns.items() if len(r) >= args.min_group}
        n_drop = Mv.shape[0] - sum(len(r) for r in kept.values())
        patterns = kept
        print(f"  marginalized: {len(patterns)} observed-patterns kept (>={args.min_group} patients); "
              f"{n_drop} patients in rare patterns dropped")

    with pm.Model() as model:
        nu = pm.Normal("nu", 0.0, 1.0, shape=J)
        if args.model == "marg":
            # MARGINALIZED Gaussian factor model: integrate the per-patient factors OUT (no latents ->
            # no funnel). Likelihood = MVN(nu, Lam Phi Lam^T + diag(psi)) on each patient's OBSERVED
            # cells, grouped by missingness pattern. Still NO imputation. Params ~ 3J + F(F-1)/2.
            lam = pm.HalfNormal("lam", 0.6, shape=J)
            psi = pm.HalfNormal("psi", 1.0, shape=J)            # uniqueness variances
            corr_raw = pm.LKJCorr("Phi", n=F, eta=2.0)          # stored lower-triangular
            corr = pt.tril(corr_raw, -1) + pt.tril(corr_raw, -1).T + pt.eye(F)   # -> symmetric correlation
            Lam = lam[:, None] * pt.as_tensor(onehot)           # (J,F) simple structure
            Sigma = Lam @ corr @ Lam.T + pt.diag(psi)           # (J,J) marginal covariance
            ll = 0.0
            for o, rows in patterns.items():
                m, oi = len(o), list(o)
                S = np.zeros((m, J)); S[np.arange(m), oi] = 1.0  # selection matrix (avoids buggy adv-index)
                St = pt.as_tensor(S)
                Soo = St @ Sigma @ St.T                          # (m,m) observed submatrix, via matmul
                Xg = pt.as_tensor(Mv[np.ix_(rows, oi)])          # (n_g, m) observed data (constant)
                Lc = pt.linalg.cholesky(Soo + 1e-6 * pt.eye(m))
                sol = pt.linalg.solve_triangular(Lc, (Xg - St @ nu).T, lower=True)   # (m, n_g)
                ll = ll + (-0.5 * (m * np.log(2 * np.pi)
                                   + 2 * pt.log(pt.diag(Lc)).sum()
                                   + (sol ** 2).sum(axis=0))).sum()
            pm.Potential("obs_ll", ll)
            load_vars = ["lam", "psi", "Phi"]
        else:
            sigma = pm.HalfNormal("sigma", 1.0, shape=J)
            if args.model == "bifactor":
                lamG = pm.HalfNormal("lamG", 0.6, shape=J)
                lamS = pm.HalfNormal("lamS", 0.6, shape=J)
                G = pm.Normal("G", 0.0, 1.0, shape=d["N"])
                D = pm.Normal("D", 0.0, 1.0, shape=(F, d["N"]))
                mu = nu[jj] + lamG[jj] * G[ii] + lamS[jj] * D[facidx[jj], ii]
                a_cog = pm.Normal("a_cog", 0.0, 1.5)
                b_cog = pm.Normal("b_cog", 0.0, 1.0)
                pm.Bernoulli("cog_R", logit_p=a_cog + b_cog * G, observed=d["cog_obs"])
                load_vars = ["lamG", "lamS", "sigma"]
            else:  # corr: explicit-latent correlated factors (kept for comparison; divergent)
                lam = pm.HalfNormal("lam", 0.6, shape=J)
                corr = pm.LKJCorr("Phi", n=F, eta=2.0)
                Lc = pt.linalg.cholesky(corr + 1e-6 * pt.eye(F))
                z = pm.Normal("z", 0.0, 1.0, shape=(d["N"], F))
                Fsc = pm.Deterministic("Fscore", z @ Lc.T)
                mu = nu[jj] + lam[jj] * Fsc[ii, facidx[jj]]
                load_vars = ["lam", "sigma"]
            pm.Normal("x", mu, sigma[jj], observed=vv)
        idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains, cores=1,
                          target_accept=args.ta, random_seed=SEED, progressbar=False,
                          idata_kwargs={"log_likelihood": False})   # cores=1: avoid macOS mp hang on the pattern graph

    # ---- diagnostics ----
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    summ = az.summary(idata, var_names=load_vars)
    rhat_col = "r_hat" if "r_hat" in summ.columns else ("rhat" if "rhat" in summ.columns else None)
    ess_col = next((c for c in summ.columns if c.startswith("ess")), None)
    rhat_max = float(pd.to_numeric(summ[rhat_col], errors="coerce").max()) if rhat_col else float("nan")
    ess_min = float(pd.to_numeric(summ[ess_col], errors="coerce").min()) if ess_col else float("nan")
    if args.model in ("corr", "marg"):   # fold in the factor-correlation R-hat (off-diagonal of Phi)
        try:
            rhat_max = max(rhat_max, float(np.nanmax(az.rhat(idata, var_names=["Phi"])["Phi"].values)))
        except Exception:
            pass
    converged = (rhat_max < 1.05) and (div == 0)
    verdict = "**CONVERGED**" if converged else "**NOT CONVERGED — numbers PROVISIONAL**"
    post = idata.posterior

    OUT.mkdir(parents=True, exist_ok=True)
    extra_md = []
    if args.model in ("corr", "marg"):
        lam_m = post["lam"].mean(("chain", "draw")).values
        load = pd.DataFrame({"indicator": names, "factor": [SPEC[n][0] for n in names],
                             "loading": np.round(lam_m, 2)})
        Phi = post["Phi"].mean(("chain", "draw")).values
        Phi = np.tril(Phi, -1) + np.tril(Phi, -1).T + np.eye(F)   # symmetrize (LKJCorr stores lower-tri)
        phi = pd.DataFrame(np.round(Phi, 2), index=FACTORS, columns=FACTORS)
        if args.model == "corr":
            Fm = post["Fscore"].mean(("chain", "draw")).values  # (N,F)
        else:  # marg: posterior-mean regression (Thomson) scores  E[F|x_o] = Phi Lam_o^T Sig_oo^-1 (x_o-nu_o)
            psi_m = post["psi"].mean(("chain", "draw")).values
            nu_m = post["nu"].mean(("chain", "draw")).values
            Lam_m = lam_m[:, None] * onehot
            Sig = Lam_m @ Phi @ Lam_m.T + np.diag(psi_m)
            Fm = np.full((d["N"], F), np.nan)
            for o, rows in patterns.items():
                oi = list(o)
                B = Phi @ Lam_m[oi].T @ np.linalg.pinv(Sig[np.ix_(oi, oi)])   # (F, m)
                Fm[rows] = (Mv[np.ix_(rows, oi)] - nu_m[oi]) @ B.T
        scores = pd.DataFrame(Fm, columns=[f"F_{f}" for f in FACTORS], index=M.index)
        scores.insert(0, "cohort", coh.values)
        r_mi = float(Phi[FACTORS.index("metabolic"), FACTORS.index("inflammatory")])
        mean_offdiag = float(np.abs(Phi[np.triu_indices(F, 1)]).mean())
        extra_md = ["## Factor correlation matrix Φ (the structural read-out)",
                    phi.to_markdown(), "",
                    f"- mean |off-diagonal| = **{mean_offdiag:.2f}** "
                    f"({'weakly-correlated factors — a loose backbone, not one general factor' if mean_offdiag < 0.4 else 'strongly correlated — a general factor may be warranted'}).",
                    f"- corr(metabolic, inflammatory) = **{r_mi:+.2f}** "
                    f"({'SEPARABLE → supports the split' if abs(r_mi) < 0.5 else 'entangled → may not split'})."]
        print_lines = [phi.to_string()]
    else:
        lamG_m = post["lamG"].mean(("chain", "draw")).values
        lamS_m = post["lamS"].mean(("chain", "draw")).values
        load = pd.DataFrame({"indicator": names, "factor": [SPEC[n][0] for n in names],
                             "lambda_G": np.round(lamG_m, 2), "lambda_specific": np.round(lamS_m, 2)})
        scores = pd.DataFrame({"cohort": coh.values, "G": post["G"].mean(("chain", "draw")).values}, index=M.index)
        Dm = post["D"].mean(("chain", "draw")).values
        for f, fac in enumerate(FACTORS):
            scores[f"D_{fac}"] = Dm[f]
        bcm = float(post["b_cog"].mean()); blo, bhi = np.percentile(post["b_cog"].values, [3, 97])
        extra_md = ["## Cognition MNAR arm",
                    f"- b_cog (burden→P(neuropsych observed)) = **{bcm:+.2f}** [{blo:+.2f}, {bhi:+.2f}] "
                    f"— {'trust only if CONVERGED' if not converged else ('confirms burden→non-completion' if bhi < 0 else 'inconclusive')}."]
        print_lines = []

    load.to_csv(OUT / "loadings.csv", index=False)
    scores.round(3).to_csv(OUT / "factor_scores.csv")
    try:
        idata.to_netcdf(str(OUT / "idata_core.nc"))
    except Exception as e:
        print("  (idata not saved:", type(e).__name__, ")")

    md = [f"# V3 Bayesian core — {args.model} model (Phase F prototype)", "",
          f"Patient-level observed-likelihood (NO imputation). N={d['N']} (cohort-stratified subsample"
          f"{' — SMOKE' if args.smoke else ''}), J={d['J']} continuous core indicators, "
          f"{d['n_obs']:,} observed cells. NUTS {args.chains}×{args.draws} (+{args.tune} tune), target_accept {args.ta}.", "",
          f"## Convergence — {verdict}",
          f"- max R-hat = {rhat_max:.3f} (want < 1.01) · min ESS = {ess_min:.0f} · divergences = {div}",
          ("" if converged else "- ⚠️ Treat all loadings/correlations below as PROVISIONAL until this is fixed."), "",
          "## Loadings (posterior mean; higher indicator value = more burden)",
          load.to_markdown(index=False), ""] + extra_md + ["",
          "Artifacts: `loadings.csv`, `factor_scores.csv`, `idata_core.nc`.", "",
          "### Next iteration",
          "- add suicidality (ordinal/Bernoulli/count) + affective/anhedonia BP/DR extension + the "
          "cognition MNAR arm; ESEM soft cross-loadings; scale to full N (ADVI/NumPyro); "
          "posterior-predictive checks; measurement invariance (Phase H)."]
    (OUT / "core_model.md").write_text("\n".join(md))

    print(f"\n{args.model} model — max R-hat={rhat_max:.3f}  min ESS={ess_min:.0f}  divergences={div}  -> {verdict}")
    print("\nloadings:\n", load.to_string(index=False))
    for ln in print_lines:
        print("\nΦ (factor correlations):\n", ln)
    print("\nwrote:", OUT.relative_to(ROOT), "/ {core_model.md, loadings.csv, factor_scores.csv, idata_core.nc}")


if __name__ == "__main__":
    main()
