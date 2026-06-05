"""V3 Phase F (extended) — measurement model + affective block + mixed-likelihood suicidality + MNAR.

Builds on the certified marginalized core (03):
  • MARGINALIZED Gaussian block, 5 factors: cognition · metabolic · inflammatory · sleep · AFFECTIVE
    (affective = MADRS/QIDS/STAI/anhedonia, BP/DR; SZ contributes no affective cells — observed-likelihood,
    no imputation). Putting affective IN the marginalized block estimates the **symptom⊥biology**
    correlations directly (the V2 headline), not post-hoc.
  • EXPLICIT-latent SUICIDALITY module with **mixed likelihoods**: ISF binary items (Bernoulli) + attempt
    count (negative-binomial). 3-cohort. Its correlation with the 5 factors is read post-hoc from scores
    (V2 found suicidality orthogonal/standalone — we retest).
  • Post-hoc **MNAR diagnostic**: does affective severity predict NON-completion of cognition / suicide
    items? (the V3-2/V3-4 informative-missingness finding, now using the model's own severity).

Cohort-balanced (500 most-complete/cohort) · V0 only · no imputation. Aggregate outputs only:
  results/v3/bayesian_ext/{extended_model.md, loadings.csv, phi.csv, factor_scores.csv}
Run:  python3 scripts/v3/04_extended_model.py [--smoke] [--n-per-cohort 500] [--draws 600] [--min-group 10]
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

from v3.data import build_unified_dataframe, load_variables, to_harmonized_dataset  # noqa: E402

OUT = ROOT / "results" / "v3" / "bayesian_ext"
SEED = 20260605

# continuous indicators -> (factor, orient, log); oriented so higher = more burden
CONT = {
    "cvlt_total_recall": ("cognition", -1, False), "wais_code_std": ("cognition", -1, False),
    "wais_digitspan_std": ("cognition", -1, False), "verbal_fluency_semantic": ("cognition", -1, False),
    "tmt_b_time_sec": ("cognition", +1, True),
    "bmi": ("metabolic", +1, False), "wstcir": ("metabolic", +1, False), "trig": ("metabolic", +1, True),
    "gluc": ("metabolic", +1, True), "hdl": ("metabolic", -1, False), "sysbpsupine": ("metabolic", +1, False),
    "crp": ("inflammatory", +1, True), "wbc": ("inflammatory", +1, True), "neut": ("inflammatory", +1, True),
    "plat": ("inflammatory", +1, True), "mono_lbstresc": ("inflammatory", +1, True),
    "psqi": ("sleep", +1, False), "psqi11": ("sleep", +1, False), "psqi12": ("sleep", +1, False),
    "psqi13": ("sleep", +1, False), "psqi14": ("sleep", +1, False), "psqi15": ("sleep", +1, False),
    # AFFECTIVE (BP/DR; higher = worse mood/anxiety/anhedonia)
    "madrs": ("affective", +1, False), "qidsr120": ("affective", +1, False),
    "staya": ("affective", +1, False), "qids_anhedonia_interest": ("affective", +1, False),
}
FACTORS = ["cognition", "metabolic", "inflammatory", "sleep", "affective"]
# sleep specifications (V3-7): 'full' = V3-6 set; 'objective' = sleep-parameter items only (less
# affect-overlap — daytime-dysfunction/quality drive the sleep×affect coupling, see V3-7 sensitivity).
SLEEP_SETS = {"full": ["psqi", "psqi11", "psqi13", "psqi15"], "objective": ["psqi11", "psqi12", "psqi14"]}
SUIC_BIN = ["isf01", "isf02", "isf03", "isf04", "isf05", "isf08", "isf09"]   # Bernoulli
SUIC_COUNT = "isf09a"                                                         # negative-binomial (attempt count)


def prep(args):
    variables = load_variables(str(ROOT / "data" / "face-common-vars.xlsx"))
    df = build_unified_dataframe("data", str(ROOT / "data" / "face-common-vars.xlsx"),
                                 readiness=["READY", "PARTIAL"], format="long")
    ds = to_harmonized_dataset(df, variables, visit="V0", normalize=False, apply_skip_logic=True)
    X = ds.X
    cohort = pd.Series(X.index.get_level_values("cohort"), index=X.index)

    sleep_keep = set(SLEEP_SETS[args.sleep])
    cont = [c for c in CONT if c in X.columns and (CONT[c][0] != "sleep" or c in sleep_keep)]
    M = pd.DataFrame(index=X.index)
    for c in cont:
        _, orient, dolog = CONT[c]
        v = pd.to_numeric(X[c], errors="coerce").astype(float)
        if dolog:
            v = np.log1p(v - np.nanmin(v) + 1e-6) if np.nanmin(v) <= 0 else np.log(v)
        v = orient * v
        sd = v.std()
        M[c] = (v - v.mean()) / sd if sd and sd > 0 else np.nan
    # suicidality raw (binary 0/1, count >=0)
    sb = [c for c in SUIC_BIN if c in X.columns]
    S = pd.DataFrame({c: pd.to_numeric(X[c], errors="coerce") for c in sb}, index=X.index)
    cnt = pd.to_numeric(X[SUIC_COUNT], errors="coerce") if SUIC_COUNT in X.columns else pd.Series(np.nan, index=X.index)

    # cohort-balanced subsample: 500 MOST-COMPLETE (on continuous) per cohort
    keep = M.notna().any(axis=1)
    M, S, cnt, coh = M[keep], S[keep], cnt[keep], cohort[keep]
    obs_count = M.notna().sum(axis=1).values
    idx = []
    for c in ["bp", "sz", "dr"]:
        pool = np.where(coh.values == c)[0]
        take = min(args.n_per_cohort, len(pool))
        order = pool[np.argsort(-obs_count[pool], kind="stable")]
        idx.extend(order[:take])
    idx = np.sort(np.array(idx))
    M, S, cnt, coh = M.iloc[idx], S.iloc[idx], cnt.iloc[idx], coh.iloc[idx]
    return M, S, cnt, coh, cont, sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-per-cohort", type=int, default=500)
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--tune", type=int, default=800)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--ta", type=float, default=0.95)
    ap.add_argument("--min-group", type=int, default=10)
    ap.add_argument("--sleep", choices=["objective", "full"], default="objective",
                    help="sleep factor indicators: 'objective' = sleep-parameter items only (CANONICAL, "
                         "V3-7 — less affect-overlap); 'full' = all PSQI items (over-couples with affect)")
    args = ap.parse_args()
    if args.smoke:
        args.n_per_cohort, args.draws, args.tune, args.chains = 150, 80, 150, 2

    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    M, S, cnt, coh, cont, sb = prep(args)
    N, J = M.shape
    F = len(FACTORS)
    facidx = np.array([FACTORS.index(CONT[c][0]) for c in cont])
    onehot = np.zeros((J, F)); onehot[np.arange(J), facidx] = 1.0
    Mv = M.to_numpy()
    # observed-pattern groups (continuous block), filtered to common patterns
    patterns = {}
    for i in range(N):
        o = tuple(np.flatnonzero(~np.isnan(Mv[i])))
        if o:
            patterns.setdefault(o, []).append(i)
    patterns = {o: r for o, r in patterns.items() if len(r) >= args.min_group}
    n_drop = N - sum(len(r) for r in patterns.values())
    # suicidality observed cells (binary + count)
    Sv = S.to_numpy()
    cntv = cnt.to_numpy()
    print(f"N={N} · cont J={J} · {len(patterns)} cont-patterns (dropped {n_drop}) · "
          f"suic items={len(sb)}+count · cohorts={dict(coh.value_counts())}")

    with pm.Model() as model:
        # ---- marginalized Gaussian block (5 factors incl. affective) ----
        nu = pm.Normal("nu", 0.0, 1.0, shape=J)
        lam = pm.HalfNormal("lam", 0.6, shape=J)
        psi = pm.HalfNormal("psi", 1.0, shape=J)
        corr_raw = pm.LKJCorr("Phi", n=F, eta=2.0)
        corr = pt.tril(corr_raw, -1) + pt.tril(corr_raw, -1).T + pt.eye(F)
        Lam = lam[:, None] * pt.as_tensor(onehot)
        Sigma = Lam @ corr @ Lam.T + pt.diag(psi)
        ll = 0.0
        for o, rows in patterns.items():
            m, oi = len(o), list(o)
            Sel = np.zeros((m, J)); Sel[np.arange(m), oi] = 1.0
            St = pt.as_tensor(Sel)
            Lc = pt.linalg.cholesky(St @ Sigma @ St.T + 1e-6 * pt.eye(m))
            sol = pt.linalg.solve_triangular(Lc, (pt.as_tensor(Mv[np.ix_(rows, oi)]) - St @ nu).T, lower=True)
            ll = ll + (-0.5 * (m * np.log(2 * np.pi) + 2 * pt.log(pt.diag(Lc)).sum() + (sol ** 2).sum(axis=0))).sum()
        pm.Potential("cont_ll", ll)

        # ---- explicit-latent suicidality module (mixed likelihoods) ----
        Ds = pm.Normal("D_suic", 0.0, 1.0, shape=N)
        for k, col in enumerate(sb):
            obs = np.flatnonzero(~np.isnan(Sv[:, k]))
            a = pm.Normal(f"a_{col}", 0.0, 1.5)
            lj = pm.HalfNormal(f"lam_{col}", 0.8)
            pm.Bernoulli(f"y_{col}", logit_p=a + lj * Ds[obs], observed=Sv[obs, k].astype("int8"))
        oc = np.flatnonzero(~np.isnan(cntv))
        if len(oc) > 30:
            ac = pm.Normal("a_cnt", 0.0, 1.5); lc = pm.HalfNormal("lam_cnt", 0.8)
            alpha = pm.HalfNormal("alpha_cnt", 2.0)
            pm.NegativeBinomial("y_cnt", mu=pt.exp(ac + lc * Ds[oc]), alpha=alpha,
                                observed=np.rint(cntv[oc]).astype("int64"))

        idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains, cores=1,
                          target_accept=args.ta, random_seed=SEED, progressbar=False,
                          idata_kwargs={"log_likelihood": False})

    # ---- diagnostics ----
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    summ = az.summary(idata, var_names=["lam", "psi", "Phi", "D_suic"])
    rc = "r_hat" if "r_hat" in summ.columns else "rhat"
    ec = next((c for c in summ.columns if c.startswith("ess")), None)
    rhat_max = float(pd.to_numeric(summ[rc], errors="coerce").max())
    ess_min = float(pd.to_numeric(summ[ec], errors="coerce").min())
    converged = rhat_max < 1.05 and div == 0
    post = idata.posterior

    # ---- continuous loadings + Phi(5) ----
    lam_m = post["lam"].mean(("chain", "draw")).values
    Phi = post["Phi"].mean(("chain", "draw")).values
    Phi = np.tril(Phi, -1) + np.tril(Phi, -1).T + np.eye(F)
    psi_m = post["psi"].mean(("chain", "draw")).values
    nu_m = post["nu"].mean(("chain", "draw")).values
    # post-hoc 5-factor regression scores (Thomson) per pattern
    Lam_m = lam_m[:, None] * onehot
    Sig = Lam_m @ Phi @ Lam_m.T + np.diag(psi_m)
    Fsc = np.full((N, F), np.nan)
    for o, rows in patterns.items():
        oi = list(o)
        B = Phi @ Lam_m[oi].T @ np.linalg.pinv(Sig[np.ix_(oi, oi)])
        Fsc[rows] = (Mv[np.ix_(rows, oi)] - nu_m[oi]) @ B.T
    Ds_m = post["D_suic"].mean(("chain", "draw")).values

    # ---- 6x6 correlation (5 factors + suicidality), post-hoc on scores ----
    allsc = np.column_stack([Fsc, Ds_m])
    labels = FACTORS + ["suicidality"]
    C = pd.DataFrame(allsc, columns=labels).corr()

    # ---- MNAR diagnostic: does affective severity predict NON-completion? ----
    # NOTE: the most-complete subsample has ~no missingness in cognition/suicide by construction, so this
    # is usually NOT identifiable here — the proper MNAR test is the full-sample atlas (V3-2/V3-4). We
    # report it only when there is real variation.
    import statsmodels.api as sm
    aff = Fsc[:, FACTORS.index("affective")]
    cog_obs = (~np.isnan(Mv[:, facidx == FACTORS.index("cognition")])).any(axis=1).astype(int)
    suic_obs = (~np.isnan(Sv)).any(axis=1).astype(int)
    mnar = {}
    for nm, y in [("cognition", cog_obs), ("suicide_items", suic_obs)]:
        frac = float(y.mean())
        if min(frac, 1 - frac) < 0.02:
            mnar[nm] = f"not identifiable here ({frac:.0%} observed in the most-complete subsample) — see atlas V3-2/V3-4"
            continue
        ok = ~np.isnan(aff)
        try:
            r = sm.Logit(y[ok], sm.add_constant(aff[ok])).fit(disp=0)
            b, p = float(r.params.iloc[1]), float(r.pvalues.iloc[1])
            mnar[nm] = f"β={b:+.2f} (p={p:.3f}) — {'severity↓completion (MNAR)' if b < 0 else 'severity↑completion'}"
        except Exception:
            mnar[nm] = "fit failed"

    # ---- write ----
    OUT.mkdir(parents=True, exist_ok=True)
    load = pd.DataFrame({"indicator": cont, "factor": [CONT[c][0] for c in cont], "loading": np.round(lam_m, 2)})
    suic_load = pd.DataFrame({"indicator": sb, "factor": "suicidality",
                              "loading": [round(float(post[f"lam_{c}"].mean()), 2) for c in sb]})
    pd.concat([load, suic_load]).to_csv(OUT / "loadings.csv", index=False)
    C.round(3).to_csv(OUT / "phi.csv")
    sc = pd.DataFrame(allsc, columns=[f"F_{x}" for x in labels], index=M.index)
    sc.insert(0, "cohort", coh.values); sc.round(3).to_csv(OUT / "factor_scores.csv")
    try:
        idata.to_netcdf(str(OUT / "idata_ext.nc"))
    except Exception:
        pass

    suic_corr = C["suicidality"].drop("suicidality")
    md = ["# V3 extended measurement model (5 Gaussian factors + mixed-likelihood suicidality)", "",
          f"N={N} (500 most-complete/cohort), V0, no imputation. Marginalized Gaussian block "
          f"({len(patterns)} patterns, {n_drop} dropped) + explicit suicidality ({len(sb)} Bernoulli + 1 NB count).",
          f"NUTS {args.chains}×{args.draws}.", "",
          f"## Convergence — {'**CONVERGED**' if converged else '**NOT CONVERGED — provisional**'}",
          f"- max R-hat {rhat_max:.3f} · min ESS {ess_min:.0f} · divergences {div}", "",
          "## Factor correlation matrix (5 factors + suicidality)",
          C.round(2).to_markdown(), "",
          f"- mean |off-diag| (5 cont factors) = **{np.abs(Phi[np.triu_indices(F,1)]).mean():.2f}**",
          f"- **symptom (affective) ⊥ biology:** affective×metabolic {C.loc['affective','metabolic']:+.2f}, "
          f"affective×inflammatory {C.loc['affective','inflammatory']:+.2f}, "
          f"affective×cognition {C.loc['affective','cognition']:+.2f}, affective×sleep {C.loc['affective','sleep']:+.2f}",
          "- **suicidality correlations:** " + ", ".join(f"{k} {v:+.2f}" for k, v in suic_corr.items()), "",
          "## MNAR diagnostic (affective severity → P(items observed))",
          "\n".join(f"- {k}: {v}" for k, v in mnar.items()),
          "_(The full-sample MNAR analysis is the atlas, V3-2/V3-4: cognition partly BP-driven; robust"
          " informative-missingness in suicidality + self-reports.)_", "",
          "## Loadings", pd.concat([load, suic_load]).to_markdown(index=False), "",
          "Artifacts: `loadings.csv`, `phi.csv`, `factor_scores.csv`, `idata_ext.nc`."]
    (OUT / "extended_model.md").write_text("\n".join(md))

    print(f"\nmax R-hat {rhat_max:.3f} · min ESS {ess_min:.0f} · div {div} · "
          f"{'CONVERGED' if converged else 'NOT CONVERGED'}")
    print("\ncorrelation matrix:\n", C.round(2).to_string())
    print("\nsuicidality vs factors:\n", suic_corr.round(2).to_string())
    print("\nMNAR (affective->completion):", mnar)
    print("wrote:", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
