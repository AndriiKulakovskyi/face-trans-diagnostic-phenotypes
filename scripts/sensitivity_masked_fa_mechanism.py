"""WHY does the 6th varimax axis fail to reproduce imputation-free? (mechanism probe)

Companion to sensitivity_masked_fa.py. That script showed 5 of 6 axes reproduce on a
pairwise-complete (masked) correlation matrix, but the 6th (mean-fill: WURS/BIS/CTQ =
ADHD/impulsivity/trauma) does not — without imputation a socio-occupational/work-disability
factor occupies that slot instead.

Hypothesis: mean-filling missing cells with the (zero) standardized mean makes each Pearson
correlation a CO-OBSERVATION-WEIGHTED version of the true one. For standardized data filled
to 0,
        corr_fill(A,B)  ~=  O_AB * corr_masked(A,B),   O_AB = n_AB / sqrt(n_A n_B) <= 1,
so the matrix fed to the factor model is the true correlation matrix reweighted (Hadamard)
by the co-observation overlap O. Where O is block-structured (differential missingness), this
reshapes the covariance and can spawn/destroy the weakest factor — here, the 6th.

This script (1) verifies the identity across all domain pairs, (2) shows the overlap block
structure, (3) traces it to per-cohort missingness, and (4) inspects the affected axis's
domains directly. Read-only; writes results/sensitivity_masked_fa_mechanism.json.
Run:  python3 scripts/sensitivity_masked_fa_mechanism.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_DIR = REPO_ROOT / "results"
SCORES_PATH = RESULTS_DIR / "cluster_domains_scores.parquet"

# the two competing 6th-axis domain groups (from sensitivity_masked_fa.py output)
G_FILL = ["wurs", "bis", "ctq", "prism", "ess", "isf"]                     # mean-fill axis6 (ADHD/impuls./trauma)
G_MASK = ["hooccur_arret_travail_actuel", "hooccur_arret_travail",          # masked axis6 (work-disability)
          "stprof", "edulevel", "pregnn_rporres", "fagers"]


def main() -> int:
    sc = pd.read_parquet(SCORES_PATH)
    domains = list(sc.columns)
    p = len(domains)
    N = len(sc)
    cohort = pd.Index(sc.index.get_level_values("cohort").astype(str))

    M = sc.notna().to_numpy(float)            # N x p observation mask
    n = M.sum(0)                              # per-domain observed counts
    n_ab = M.T @ M                            # p x p co-observed counts
    O = n_ab / np.sqrt(np.outer(n, n))        # overlap factor, <= 1

    Rm = sc.corr().to_numpy(float)            # masked pairwise-complete correlation
    z = (sc - sc.mean()) / sc.std(ddof=0)
    Rf = np.corrcoef(z.fillna(0.0).to_numpy(np.float64), rowvar=False)  # mean-fill correlation

    iu = np.triu_indices(p, 1)
    obs = Rf[iu]
    pred = (O * Rm)[iu]                        # the identity's prediction
    # how well does corr_fill ~= O * corr_masked hold? (R^2 through the origin)
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum(obs ** 2))
    r2_identity = 1.0 - ss_res / ss_tot
    slope = float(np.sum(obs * pred) / np.sum(pred ** 2))
    corr_op = float(np.corrcoef(obs, pred)[0, 1])

    # naive (wrong) baseline: does corr_fill ~= corr_masked WITHOUT the overlap reweight?
    r2_naive = 1.0 - float(np.sum((obs - Rm[iu]) ** 2)) / ss_tot

    di = {d: i for i, d in enumerate(domains)}
    gf = [d for d in G_FILL if d in di]
    gm = [d for d in G_MASK if d in di]

    def block_overlap(group):
        idx = [di[d] for d in group]
        within = O[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
        rest = [j for j in range(p) if j not in idx]
        across = O[np.ix_(idx, rest)].ravel()
        return float(np.mean(within)), float(np.mean(across))

    wf, af = block_overlap(gf)
    wm, am = block_overlap(gm)

    # per-cohort observed fraction for the two groups -> reveals the missingness blocks
    obs_by_cohort = sc.notna().groupby(cohort).mean()
    cohorts = list(obs_by_cohort.index)

    # direct pair trace within the mean-fill 6th-axis group
    pair_rows = []
    for a in range(len(gf)):
        for b in range(a + 1, len(gf)):
            ia, ib = di[gf[a]], di[gf[b]]
            pair_rows.append({
                "pair": f"{gf[a]}~{gf[b]}",
                "n_co": int(n_ab[ia, ib]), "overlap_O": round(float(O[ia, ib]), 3),
                "corr_masked": round(float(Rm[ia, ib]), 3),
                "corr_fill": round(float(Rf[ia, ib]), 3),
                "O*masked": round(float(O[ia, ib] * Rm[ia, ib]), 3),
            })

    # ---- print ----
    print(f"matrix {N:,} x {p} | testing  corr_fill ~= O * corr_masked  (O = n_AB / sqrt(n_A n_B))")
    print(f"  identity fit:  R^2 = {r2_identity:.3f}  slope = {slope:.3f}  corr(obs,pred) = {corr_op:.3f}")
    print(f"  naive 'corr_fill ~= corr_masked' (no overlap reweight):  R^2 = {r2_naive:.3f}")
    print("  --> the mean-fill correlation IS the masked one reweighted by co-observation.\n")

    print("co-observation block structure (mean overlap O within group vs to the rest):")
    print(f"  mean-fill 6th axis  {gf}\n      within={wf:.3f}  to-rest={af:.3f}  ratio={wf/af:.2f}")
    print(f"  masked   6th axis  {gm}\n      within={wm:.3f}  to-rest={am:.3f}  ratio={wm/am:.2f}\n")

    print("per-cohort observed fraction (why O is block-structured):")
    hdr = "  domain".ljust(34) + "".join(f"{c:>8}" for c in cohorts) + "    all"
    print(hdr)
    for d in gf + gm:
        if d in obs_by_cohort.columns:
            row = "".join(f"{obs_by_cohort.loc[c, d]:8.2f}" for c in cohorts)
            print(f"  {d:<32}{row}{float(sc[d].notna().mean()):7.2f}")

    print("\nwithin-group pairs of the mean-fill 6th axis (corr_fill vs masked):")
    for r in pair_rows:
        print(f"  {r['pair']:<16} n_co={r['n_co']:>5} O={r['overlap_O']:.2f}  "
              f"masked={r['corr_masked']:+.2f}  fill={r['corr_fill']:+.2f}  O*masked={r['O*masked']:+.2f}")

    out = {
        "identity_r2": round(r2_identity, 4), "identity_slope": round(slope, 4),
        "identity_corr_obs_pred": round(corr_op, 4), "naive_r2_no_overlap": round(r2_naive, 4),
        "fill_axis_overlap_within": round(wf, 4), "fill_axis_overlap_to_rest": round(af, 4),
        "mask_axis_overlap_within": round(wm, 4), "mask_axis_overlap_to_rest": round(am, 4),
        "cohorts": cohorts,
        "obs_fraction_by_cohort": {d: {c: round(float(obs_by_cohort.loc[c, d]), 3) for c in cohorts}
                                   for d in gf + gm if d in obs_by_cohort.columns},
        "within_group_pairs": pair_rows,
        "interpretation": ("corr_fill = O (.) corr_masked: mean-fill reweights every correlation "
                           "by co-observation overlap. Block-structured missingness therefore "
                           "reshapes the FA input and can manufacture/destroy the weakest factor."),
    }
    (RESULTS_DIR / "sensitivity_masked_fa_mechanism.json").write_text(json.dumps(out, indent=2))
    print("\nWrote results/sensitivity_masked_fa_mechanism.json. Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
