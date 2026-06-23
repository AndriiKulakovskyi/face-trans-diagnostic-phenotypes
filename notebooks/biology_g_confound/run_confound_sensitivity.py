#!/usr/bin/env python3
"""Biology⊥G confound-sensitivity: is the metabolic/inflammatory ⊥ severity headline a real biological
signal, or an artifact of medication / adiposity / site?

The project's load-bearing claim is that metabolic and inflammatory load are the *least* severity-
entangled domains (Φ(G, metabolic) ≈ 0.12, Φ(G, inflammatory) ≈ 0.07, vs 0.39 cognition / 0.42 sleep) —
the reason the map is "biology-aware" rather than a re-dressed severity scale. A reviewer's hardest
critique is that this biology axis is really a proxy for *who is on antipsychotics* (which cause metabolic
syndrome), for *body weight*, or for *which site/platform ran the bloodwork*. This script re-derives
Φ(G, ·) on the same correlated-G marginalized model (``scripts/10``/``s5_corrg``) under a ladder of
adjustments, partialling each continuous item (FWL) on progressively more confounders:

    A0  unadjusted                                  the reported 0.12 / 0.07
    A1  + age(spline) + sex + edu + site            reproduces scripts/10 (demographics + site)
    A2  + antipsychotic exposure                    HEADLINE — is biology a medication signal?
    A3  + BMI (BMI moved to the covariate block)    EXPLORATORY / partly circular (BMI is itself a
                                                     metabolic indicator); is metabolic-G just adiposity?

Verdict logic (conservative, on A2): if metabolic and inflammatory stay the least G-entangled
(Φ small and below cognition/sleep) the headline is confound-robust; if they inflate, the claim is
honestly downgraded to "a medication/adiposity-linked biological axis".

Honest limits recorded in the report: antipsychotic coverage ~54 % (NaN mean-imputed for the design, BP
lifetime vs SZ/DR current); adjusting for antipsychotic is conservative-to-over-conservative (it is on the
causal path to metabolic load, not only a confounder); a site dummy is coarser than full assay/batch
harmonization; A3 is partly circular. Internal sensitivity, not external validation.

    PYTHONPATH=$PWD/src python notebooks/biology_g_confound/run_confound_sensitivity.py --smoke
    PYTHONPATH=$PWD/src python notebooks/biology_g_confound/run_confound_sensitivity.py            # real
    # detached:  python scripts/run_job.py biology_g_confound -- \
    #              python -u notebooks/biology_g_confound/run_confound_sensitivity.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from face.confirm import corr_no_g_prep  # noqa: E402
from face.io import manifest, progress  # noqa: E402
from face.models.bayesian.continuous_core import S1_FACTORS, prepare  # noqa: E402
from face.runner import quick_diag, sample_marginalized  # noqa: E402

REPORTS = REPO / "reports"
OUT = REPO / "results" / "face" / "biology_g_confound"
# (arm label, covariate_adjust, extra confounder columns beyond age/sex/edu/site)
ARMS = [
    ("A0_unadjusted", False, ()),
    ("A1_demo_site", True, ()),
    ("A2_antipsychotic", True, ("on_antipsychotic",)),
    ("A3_bmi", True, ("on_antipsychotic", "bmi")),
]
SPECS = [f for f in S1_FACTORS if f != "overall_severity"]


def _fit_arm(arm, adjust, extra, n, seeds, draws, tune, chains):
    import arviz as az
    cache = OUT / arm
    cache.mkdir(parents=True, exist_ok=True)
    phis, diags = {}, []
    for i in range(seeds):
        seed = 20260623 + i
        nc = cache / f"s{i + 1}_n{n}.nc"                # N-aware cache key
        progress.heartbeat(stage=f"{arm} seed {i + 1}/{seeds}", frac=None, msg=f"{arm}, seed {i + 1}")
        if nc.exists():
            idata = az.from_netcdf(str(nc))
            print(f"  [cached] {arm} seed {i + 1}", flush=True)
        else:
            base = prepare(S1_FACTORS, correlated=True, windows=False, g_correlated=True,
                           balanced=True, n_subsample=n, seed=seed,
                           covariate_adjust=adjust, covariate_extra_cols=extra)
            prep = corr_no_g_prep(base)
            idata = sample_marginalized(prep, draws=draws, tune=tune, chains=chains, seed=seed,
                                        target_accept=0.92, label=f"confound-{arm} s{i + 1}",
                                        step=f"[{arm} {i + 1}/{seeds}] ")
            try:
                idata.to_netcdf(str(nc))
            except Exception:
                pass
            d = quick_diag(idata)
            manifest.write_manifest(f"confound_{arm}_s{i + 1}", out_dir=cache, N=base.M.shape[0],
                                    index=base.index, cohort=base.cohort, seed=seed,
                                    diagnostics={k: float(v) for k, v in d.items()},
                                    extra={"arm": arm, "covariate_adjust": adjust,
                                           "extra_cols": list(extra)})
        fcols = ["overall_severity"] + SPECS
        Phi = idata.posterior["Phi"].mean(("chain", "draw")).values
        g = fcols.index("overall_severity")
        phis[i] = {fcols[c]: float(Phi[g, c]) for c in range(len(fcols)) if c != g}
        d = quick_diag(idata)
        diags.append({"arm": arm, "seed": f"s{i + 1}",
                      **{k: round(v, 3) if k == "rhat" else int(v) for k, v in d.items()}})
        print(f"    → {diags[-1]} · Φ(G,·) {({k: round(v, 3) for k, v in phis[i].items()})}", flush=True)
    return phis, diags


def main(n=2000, seeds=2, draws=600, tune=800, chains=2, smoke=False):
    if smoke:
        n, seeds, draws, tune, chains = 500, 1, 150, 200, 2
    print(f"biology⊥G confound sensitivity: N≈{n} balanced · {seeds} seed(s) · {len(ARMS)} arms\n", flush=True)
    arm_phi, all_diags = {}, []
    for arm, adjust, extra in ARMS:
        phis, diags = _fit_arm(arm, adjust, extra, n, seeds, draws, tune, chains)
        arm_phi[arm] = {f: float(np.mean([phis[i][f] for i in range(seeds)])) for f in SPECS}
        all_diags += diags

    arm_names = [a for a, *_ in ARMS]
    rows = [dict(domain=f, **{a: round(arm_phi[a][f], 3) for a in arm_names}) for f in SPECS]
    tab = pd.DataFrame(rows).sort_values(arm_names[-1])
    REPORTS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(REPORTS / "12_biology_g_confound.csv", index=False)

    # verdict on A2 (antipsychotic — the conservative headline arm)
    head = "A2_antipsychotic"
    bio = {f: arm_phi[head][f] for f in ("metabolic", "inflammatory")}
    ref = min(arm_phi[head]["cognition"], arm_phi[head]["sleep"])
    survives = bool(max(abs(v) for v in bio.values()) < 0.15 and max(bio.values()) < ref)
    a3_ok = all(d["rhat"] <= 1.05 for d in all_diags if d["arm"] == "A3_bmi")

    md = ["# 12 — biology⊥G confound sensitivity (medication / adiposity / site)", "",
          f"Correlated-G marginalized model, N≈{n} balanced, {seeds} seed(s). Each continuous item is "
          "partialled (FWL) on a growing covariate design before the factor model; Φ(G,·) compares the "
          "G-entanglement of each domain across the adjustment ladder.", "",
          "| arm | adjusts for |", "|---|---|",
          "| A0_unadjusted | nothing (the reported value) |",
          "| A1_demo_site | age(spline) + sex + education + site |",
          "| A2_antipsychotic | A1 + antipsychotic exposure **(conservative headline)** |",
          "| A3_bmi | A2 + BMI moved to the covariate block **(exploratory / partly circular)** |", "",
          "## Φ(G, domain) across the adjustment ladder", tab.to_markdown(index=False), "",
          "## Convergence", pd.DataFrame(all_diags).to_markdown(index=False), "",
          "## Verdict (on A2 — antipsychotic-adjusted)",
          (f"- **Biology⊥G survives medication + site adjustment**: metabolic "
           f"{bio['metabolic']:+.3f} and inflammatory {bio['inflammatory']:+.3f} remain the least "
           f"severity-entangled domains (both below cognition/sleep, min {ref:.3f}). The headline is "
           "confound-robust." if survives else
           f"- **⚠ Biology⊥G shifts under adjustment**: metabolic {bio['metabolic']:+.3f}, inflammatory "
           f"{bio['inflammatory']:+.3f}. Reframe the claim as a medication/adiposity-linked biological axis."),
          (f"- A3 (BMI-as-covariate) {'converged (metabolic still identified on its non-BMI indicators)' if a3_ok else 'shows convergence trouble — read as a circularity/identification flag, not a result'}; "
           "it is exploratory because BMI is itself a metabolic indicator."), "",
          "## Honest limits",
          "- Antipsychotic coverage ~54 % (NaN mean-imputed for the design; BP lifetime vs SZ/DR current).",
          "- Antipsychotic is on the causal path to metabolic load, so adjusting for it is "
          "conservative-to-over-conservative (it can remove real signal).",
          "- A site dummy is coarser than full cross-platform assay/batch harmonization.",
          "- Internal sensitivity on the correlated-G measurement structure; not external validation.", ""]
    (REPORTS / "12_biology_g_confound_report.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nwrote reports/12_biology_g_confound_report.md (+ .csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--tune", type=int, default=800)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(n=a.n, seeds=a.seeds, draws=a.draws, tune=a.tune, chains=a.chains, smoke=a.smoke)
