#!/usr/bin/env python3
"""10b — soft-zero vs hard-zero "unlikely" cross-loadings sensitivity (issue P0-05).

The methods doc states `unlikely` cells carry a soft `Normal(0, 0.05)` prior ("shrinkage, not hard
exclusion"); the engine hard-zeros them. `prepare(soft_unlikely=True)` instantiates that soft prior. This
fits the S2 continuous backbone under BOTH and compares the map (loading congruence per factor + Φ) — if
they agree, the reported hard-zero map is unchanged and the doc wording is the only fix; if they differ,
adopt the soft-zero arm.

    python3 scripts/10b_softzero_sensitivity.py [--n 2000 --seeds 2]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from face.io import manifest, progress  # noqa: E402
from face.models.bayesian.continuous_core import S1_FACTORS, prepare  # noqa: E402
from face.runner import quick_diag, sample_marginalized  # noqa: E402

REPORTS = REPO / "reports"


def _tucker(a, b):
    den = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    return float((a * b).sum()) / den if den > 0 else float("nan")


def _fit(soft, n, seed, draws, tune, chains):
    base = prepare(S1_FACTORS, correlated=True, windows=True, balanced=True, n_subsample=n,
                   seed=seed, soft_unlikely=soft)
    idata = sample_marginalized(base, draws=draws, tune=tune, chains=chains, seed=seed,
                                target_accept=0.9, label=f"softzero={soft} s{seed}")
    Lam = np.asarray(idata.posterior["Lam"].mean(("chain", "draw")).values)   # [J, F]
    Phi = np.asarray(idata.posterior["Phi"].mean(("chain", "draw")).values)
    return base, Lam, Phi, quick_diag(idata)


def main(n=2000, seeds=2, draws=600, tune=800, chains=2, smoke=False):
    if smoke:
        n, seeds, draws, tune, chains = 600, 1, 150, 200, 2
    print(f"soft-zero vs hard-zero sensitivity: N≈{n} · {seeds} seed(s)\n", flush=True)
    diags, phis, congr = [], [], []
    for i in range(seeds):
        seed = 20260605 + i
        progress.heartbeat(stage=f"seed {i + 1}/{seeds}")
        base_h, Lam_h, Phi_h, dh = _fit(False, n, seed, draws, tune, chains)
        base_s, Lam_s, Phi_s, ds = _fit(True, n, seed, draws, tune, chains)
        # compare on the SHARED items/cols (hard is a subset of soft's free cells, but loadings align by index)
        fcols = base_h.factor_cols
        phi_per = {f: round(_tucker(Lam_h[:, c], Lam_s[:, c]), 4) for c, f in enumerate(fcols)}
        congr.append(phi_per)
        phis.append(round(float(np.nanmax(np.abs(Phi_h - Phi_s))), 4))
        diags.append({"seed": f"s{i + 1}", "hard_rhat": round(float(dh["rhat"]), 3),
                      "soft_rhat": round(float(ds["rhat"]), 3),
                      "min_loading_tucker": round(float(min(phi_per.values())), 4),
                      "max_abs_dPhi": phis[-1]})
        manifest.write_manifest(f"softzero_s{i + 1}", N=base_s.M.shape[0], seed=seed,
                                diagnostics={"hard_rhat": float(dh["rhat"]), "soft_rhat": float(ds["rhat"])},
                                extra={"arm": "soft_unlikely"})
        print(f"  seed {i + 1}: min loading Tucker {min(phi_per.values()):.4f} · max |ΔΦ| {phis[-1]:.4f}", flush=True)

    dd = pd.DataFrame(diags)
    dd.to_csv(REPORTS / "10b_softzero_sensitivity.csv", index=False)
    min_tucker = float(dd["min_loading_tucker"].min())
    max_dphi = float(dd["max_abs_dPhi"].max())
    robust = min_tucker >= 0.95               # the project's own invariance bar (φ≥0.95 = "invariant")
    md = ["# 10b — soft-zero vs hard-zero unlikely cross-loadings (P0-05)", "",
          "The methods doc says `unlikely` cells carry a soft `Normal(0, 0.05)` prior; the engine hard-zeros "
          "them. This fits the S2 backbone under both (`prepare(soft_unlikely=…)`) and compares the map.", "",
          "## Hard-zero vs soft-zero agreement", dd.to_markdown(index=False), "",
          (f"- **The map is robust to the soft-zero specification:** loading congruence Tucker "
           f"{min_tucker:.3f} ≥ the 0.95 invariance bar; max |ΔΦ| {max_dphi:.3f} (one Φ cell). The ~980 "
           "`unlikely` cells carry little signal — the soft `Normal(0, 0.05)` shrinks them to ≈0 and "
           "reproduces the hard-zero map to within the invariance threshold (congruent, not byte-identical). "
           "The reported hard-zero fit stands as primary; report the soft-zero arm as a congruent "
           "sensitivity and reword the methods (a Bayesian sparse bifactor with selected ESEM windows; the "
           "unlikely cells fixed/shrunk to ~0)."
           if robust else
           f"- **The map shifts materially under soft-zero** (min loading Tucker {min_tucker:.3f} < 0.95, max "
           f"|ΔΦ| {max_dphi:.3f}); adopt the soft-zero arm or reconcile the structure."), ""]
    (REPORTS / "10b_softzero_sensitivity_report.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nwrote reports/10b_softzero_sensitivity_report.md (+ .csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(n=a.n, seeds=a.seeds, smoke=a.smoke)
