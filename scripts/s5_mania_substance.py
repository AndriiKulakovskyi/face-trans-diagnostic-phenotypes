#!/usr/bin/env python3
"""Adjudicate the deferred candidates — mania_activation + substance (§6), closing the M1 gap.

Adds both to the marginalized bifactor (G + cognition/metabolic/inflammatory/sleep), treating
substance's binary/count indicators (alcohol/cannabis lifetime SUD, cigarettes) as z-scored continuous
— an adjudication approximation that answers the §6 questions (distinct? identified? indicators load
≥0.30? reducible to G?). A proper mixed-likelihood re-test follows only if they survive. 2 seeds.

    python3 scripts/s5_mania_substance.py

Writes reports/10_mania_substance_report.md.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from face.models.bayesian.continuous_core import S1_FACTORS, prepare  # noqa: E402
from face.runner import quick_diag, sample_marginalized  # noqa: E402

REPORTS = REPO / "reports"
NEW = ["mania_activation", "substance"]
FACTORS = S1_FACTORS + NEW


def main(seeds: int = 2):
    rows = {f: [] for f in NEW}            # per-factor: list of (seed) loading/G/phiG dicts
    diags = []
    for si in range(seeds):
        prep = prepare(FACTORS, correlated=True, windows=False, force_factors_continuous=["substance"],
                       balanced=True, n_subsample=2000, seed=20260605 + si)
        idata = sample_marginalized(prep, label=f"mania+substance s{si+1}", step=f"[{si+1}/{seeds}] ",
                                    seed=20260605 + si, draws=600, tune=800, target_accept=0.92)
        d = quick_diag(idata); diags.append({"seed": f"s{si+1}", **{k: round(v, 3) if k == "rhat"
                                                                    else int(v) for k, v in d.items()}})
        Lam = idata.posterior["Lam"].mean(("chain", "draw")).values
        Phi = idata.posterior["Phi"].mean(("chain", "draw")).values
        col = {f: i for i, f in enumerate(prep.factor_cols)}
        g = col["overall_severity"]
        for f in NEW:
            its = [j for j, h in enumerate(prep.home) if h == f]
            prim = [abs(Lam[j, col[f]]) for j in its]                    # |primary loading| per item
            gload = [abs(Lam[j, g]) for j in its]                        # |loading on G| (bifactor)
            rows[f].append(dict(n_items=len(its), mean_primary=float(np.mean(prim)) if prim else np.nan,
                                min_primary=float(np.min(prim)) if prim else np.nan,
                                mean_G=float(np.mean(gload)) if gload else np.nan,
                                phi_max=float(np.max(np.abs([Phi[col[f], c] for c in range(len(prep.factor_cols))
                                                             if c not in (col[f], g)])))))
    # ---- verdict per factor ----
    def verdict(f):
        a = pd.DataFrame(rows[f])
        mp, mg = a.mean_primary.mean(), a.mean_G.mean()
        rhat_ok = all(pd.DataFrame(diags).rhat <= 1.05)
        if mp >= 0.30 and mg < mp and rhat_ok:
            return "confirmed", a, mp, mg
        if mp < 0.20 or not rhat_ok:
            return "rejected (weak/non-identified)", a, mp, mg
        return "weak / reducible-to-G", a, mp, mg

    md = ["# 10 — adjudication of the deferred candidates: mania_activation + substance (§6)", "",
          f"Added to the marginalized bifactor (G + cognition/metabolic/inflammatory/sleep), {seeds} seeds, "
          "N≈2,000 balanced. Substance's binary/count indicators (alcohol/cannabis SUD, cigarettes) treated "
          "as z-scored continuous (adjudication approximation). Verdict: distinct + identified + primary "
          "|λ| ≥ 0.30 + not reducible to G.", "",
          "## Convergence", pd.DataFrame(diags).to_markdown(index=False), ""]
    verdicts = {}
    for f in NEW:
        v, a, mp, mg = verdict(f); verdicts[f] = v
        md += [f"## {f} — **{v}**",
               f"- {int(a.n_items.iloc[0])} indicators · mean primary |λ| = **{mp:.2f}** (min "
               f"{a.min_primary.mean():.2f}) · mean |loading on G| = **{mg:.2f}** · max |Φ| with other "
               f"factors = {a.phi_max.mean():.2f}", ""]
    md += ["## Verdict (→ docs/ADJUDICATION.md)",
           f"- **mania_activation: {verdicts['mania_activation']}**",
           f"- **substance: {verdicts['substance']}**",
           "\n(Approximation note: substance's SUD/count items were modelled as continuous; a "
           "mixed-likelihood re-test is the proper confirmation only if a candidate survives here.)"]
    (REPORTS / "10_mania_substance_report.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nwrote reports/10_mania_substance_report.md")


if __name__ == "__main__":
    main()
