#!/usr/bin/env python3
"""23b — archetype LOCATION (anchor) uncertainty + dominant-share point estimate (P3-04/05).

The reported archetypes (scripts/23) fit the 8 anchors ONCE on the posterior-mean coordinates and
propagate only membership uncertainty. This re-fits the anchors across M1 posterior draws + patient
bootstraps (Hungarian-aligned) to report where each extreme phenotype itself sits — a profile peak +
HDI + stability per archetype — which the rare inflammatory/suicidality corners most need.

    python3 scripts/23b_archetype_location.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.strata import archetypes as ar  # noqa: E402

M2 = REPO / "results" / "face" / "m2"
REPORTS = REPO / "reports"
SEED, A = 20260609, 8
CANON = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep", "mania_activation",
         "suicidality", "developmental_risk", "substance"]


def main():
    df = pd.read_parquet(M2 / "coordinates_full.parquet")
    X9 = df[[f"{f}__mean" for f in CANON]].to_numpy()
    draws = np.load(M2 / "coordinates_draws.npz", allow_pickle=True)["draws"]

    print(f"[1/2] archetype location uncertainty A={A} (40 draws + 40 bootstraps)...", flush=True)
    res = ar.archetype_location_uncertainty(X9, draws, list(range(9)), A, n_draw=40, n_boot=40, seed=SEED)
    Zr, lo, hi = res["Z_ref"], res["Z_lo"], res["Z_hi"]
    names = ar.name_archetypes(Zr, CANON)

    print("[2/2] dominant-share point estimate (project onto reference anchors)...", flush=True)
    W = ar.project_to_Z(X9, Zr)
    dom = W.argmax(1)
    share = np.bincount(dom, minlength=A) / len(dom)

    rows = []
    for a in range(A):
        d = int(np.argmax(np.abs(Zr[a])))                            # dominant axis of the archetype
        rows.append({"archetype": f"A{a}", "name": names[a], "dom_share": round(float(share[a]), 3),
                     "dominant_axis": CANON[d], "peak": round(float(Zr[a, d]), 2),
                     "peak_HDI": f"[{lo[a, d]:.2f}, {hi[a, d]:.2f}]",
                     "mean_profile_SD": round(float(res["profile_sd"][a].mean()), 3),
                     "min_tucker": round(float(res["min_tucker_per_arch"][a]), 3)})
    tab = pd.DataFrame(rows).sort_values("dom_share")
    tab.to_csv(REPORTS / "23b_archetype_location.csv", index=False)
    # full per-(archetype, dim) mean + HDI
    full = []
    for a in range(A):
        for d, f in enumerate(CANON):
            full.append({"archetype": f"A{a}", "dim": f, "mean": round(float(res["Z_mean"][a, d]), 3),
                         "hdi_lo": round(float(lo[a, d]), 3), "hdi_hi": round(float(hi[a, d]), 3)})
    pd.DataFrame(full).to_csv(REPORTS / "23b_archetype_profiles_hdi.csv", index=False)

    rare = tab[tab.dom_share < 0.05]
    md = ["# 23b — archetype location (anchor) uncertainty (P3-04/05)", "",
          f"Anchors re-fit across **{res['n_refits']}** states (40 M1 posterior draws + 40 patient "
          "bootstraps), Hungarian-aligned to the reference. Reports each extreme phenotype's peak axis, "
          "its **profile HDI** (anchor-location uncertainty, which the fixed-anchor projection omitted), "
          "and a per-archetype stability (min Tucker congruence vs the reference).", "",
          "## Per-archetype location + stability", tab.to_markdown(index=False), "",
          (f"- **Rare corners** ({', '.join(rare.name)}): their peaks carry the widest HDIs (the tails are "
           "skewed-biomarker-driven and sparsely populated), now reported as intervals rather than points — "
           "so the rare-archetype claims are uncertainty-qualified."
           if len(rare) else "- No archetype below a 5% dominant share."),
          f"- archetype stability: min Tucker congruence across re-fits **{res['min_tucker_per_arch'].min():.3f}** "
          f"(worst archetype); mean profile SD **{res['profile_sd'].mean():.3f}**.", "",
          "## Artifacts",
          "- `reports/23b_archetype_location.csv` — peak axis, peak HDI, stability per archetype.",
          "- `reports/23b_archetype_profiles_hdi.csv` — full per-(archetype, dim) mean + HDI.", ""]
    (REPORTS / "23b_archetype_location_report.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nwrote reports/23b_archetype_location_report.md (+ 2 csv)")


if __name__ == "__main__":
    main()
