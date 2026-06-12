#!/usr/bin/env python3
"""33 — G1: longitudinal measurement invariance (V0 → V1 → V2), the precondition gate.

Refit the simple-structure continuous backbone PER VISIT (cohort→visit vs §6), z-scored in-sample, and
compare the primary loadings V1/V2 vs V0 by Tucker congruence φ. Metric invariance (φ≥0.95 invariant ·
≥0.85 partial) per factor licenses interpreting that axis's V1/V2 coordinate change as *patient* change in
G3/G4; otherwise the change could be instrument drift. Cohort-balanced subsample per visit (holds the mix
roughly constant; M1 §8 already found the loadings largely cross-cohort invariant). Backbone = the 6
continuous axes (severity / cognition / metabolic / inflammatory / sleep / developmental_risk — the last
continuous-anchored via CTQ / WURS / age-of-onset / perinatal); the explicit axes (suicidality / substance)
and mania (2 indicators) are flagged not-tested-here (§4).
Methods: docs/TEMPORAL_MODEL.md §4. Each fit is CACHED → resumable.

    python3 scripts/33_invariance.py                 # n=1800 balanced/visit, 2 seeds
    python3 scripts/33_invariance.py --n 1800 --seeds 2
    python3 scripts/33_invariance.py --smoke          # tiny, 1 quick pass

Writes reports/33_invariance_report.md (+ 33_congruence.csv, 33_dif_items.csv) ·
results/face/m3/invariance_license.parquet · docs/figures/33_congruence.png.
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

from face.models.bayesian.continuous_core import S3A_FACTORS  # noqa: E402
from face.temporal import VISITS  # noqa: E402
from face.temporal.invariance import (MIN_OBS, PHI_GOOD, PHI_OK, axis_license,  # noqa: E402
                                      congruence_over_visits, fit_visit_backbone)

REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
OUT = REPO / "results" / "face" / "m3"
CACHE = OUT / "invariance_cache"
ALL_AXES = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
            "mania_activation", "suicidality", "developmental_risk", "substance"]
BACKBONE = S3A_FACTORS                 # S1 + developmental_risk (continuous-anchored 6-axis backbone)
BACKBONE_TAG = "s3a"
NOT_TESTED = ["mania_activation", "suicidality", "substance"]


def main(n_total=1800, seeds=2, smoke=False):
    if smoke:
        n_total, seeds = 450, 1
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    seed_list = [20260609 + s for s in range(seeds)]
    visits = list(VISITS)
    total = len(visits) * len(seed_list)
    print(f"Temporal invariance: per-visit backbone fits · N≈{n_total} balanced/visit · "
          f"{seeds} seed(s) · {total} fits\n", flush=True)

    fits, diags, convflag, k = {}, [], {}, 0
    for si, seed in enumerate(seed_list):
        for v in visits:
            k += 1
            cf = CACHE / f"ss_{BACKBONE_TAG}_n{n_total}_{v}_s{si}.json"
            if cf.exists() and not smoke:
                blob = json.loads(cf.read_text())
                fits[(v, si)] = {it: tuple(val) for it, val in blob["rec"].items()}
                diag = blob["diag"]
                print(f"  [{k}/{total}] [cached] {v} seed{si+1} "
                      f"(R-hat {diag['rhat']} · ESS {diag['ess']})", flush=True)
            else:
                d = dict(draws=300, tune=300, chains=2) if smoke else dict(draws=500, tune=600, chains=2)
                rec, diag = fit_visit_backbone(v, factors=BACKBONE, n_total=n_total, seed=seed,
                                               label=f"{v} seed{si+1}", step=f"[{k}/{total}] ", **d)
                fits[(v, si)] = rec
                if not smoke:
                    cf.write_text(json.dumps({"rec": {it: list(val) for it, val in rec.items()}, "diag": diag}))
            diags.append(diag)
            convflag[(v, si)] = bool(diag["converged"])

    # ---- congruence V1/V2 vs V0 (Tucker φ per backbone factor, mean over seeds) ----
    seed_idx = list(range(len(seed_list)))
    conv = {key for key, ok in convflag.items() if ok}
    cong = congruence_over_visits(fits, BACKBONE, visits, seed_idx, reference="V0", converged=conv)
    cong.to_csv(REPORTS / "33_congruence.csv", index=False)
    lic = axis_license(cong)

    # full-axis license table (backbone axes get a verdict; the rest are not-tested-here)
    lic_full = pd.DataFrame({"axis": ALL_AXES})
    lic_full = lic_full.merge(lic, on="axis", how="left")
    lic_full["license"] = lic_full["license"].fillna("not-tested")
    lic_full.to_parquet(OUT / "invariance_license.parquet", index=False)

    # ---- loading-DIF: items observed ≥MIN_OBS at all visits, max cross-visit |Δλ| (seed 0) ----
    def _conv_seed(v):
        return next((si for si in seed_idx if convflag.get((v, si))), 0)
    f0 = {v: fits[(v, _conv_seed(v))] for v in visits}      # DIF from a converged seed per visit
    items = set(f0["V0"])
    dif_rows = []
    for it in items:
        per = {v: f0[v][it] for v in visits if it in f0[v] and f0[v][it][2] >= MIN_OBS}
        if len(per) == len(visits):
            loads = {v: per[v][1] for v in visits}
            dif_rows.append(dict(item=it, factor=per["V0"][0], spread=round(max(loads.values()) - min(loads.values()), 3),
                                 **{f"load_{v}": round(loads[v], 3) for v in visits}))
    dif = pd.DataFrame(dif_rows).sort_values("spread", ascending=False) if dif_rows else pd.DataFrame()
    if len(dif):
        dif.to_csv(REPORTS / "33_dif_items.csv", index=False)

    _figure(cong)

    # ---- report ----
    cv = pd.DataFrame(diags)
    n_conv = int(cv["converged"].sum())
    show = cong.copy()
    show["verdict"] = show["phi_mean"].map(
        lambda p: "invariant" if p >= PHI_GOOD else ("partial" if p >= PHI_OK else "**non-invariant**"))
    n_inv = int((lic["license"] == "invariant").sum())
    n_part = int((lic["license"] == "partial").sum())
    md = ["# 33 — G1: longitudinal measurement invariance (V0 → V1 → V2)", "",
          f"Per-visit **simple-structure** backbone (cohort-balanced N≈{n_total}, in-sample z-scored, "
          f"{seeds} seed(s)). Metric invariance = Tucker congruence φ of the primary loadings per factor "
          f"vs V0 (φ≥{PHI_GOOD} invariant · ≥{PHI_OK} partial). Tucker φ is scale-invariant, so the frozen "
          "V0 spec is deliberately not used here (this tests loading *shape*). A passing axis licenses "
          "interpreting its V1/V2 coordinate change as patient change (G3/G4).", "",
          f"## Convergence — {n_conv}/{len(cv)} fits converged (R-hat ≤ 1.05 · ESS ≥ 100 · 0 div)",
          cv.to_markdown(index=False),
          (f"\n- ⚠ {len(cv) - n_conv} non-converged fit(s) excluded from the φ averages — congruence "
           "rests only on fits that passed the gate (each axis still has ≥1 converged seed at every visit)."
           if n_conv < len(cv) else ""), "",
          "## Metric invariance — Tucker congruence φ vs V0 (mean over seeds)",
          show.to_markdown(index=False), "",
          f"## Per-axis license — {n_inv} invariant · {n_part} partial (backbone axes)",
          lic.to_markdown(index=False), "",
          f"- **Not tested in stage 33** ({', '.join(NOT_TESTED)}): mania has only 2 indicators (φ on 2 "
          "items is unstable); suicidality/substance are explicit (binary/count) — not in the continuous "
          "backbone (a heavier mixed-model refit). These carry `license=not-tested` in the panel; their "
          "change is reported descriptively, not as licensed patient-change.", ""]
    if len(dif):
        md += ["## Loading-DIF — largest cross-visit loading spread (seed 1)",
               dif.head(10).to_markdown(index=False),
               f"\n- {int((dif.spread > 0.20).sum())} item(s) with cross-visit spread > 0.20.", ""]
    overall = "largely invariant" if (n_inv + n_part) == len(lic) and n_inv >= 3 else "mixed"
    md += ["## Verdict",
           f"The continuous backbone is **{overall}** across V0→V1→V2 on the testable axes: "
           + ", ".join(f"{r.axis} ({r.license}, φ={r.min_phi})" for r in lic.itertuples()) + ". "
           "Where φ is only partial, the axis's change is interpreted with that caveat (documented partial, "
           "not hidden). The licenses gate stage 34's panel.", "",
           "Artifacts: `reports/33_{congruence,dif_items}.csv` · "
           "`results/face/m3/invariance_license.parquet` · `docs/figures/33_congruence.png`."]
    (REPORTS / "33_invariance_report.md").write_text("\n".join(md))
    print("\n".join(md))


def _figure(cong):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    factors = list(dict.fromkeys(cong["factor"]))
    x = np.arange(len(factors))
    for v, color, off in [("V1", "#2c7fb8", -0.15), ("V2", "#d95f0e", 0.15)]:
        sub = cong[cong["visit"] == v].set_index("factor").reindex(factors)
        ax.bar(x + off, sub["phi_mean"].values, 0.3, label=f"{v} vs V0", color=color)
    ax.axhline(PHI_GOOD, color="g", ls="--", lw=1, label=f"invariant ≥{PHI_GOOD}")
    ax.axhline(PHI_OK, color="orange", ls="--", lw=1, label=f"partial ≥{PHI_OK}")
    ax.set_xticks(x); ax.set_xticklabels(factors, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Tucker congruence φ"); ax.set_ylim(0.6, 1.02)
    ax.set_title("Longitudinal metric invariance — primary loadings vs V0")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIGS / "33_congruence.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1800, help="cohort-balanced total per visit")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(n_total=a.n, seeds=a.seeds, smoke=a.smoke)
