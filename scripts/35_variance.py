#!/usr/bin/env python3
"""35 — G3: trait vs state decomposition (the first scientific headline).

For each of the 9 axes, decompose the longitudinal coordinate into between-patient *trait* variance, genuine
within-person *state* variance, and the KNOWN M1 measurement variance (plugged, not estimated — so a
low-reliability axis can't masquerade as state). Trait ratio ICC = σ²_b/(σ²_b+σ²_w); §5 verdict bands
trait ≥0.6 / state ≤0.4 (CI clearing 0.5) / mixed. Tests the §1.4 prediction: the **spine (severity) most
state-like**, the **biology corners (cognition/metabolic/developmental) most trait-like** — the variance
route to the headline that G4 (stage 36) will corroborate geometrically. Reports all-available AND
completers (the survivorship sensitivity, conditioned on G6) + a raw-ICC triangulation. Methods:
docs/TEMPORAL_MODEL.md §5.

    python3 scripts/35_variance.py

Writes reports/35_variance_report.md (+ 35_trait_state.csv) · docs/figures/35_trait_state.png.
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

from face.temporal import CANON  # noqa: E402
from face.temporal.variance import decompose, patient_patterns, raw_icc  # noqa: E402

M3 = REPO / "results" / "face" / "m3"
REPORTS, FIGS = REPO / "reports", REPO / "docs" / "figures"
# the §1.4 geometric prediction (spine state; biology corners trait; the rest intermediate)
PREDICTED = {"overall_severity": "state", "mania_activation": "state", "suicidality": "state",
             "cognition": "trait", "developmental_risk": "trait", "metabolic": "trait",
             "inflammatory": "mixed", "sleep": "mixed", "substance": "mixed"}


def main():
    REPORTS.mkdir(exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(M3 / "panel_coords.parquet")
    lic = {ax: panel[f"{ax}__license"].iloc[0] for ax in CANON}
    patterns = patient_patterns(panel)
    n_comp = len(patterns.get((0, 1, 2), []))
    n_multi = sum(len(v) for k, v in patterns.items() if len(k) >= 2)

    print(f"[35] trait/state decomposition · {n_multi:,} multi-visit patients drive the split "
          f"({n_comp:,} completers) · 9 axes × (all-available + completers)\n", flush=True)
    allv = decompose(panel, CANON, patterns)                              # all data, no completeness selection
    comp = decompose(panel, CANON, patterns, keys=[(0, 1, 2)])            # completers sensitivity
    rawicc = raw_icc(panel, CANON)

    allv["license"] = allv["axis"].map(lic)
    allv["predicted"] = allv["axis"].map(PREDICTED)
    allv["icc_completers"] = allv["axis"].map(comp.set_index("axis")["icc"])
    allv["icc_raw"] = allv["axis"].map(rawicc)
    allv = allv.sort_values("icc").reset_index(drop=True)
    allv.to_csv(REPORTS / "35_trait_state.csv", index=False)
    _figure(allv)

    # §1.4 scorecard on the LICENSED axes (where the trait/state claim is measurement-backed)
    licensed = allv[allv["license"].isin(["invariant", "partial"])]
    match = licensed[licensed["verdict"] == licensed["predicted"]]
    spine = allv[allv.axis == "overall_severity"].iloc[0]
    corners = allv[allv.axis.isin(["cognition", "metabolic", "developmental_risk"])]

    md = ["# 35 — G3: trait vs state decomposition (the first headline)", "",
          "Per axis: between-patient **trait** σ²_b vs within-person **state** σ²_w, with the known M1 "
          "measurement variance **plugged** (so reliability can't be mistaken for state). "
          f"ICC = σ²_b/(σ²_b+σ²_w); trait ≥{0.6} · state ≤{0.4} (94% HDI clearing 0.5) · else mixed. "
          f"{n_multi:,} multi-visit patients drive the split; all data used (no completeness selection).", "",
          "## Trait/state profile — axes ordered by ICC (state → trait)", "",
          "`license`: G1 status (trait/state claim is measurement-backed for invariant/partial; the "
          "not-tested axes are descriptive). `predicted`: the §1.4 geometric expectation.", ""]
    show = allv[["axis", "icc", "icc_lo", "icc_hi", "verdict", "predicted", "pop_slide",
                 "var_between", "var_within", "var_meas", "license", "icc_completers"]]
    md += [show.to_markdown(index=False), "",
           "- `pop_slide` = the cohort's V0→V2 population trend on that axis, **removed by the visit fixed "
           "effects before the ICC** — so ICC measures *individual* trait/state on top of any shared slide. "
           "A large slide with a high ICC means the cohort moves but individual *ranks* are preserved.", "",
           "## The §1.4 test — does trait/state align with spine/corner?",
           f"- **Spine (overall_severity): ICC {spine.icc:.2f} [{spine.icc_lo:.2f}, {spine.icc_hi:.2f}] → "
           f"{spine.verdict.upper()}** (predicted state). The severity spine is "
           + ("the state axis the cloud slides along — confirmed." if spine.verdict == "state"
              else f"**{spine.verdict}** — not the predicted pure state; see caveats.") + "",
           "- **Biology corners (cognition / metabolic / developmental_risk):** "
           + ", ".join(f"{r.axis} {r.icc:.2f}→{r.verdict}" for r in corners.itertuples())
           + " (predicted trait). " + ("All trait-like — confirmed."
              if (corners["verdict"] == "trait").all() else "Mixed result — see table."),
           f"- **Licensed-axis scorecard:** {len(match)}/{len(licensed)} licensed axes match the §1.4 "
           "prediction (the measurement-backed verdicts).",
           "- **Measurement-error correction matters:** corrected ICC vs raw ICC (which charges measurement "
           "noise to state) diverges most where reliability is low — e.g. "
           + ", ".join(f"{r.axis} {r.icc_raw:.2f}→{r.icc:.2f}"
                       for r in allv.sort_values("icc").head(3).itertuples()) + ".", "",
           "## Survivorship (completers vs all-available)",
           "- `icc_completers` (col above) refits on the V0+V1+V2 completers only. Large upward shifts vs "
           "`icc` would flag that the stable patients are retained (dropout biasing toward trait, per G6). "
           f"Max |Δ| = {float((allv['icc'] - allv['icc_completers']).abs().max()):.2f} → "
           + ("mild — the trait/state verdicts are robust to attrition."
              if float((allv['icc'] - allv['icc_completers']).abs().max()) < 0.15
              else "non-trivial — flagged per axis."), "",
           "## Verdict",
           "The trait/state profile is the **variance route** to the §1.4 prediction; G4 (stage 36) supplies "
           "the geometric route, and their agreement is the headline. Trait/state claims are strong on the "
           "G1-licensed axes (severity/cognition/metabolic/sleep/developmental), caveated on inflammatory "
           "(partial), descriptive on the not-tested explicit axes.", "",
           "Artifacts: `reports/35_trait_state.csv` · `docs/figures/35_trait_state.png`."]
    (REPORTS / "35_variance_report.md").write_text("\n".join(md))
    print("\n".join(md))


def _figure(allv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = allv.copy()
    y = np.arange(len(a))
    ax[0].barh(y, a["var_between"], color="#2c7fb8", label="trait σ²_b")
    ax[0].barh(y, a["var_within"], left=a["var_between"], color="#d95f0e", label="state σ²_w")
    ax[0].barh(y, a["var_meas"], left=a["var_between"] + a["var_within"], color="#cccccc",
               label="measurement σ²_e")
    ax[0].set_yticks(y); ax[0].set_yticklabels(a["axis"], fontsize=8)
    ax[0].set_xlabel("variance"); ax[0].set_title("Variance decomposition per axis")
    ax[0].legend(fontsize=8)
    colors = {"trait": "#2c7fb8", "state": "#d95f0e", "mixed": "#999999", "uninformative": "#dddddd"}
    ax[1].barh(y, a["icc"], xerr=[a["icc"] - a["icc_lo"], a["icc_hi"] - a["icc"]],
               color=[colors.get(v, "#999999") for v in a["verdict"]], capsize=2)
    ax[1].axvline(0.5, color="k", ls="--", lw=1)
    ax[1].axvline(0.6, color="#2c7fb8", ls=":", lw=1); ax[1].axvline(0.4, color="#d95f0e", ls=":", lw=1)
    ax[1].set_yticks(y); ax[1].set_yticklabels(a["axis"], fontsize=8)
    ax[1].set_xlabel("ICC = trait / (trait+state)  [94% HDI]"); ax[1].set_xlim(0, 1)
    ax[1].set_title("Trait (→1) vs state (→0) per axis")
    fig.tight_layout(); fig.savefig(FIGS / "35_trait_state.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
