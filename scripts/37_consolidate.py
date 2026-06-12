#!/usr/bin/env python3
"""37 — M3.7: consolidate the longitudinal panel into the M4 hand-off + the axis-level coherence summary.

Joins the G3 trait/state verdict onto the per-(patient, visit) panel (so the hand-off is self-contained:
coords + uncertainty + memberships + G1 license + G3 trait/state), and writes the axis-level M3 summary
(per axis: ICC, trait/state, population slide, G4 reliable-change rate, G1 license). This is the substrate
M4 (prognosis) consumes — telling it which axes are durable (worth stratifying/predicting on) vs which
fluctuate (worth monitoring). Methods: docs/TEMPORAL_MODEL.md §3/§11.

    python3 scripts/37_consolidate.py

Writes results/face/patient_panel.parquet (gitignored) · reports/37_axis_summary.csv · reports/37_panel.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.temporal import CANON, VISITS  # noqa: E402

M3 = REPO / "results" / "face" / "m3"
REPORTS = REPO / "reports"


def main():
    panel = pd.read_parquet(M3 / "panel_coords.parquet")
    g3 = pd.read_csv(REPORTS / "35_trait_state.csv").set_index("axis")
    g4 = pd.read_csv(REPORTS / "36_change_rates.csv").set_index("axis")

    # ---- axis-level M3 coherence summary (one row per axis) ----
    summary = pd.DataFrame({"axis": list(CANON)})
    summary["g1_license"] = summary["axis"].map(lambda a: panel[f"{a}__license"].iloc[0])
    summary["icc_trait"] = summary["axis"].map(g3["icc"])
    summary["trait_state"] = summary["axis"].map(g3["verdict"])
    summary["pop_slide_v0v2"] = summary["axis"].map(g3["pop_slide"])
    summary["reliable_change_rate"] = summary["axis"].map(g4["frac_reliable"])
    # durable stratify-on axes = G1-licensed + G3-trait + LOW movement (the stable biology corners).
    # severity is excluded by the movement filter: it is rank-stable but the population slides, so it is
    # the spine / monitoring axis (§1.4), not a stratify-on dimension.
    summary["durable_for_m4"] = ((summary["g1_license"].isin(["invariant", "partial"]))
                                 & (summary["trait_state"] == "trait")
                                 & (summary["reliable_change_rate"] < 0.20))
    summary.to_csv(REPORTS / "37_axis_summary.csv", index=False)

    # ---- the M4 hand-off: per-(patient, visit) panel + the broadcast trait/state verdict ----
    for ax in CANON:
        panel[f"{ax}__trait_state"] = g3.loc[ax, "verdict"]
    panel.to_parquet(M3.parent / "patient_panel.parquet", index=False)

    durable = summary[summary["durable_for_m4"]]["axis"].tolist()
    n_pat = panel["patient_uid"].nunique()
    md = ["# 37 — M3 consolidation: the longitudinal hand-off + axis coherence summary", "",
          f"**M3 complete.** Per-(patient, visit) panel over {', '.join(VISITS)} for **{n_pat:,} patients** "
          f"({len(panel):,} rows) → `results/face/patient_panel.parquet` (the M4 substrate). Axis-level "
          "coherence summary below.", "",
          "## Axis-level M3 summary (the temporal verdict per dimension)",
          summary.to_markdown(index=False), "",
          f"- **Durable stratify-on axes** (G1-licensed + G3-trait + stable over time — the biology corners "
          f"worth stratifying / predicting on): **{', '.join(durable)}**.",
          "- **Spine / monitoring axes** (move over time → track, don't stratify): **severity** "
          "(rank-stable but the cohort slides — the spine), suicidality, sleep.",
          "- **Caveats carried forward:** developmental_risk's apparent state is CTQ recall noise (trait by "
          "design); inflammatory is partial-invariant; substance is uninformative (signal ≪ noise); mania / "
          "suicidality / substance are not G1-tested (explicit block).", "",
          "## The M3 coherence verdict (G1–G6)",
          "- **G1 (invariance):** the V0 map measures the same constructs at V1/V2 — severity, cognition, "
          "metabolic, sleep, developmental invariant; inflammatory partial. The precondition holds.",
          "- **G3 ⟷ G4 (trait/state ⟷ geometry):** both routes agree — biology/cognition are durable "
          "(trait, ranks/positions/archetype-identity persist) while severity + symptoms move (state, "
          "population slide). The M2 geometry is temporally coherent.",
          "- **G6 (attrition):** dropout is mild/cognition-leaning; verdicts robust to survivorship.",
          "- **Bottom line:** the transdiagnostic map and strata are **temporally coherent** — *stratify on "
          "the durable biology, monitor the moving symptoms.* Persists ≠ predicts (M4).", "",
          "Docs: `docs/TEMPORAL_FINDINGS.md` (paper-facing) · `docs/TEMPORAL_MODEL.md` (methods) · "
          "`docs/TEMPORAL_RESULTS.md` (per-stage). Hand-off: `results/face/patient_panel.parquet`."]
    (REPORTS / "37_panel.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n[done] wrote patient_panel.parquet ({panel.shape}) + 37_axis_summary.csv")


if __name__ == "__main__":
    main()
