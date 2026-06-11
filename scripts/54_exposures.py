#!/usr/bin/env python3
"""54 — M5.1 build the harmonized treatment-exposure table + join QC.

Runs the harmonization layer (face.treatment.medications) over the raw per-cohort CSVs to produce one
row per (cohort, patient_id) of common drug-class exposures at V0 (the moderation baseline), then QCs
the join to the fixed M5 frame and reports the powered questions. No modelling, no imputation. Methods:
docs/TREATMENT_MODEL.md (§2, §4, M5.1).

    python3 scripts/54_exposures.py

Writes results/face/m5/treatment_exposures.parquet, reports/54_exposures.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.treatment.medications import CLASSES, build_treatment_exposures  # noqa: E402

M5 = REPO / "results" / "face" / "m5"
REPORTS = REPO / "reports"
COHORTS = ("bp", "sz", "dr")


def main() -> None:
    M5.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    exp = build_treatment_exposures(REPO / "data", visit="V0")
    exp.to_parquet(M5 / "treatment_exposures.parquet", index=False)

    # coverage: n exposed per class × cohort (at V0)
    rows = []
    for c in COHORTS:
        sub = exp[exp["cohort"] == c]
        rec = {"cohort": c, "temporality": sub["temporality"].dropna().iloc[0] if len(sub.dropna(subset=["temporality"])) else "—",
               "n_with_med_record": int(sub[[f"on_{k}" for k in CLASSES]].notna().any(axis=1).sum())}
        for k in CLASSES:
            rec[k] = int(sub[f"on_{k}"].fillna(0).sum())
        rec["clozapine"] = int(sub["on_clozapine"].fillna(0).sum())
        rows.append(rec)
    cover = pd.DataFrame(rows)

    # join QC: do exposures land on the M5 frame patients?
    frame = pd.read_parquet(M5 / "analysis_frame.parquet")[["cohort", "patient_id"]].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    exp["patient_id"] = exp["patient_id"].astype(str)
    j = frame.merge(exp, on=["cohort", "patient_id"], how="left")
    has_any = j[[f"on_{k}" for k in CLASSES]].notna().any(axis=1)
    join_by_cohort = {c: f"{int((has_any & (j.cohort == c)).sum())}/{int((frame.cohort == c).sum())}" for c in COHORTS}

    # the powered questions (exposed / total-with-record, within cohort)
    bp = exp[exp.cohort == "bp"]; sz = exp[exp.cohort == "sz"]
    questions = [
        ("lithium-response-in-BP", f"BP on lithium {int(bp.on_lithium.fillna(0).sum())} / "
         f"off {int((bp.on_lithium == 0).sum())}; +plasma n={int(bp.lithium_plasma.notna().sum())}"),
        ("clozapine-in-SZ", f"SZ on clozapine {int(sz.on_clozapine.fillna(0).sum())} / "
         f"off {int((sz.on_clozapine == 0).sum())}"),
        ("antipsychotic (BP/SZ/DR)", " · ".join(f"{c}={int(exp[exp.cohort==c].on_antipsychotic.fillna(0).sum())}" for c in COHORTS)),
        ("antidepressant (BP/SZ/DR)", " · ".join(f"{c}={int(exp[exp.cohort==c].on_antidepressant.fillna(0).sum())}" for c in COHORTS)),
    ]

    md = [
        "# 54 — M5.1 harmonized treatment-exposure table", "",
        "One row per (cohort, patient_id) of common drug-class exposures at **V0** (the moderation "
        "baseline), harmonized across the three capture mechanisms (ATC / class-string / lifetime-flag). "
        "No imputation — a patient with no medication record at V0 is NaN, not unexposed.", "",
        f"- **Rows:** {len(exp)} ; with a V0 medication record: "
        + " · ".join(f"{r['cohort']}={r['n_with_med_record']}" for _, r in cover.iterrows()) + ".",
        f"- **Join to the M5 frame** (exposed-or-recorded / frame patients): "
        + " · ".join(f"{c} {v}" for c, v in join_by_cohort.items()) + ".", "",
        "## Exposure coverage (n exposed) by class × cohort (V0)", "",
        cover.to_markdown(index=False), "",
        "## The powered moderation questions", "",
        "\n".join(f"- **{q}** — {d}" for q, d in questions), "",
        "- **Temporality**: SZ/DR are **current** (the V0 medication); BP is **lifetime** (`cmoccur_*`, "
        "ever-by-baseline) — the BP exposures are illness-history-confounded and carry the target-trial "
        "caveat in the M5.2 design.", "",
        "## Decision for the gate",
        "Confirm the exposure table (coverage, join rate, the on/off split per question) before the "
        "propensity models (M5.2 / scripts 55) + the stratum × treatment moderation (56).", "",
        "Artifact: `results/face/m5/treatment_exposures.parquet`.",
    ]
    (REPORTS / "54_exposures.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
