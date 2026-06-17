#!/usr/bin/env python3
"""51 — M5.1 build the M5 analysis frame (M4 predictor side + treatment-response endpoints).

Joins the treatment-response endpoints (from the raw CGI signals at the horizon) onto the fixed M4
predictor side (coordinates + SD, archetypes + tessellation, covariates, baseline CGI-S + G, IPW).
Nothing re-scored, nothing imputed. Methods: docs/TREATMENT_MODEL.md (M5.1).

    python3 scripts/51_frame.py

Writes results/face/m5/analysis_frame.parquet, reports/51_frame.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.treatment.endpoints import ENDPOINTS, load_m5_config  # noqa: E402
from face.treatment.frame import build_m5_frame  # noqa: E402

CONFIG = REPO / "configs" / "m5_outcomes.yaml"
M5 = REPO / "results" / "face" / "m5"
REPORTS = REPO / "reports"
COHORTS = ("bp", "sz", "dr")


def main() -> None:
    M5.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    cfg = load_m5_config(CONFIG)
    horizon = cfg["meta"].get("primary_horizon", "V2")
    frame = build_m5_frame(horizon=horizon, mars_low=cfg["meta"].get("mars_low_threshold", 5),
                           resistance_cgis=cfg["meta"].get("resistance_cgis_min", 4))
    frame.to_parquet(M5 / "analysis_frame.parquet", index=False)

    # QC: endpoint coverage by cohort + predictor presence
    coh = frame["cohort"]
    rows = []
    for e in ENDPOINTS:
        col = f"ep_{e.name}"
        rec = {"endpoint": e.name, "n_total": int(frame[col].notna().sum())}
        for c in COHORTS:
            rec[c] = int(frame.loc[coh == c, col].notna().sum())
        rows.append(rec)
    cover = pd.DataFrame(rows)
    durable_ok = int(frame[["cognition__mean", "metabolic__mean", "inflammatory__mean"]].notna().all(axis=1).sum())
    sev_ok = int(frame[["overall_severity__mean", "cgi_s__V0"]].notna().all(axis=1).sum())

    md = [
        "# 51 — M5.1 analysis-frame build", "",
        "The fixed M4 predictor side + the treatment-response endpoints at the horizon. One row per "
        "V0-roster patient; nothing re-scored or imputed.", "",
        f"- **Rows:** {len(frame)} (V0 roster); columns {frame.shape[1]}.",
        f"- **Predictors present:** durable trio (cognition/metabolic/inflammatory mean) for "
        f"{durable_ok}; baseline severity (G + CGI-S) for {sev_ok}.",
        "- **Map representations + covariates + IPW** carried from `results/face/m4/analysis_frame.parquet`.",
        "", "## Endpoint coverage by cohort (non-missing at the horizon)", "",
        cover.to_markdown(index=False), "",
        "- The CGI response endpoints are **BP/SZ only** (DR `n=0` — no CGI efficacy index); "
        "`low_adherence` excludes DR (MARS mis-scaled, M5.0). The modelling sample for the tolerability "
        "test is BP/SZ.", "",
        "## Decision for the gate",
        "Confirm the frame (endpoint coverage, predictor presence) before the tolerability test (stage 52).", "",
        "Artifact: `results/face/m5/analysis_frame.parquet`.",
    ]
    (REPORTS / "51_frame.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
