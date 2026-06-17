#!/usr/bin/env python3
"""48 — M4.8 consolidation + the M5 hand-off.

Closes out M4: a consolidated per-outcome verdict table and a per-patient prognostic-risk object that
M5 (treatment) consumes. The verdict table records what the map predicts, beyond what, and how robustly
(the locked M4.1–4.7 findings). The per-patient risk recomputes out-of-fold (5-fold CV) predicted
probabilities for the two headline functional endpoints from the clinician-reference + map model, so
every modelled patient carries a functional-remission and functional-deterioration risk alongside their
archetype. Methods of record: docs/PROGNOSIS_MODEL.md.

    python3 scripts/48_consolidate.py

Writes results/face/m4/{prognosis_summary.csv, prognosis_patient_risk.parquet}, reports/48_consolidate.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.prognosis.clinical_value import cv_predict  # noqa: E402
from face.prognosis.endpoints import build_endpoints  # noqa: E402
from face.prognosis.frame import load_outcome_config  # noqa: E402
from face.prognosis.reference import (  # noqa: E402
    arm_block,
    armB_block,
    foundation_design,
    severity_column,
)

CONFIG = REPO / "configs" / "m4_outcomes.yaml"
M4 = REPO / "results" / "face" / "m4"
PROFILES = REPO / "results" / "face" / "m2" / "archetype_profiles.csv"
REPORTS = REPO / "reports"
CGI_BASELINE = "cgi_s__V0"
COVARS = ("age", "sex", "siteid_city", "arm")

# The locked M4 verdicts (M4.3 incremental · M4.4 head-to-head · M4.6 clinical value · M4.7 robustness).
SUMMARY = [
    {"outcome": "EGF (functioning)", "map_vs_reference_elpd": "+46 (archetypes) / +7 (durable coords)",
     "durable_signal": "metabolic β −0.062, inflammatory −0.060 (both exclude 0; metabolic survives "
                       "error-corrected-G severity)", "vs_dsm5": "co-informative (B−A +47, B−C +40)",
     "clinical_auc": "ref 0.76 → +map 0.78 (ΔAUC +0.017, CI excl 0)",
     "generalization": "course-dependent: BP/DR yes, SZ null (foundation saturation)",
     "robust": "survives IPW + reliability + permutation (p=0.001); weakens dropping BP",
     "verdict": "PREDICTIVE (functional), robust, complementary to DSM-5"},
    {"outcome": "CGI-S (severity)", "map_vs_reference_elpd": "+13 (archetypesA) / −1 (durable coords)",
     "durable_signal": "all durable β straddle 0 (autoregression-saturated)",
     "vs_dsm5": "co-informative, DSM-5-leaning (B−C +35 > B−A +15)",
     "clinical_auc": "ref 0.87 → +map 0.87 (ΔAUC +0.002, ns)",
     "generalization": "BP only; SZ/DR null", "robust": "n/a (no headline gain)",
     "verdict": "NOT incremental (severity is baseline-determined)"},
]


def _patient_risk(f, cfg, horizon):
    """Per-patient out-of-fold (5-fold CV) functional-remission & -deterioration risk from the
    clinician-reference + map model, with archetype + cohort. The patient-level M5 hand-off."""
    spec = cfg.by_name("egf")
    sev = severity_column(spec, cgi_baseline_col=CGI_BASELINE)
    out = []
    for ep, pcol in [("egf_remission", "p_remission"), ("egf_deterioration", "p_deterioration")]:
        sub = f.dropna(subset=[f"ep_{ep}", *COVARS, sev, f"{spec.name}__V0"]).copy()
        y = sub[f"ep_{ep}"].to_numpy("int64")
        found, _ = foundation_design(sub, spec, severity_col=sev, horizon=horizon)
        arm, _ = arm_block(sub)
        archB, _ = armB_block(sub, profiles_path=PROFILES)
        p = cv_predict(np.column_stack([found, arm, archB]), y, seed=int(cfg.meta.get("seed", 20260610)))
        out.append(sub[["cohort", "patient_id", "arch_dominant_name"]].assign(**{pcol: np.round(p, 3)}))
    risk = out[0].merge(out[1][["cohort", "patient_id", "p_deterioration"]], on=["cohort", "patient_id"],
                        how="outer")
    return risk.rename(columns={"arch_dominant_name": "archetype"})


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    cfg = load_outcome_config(CONFIG)
    horizon = cfg.meta.get("primary_horizon", "V2")
    f = build_endpoints(pd.read_parquet(M4 / "analysis_frame.parquet"))

    summary = pd.DataFrame(SUMMARY)
    summary.to_csv(M4 / "prognosis_summary.csv", index=False)
    risk = _patient_risk(f, cfg, horizon)
    risk.to_parquet(M4 / "prognosis_patient_risk.parquet", index=False)

    md = [
        "# 48 — M4.8 consolidation + the M5 hand-off", "",
        "Closes M4. The consolidated per-outcome verdict and the per-patient prognostic-risk object for M5.", "",
        "## Consolidated verdict (M4.1–4.7)", "",
        summary[["outcome", "verdict", "vs_dsm5", "generalization", "robust"]].to_markdown(index=False), "",
        "Full detail (ELPD / β / AUC) in `results/face/m4/prognosis_summary.csv`.", "",
        f"## Per-patient prognostic risk (M5 hand-off) — {len(risk)} patients", "",
        "Out-of-fold (5-fold CV) `reference + map` predicted probabilities, with each patient's "
        "archetype + cohort:", "",
        risk.head(6).to_markdown(index=False), "",
        "- Columns: `cohort, patient_id, archetype, p_remission, p_deterioration`.",
        "- `results/face/m4/prognosis_patient_risk.parquet` — the patient-level object M5 (treatment) "
        "consumes: stratum + prognostic risk per patient.", "",
        "## M4 is complete", "",
        "- **Map adds robust prognostic value for functioning** beyond diagnosis+severity (metabolic/"
        "inflammatory ⊥G; archetypes stratify 14%→60% functional remission), surviving "
        "attrition/reliability/permutation.",
        "- **Co-informative with DSM-5** (complements, not replaces) and **course-dependent** "
        "(episodic BP/DR, not baseline-saturated SZ).",
        "- **Severity (CGI-S) is autoregression-determined** — the map adds little there.",
        "- Honest limits: scale trajectories not events; internal validity; 2-year horizon. Next: **M5 "
        "treatment** (does stratum moderate treatment response?).", "",
        "Artifacts: `results/face/m4/{prognosis_summary.csv, prognosis_patient_risk.parquet}`. "
        "Docs: `docs/PROGNOSIS_{MODEL,FINDINGS,RESULTS,ATLAS}.md`.",
    ]
    (REPORTS / "48_consolidate.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
