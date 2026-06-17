#!/usr/bin/env python3
"""57 — M5.2c treatment-as-confounder for M4 (does the prognosis survive treatment adjustment?).

M4's headline: the durable axes (metabolic / inflammatory) predict future FUNCTIONING (EGF) beyond
diagnosis + severity + baseline functioning. A standing objection — "that's just unmodelled treatment"
(sicker-biology patients are treated differently, and treatment drives the outcome). This stage re-fits
the M4 functioning prognosis on the treatment-data subset, WITH vs WITHOUT the harmonized drug-class
exposures as covariates, on the SAME sample (so the contrast isolates treatment adjustment, not sample
change). The durable-axis coefficient surviving = the map's forecast is not a treatment proxy. EIV +
site random-effect throughout. Methods: docs/TREATMENT_MODEL.md §4.5 (Q2).

    python3 scripts/57_confounder.py [--smoke]

Writes results/face/m5/confounder.csv, reports/57_confounder.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import numpyro
import pandas as pd

numpyro.set_host_device_count(4)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.prognosis import DURABLE  # noqa: E402
from face.prognosis.frame import OutcomeSpec  # noqa: E402
from face.prognosis.glm import fit_glm  # noqa: E402
from face.prognosis.reference import (  # noqa: E402
    arm_block,
    coord_eiv_block,
    foundation_design,
    site_index,
)

M5 = REPO / "results" / "face" / "m5"
REPORTS = REPO / "reports"
SEV = "overall_severity__mean"
TREAT_COLS = ["on_antipsychotic", "on_antidepressant", "on_mood_stabilizer", "on_lithium", "on_anxiolytic"]
SPEC_EGF = OutcomeSpec(name="egf", label="egf", source_var="egf", family="gaussian",
                       direction="higher_better", cohort_scope=("bp", "sz", "dr"), severity_anchor="G", role="primary")


def main(smoke=False) -> None:
    M5.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    fit_kw = dict(draws=150, tune=150, chains=2, seed=20260611) if smoke else dict(draws=800, tune=800, chains=4, seed=20260611)
    frame = pd.read_parquet(M5 / "analysis_frame.parquet"); frame["patient_id"] = frame["patient_id"].astype(str)
    exp = pd.read_parquet(M5 / "treatment_exposures.parquet"); exp["patient_id"] = exp["patient_id"].astype(str)
    merged = frame.merge(exp[["cohort", "patient_id", *TREAT_COLS]], on=["cohort", "patient_id"], how="left")

    # treatment-data subset = patients with a V0 medication record; within it, absence of a class = 0 (full med list)
    sub = merged[merged["on_antipsychotic"].notna()].copy()
    for c in TREAT_COLS:
        sub[c] = sub[c].fillna(0.0)
    need = ["egf__V2", "egf__V0", SEV, "age", "sex", "siteid_city", "arm",
            *[f"{a}__mean" for a in DURABLE], *[f"{a}__sd" for a in DURABLE]]
    sub = sub.dropna(subset=need).copy()
    y = sub["egf__V2"].to_numpy(float); y = (y - y.mean()) / (y.std() or 1.0)
    found, _ = foundation_design(sub, SPEC_EGF, severity_col=SEV, horizon="V2")
    arm, _ = arm_block(sub)
    grp, ng = site_index(sub)
    ob, sd, _ = coord_eiv_block(sub, DURABLE)
    treat = sub[TREAT_COLS].to_numpy(float)
    base = dict(family="gaussian", group=grp, n_groups=ng, eiv_obs=ob, eiv_sd=sd, **fit_kw)

    print(f"  treatment-data subset N={len(sub)} (cohorts {dict(sub.cohort.value_counts())}); fitting ...", flush=True)
    fit_no = fit_glm(y, np.column_stack([found, arm]), **base)                       # M4-style (no treatment)
    fit_tx = fit_glm(y, np.column_stack([found, arm, treat]), **base)                # + drug-class exposures

    def betas(fit):
        c = fit["coef"].set_index("term")
        return {a: (float(c.loc[f"beta_eiv[{i}]", "mean"]), float(c.loc[f"beta_eiv[{i}]", "eti_lo"]),
                    float(c.loc[f"beta_eiv[{i}]", "eti_hi"])) for i, a in enumerate(DURABLE)}
    b_no, b_tx = betas(fit_no), betas(fit_tx)
    rows = []
    for a in DURABLE:
        rows.append({"axis": a, "beta_no_treat": round(b_no[a][0], 3), "hdi_no": f"[{b_no[a][1]:+.3f},{b_no[a][2]:+.3f}]",
                     "beta_with_treat": round(b_tx[a][0], 3), "hdi_with": f"[{b_tx[a][1]:+.3f},{b_tx[a][2]:+.3f}]",
                     "survives": bool(b_tx[a][1] > 0 or b_tx[a][2] < 0),
                     "attenuation_%": round(100 * (1 - abs(b_tx[a][0]) / (abs(b_no[a][0]) or 1e-9)), 1)})
    res = pd.DataFrame(rows)
    res.to_csv(M5 / "confounder.csv", index=False)
    _report(res, len(sub), dict(sub.cohort.value_counts()))


def _report(res, n, cohorts):
    md = [
        "# 57 — M5.2c treatment-as-confounder for M4", "",
        f"M4's functioning prognosis re-fit on the **treatment-data subset** (N={n}; "
        f"{', '.join(f'{k}={v}' for k, v in cohorts.items())}), WITH vs WITHOUT the harmonized drug-class "
        "exposures (antipsychotic/antidepressant/mood-stab/lithium/anxiolytic) as covariates — same "
        "sample, so the contrast isolates treatment adjustment. EGF z-scored; durable axes EIV.", "",
        "## Durable-axis effect on future functioning, with vs without treatment adjustment", "",
        res.to_markdown(index=False), "",
        "## Read",
        "- **survives** = the durable-axis coefficient's 94% HDI still excludes 0 after adjusting for "
        "treatment; **attenuation_%** = how much the point estimate shrank.",
        "- A surviving metabolic / inflammatory effect answers the standing objection: **the map's "
        "functional forecast is not merely unmodelled treatment** — it holds controlling for the drug "
        "classes the patient was on. (Treatment here is the observed exposure; residual/unmeasured "
        "prescribing is bounded by the M5.2b E-values, not eliminated.)", "",
        "Artifact: `results/face/m5/confounder.csv`.",
    ]
    (REPORTS / "57_confounder.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
