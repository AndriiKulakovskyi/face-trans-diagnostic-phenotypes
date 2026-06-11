#!/usr/bin/env python3
"""55 — M5.2a treatment propensity + overlap (the identification gate).

For each treatment question × contrast mode (active-comparator primary, on/off sensitivity): define the
exposure, fit P(treat | severity + diagnosis + demographics + the 9 map coords), and report common
support + covariate balance (SMD) before vs after stabilized IPTW. **Overlap decides what is estimable**
— a channeled treatment (e.g. clozapine) with no comparable controls is reported as non-estimable, not
forced. Persists per-patient PS + IPTW for the moderation stage (56). Methods: docs/TREATMENT_MODEL.md §4.

    python3 scripts/55_propensity.py

Writes results/face/m5/propensity_{question}_{mode}.parquet, reports/55_propensity.md,
docs/figures/55_overlap.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.treatment.propensity import (QUESTIONS, confounder_matrix, define_exposure,  # noqa: E402
                                       overlap, propensity_score, smd, stabilized_iptw)

M5 = REPO / "results" / "face" / "m5"
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
MODES = ("active_comparator", "on_off")
SEED = 20260611


def _merged():
    frame = pd.read_parquet(M5 / "analysis_frame.parquet")
    exp = pd.read_parquet(M5 / "treatment_exposures.parquet")
    frame["patient_id"] = frame["patient_id"].astype(str)
    exp["patient_id"] = exp["patient_id"].astype(str)
    return frame.merge(exp.drop(columns=["temporality"]), on=["cohort", "patient_id"], how="left")


def _verdict(diag, smd_after):
    if min(diag["n_treated"], diag["n_control"]) < 30:
        return "non-estimable (arm < 30)"
    if diag["frac_in_support"] < 0.5:
        return "channeled (poor overlap)"
    if smd_after > 0.25:
        return "estimable — residual imbalance (caution)"
    return "estimable"


def main() -> None:
    M5.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    merged = _merged()
    rows, ps_store = [], {}
    for q in QUESTIONS:
        for mode in MODES:
            sub, treat = define_exposure(merged, q, mode)
            if treat.sum() < 5 or (treat == 0).sum() < 5:
                rows.append({"question": q, "mode": mode, "n_treated": int(treat.sum()),
                             "n_control": int((treat == 0).sum()), "verdict": "no contrast"}); continue
            X, names, row_ok = confounder_matrix(sub)
            tr = treat[row_ok]
            if tr.sum() < 5 or (tr == 0).sum() < 5:
                rows.append({"question": q, "mode": mode, "verdict": "no contrast (after NaN drop)"}); continue
            ps = propensity_score(X, tr, seed=SEED)
            diag = overlap(ps, tr)
            w, keep = stabilized_iptw(ps, tr)
            smd_before, smd_after = float(smd(X, tr).max()), float(smd(X[keep], tr[keep], w[keep]).max())
            verdict = _verdict(diag, smd_after)
            rows.append({"question": q, "mode": mode, **{k: diag[k] for k in ("n_treated", "n_control", "frac_in_support")},
                         "max_smd_before": round(smd_before, 3), "max_smd_after": round(smd_after, 3), "verdict": verdict})
            sub_ok = sub[row_ok]
            ps_store[(q, mode)] = (pd.DataFrame({"cohort": sub_ok["cohort"].values, "patient_id": sub_ok["patient_id"].values,
                                                 "treat": tr, "ps": ps, "iptw": w, "in_support": keep}), diag)
            ps_store[(q, mode)][0].to_parquet(M5 / f"propensity_{q}_{mode}.parquet", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(M5 / "propensity_summary.csv", index=False)
    _figure(ps_store)
    _report(summary)


def _figure(ps_store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qs = [q for q in QUESTIONS if (q, "active_comparator") in ps_store]
    fig, axes = plt.subplots(1, max(len(qs), 1), figsize=(4.4 * max(len(qs), 1), 4))
    axes = np.atleast_1d(axes)
    for ax, q in zip(axes, qs):
        df, diag = ps_store[(q, "active_comparator")]
        ax.hist(df.ps[df.treat == 1], bins=20, alpha=0.6, density=True, label=f"treated (n={diag['n_treated']})", color="#d73027")
        ax.hist(df.ps[df.treat == 0], bins=20, alpha=0.6, density=True, label=f"comparator (n={diag['n_control']})", color="#4575b4")
        ax.axvspan(diag["common_lo"], diag["common_hi"], color="grey", alpha=0.12)
        ax.set_title(f"{QUESTIONS[q]['label']}\noverlap {diag['frac_in_support']:.0%}", fontsize=9)
        ax.set_xlabel("propensity score"); ax.legend(fontsize=7)
    fig.suptitle("Propensity overlap by treatment arm (active-comparator)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "55_overlap.png", dpi=130)
    plt.close(fig)


def _report(summary):
    md = [
        "# 55 — M5.2a propensity + overlap (the identification gate)", "",
        "`P(treat | severity[CGI-S + error-corrected G] + DSM-5 arm + demographics + the 9 map "
        "coordinates)` per question × contrast. **Overlap decides estimability**; balance (max |SMD|) "
        "before vs after stabilized IPTW shows whether weighting can render the arms comparable.", "",
        "## Overlap + balance by question × mode", "",
        summary.to_markdown(index=False), "",
        "## Read",
        "- **Active-comparator** is the primary contrast (both arms treated → indication more similar). "
        "`on_off` is the higher-powered sensitivity.",
        "- A `max_smd_after` ≤ 0.1 is good balance, ≤ 0.25 acceptable; `frac_in_support` is the share of "
        "patients inside the common propensity range.",
        "- **Channeled** (poor overlap) questions are reported as **non-estimable** — the honest outcome "
        "of confounding by indication, not a failure to find an effect.", "",
        "## Decision for the gate",
        "Carry the **estimable** question×mode cells into the moderation stage (56); report channeled "
        "ones as non-estimable. Per-patient PS + IPTW persisted to `results/face/m5/propensity_*.parquet`.", "",
        "Artifacts: `results/face/m5/propensity_summary.csv` · `docs/figures/55_overlap.png`.",
    ]
    (REPORTS / "55_propensity.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
