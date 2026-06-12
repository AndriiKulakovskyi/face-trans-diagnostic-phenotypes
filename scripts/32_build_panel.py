#!/usr/bin/env python3
"""32 — per-visit model-ready tables (V1, V2) + the V0 standardization spec (the load-bearing step).

To score follow-up on the FIXED M1 scale, follow-up cells must be standardized with the **V0** transform,
not re-standardized per visit (§3.1). This step (1) captures that transform at its source
(`prepare(emit_moments=True)`), (2) PROVES it reproduces `prepare()`'s V0 matrix to 1e-6 — the must-pass
round-trip; if it fails, every level claim downstream is wrong — and (3) persists the V1/V2 raw tables
(same contract as `baseline_v0.parquet`) and reports coverage + out-of-V0-support cells (no imputation).
Methods: docs/TEMPORAL_MODEL.md §3.

    python3 scripts/32_build_panel.py

Writes data/processed/{baseline_v1,baseline_v2}.parquet + v0_standardization_spec.json (gitignored) ·
reports/32_build_panel.md (+ 32_coverage_by_visit.csv) · docs/figures/32_coverage.png.
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

from face.data import load_variables  # noqa: E402
from face.models.bayesian.continuous_core import S5_FACTORS, prepare  # noqa: E402
from face.temporal import VISITS  # noqa: E402
from face.temporal.panel import build_visit_table, load_long, modeled_items  # noqa: E402
from face.temporal.standardize import apply_spec, capture_v0_spec, save_spec  # noqa: E402

PROC = REPO / "data" / "processed"
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
XLSX = REPO / "data" / "face-common-vars.xlsx"
FOLLOWUP = [v for v in VISITS if v != "V0"]            # V1, V2


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)

    # ---- 1) capture + persist the V0 standardization spec ----
    spec = capture_v0_spec(S5_FACTORS)
    save_spec(spec, PROC / "v0_standardization_spec.json")
    n_log = sum(1 for it in spec.items if spec.family[it] == "lognormal")

    # ---- 2) MUST-PASS round-trip QC: apply_spec(V0) reproduces prepare().M ----
    B0 = pd.read_parquet(PROC / "baseline_v0.parquet")
    prep = prepare(S5_FACTORS, correlated=True, windows=True)
    M_prep = prep.M
    M_spec = apply_spec(spec, B0)[prep.items].to_numpy()
    nan_match = bool((np.isnan(M_prep) == np.isnan(M_spec)).all())
    maxdiff = float(np.nanmax(np.abs(M_spec - M_prep))) if nan_match else float("nan")
    qc_ok = nan_match and maxdiff < 1e-6
    assert qc_ok, f"V0 round-trip FAILED — nan_match={nan_match}, maxdiff={maxdiff:.2e}"

    # ---- 3) build + persist the V1/V2 raw tables; measure coverage on the frozen scale ----
    long = load_long()
    variables = load_variables(str(XLSX))
    items = modeled_items()
    cov_rows = []
    tables = {"V0": B0[[it for it in spec.items if it in B0.columns]]}
    for v in FOLLOWUP:
        B = build_visit_table(long, v, variables=variables, items=items)
        B.to_parquet(PROC / f"baseline_{v.lower()}.parquet")
        tables[v] = B
    for v in VISITS:
        B = tables[v]
        common = [it for it in spec.items if it in B.columns]
        Bz = apply_spec(spec, B)
        raw_obs = B[common].notna().to_numpy()
        oos = int((raw_obs & ~Bz[common].notna().to_numpy()).sum())   # raw value outside V0 support → NaN
        coh = pd.Series(B.index.get_level_values("cohort"))
        cov_rows.append(dict(visit=v, n_patients=len(B), n_items=len(common),
                             obs_cells=int(raw_obs.sum()),
                             mean_coverage=round(float(B[common].notna().mean().mean()), 3),
                             out_of_v0_support_cells=oos,
                             **{f"cov_{c}": round(float(B[common].loc[coh.values == c].notna().mean().mean()), 3)
                                for c in ("bp", "sz", "dr")}))
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(REPORTS / "32_coverage_by_visit.csv", index=False)
    _figure(cov)

    # ---- report ----
    md = ["# 32 — per-visit tables (V1, V2) + the V0 standardization spec", "",
          f"**V0 round-trip QC — {'✅ PASS' if qc_ok else '❌ FAIL'}** "
          f"(max |apply_spec(V0) − prepare().M| = {maxdiff:.1e}, NaN masks identical = {nan_match}). "
          "The frozen V0 transform reproduces the fitted V0 matrix exactly, so V1/V2 are scored on the "
          "same scale the certified loadings live on (genuine change is preserved, not re-centred).", "",
          f"- Spec: **{len(spec.items)} indicators** ({n_log} lognormal with a frozen V0 log-min) → "
          "`data/processed/v0_standardization_spec.json` (family / sign / log-min / mean / sd per item).", "",
          "## Per-visit coverage (modeled continuous block, on the frozen V0 scale)",
          cov.to_markdown(index=False), "",
          "- `out_of_v0_support_cells` = follow-up cells whose raw value falls outside V0's lognormal "
          "support (→ NaN, treated as missing — never imputed, never clipped).",
          f"- Data density thins with the panel (mean coverage V0 {cov.loc[0,'mean_coverage']:.2f} → "
          f"V1 {cov.loc[1,'mean_coverage']:.2f} → V2 {cov.loc[2,'mean_coverage']:.2f}); every patient is "
          "still scored from their own observed cells, with uncertainty propagated.", "",
          "## Artifacts",
          "- `data/processed/baseline_v{1,2}.parquet` — raw harmonized modeled indicators (gitignored).",
          "- `data/processed/v0_standardization_spec.json` — the frozen V0 transform (gitignored).",
          "- `reports/32_coverage_by_visit.csv` · `docs/figures/32_coverage.png`.", "",
          "Next: stage 33 (G1 longitudinal measurement invariance) → then stage 34 scores this panel."]
    (REPORTS / "32_build_panel.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n[done] V0 round-trip {'PASS' if qc_ok else 'FAIL'} (maxdiff {maxdiff:.1e}); "
          f"wrote baseline_v1/v2.parquet + spec.")


def _figure(cov):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis = list(cov["visit"])
    x = np.arange(len(vis))
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.2
    for i, (c, color) in enumerate([("bp", "#2c7fb8"), ("sz", "#d95f0e"), ("dr", "#31a354")]):
        ax.bar(x + (i - 1) * w, cov[f"cov_{c}"].values, w, label=c.upper(), color=color)
    ax.plot(x, cov["mean_coverage"].values, "ks--", label="overall", lw=2)
    ax.set_xticks(x); ax.set_xticklabels(vis)
    ax.set_ylabel("mean cell coverage (modeled block)"); ax.set_ylim(0, 1)
    ax.set_title("Data density across the M3 panel (V0 → V1 → V2)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "32_coverage.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
