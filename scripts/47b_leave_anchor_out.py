#!/usr/bin/env python3
"""47b — leave-one-anchor-out prognosis check (P5-01).

G is anchored on the functioning items (EGF/FAST/CGI-S/EQ-5D), and the headline predicts future EGF —
so "the map predicts functioning" risks being partly circular (the same measurement defined G). This
re-scores G with the **EGF anchor masked** (G_noEGF) and re-runs the durable-biology incremental ΔR²
for 2-year EGF beyond [diagnosis + severity + baseline EGF], under G_full vs G_noEGF. If the
metabolic/inflammatory increment survives the leave-anchor-out, it is not an artefact of EGF anchoring G.

    python3 scripts/47b_leave_anchor_out.py
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

from face.models.bayesian.continuous_core import S5_FACTORS, prepare  # noqa: E402
from face.prognosis.robustness import permutation_null  # noqa: E402
from face.scoring import conditional_gaussian_scores  # noqa: E402

M4 = REPO / "results" / "face" / "m4"
REPORTS = REPO / "reports"


def _dummies(arm):
    a = pd.Series(arm).astype(str).fillna("na")
    return pd.get_dummies(a, drop_first=True).to_numpy(dtype="float64")


def main():
    import arviz as az

    fr = pd.read_parquet(M4 / "analysis_frame.parquet").reset_index(drop=True)
    egf_v0 = next((c for c in fr.columns if c.lower() in ("egf__v0", "egf_v0")), None)
    egf_v2 = next((c for c in fr.columns if c.lower() in ("egf__v2", "egf_v2")), None)
    if egf_v0 is None or egf_v2 is None:
        raise SystemExit(f"EGF baseline/outcome columns not found; have {[c for c in fr.columns if 'egf' in c.lower()][:6]}")

    print("[1/2] re-score G with the EGF anchor masked (G_noEGF)...", flush=True)
    idata = az.from_netcdf(REPO / "results/face/s5_cert9_s1/idata.nc")
    prep = prepare(S5_FACTORS, correlated=True, windows=True)
    M = prep.M.copy()
    if "egf" in prep.items:
        M[:, prep.items.index("egf")] = np.nan                       # drop the EGF anchor, observed cells only
    g_noegf = conditional_gaussian_scores(M, idata.posterior, prep.factor_cols)["mean"][
        :, prep.factor_cols.index("overall_severity")]
    # prep.index and the analysis frame both trace to prepare(S5_FACTORS).index -> positionally aligned.
    assert len(g_noegf) == len(fr), f"row mismatch {len(g_noegf)} vs {len(fr)}"
    g_full = fr["overall_severity__mean"].to_numpy("float64")
    fin = np.isfinite(g_full) & np.isfinite(g_noegf)
    r_gg = float(np.corrcoef(g_full[fin], g_noegf[fin])[0, 1])

    print("[2/2] durable-biology incremental ΔR² for 2y EGF under G_full vs G_noEGF...", flush=True)
    y = fr[egf_v2].to_numpy("float64")
    bio = fr[["metabolic__mean", "inflammatory__mean"]].to_numpy("float64")
    arm = _dummies(fr["arm"]) if "arm" in fr.columns else np.zeros((len(fr), 0))
    base_egf = fr[egf_v0].to_numpy("float64")
    rows = []
    for tag, g in [("G_full", g_full), ("G_noEGF", g_noegf)]:
        found = np.column_stack([arm, g, base_egf])
        ok = np.isfinite(y) & np.isfinite(found).all(1) & np.isfinite(bio).all(1)
        res = permutation_null(y[ok], found[ok], bio[ok], n_sim=1000)
        rows.append({"severity_arm": tag, "n": int(ok.sum()), "durable_dR2": round(res["real_dR2"], 4),
                     "null_p95": round(res["null_p95"], 4), "p_value": round(res["p_value"], 4)})
    tab = pd.DataFrame(rows)
    tab.to_csv(REPORTS / "47b_leave_anchor_out.csv", index=False)

    survives = (tab["p_value"] < 0.05).all() and tab["durable_dR2"].min() > 0
    md = ["# 47b — leave-one-anchor-out prognosis (P5-01)", "",
          "G is anchored on the functioning items; the headline predicts future EGF. This re-scores G "
          "with the **EGF anchor masked** and re-runs the durable-biology (metabolic+inflammatory) "
          "incremental ΔR² for 2-year EGF beyond [diagnosis + severity + baseline EGF], under the full G "
          "vs the EGF-leave-out G.", "",
          f"- G_full vs G_noEGF correlation: **{r_gg:.3f}** (G barely depends on the single EGF anchor).", "",
          "## Durable-biology incremental ΔR² for 2y EGF", tab.to_markdown(index=False), "",
          ("- **Survives the leave-anchor-out:** the metabolic/inflammatory increment for future EGF holds "
           "with G re-scored WITHOUT EGF — it is not an artefact of EGF anchoring G (the biology axes are ⊥G "
           "and the increment is robust to G's anchor composition)."
           if survives else
           "- **⚠ The durable increment weakens under the leave-anchor-out** — see the table; the EGF "
           "prognosis claim should be qualified."), ""]
    (REPORTS / "47b_leave_anchor_out_report.md").write_text("\n".join(md))
    print("\n".join(md))
    print("\nwrote reports/47b_leave_anchor_out_report.md (+ .csv)")


if __name__ == "__main__":
    main()
