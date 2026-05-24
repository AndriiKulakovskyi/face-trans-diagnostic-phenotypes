"""§3.9 — The FACE profile: two clinically-actionable trans-diagnostic indices.

Builds FACE-D (affective distress) and FACE-M (cardiometabolic load) from routine instruments
(`trans_diag.face_score`) and validates the translational proposal:
  1. **Parsimony** — the short scores reproduce the depression / metabolic axes (correlation).
  2. **Retained value** — added to DSM diagnosis, the 2-score profile recovers most of the full
     6-axis out-of-sample advantage for patient-reported outcomes (leakage-safe nested CV, exactly
     as in 10_phase5_outcomes: V0 predictors, V1 outcomes, baseline+age+sex adjusted, shuffled CV).

Artifacts: results/face_score_validation.json, reports/face_score.html.
Run:  python3 scripts/19_face_score.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402

from trans_diag import (  # noqa: E402
    AXIS_NAMES,
    FACE_D_ITEMS,
    FACE_M_ITEMS,
    build_unified_dataframe,
    compute_face_scores,
)
from trans_diag.outcomes import OUTCOMES, cv_metric  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
FIGDIR = REPORTS_DIR / "figures"
COH_COLOR = {"bp": "#1f77b4", "sz": "#d62728", "dr": "#2ca02c"}
COH_LABEL = {"bp": "Bipolar", "sz": "Schizophrenia", "dr": "Depression"}


def _face_figure(face: pd.DataFrame, axes: pd.DataFrame) -> None:
    """Fig 7 — FACE profile: parsimony (a,b), the clinical FACE plane (c), by-diagnosis (d)."""
    df = pd.DataFrame({"FACE_D": face["FACE_D"].to_numpy(),
                       "FACE_M": face["FACE_M"].to_numpy(),
                       "dep": axes["depression_severity"].reindex(face.index).to_numpy(),
                       "met": axes["metabolic"].reindex(face.index).to_numpy(),
                       "cohort": face.index.get_level_values("cohort")})
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    a = df.dropna(subset=["FACE_D", "dep"])
    ax[0, 0].hexbin(a["dep"], a["FACE_D"], gridsize=40, cmap="Blues", mincnt=1)
    ax[0, 0].set(xlabel="depression axis (masked-FA score)", ylabel="FACE-D",
                 title=f"(a) FACE-D ≈ depression axis (r={np.corrcoef(a['dep'],a['FACE_D'])[0,1]:.2f})")
    b = df.dropna(subset=["FACE_M", "met"])
    ax[0, 1].hexbin(b["met"], b["FACE_M"], gridsize=40, cmap="Oranges", mincnt=1)
    ax[0, 1].set(xlabel="metabolic axis (masked-FA score)", ylabel="FACE-M",
                 title=f"(b) FACE-M ≈ metabolic axis (r={np.corrcoef(b['met'],b['FACE_M'])[0,1]:.2f})")
    groups = ["bp", "sz", "dr"]
    # (c) FACE-M by diagnosis — transdiagnostic (metabolic labs collected in all three networks)
    bm = ax[1, 0].boxplot([df.loc[df["cohort"] == c, "FACE_M"].dropna().to_numpy() for c in groups],
                          patch_artist=True, widths=0.6, showfliers=False)
    for patch, c in zip(bm["boxes"], groups, strict=False):
        patch.set_facecolor(COH_COLOR[c]); patch.set_alpha(0.5)
    ax[1, 0].axhline(0, color="#999", lw=0.7)
    ax[1, 0].set_xticklabels([COH_LABEL[c] for c in groups], fontsize=8)
    ax[1, 0].set(ylabel="FACE-M (cardiometabolic, z)",
                 title="(c) FACE-M — transdiagnostic (all three diagnoses)")
    # (d) FACE-D by diagnosis — BP/DR only (SZ battery has no self-report depression scales)
    dd = {c: df.loc[df["cohort"] == c, "FACE_D"].dropna().to_numpy() for c in groups}
    have = [(i + 1, c) for i, c in enumerate(groups) if len(dd[c]) > 5]
    bd = ax[1, 1].boxplot([dd[c] for _, c in have], positions=[p for p, _ in have],
                          patch_artist=True, widths=0.6, showfliers=False)
    for patch, (_, c) in zip(bd["boxes"], have, strict=False):
        patch.set_facecolor(COH_COLOR[c]); patch.set_alpha(0.5)
    ax[1, 1].axhline(0, color="#999", lw=0.7)
    ax[1, 1].set_xticks([1, 2, 3]); ax[1, 1].set_xticklabels([COH_LABEL[c] for c in groups], fontsize=8)
    for i, c in enumerate(groups):
        if len(dd[c]) <= 5:
            ax[1, 1].text(i + 1, 0, "not measured\n(no QIDS/MADRS/\nSTAI in SZ battery)",
                          ha="center", va="center", fontsize=6, color="#d62728")
    ax[1, 1].set(ylabel="FACE-D (affective distress, z)",
                 title="(d) FACE-D — BP/DR only (SZ measurement gap)")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGDIR / f"fig7_face_profile.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pid_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.MultiIndex.from_arrays(
        [df.index.get_level_values("cohort").astype(str),
         df.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    return df


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    domains = _pid_index(pd.read_parquet(RESULTS_DIR / "cluster_domains_scores.parquet"))
    axes = _pid_index(pd.read_parquet(RESULTS_DIR / "dimensional_final_scores.parquet"))
    axes.columns = AXIS_NAMES
    face = compute_face_scores(domains)                       # FACE_D, FACE_M

    # 1. parsimony — do the short scores reproduce the axes they target?
    j = face.index.intersection(axes.index)
    def _corr(a, b):
        m = a.notna() & b.notna()
        return float(np.corrcoef(a[m], b[m])[0, 1])
    r_d = _corr(face.loc[j, "FACE_D"], axes.loc[j, "depression_severity"])
    r_m = _corr(face.loc[j, "FACE_M"], axes.loc[j, "metabolic"])
    print(f"parsimony: corr(FACE-D, depression axis)={r_d:.2f} | corr(FACE-M, metabolic axis)={r_m:.2f}")

    # build pid keys for the outcome join
    for d in (face, axes):
        d["pid"] = (d.index.get_level_values("cohort") + "::" + d.index.get_level_values("patient_id"))
    F = face.set_index("pid")[["FACE_D", "FACE_M"]]
    A = axes.set_index("pid")[AXIS_NAMES]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
    df["pid"] = df["cohort"].str.lower() + "::" + df["usubjid_patients"].astype(str)
    v0 = df[df["visit"] == "V0"].set_index("pid")
    vk = df[df["visit"] == "V1"].set_index("pid")
    base = pd.DataFrame(index=v0.index)
    base["age"] = pd.to_numeric(v0["age"], errors="coerce")
    base["sex"] = pd.to_numeric(v0["sex"], errors="coerce")
    dsm = pd.get_dummies(v0["arm"].astype(str), drop_first=True).astype(float)
    dsm_cols = list(dsm.columns)
    base = base.join(dsm)

    # 2. head-to-head: does DSM + the 2 FACE scores recover the DSM + 6-axis advantage?
    rows = []
    for name, kind, col, tf in OUTCOMES:
        if col not in df.columns:
            continue
        y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
        yk = pd.to_numeric(vk[col], errors="coerce")
        if tf is not None:
            yk = tf(yk)
        d = (base.join(y0).join(F).join(A).join(yk.rename("y"))
             .dropna(subset=["y", "baseline", "age", "sex", "FACE_D", "FACE_M"] + AXIS_NAMES))
        if kind == "binary" and (d["y"].nunique() < 2 or d["y"].mean() < 0.02):
            continue
        if len(d) < 200:
            continue
        bc = ["baseline", "age", "sex"]
        y = d["y"].to_numpy(float)
        m_dsm = cv_metric(d[bc + dsm_cols].to_numpy(float), y, kind)
        m_face = cv_metric(d[bc + dsm_cols + ["FACE_D", "FACE_M"]].to_numpy(float), y, kind)
        m_full = cv_metric(d[bc + dsm_cols + AXIS_NAMES].to_numpy(float), y, kind)
        metric = "R2" if kind == "continuous" else "AUC"
        rows.append({"outcome": name, "n": len(d), "metric": metric,
                     "DSM": round(m_dsm, 3), "DSM+FACE": round(m_face, 3), "DSM+6axes": round(m_full, 3),
                     "FACE_gain": round(m_face - m_dsm, 3), "full_gain": round(m_full - m_dsm, 3),
                     "FACE_recovers_pct": (round(100 * (m_face - m_dsm) / (m_full - m_dsm))
                                           if abs(m_full - m_dsm) > 1e-6 else None)})
        print(f"{name}: n={len(d)} {metric}  DSM={m_dsm:.3f}  +FACE={m_face:.3f}  +6axes={m_full:.3f}  "
              f"(FACE gain {m_face-m_dsm:+.3f} vs full {m_full-m_dsm:+.3f})")

    # cohort gradients (interpretation) + static figure (Fig 7)
    fa = pd.DataFrame({"FACE_D": face["FACE_D"].to_numpy(), "FACE_M": face["FACE_M"].to_numpy(),
                       "cohort": face.index.get_level_values("cohort")})
    cohmean = fa.groupby("cohort")[["FACE_D", "FACE_M"]].mean().round(3)
    print("\nFACE means by cohort (z):\n" + cohmean.to_string())
    _face_figure(face, axes)
    print("  wrote reports/figures/fig7_face_profile.png/.svg")

    out = {"parsimony": {"corr_FACE_D_depression": round(r_d, 3), "corr_FACE_M_metabolic": round(r_m, 3)},
           "face_means_by_cohort": {c: {"FACE_D": float(cohmean.loc[c, "FACE_D"]),
                                        "FACE_M": float(cohmean.loc[c, "FACE_M"])}
                                    for c in cohmean.index},
           "FACE_D_items": list(FACE_D_ITEMS),
           "FACE_M_items": list(FACE_M_ITEMS),
           "headtohead": rows,
           "note": "Translational proposal (MANUSCRIPT §3.9); needs prospective validation. FACE-D "
                   "targets patient-reported outcomes; FACE-M is a trait-stable cardiometabolic risk flag."}
    (RESULTS_DIR / "face_score_validation.json").write_text(json.dumps(out, indent=2))

    # report
    hh = pd.DataFrame(rows)
    bars = go.Figure()
    for c, color in [("DSM", "#888"), ("DSM+FACE", "#2b8cbe"), ("DSM+6axes", "#08589e")]:
        bars.add_bar(name=c, x=hh["outcome"], y=hh[c], marker_color=color)
    bars.update_layout(barmode="group", title="FACE profile vs DSM vs full 6-axis model (CV R²/AUC)",
                       height=380, margin=dict(t=46), yaxis_title="cross-validated R² / AUC")
    css = "body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}h1{color:#2b3a55}.c{background:#eef6fb;border-left:4px solid #2b8cbe;padding:10px 14px;margin:12px 0}"
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>The FACE profile — clinically-actionable trans-diagnostic indices</h1>",
            f"<div class='c'>FACE-D (QIDS+MADRS+STAI) reproduces the depression axis r={r_d:.2f}; "
            f"FACE-M (metabolic syndrome+cholesterol+inflammation) the metabolic axis r={r_m:.2f}. "
            "Below: added to DSM, does the 2-score profile recover the full 6-axis advantage?</div>",
            hh.to_html(index=False, border=0),
            pio.to_html(bars, include_plotlyjs="cdn", full_html=False), "</body></html>"]
    (REPORTS_DIR / "face_score.html").write_text("\n".join(html), encoding="utf-8")
    print("\nWrote results/face_score_validation.json + reports/face_score.html. Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
