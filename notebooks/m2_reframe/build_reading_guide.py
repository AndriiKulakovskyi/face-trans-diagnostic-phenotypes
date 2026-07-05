#!/usr/bin/env python3
"""M2 reading guide — the interpretation layer of the reframed M2 (coordinate system + reading guide).

The M2 verdict is a CONTINUUM (no biotypes). The load-bearing object is therefore the continuous 9-dim
copula coordinates, not a typology. This script builds the human-readable *reading guide* for those
coordinates: the four archetype CORNERS (the poles of the cloud, which carry the biology⊥severity structure),
and for each corner a clinically plain label + a "typical representative" (the centroid of patients whose
dominant pull is that corner) + a real MEDOID exemplar patient. Corners/centroids are interpretation lenses on
the continuum — NOT discovered subgroups.

Reads existing strata_oop artifacts only (no fitting). Writes:
  - results/m2_strata/reading_guide/reading_guide.csv
  - docs/figures/strata_oop/reading_guide.png

    PYTHONPATH=$PWD/src python notebooks/m2_reframe/build_reading_guide.py
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
R = REPO / "results" / "face" / "strata_oop"
OUTD = R / "reading_guide"; OUTD.mkdir(parents=True, exist_ok=True)
FIGD = REPO / "docs" / "figures" / "strata_oop"; FIGD.mkdir(parents=True, exist_ok=True)

AX = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
      "mania_activation", "suicidality", "developmental_risk", "substance"]
PRETTY = {"overall_severity": "severity (G)", "cognition": "cognition", "metabolic": "metabolic",
          "inflammatory": "inflammatory", "sleep": "sleep", "mania_activation": "mania",
          "suicidality": "suicidality", "developmental_risk": "developmental", "substance": "substance"}

# Plain clinical labels, curated from the corner z-profiles (handle | one-line clinical descriptor).
# A0/A2 are the biology⊥severity pair: both elevated severity, biologically inverse.
LABELS = {
    0: ("High biological load",
        "elevated cardiometabolic markers, systemic inflammation and substance use, with above-average overall severity"),
    1: ("Low burden (relatively well)",
        "mild across symptoms and biology — comparatively well-functioning"),
    2: ("Severe, low biological load",
        "high overall clinical severity without the metabolic / inflammatory / substance signature"),
    3: ("Symptom-driven",
        "disrupted sleep/circadian, high early-life adversity and mood activation/suicidality, with low metabolic load"),
}


def main() -> None:
    prof = pd.read_csv(R / "consolidate" / "archetype_profiles.csv")
    prof = prof[prof.arm == "A_all9"].set_index("archetype")
    ps = pd.read_parquet(R / "consolidate" / "patient_strata.parquet").reset_index()
    co = pd.read_parquet(R / "coordinates" / "coordinates_full.parquet").reset_index()
    keep = ["cohort", "patient_id", "arch_w0", "arch_w1", "arch_w2", "arch_w3", "arch_dominant"]
    m = co.merge(ps[keep], on=["cohort", "patient_id"], how="inner")
    assert len(m) == len(co), f"merge blew up: {len(m)} vs {len(co)} (patient_id not unique across cohorts)"
    Xc = m[[f"{a}__mean" for a in AX]].to_numpy()

    rows, cent_mat = [], []
    for k in range(4):
        pole = prof.loc[k, AX].to_numpy(float)
        dom = m[m.arch_dominant == k]
        idx = dom.index.to_numpy()
        cent = Xc[idx].mean(0)
        med = idx[int(np.argmin(((Xc[idx] - cent) ** 2).sum(1)))]
        mr = m.loc[med]
        cohorts = {c: int((dom.cohort == c).mean() * 100) for c in ["bp", "sz", "dr"]}
        handle, desc = LABELS[k]
        rows.append({
            "archetype": k, "label": handle, "clinical_descriptor": desc,
            "n": len(dom), "pct": round(len(dom) * 100 / len(m), 1),
            "cohort_bp": cohorts["bp"], "cohort_sz": cohorts["sz"], "cohort_dr": cohorts["dr"],
            "medoid_cohort": mr.cohort, "medoid_patient_id": mr.patient_id,
            "medoid_weight": round(float(mr[f"arch_w{k}"]), 2),
            **{f"pole_{a}": round(float(pole[i]), 3) for i, a in enumerate(AX)},
            **{f"typical_{a}": round(float(cent[i]), 3) for i, a in enumerate(AX)},
        })
        cent_mat.append(cent)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTD / "reading_guide.csv", index=False)
    print(tab[["archetype", "label", "n", "pct", "cohort_bp", "cohort_sz", "cohort_dr",
               "medoid_cohort", "medoid_patient_id"]].to_string(index=False))

    # --- figure: the four representative profiles (typical-member centroids), plain-labelled ---
    DIV = LinearSegmentedColormap.from_list("d", ["#B42318", "#f6e9e6", "#f7f7f7", "#dbe5f3", "#2B4C8C"])
    C = np.array(cent_mat)                                   # [4,9] but plot severity-independent ordering
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    im = ax.imshow(C, cmap=DIV, norm=TwoSlopeNorm(0, vmin=-1.0, vmax=1.0), aspect="auto")
    ax.set_xticks(range(len(AX))); ax.set_xticklabels([PRETTY[a] for a in AX], rotation=35, ha="right", fontsize=8.5)
    ylab = [f"A{k} · {LABELS[k][0]}\n({rows[k]['pct']:.0f}% · bp{rows[k]['cohort_bp']}/sz{rows[k]['cohort_sz']}/dr{rows[k]['cohort_dr']})"
            for k in range(4)]
    ax.set_yticks(range(4)); ax.set_yticklabels(ylab, fontsize=8)
    for k in range(4):
        for i in range(len(AX)):
            v = C[k, i]
            if abs(v) >= 0.25:
                ax.text(i, k, f"{v:+.2f}", ha="center", va="center", fontsize=6.6,
                        color="white" if abs(v) > 0.6 else "#14181F")
    ax.set_title("M2 reading guide — four representative profiles on the continuum (typical-member centroids)",
                 fontsize=10.5, fontweight="bold", color="#1E366B", pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("mean coordinate (SD units)", fontsize=8)
    fig.text(0.5, -0.02, "A0 and A2 sit at the same severity but are biologically inverse — the biology⊥severity result, "
             "as two representative patients. Poles (pure extremes) are ~3× stronger; these are the typical members.",
             ha="center", fontsize=7.2, color="#5B6573")
    fig.tight_layout()
    for p in [FIGD / "reading_guide.png", REPO / "report" / "figures" / "m2_reading_guide.png"]:
        fig.savefig(p, bbox_inches="tight", facecolor="white", dpi=150)
    print(f"\nwrote {OUTD/'reading_guide.csv'}\nwrote {FIGD/'reading_guide.png'} (+ report/figures/m2_reading_guide.png)")


if __name__ == "__main__":
    main()
