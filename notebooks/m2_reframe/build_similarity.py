#!/usr/bin/env python3
"""M2 patient-positioning + similarity tool — the continuum-honest core of the reframed M2.

The map is a continuum, so the most faithful clinical product is not a typology but: for any patient, (a) their
position on the validated axes (already in the coordinates) + their corner weights (the reading guide), and
(b) their NEAREST REAL NEIGHBOURS — "patients most like this one." This makes the archetype corners / tessellation
genuinely interpretation lenses, not the load-bearing object.

Uncertainty-aware distance (diagonal posteriors N(μ_i, σ_i²)):

    d²(i,j) = Σ_d (μ_i,d − μ_j,d)² / (σ_i,d² + σ_j,d² + ε)

i.e. a per-dimension standardized distance under the combined measurement uncertainty: an axis a patient is
uncertain about contributes little, so an ill-measured patient gets a *fuzzy* (indistinct) neighbourhood — exactly
as it should. Computed two ways: ARM A = all 9 axes ("similar overall, incl. severity"); ARM B = the 8 specifics
with severity dropped ("similar in clinical kind, regardless of how ill").

Reads existing strata_oop coordinates (means + SDs) — no fitting. Writes the top-k neighbour hand-off and prints a
demonstration on the four reading-guide medoid exemplars.

    PYTHONPATH=$PWD/src python notebooks/m2_reframe/build_similarity.py
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
R = REPO / "results" / "face" / "strata_oop"
OUTD = R / "similarity"; OUTD.mkdir(parents=True, exist_ok=True)

AX = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
      "mania_activation", "suicidality", "developmental_risk", "substance"]
SPECIFICS = [a for a in AX if a != "overall_severity"]          # arm B = severity removed
SH = {"overall_severity": "sev", "cognition": "cog", "metabolic": "metab", "inflammatory": "inflam",
      "sleep": "sleep", "mania_activation": "mania", "suicidality": "suic", "developmental_risk": "devel",
      "substance": "subst"}
EPS = 1e-3
K = 10


def _knn(mu, sd, cols, k=K, chunk=512):
    """Top-k uncertainty-weighted neighbours for every patient over the given column indices."""
    M, V = mu[:, cols], sd[:, cols] ** 2
    N = M.shape[0]
    nbr = np.zeros((N, k), dtype=int); dst = np.zeros((N, k))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        # d2[q, N] = Σ_d (Mq - M)² / (Vq + V + eps)
        num = (M[s:e, None, :] - M[None, :, :]) ** 2
        den = V[s:e, None, :] + V[None, :, :] + EPS
        d2 = (num / den).sum(-1)
        for r in range(e - s):
            d2[r, s + r] = np.inf                                # exclude self
        part = np.argpartition(d2, k, axis=1)[:, :k]
        for r in range(e - s):
            order = part[r][np.argsort(d2[r, part[r]])]
            nbr[s + r] = order; dst[s + r] = d2[r, order]
    return nbr, dst


def main() -> None:
    co = pd.read_parquet(R / "coordinates" / "coordinates_full.parquet").reset_index()
    ps = pd.read_parquet(R / "consolidate" / "patient_strata.parquet").reset_index()
    m = co.merge(ps[["cohort", "patient_id", "arch_dominant", "arch_dominant_name", "arch_entropy"]],
                 on=["cohort", "patient_id"], how="inner")
    assert len(m) == len(co), f"merge blew up: {len(m)} vs {len(co)} (patient_id not unique across cohorts)"
    mu = m[[f"{a}__mean" for a in AX]].to_numpy("float64")
    sd = m[[f"{a}__sd" for a in AX]].to_numpy("float64")
    pid = pd.to_numeric(m["patient_id"], errors="coerce").astype("Int64").to_numpy()
    coh = m["cohort"].astype(str).to_numpy(); dom = m["arch_dominant"].to_numpy()
    uid = np.array([f"{c}/{p}" for c, p in zip(coh, pid)])      # unambiguous key (patient_id repeats across cohorts)
    colA = list(range(len(AX)))
    colB = [AX.index(a) for a in SPECIFICS]

    out = {"A": _knn(mu, sd, colA), "B": _knn(mu, sd, colB)}
    # persist hand-off (neighbour patient_ids + distances per arm)
    rec = {"uid": uid, "cohort": coh, "patient_id": pid, "arch_dominant": dom}
    for arm in ("A", "B"):
        nbr, dst = out[arm]
        rec[f"neighbors_{arm}"] = [list(uid[nbr[i]]) for i in range(len(uid))]
        rec[f"dist_{arm}"] = [list(np.round(dst[i], 3)) for i in range(len(uid))]
    pd.DataFrame(rec).to_parquet(OUTD / "neighbors.parquet")
    print(f"wrote {OUTD/'neighbors.parquet'}  ({len(pid)} patients × top-{K}, two arms)")

    # --- demonstration on the four reading-guide medoid exemplars ---
    LAB = {0: "High biological load", 1: "Low burden", 2: "Severe, low-biology", 3: "Symptom-driven"}
    rg = pd.read_csv(R / "reading_guide" / "reading_guide.csv")
    pos = {(c, int(p)): i for i, (c, p) in enumerate(zip(coh, pid))}

    def prof(i, cols):
        v = mu[i]; order = sorted(cols, key=lambda d: -abs(v[d]))[:4]
        return "  ".join(f"{('+' if v[d] >= 0 else '')}{v[d]:.2f}{SH[AX[d]]}" for d in order)

    for _, r in rg.iterrows():
        q = pos[(r.medoid_cohort, int(r.medoid_patient_id))]
        print(f"\n{'='*96}\nEXEMPLAR  A{int(r.archetype)} «{LAB[int(r.archetype)]}»  — {coh[q]}/{pid[q]}  "
              f"(corner-entropy {m.loc[q,'arch_entropy']:.2f})")
        print(f"  profile (all 9): {prof(q, colA)}")
        for arm, cols, tag in [("A", colA, "similar OVERALL (all 9)"), ("B", colB, "similar IN KIND (no severity)")]:
            nbr, dst = out[arm]
            ns = nbr[q][:5]
            cohmix = "/".join(f"{c}{sum(coh[ns]==c)}" for c in ['bp','sz','dr'])
            dommix = "/".join(f"A{k}:{sum(dom[ns]==k)}" for k in range(4))
            print(f"  → {tag}: nearest5 cohorts {cohmix} · corners {dommix} · d²[{dst[q][0]:.2f}–{dst[q][4]:.2f}]")
    print("\n(neighbourhood d² small+tight = a well-located patient; large/flat = uncertain or sparse region.)")

    # --- illustration figure: the biological-pole exemplar and its nearest real neighbours ---
    qrow = rg[rg.archetype == 0].iloc[0]
    q = pos[(qrow.medoid_cohort, int(qrow.medoid_patient_id))]
    COHC = {"bp": "#2B4C8C", "sz": "#B42318", "dr": "#0F766E"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), sharey=True)
    for ax, (arm, cols, title) in zip(axes, [("A", colA, "similar OVERALL (all 9 axes)"),
                                             ("B", colB, "similar IN KIND (severity removed)")]):
        nbr = out[arm][0][q][:8]
        xs = list(range(len(cols)))
        for j in nbr:
            ax.plot(xs, [mu[j][d] for d in cols], "-", color=COHC.get(coh[j], "#999"), lw=0.9, alpha=0.55)
        ax.plot(xs, [mu[q][d] for d in cols], "-o", color="#14181F", lw=2.6, ms=4, zorder=5)
        ax.axhline(0, color="#5B6573", lw=0.6, ls=":")
        ax.set_xticks(xs); ax.set_xticklabels([SH[AX[d]] for d in cols], rotation=40, ha="right", fontsize=8)
        cm = "  ".join(f"{c.upper()}:{int(sum(coh[nbr] == c))}" for c in ["bp", "sz", "dr"])
        ax.set_title(f"{title}\nnearest 8 neighbours — {cm}", fontsize=9.5)
    axes[0].set_ylabel("coordinate (SD units)")
    axes[1].legend(handles=[Line2D([0], [0], color=COHC[c], lw=2, label=c.upper()) for c in ["bp", "sz", "dr"]]
                   + [Line2D([0], [0], color="#14181F", lw=2.6, marker="o", label="query patient")],
                   frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Patient similarity — a biological-pole patient and its nearest real neighbours (transdiagnostic)",
                 y=1.02, fontsize=11, fontweight="bold", color="#1E366B")
    fig.tight_layout()
    for p in [REPO / "report" / "figures" / "m2_similarity.png",
              REPO / "docs" / "figures" / "strata_oop" / "similarity.png"]:
        fig.savefig(p, bbox_inches="tight", facecolor="white", dpi=150)
    print(f"wrote {REPO/'report'/'figures'/'m2_similarity.png'} (+ docs/figures/strata_oop/similarity.png)")


if __name__ == "__main__":
    main()
