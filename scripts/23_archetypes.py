#!/usr/bin/env python3
"""23 — M2.3 archetypal analysis: soft archetype membership (the LEAD view; §3.3).

Continuum verdict (M2.1) ⇒ represent each patient as a convex blend of a few EXTREME PHENOTYPES
(archetypes), not a hard cluster. Soft simplex weights = the continuum-honest probabilistic decision
regions. Fit on both G-arms (A: all 9 = full phenotype; B: 8 specifics = pure profile), uncertainty-aware
(M1 draws projected onto fixed anchor archetypes), with the archetype atlas + diagnostic composition at
both granularities (cohort + 7 DSM-5 subtypes).

    python3 scripts/23_archetypes.py

Reads results/face/m2/{coordinates_full.parquet, coordinates_draws.npz, validation_table.parquet}.
Writes results/face/m2/{archetypes.parquet, archetype_profiles.csv} + reports/23_archetypes.md
      + docs/figures/23_{scree,profiles,membership}.png.
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

REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
M2 = REPO / "results" / "face" / "m2"
SEED = 20260609
CANON = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep", "mania_activation",
         "suicidality", "developmental_risk", "substance"]


def main():
    from face.strata import archetypes as ar

    df = pd.read_parquet(M2 / "coordinates_full.parquet")
    vt = pd.read_parquet(M2 / "validation_table.parquet")
    dz = np.load(M2 / "coordinates_draws.npz", allow_pickle=True)
    draws = dz["draws"]
    X9 = df[[f"{f}__mean" for f in CANON]].to_numpy()
    axesB = [a for a in CANON if a != "overall_severity"]
    colsB = [CANON.index(a) for a in axesB]
    X8 = X9[:, colsB]

    # ---- choose A on Arm A (scree / knee) ----
    import time
    t0 = time.time()
    print("[1/4] selecting number of archetypes A (scree on Arm A)...", flush=True)
    selA = ar.select_A(X9, range(2, 9), seed=SEED)              # scree (no elbow → continuum; see 23b)
    A = 8                                                        # PI-confirmed at the gate (2026-06-09):
    # the only A resolving BOTH biology corners (metabolic + inflammatory) — the biology⊥G headline; one
    # extreme phenotype per specific axis + a low-burden pole (severity = spine, substance absorbed). 23b.
    print(f"  A_knee={selA['A_knee']} → using A={A}; ev(A)={selA['explained_variance'].get(A):.3f} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- anchor fit (Arm A = full phenotype) ----
    print(f"[2/4] anchor archetypes A={A} (Arm A) + stability + draw-uncertainty...", flush=True)
    _, Z, W, rss = ar.fit_aa(X9, A, seed=SEED, n_init=4)
    ev = ar.explained_variance(X9, rss)
    stab = ar.stability(X9, A, seeds=(0, 1, 2))
    names = ar.name_archetypes(Z, CANON)
    unc = ar.project_draws(Z, draws, list(range(9)), n_draw=40, seed=SEED)
    Wsd = unc["sd"]
    print(f"  [{time.time()-t0:.0f}s] ev={ev:.3f} min-congruence={stab['min_tucker_congruence']:.3f}", flush=True)

    # ---- Arm B (pure profile) at same A ----
    _, ZB, WB, rssB = ar.fit_aa(X8, A, seed=SEED, n_init=4)
    namesB = ar.name_archetypes(ZB, axesB)
    print(f"[3/4] building atlas + figures [{time.time()-t0:.0f}s]...", flush=True)

    # ---- per-patient membership ----
    dom = W.argmax(1)
    ent = (-(W * np.log(np.clip(W, 1e-9, 1))).sum(1) / np.log(A))   # normalized entropy 0..1
    out = df[["cohort", "patient_id"]].copy()
    for a in range(A):
        out[f"w{a}_mean"] = W[:, a].round(4)
        out[f"w{a}_sd"] = Wsd[:, a].round(4)
    out["dominant"] = dom
    out["dominant_name"] = [names[d] for d in dom]
    out["entropy"] = ent.round(3)
    out.to_parquet(M2 / "archetypes.parquet")

    prof = pd.DataFrame(Z, columns=CANON); prof.insert(0, "name", names); prof.insert(0, "arm", "A_all9")
    profB = pd.DataFrame(ZB, columns=axesB); profB.insert(0, "name", namesB); profB.insert(0, "arm", "B_specifics")
    pd.concat([prof, profB], ignore_index=True).round(3).to_csv(M2 / "archetype_profiles.csv", index=False)

    # ---- figures ----
    _fig_scree(selA, A)
    _fig_profiles(Z, names, A)
    diagA = _fig_membership(X9, df, vt, W, dom, names, A)

    # ---- diagnostic composition (two granularities) ----
    comp_coh = pd.crosstab(pd.Series(dom, name="archetype"), vt["cohort"])
    comp_arm = pd.crosstab(pd.Series([names[d] for d in dom], name="archetype"), vt["arm"].astype(str))
    share = pd.Series(np.bincount(dom, minlength=A) / len(dom), index=range(A)).round(3)
    clear = float((W.max(1) >= 0.5).mean())

    md = ["# 23 — M2.3 archetypal analysis (soft archetype membership — the lead view)", "",
          "Continuum verdict (M2.1) ⇒ each patient = a convex blend of **extreme phenotypes**, not a hard "
          "cluster. Soft simplex weights = the probabilistic decision regions. Native latent z-scale; "
          "uncertainty from projecting M1 draws onto fixed archetypes. **No biotype claim.**", "",
          f"## Number of archetypes A = **{A}** (data-driven knee on Arm A = {selA['A_knee']})",
          f"- explained variance at A={A}: Arm A **{ev:.3f}** · Arm B {ar.explained_variance(X8, rssB):.3f}",
          f"- stability across seeds: min Tucker congruence **{stab['min_tucker_congruence']:.3f}**, "
          f"mean profile SD {stab['profile_across_seed_sd']:.3f}",
          f"- membership: **{clear:.0%}** of patients have a clear dominant archetype (max weight ≥0.5); "
          f"mean normalized entropy {ent.mean():.2f} (1 = fully blended — expected on a continuum)", "",
          "## Archetype profiles — Arm A (full phenotype, z-units; the extreme phenotypes)",
          pd.DataFrame(Z, columns=CANON, index=[f"A{a}: {names[a]}" for a in range(A)]).round(2).to_markdown(), "",
          f"Population share by dominant archetype: {share.to_dict()}", "",
          "## Archetype profiles — Arm B (pure profile, G removed)",
          pd.DataFrame(ZB, columns=axesB, index=[f"B{a}: {namesB[a]}" for a in range(A)]).round(2).to_markdown(), "",
          "## Diagnostic composition (Q3 preview — two granularities; validation-only)",
          "By cohort (counts):", comp_coh.to_markdown(), "",
          "By DSM-5 subtype (counts):", comp_arm.to_markdown(), "",
          "## Reading",
          "- Archetypes are **corners of the data's convex hull** — extreme phenotypes that span the "
          "continuum; most patients are blends (high entropy), consistent with M2.1.",
          "- This is the lead M2 deliverable; the mixture (M2.2) will overlay a soft tessellation. Whether "
          "archetypes are **better than DSM-5** is the M4/M5 predictive/treatment head-to-head (§1.7), not "
          "decided here.", "",
          "## Artifacts",
          "- `results/face/m2/archetypes.parquet` — per-patient weights (mean+sd) · dominant · entropy.",
          "- `results/face/m2/archetype_profiles.csv` — both arms' archetype profiles.",
          "- Figures: `docs/figures/23_{scree,profiles,membership}.png`."]
    (REPORTS / "23_archetypes.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n[done] A={A}, explained var {ev:.3f}, {clear:.0%} clear-dominant")


def _fig_scree(selA, A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ks = sorted(selA["explained_variance"])
    ax.plot(ks, [selA["explained_variance"][k] for k in ks], "-o", label="A (all 9)", color="#2c7fb8")
    ax.axvline(A, ls="--", c="grey", label=f"chosen A={A}")
    ax.set_xlabel("number of archetypes A"); ax.set_ylabel("explained variance")
    ax.set_title("M2.3 — archetype scree (knee → A)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "23_scree.png", dpi=130); plt.close(fig)


def _fig_profiles(Z, names, A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 0.7 * A + 2))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(CANON))); ax.set_xticklabels(CANON, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(range(A)); ax.set_yticklabels([f"A{a}: {names[a]}" for a in range(A)], fontsize=9)
    for i in range(A):
        for j in range(len(CANON)):
            ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(Z[i, j]) > 1.2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="archetype coordinate (z)")
    ax.set_title("M2.3 — archetype profiles (Arm A, full phenotype)")
    fig.tight_layout(); fig.savefig(FIGS / "23_profiles.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def _fig_membership(X9, df, vt, W, dom, names, A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED).fit_transform(X9)
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    cmap = plt.cm.tab10
    for a in range(A):
        m = dom == a
        ax[0].scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.5, color=cmap(a), label=f"A{a}")
    ax[0].set_title("UMAP — dominant archetype"); ax[0].legend(markerscale=3, fontsize=7)
    # composition by cohort
    coh = vt["cohort"].to_numpy()
    cc = pd.crosstab(pd.Series(dom, name="arch"), pd.Series(coh, name="cohort"))
    cc.plot(kind="bar", stacked=True, ax=ax[1], colormap="Set2"); ax[1].set_title("dominant archetype × cohort")
    ax[1].set_xlabel("archetype"); ax[1].tick_params(axis="x", rotation=0)
    # composition by DSM-5 subtype
    arm = vt["arm"].astype(str).to_numpy()
    ca = pd.crosstab(pd.Series(dom, name="arch"), pd.Series(arm, name="dsm5"))
    ca.plot(kind="bar", stacked=True, ax=ax[2], colormap="tab20"); ax[2].set_title("dominant archetype × DSM-5 subtype")
    ax[2].set_xlabel("archetype"); ax[2].legend(fontsize=6); ax[2].tick_params(axis="x", rotation=0)
    fig.suptitle("M2.3 — archetype membership & diagnostic composition", y=1.02)
    fig.tight_layout(); fig.savefig(FIGS / "23_membership.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return {"ok": True}


if __name__ == "__main__":
    main()
