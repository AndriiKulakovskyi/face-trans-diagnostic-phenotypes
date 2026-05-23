"""Publication-quality static figures for the manuscript (PNG + SVG, 300 dpi).

Regenerates Figures 1-5 directly from the reproducible result artifacts in
results/ (not from the interactive HTML reports), so the manuscript figures are
self-contained and version-controlled.

  Fig 1  structure is dimensional   (results/structure_test.json + domain scores)
  Fig 2  six-dimension loadings     (results/dimensional_final_loadings.csv)
  Fig 3  head-to-head outcomes      (results/phase5_headtohead_V1/_V2.csv)
  Fig 4  trait-state gradient       (results/longitudinal_axes_stability.csv)
  Fig 5  cognition g + speed        (results/cognition_bpsz_loadings.csv + _corr.csv)

Output: reports/figures/*.png and *.svg
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "archive"))
RESULTS = REPO_ROOT / "results"
FIGDIR = REPO_ROOT / "reports" / "figures"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "figure.dpi": 120, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "font.family": "DejaVu Sans",
})

AXIS_NAME = {
    "axis1": "Depression /\ninternalizing", "axis2": "Later onset",
    "axis3": "Mania /\nactivation", "axis4": "Illness burden",
    "axis5": "Metabolic /\ninflammatory", "axis6": "ADHD /\nimpulsivity / trauma",
}
DOMAIN_LABEL = {
    "qidsr": "QIDS", "madrs": "MADRS", "staya": "STAI-trait", "fast": "FAST (impair.)",
    "egf": "EGF (funct.)", "eq5d": "EQ-5D", "eq": "EQ-VAS", "mars": "MARS (adher.)",
    "mathys": "Mathys", "altman": "Altman", "ymrs": "YMRS", "psqi": "PSQI (sleep)",
    "bis": "BIS (impuls.)", "wurs": "WURS (ADHD)", "ctq": "CTQ (trauma)",
    "ess": "ESS (sleepy)", "prism": "PRISM (stigma)", "isf": "ISF (suicid.)",
    "csm": "CSM (chronotype)", "remisc": "remission",
    "nboccur_hospitalisation_lt": "# hospitalizations",
    "hodur_hospitalisation_lt": "hosp. duration",
    "agedebut_hospitalisation": "age 1st hosp.", "agetrt": "age at treatment",
    "agedebutpremier_episode": "age 1st episode", "edulevel": "education",
    "metabolic_syndrome": "metabolic syndrome", "cholesterol": "cholesterol",
    "inflammation": "inflammation", "hepatic": "hepatic", "renal": "renal",
    "prolactin": "prolactin", "cardiac_qtc": "QTc",
    "hooccur_arret_travail_actuel": "current sick-leave",
}
COG_LABEL = {
    "memory_cvlt": "Verbal memory\n(CVLT)", "executive_tmt": "Executive\n(TMT)",
    "proc_speed": "Processing\nspeed", "working_memory": "Working\nmemory",
    "verbal_reasoning": "Verbal\nreasoning", "percept_reasoning": "Perceptual\nreasoning",
    "fluency": "Fluency",
}
AXIS_SHORT = ["Depression", "Later onset", "Mania", "Illness burden",
              "Metabolic", "ADHD/trauma"]
SUBTYPE_SHORT = {
    "Trouble dépressif majeur": "MDD", "Bipolaire de type 2": "BP-II",
    "Bipolaire de type 1": "BP-I", "Bipolaire non spécifié": "BP-NOS",
    "Trouble schizo-affectif": "schizoaff.", "Trouble schizophréniforme": "schizophrenif.",
    "Schizophrénie": "schizophrenia",
}
SPECTRUM = {"Trouble dépressif majeur": 0, "Bipolaire de type 2": 1,
            "Bipolaire de type 1": 2, "Bipolaire non spécifié": 3,
            "Trouble schizo-affectif": 4, "Trouble schizophréniforme": 5,
            "Schizophrénie": 6}
C_DSM, C_AX, C_COMB = "#9e9e9e", "#1f77b4", "#2ca02c"


def save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIGDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote reports/figures/{name}.png + .svg")


# ---------------------------------------------------------------- Fig 1
def fig1_structure():
    st = json.loads((RESULTS / "structure_test.json").read_text())
    fig, ax = plt.subplots(2, 2, figsize=(9, 7))

    # (a) eigenvalue spectrum
    ev = np.array(st["eigengap"]["eigenvalues"])
    ax[0, 0].plot(range(1, len(ev) + 1), ev, "o-", color="#333", ms=4)
    ax[0, 0].set(title="(a) Laplacian spectrum — no eigengap",
                 xlabel="eigenvalue index", ylabel="eigenvalue")
    ax[0, 0].annotate("smooth rise from ~0\n⇒ no natural cluster count",
                      xy=(6, ev[5]), xytext=(4.2, ev.max() * 0.62), fontsize=8,
                      arrowprops=dict(arrowstyle="->", color="#888"))

    # (b) gap vs k (monotone)
    g = pd.DataFrame(st["gap_vs_gaussian"])
    ax[0, 1].plot(g["k"], g["silhouette_real"], "o-", label="real", color=C_AX, ms=4)
    ax[0, 1].plot(g["k"], g["silhouette_null"], "s--", label="Gaussian null",
                  color="#bbb", ms=4)
    ax[0, 1].plot(g["k"], g["gap"], "^-", label="gap", color="#d62728", ms=4)
    ax[0, 1].set(title="(b) Gap statistic rises monotonically",
                 xlabel="k (clusters)", ylabel="silhouette / gap")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].annotate("gap never peaks ⇒ continuum", xy=(11, g["gap"].iloc[-2]),
                      xytext=(5.5, 0.30), fontsize=8,
                      arrowprops=dict(arrowstyle="->", color="#888"))

    # (c) bimodality of top PCs
    bc = np.array(st["bimodality_pc"])
    bars = ax[1, 0].bar(range(1, len(bc) + 1), bc, color="#7e57c2")
    ax[1, 0].axhline(0.555, ls="--", color="#d62728")
    ax[1, 0].text(len(bc) + 0.1, 0.555, "  uniform\n  benchmark", va="center",
                  color="#d62728", fontsize=8)
    ax[1, 0].set(title="(c) No axis is clearly multimodal", xlabel="principal axis",
                 ylabel="Sarle bimodality coeff.", ylim=(0, 0.8))

    # (d) DSM-subtype mood↔psychosis continuum (PC1 of standardized domains)
    try:
        from sklearn.decomposition import PCA
        from face_common import (build_unified_dataframe, load_variables,
                                  to_harmonized_dataset)
        from face_common.adapter import ADMINISTRATIVE_FEATURES
        sc = pd.read_parquet(RESULTS / "cluster_domains_scores.parquet")
        sc.index = pd.MultiIndex.from_arrays(
            [sc.index.get_level_values("cohort").astype(str),
             sc.index.get_level_values("patient_id").astype(str)],
            names=("cohort", "patient_id"))
        z = ((sc - sc.mean()) / sc.std(ddof=0)).fillna(0.0)
        pc1 = PCA(n_components=1, random_state=0).fit_transform(z.to_numpy())[:, 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            long = build_unified_dataframe(REPO_ROOT / "data",
                                           REPO_ROOT / "face-common-vars.xlsx",
                                           readiness=["READY", "PARTIAL"], format="long")
            full = to_harmonized_dataset(long, load_variables(
                REPO_ROOT / "face-common-vars.xlsx"), visit="V0",
                exclude=ADMINISTRATIVE_FEATURES)
        dsm = full.metadata.reindex(sc.index)["dsm_diagnosis"].astype(str)
        rank = dsm.map(SPECTRUM).to_numpy(float)
        m = ~np.isnan(rank)
        # orient PC1 so it increases with the mood→psychosis rank
        if spearmanr(pc1[m], rank[m]).correlation < 0:
            pc1 = -pc1
        order = sorted(SPECTRUM, key=SPECTRUM.get)
        means = [np.nanmean(pc1[(dsm == s).to_numpy() & m]) for s in order]
        sems = [np.nanstd(pc1[(dsm == s).to_numpy() & m]) /
                np.sqrt(max(1, ((dsm == s).to_numpy() & m).sum())) for s in order]
        ax[1, 1].errorbar(range(len(order)), means, yerr=sems, fmt="o-",
                          color="#e8710a", ms=5, capsize=3)
        ax[1, 1].set_xticks(range(len(order)))
        ax[1, 1].set_xticklabels([SUBTYPE_SHORT[s] for s in order], rotation=40,
                                 ha="right", fontsize=7)
        rho = st["continuum_spearman"]["pc1"]
        ax[1, 1].set(title=f"(d) DSM subtypes order on a continuum (ρ={rho:.2f})",
                     ylabel="domain PC1 (mood→psychosis)")
    except Exception as e:  # pragma: no cover
        ax[1, 1].text(0.5, 0.5, f"(d) subtype continuum\n|Spearman| "
                      f"{st['continuum_spearman']['pc1']:.2f}\n(MDD→…→schizophrenia)",
                      ha="center", va="center", transform=ax[1, 1].transAxes)
        ax[1, 1].set_axis_off()
        print(f"  [fig1d fallback] {type(e).__name__}: {e}")

    fig.suptitle("Figure 1. Trans-diagnostic structure is dimensional; the only "
                 f"discrete structure is diagnosis (HDBSCAN↔cohort ARI "
                 f"{st['hdbscan']['cohort_ari']:.2f})", fontsize=10, y=1.005)
    fig.tight_layout()
    save(fig, "fig1_structure")


# ---------------------------------------------------------------- Fig 2
def fig2_loadings():
    L = pd.read_csv(RESULTS / "dimensional_final_loadings.csv")
    W = L.pivot(index="domain", columns="axis", values="loading")
    W = W[[f"axis{i}" for i in range(1, 7)]]
    salient = W[(W.abs() >= 0.20).any(axis=1)].copy()
    salient["_a"] = salient.values.argmax(axis=1)
    salient["_m"] = salient[[f"axis{i}" for i in range(1, 7)]].max(axis=1)
    salient = salient.sort_values(["_a", "_m"], ascending=[True, False]).drop(
        columns=["_a", "_m"])
    labels = [DOMAIN_LABEL.get(d, d) for d in salient.index]

    fig, ax = plt.subplots(figsize=(7.2, max(6, 0.28 * len(salient))))
    im = ax.imshow(salient.values, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(6))
    ax.set_xticklabels([AXIS_NAME[f"axis{i}"] for i in range(1, 7)], fontsize=8)
    ax.set_yticks(range(len(salient)))
    ax.set_yticklabels(labels, fontsize=7.5)
    for i in range(len(salient)):
        for j in range(6):
            v = salient.values[i, j]
            if abs(v) >= 0.20:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.45 else "#222")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("varimax loading", fontsize=8)
    ax.set_title("Figure 2. Six reproducible, confound-free trans-diagnostic\n"
                 "dimensions (salient domain loadings, |λ|≥0.20)", fontsize=10)
    fig.tight_layout()
    save(fig, "fig2_loadings")


# ---------------------------------------------------------------- Fig 3
def fig3_headtohead():
    v1 = pd.read_csv(RESULTS / "phase5_headtohead_V1.csv")
    v2 = pd.read_csv(RESULTS / "phase5_headtohead_V2.csv")
    order = ["EQ-5D quality of life", "EGF functioning", "any hospitalization"]
    pretty = {"EQ-5D quality of life": "Quality of life\n(EQ-5D, R²)",
              "EGF functioning": "Functioning\n(EGF, R²)",
              "any hospitalization": "Hospitalization\n(AUC)"}

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, df, tag in zip(axs, (v1, v2), ("V1 (primary)", "V2 (follow-up, same cohort)")):
        df = df.set_index("outcome").reindex(order)
        x = np.arange(len(order)); w = 0.26
        ax.bar(x - w, df["DSM"], w, label="DSM diagnosis", color=C_DSM)
        ax.bar(x, df["axes"], w, label="6 dimensions", color=C_AX)
        ax.bar(x + w, df["combined"], w, label="combined", color=C_COMB)
        for xi, oc in enumerate(order):
            d = df.loc[oc, "axes"] - df.loc[oc, "DSM"]
            ax.text(xi, max(df.loc[oc, ["DSM", "axes", "combined"]]) + 0.015,
                    f"Δ(dim−DSM)\n{d:+.3f}", ha="center", fontsize=7.5,
                    color="#2e7d32" if d > 0 else "#b71c1c")
        ax.set_xticks(x); ax.set_xticklabels([pretty[o] for o in order], fontsize=8)
        ax.set_title(tag); ax.set_ylim(0, 0.9)
    axs[0].set_ylabel("cross-validated R² / AUC")
    axs[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Figure 3. Dimensions outperform DSM for quality of life, complement "
                 "it for functioning,\nand are dominated by DSM for hospitalization "
                 "(leakage-safe shuffled CV; consistent at V2, same cohort)", fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "fig3_headtohead")


# ---------------------------------------------------------------- Fig 4
def fig4_traitstate():
    s = pd.read_csv(RESULTS / "longitudinal_axes_stability.csv")
    piv = s.pivot(index="axis", columns="visit", values="pearson")
    visits = [f"V{i}" for i in range(1, 5)]
    piv = piv[visits]
    order = piv["V1"].sort_values(ascending=False).index
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, axis in enumerate(order):
        static = axis == "later_onset"
        ax.plot(visits, piv.loc[axis], "o--" if static else "o-",
                color="#bbb" if static else cmap(i / max(1, len(order) - 1)),
                lw=1.6, ms=5, label=axis.replace("_", " ") + (" (static)" if static else ""))
        ax.annotate(axis.replace("_", " "), xy=(3, piv.loc[axis, "V4"]),
                    xytext=(3.06, piv.loc[axis, "V4"]), fontsize=7.5, va="center",
                    color="#888" if static else "#333")
    ax.axhspan(0.5, 1.0, color="#e8f5e9", alpha=0.5, zorder=0)
    ax.axhspan(0.0, 0.25, color="#ffebee", alpha=0.5, zorder=0)
    ax.text(0.02, 0.55, "trait-like", color="#2e7d32", fontsize=8)
    ax.text(0.02, 0.04, "state-like", color="#b71c1c", fontsize=8)
    ax.set(xlabel="follow-up visit", ylabel="V0↔Vk test–retest (Pearson r)",
           ylim=(0, 0.75), xlim=(-0.2, 3.9))
    ax.set_title("Figure 4. Trait–state gradient across four years of follow-up",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_traitstate")


# ---------------------------------------------------------------- Fig 5
def fig5_cognition():
    L = pd.read_csv(RESULTS / "cognition_bpsz_loadings.csv")
    C = pd.read_csv(RESULTS / "cognition_bpsz_corr.csv", index_col=0)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.3),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) construct loadings on g and speed
    W = L.pivot(index="domain", columns="factor", values="loading")
    constructs = ["percept_reasoning", "verbal_reasoning", "working_memory",
                  "memory_cvlt", "fluency", "proc_speed", "executive_tmt"]
    W = W.reindex(constructs)
    x = np.arange(len(constructs)); w = 0.38
    a0.bar(x - w / 2, W["cog1"], w, label="cog1: general ability (g)", color="#1f77b4")
    a0.bar(x + w / 2, W["cog2"], w, label="cog2: processing speed", color="#ff7f0e")
    a0.axhline(0, color="#888", lw=0.8)
    a0.set_xticks(x)
    a0.set_xticklabels([COG_LABEL[c] for c in constructs], fontsize=7.5)
    a0.set_ylabel("factor loading"); a0.legend(fontsize=8)
    a0.set_title("(a) Cognitive structure: g + processing speed", fontsize=10)

    # (b) cognition × symptom-axis correlations
    C = C.loc[["cog1", "cog2"], [c for c in C.columns]]
    im = a1.imshow(C.values, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    a1.set_xticks(range(C.shape[1]))
    a1.set_xticklabels(AXIS_SHORT, rotation=40, ha="right", fontsize=7.5)
    a1.set_yticks([0, 1]); a1.set_yticklabels(["g", "speed"])
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            a1.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(C.values[i, j]) > 0.2 else "#222")
    fig.colorbar(im, ax=a1, fraction=0.046, pad=0.04).set_label("Pearson r", fontsize=8)
    a1.set_title("(b) Cognition vs symptom axes (max |r|=0.24)", fontsize=10)

    fig.suptitle("Figure 5. Cognition (BP/SZ, n=6,099) recovers g + speed and is "
                 "semi-independent of the symptom dimensions", fontsize=10, y=1.04)
    fig.tight_layout()
    save(fig, "fig5_cognition")


def main() -> int:
    print("generating manuscript figures →", FIGDIR)
    fig2_loadings()
    fig3_headtohead()
    fig4_traitstate()
    fig5_cognition()
    fig1_structure()  # last (heaviest: rebuilds harmonized dataset for panel d)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
