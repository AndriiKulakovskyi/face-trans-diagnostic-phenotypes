#!/usr/bin/env python3
"""22 — M2.2 measurement-error mixture as a SOFT TESSELLATION (§3.2).

Continuum verdict (M2.1) ⇒ the mixture is a soft tessellation of the continuum (a discrete decision-region
overlay), NOT biotypes. Fit by Extreme Deconvolution: x_i ~ Σ_k π_k N(m_k, V_k + S_i), with S_i = the M1
per-patient posterior variance — so uncertainty propagates (prior-dominated coords + DR's absent substance
self-down-weight) and the components describe the underlying noise-free cloud. K is a tessellation
granularity (continuum ⇒ no natural K); reported at K=4 (M2.1 uncertainty-aware GMM mode 4 + BIC plateau).

    python3 scripts/22_mixture.py
Reads results/face/m2/{coordinates_full.parquet, validation_table.parquet}.
Writes results/face/m2/{tessellation.parquet, tessellation_profiles.csv} + reports/22_tessellation.md
      + docs/figures/22_{bic,profiles,membership}.png.
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
REPORT_K = 4
CANON = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep", "mania_activation",
         "suicidality", "developmental_risk", "substance"]


def main():
    from face.strata.archetypes import name_archetypes
    from face.strata.mixture import bic_sweep, xd_em

    df = pd.read_parquet(M2 / "coordinates_full.parquet")
    vt = pd.read_parquet(M2 / "validation_table.parquet")
    X = df[[f"{f}__mean" for f in CANON]].to_numpy()
    S = df[[f"{f}__sd" for f in CANON]].to_numpy() ** 2           # known per-patient diagonal noise

    print("[1/3] XD BIC sweep K=2..8 (tessellation granularity)...", flush=True)
    bic = bic_sweep(X, S, range(2, 9), seed=SEED)
    print("  BIC:", {k: round(v) for k, v in bic.items()}, flush=True)

    print(f"[2/3] reported tessellation at K={REPORT_K}...", flush=True)
    fit = xd_em(X, S, REPORT_K, seed=SEED)
    r, mu = fit["resp"], fit["mu"]
    names = name_archetypes(mu, CANON, thr=0.4)                   # label components by their extreme axes
    MAP = r.argmax(1)
    ent = (-(r * np.log(np.clip(r, 1e-9, 1))).sum(1) / np.log(REPORT_K))
    share = np.bincount(MAP, minlength=REPORT_K) / len(MAP)

    out = df[["cohort", "patient_id"]].copy()
    for k in range(REPORT_K):
        out[f"r{k}"] = r[:, k].round(4)
    out["MAP"] = MAP
    out["MAP_name"] = [names[k] for k in MAP]
    out["entropy"] = ent.round(3)
    out.to_parquet(M2 / "tessellation.parquet")
    pd.DataFrame(mu, columns=CANON, index=[f"T{k}: {names[k]}" for k in range(REPORT_K)]).round(3) \
        .to_csv(M2 / "tessellation_profiles.csv")

    print("[3/3] figures...", flush=True)
    _fig_bic(bic)
    _fig_profiles(mu, names)
    _fig_membership(X, df, vt, MAP, REPORT_K)

    comp_coh = pd.crosstab(pd.Series(MAP, name="component"), vt["cohort"])
    comp_arm = pd.crosstab(pd.Series([names[k] for k in MAP], name="component"), vt["arm"].astype(str))
    clear = float((r.max(1) >= 0.5).mean())
    md = ["# 22 — M2.2 measurement-error mixture (soft TESSELLATION, not biotypes)", "",
          "Continuum verdict (M2.1) ⇒ this is a **soft tessellation** of the continuum — a discrete "
          "decision-region overlay, **not** natural-kind biotypes. Fit by Extreme Deconvolution "
          "(x_i ~ Σ_k π_k N(m_k, V_k+S_i)): **propagates the M1 per-patient uncertainty S_i**, so the "
          "components are the underlying noise-free cloud and prior-dominated / DR-absent coordinates "
          "self-down-weight.", "",
          f"## K = {REPORT_K} (tessellation granularity; M2.1 uncertainty-aware mode 4 + BIC plateau)",
          "XD BIC over K (no sharp optimum expected on a continuum):",
          "| K | " + " | ".join(str(k) for k in bic) + " |",
          "|" + "---|" * (len(bic) + 1),
          "| BIC | " + " | ".join(f"{round(v):,}" for v in bic.values()) + " |", "",
          f"- membership: **{clear:.0%}** of patients have a confident component (max responsibility ≥0.5); "
          f"mean normalized entropy {ent.mean():.2f}.",
          f"- population share by component: {dict(enumerate(share.round(3)))}", "",
          "## Tessellation component profiles (m_k, z-units; higher = more burden)",
          pd.DataFrame(mu, columns=CANON, index=[f"T{k}: {names[k]}" for k in range(REPORT_K)]).round(2).to_markdown(), "",
          "## Diagnostic composition (validation-only; two granularities)",
          "By cohort:", comp_coh.to_markdown(), "",
          "By DSM-5 subtype:", comp_arm.to_markdown(), "",
          "## Reading",
          "- The components are **regions of a continuum**, not discrete kinds — read with the archetypes "
          "(M2.3), which are the lead view. Whether the tessellation is **better than DSM-5** is the M4/M5 "
          "predictive/treatment head-to-head (§1.7), set up in M2.4.", "",
          "## Artifacts",
          "- `results/face/m2/tessellation.parquet` — per-patient soft responsibilities · MAP · entropy.",
          "- `results/face/m2/tessellation_profiles.csv` — component profiles.",
          "- Figures: `docs/figures/22_{bic,profiles,membership}.png`."]
    (REPORTS / "22_tessellation.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n[done] K={REPORT_K}, {clear:.0%} confident, shares {share.round(3)}")


def _fig_bic(bic):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ks = sorted(bic)
    ax.plot(ks, [bic[k] for k in ks], "-o", color="#2c7fb8")
    ax.axvline(REPORT_K, ls="--", c="grey", label=f"reported K={REPORT_K}")
    ax.set_xlabel("K (tessellation components)"); ax.set_ylabel("XD BIC (lower=better)")
    ax.set_title("M2.2 — measurement-error mixture BIC (continuum ⇒ no sharp optimum)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGS / "22_bic.png", dpi=130); plt.close(fig)


def _fig_profiles(mu, names):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = mu.shape[0]
    fig, ax = plt.subplots(figsize=(10, 0.8 * K + 2))
    im = ax.imshow(mu, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(CANON))); ax.set_xticklabels(CANON, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"T{k}: {names[k]}" for k in range(K)], fontsize=9)
    for i in range(K):
        for j in range(len(CANON)):
            ax.text(j, i, f"{mu[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(mu[i, j]) > 0.9 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="component mean (z)")
    ax.set_title("M2.2 — soft-tessellation component profiles (deconvolved)")
    fig.tight_layout(); fig.savefig(FIGS / "22_profiles.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def _fig_membership(X, df, vt, MAP, K):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED).fit_transform(X)
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    cmap = plt.cm.tab10
    for k in range(K):
        m = MAP == k
        ax[0].scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.5, color=cmap(k), label=f"T{k}")
    ax[0].set_title("UMAP — tessellation component (MAP)"); ax[0].legend(markerscale=3, fontsize=8)
    pd.crosstab(pd.Series(MAP, name="comp"), vt["cohort"]).plot(kind="bar", stacked=True, ax=ax[1], colormap="Set2")
    ax[1].set_title("component × cohort"); ax[1].tick_params(axis="x", rotation=0)
    pd.crosstab(pd.Series(MAP, name="comp"), vt["arm"].astype(str)).plot(kind="bar", stacked=True, ax=ax[2], colormap="tab20")
    ax[2].set_title("component × DSM-5 subtype"); ax[2].legend(fontsize=6); ax[2].tick_params(axis="x", rotation=0)
    fig.suptitle("M2.2 — tessellation membership & diagnostic composition", y=1.02)
    fig.tight_layout(); fig.savefig(FIGS / "22_membership.png", dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
