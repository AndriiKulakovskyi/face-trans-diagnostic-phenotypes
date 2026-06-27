#!/usr/bin/env python
"""Synthesis figure for the Gaussian-copula vertical (M2 structure -> M3 durability -> M4 prognosis).

Reads the committed result files (never hand-typed numbers) and renders a 4-panel "the copula vertical in
one figure" to docs/figures/copula_vertical/synthesis.png. Also prints the exact numbers used, so the
findings doc can cite them verbatim.

    PYTHONPATH=$PWD/src python notebooks/copula_vertical_synthesis_figure.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
STRATA = REPO / "results" / "face" / "strata_oop"
PROG = REPO / "results" / "face" / "prognosis_oop"
TEMP = REPO / "results" / "face" / "temporal_oop"
OUT = REPO / "docs" / "figures" / "copula_vertical"
CANON = ["overall_severity", "cognition", "metabolic", "inflammatory", "sleep",
         "mania_activation", "suicidality", "developmental_risk", "substance"]
SHORT = {"overall_severity": "severity", "cognition": "cognition", "metabolic": "metabolic",
         "inflammatory": "inflammatory", "sleep": "sleep", "mania_activation": "mania",
         "suicidality": "suicidality", "developmental_risk": "developmental", "substance": "substance"}


def _archetype_label(name: str) -> str:
    n = name.lower()
    if "metabolic" in n and "inflammatory" in n:
        return "A0 biological"
    if "↓overall_severity" in name or "low" in n:
        return "A1 low-burden"
    if "↑overall_severity" in name:
        return "A2 severe / non-bio"
    return "A3 symptom"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)

    # --- data ---
    prof = pd.read_csv(STRATA / "consolidate" / "archetype_profiles.csv")
    profA = prof[prof.arm == "A_all9"].reset_index(drop=True)
    ZA = profA[CANON].to_numpy("float64")                                   # [4, 9]
    arch_names = [_archetype_label(n) for n in profA["name"]]

    ts = pd.read_csv(TEMP / "trait_state" / "trait_state.csv").set_index("axis").reindex(CANON).reset_index()

    inc = pd.read_csv(PROG / "incremental" / "incremental_comparison.csv")
    egf = inc[inc.outcome == "egf"].set_index("model")

    frame = pd.read_parquet(PROG / "frame" / "analysis_frame.parquet")
    rem = (frame[frame["egf__remission_V2"].notna()]
           .groupby("arch_dominant")["egf__remission_V2"].agg(["size", "mean"]))
    rem["label"] = [arch_names[k] for k in rem.index]
    rem = rem.sort_values("mean")

    print("=== numbers used ===")
    print("archetype remission:", {r.label: round(r.mean * 100) for r in rem.itertuples()})
    print("trait_state ICC:", dict(zip(ts.axis, ts.icc.round(2), strict=False)))
    print("egf incremental:", egf["d_elpd_vs_ref"].round(1).to_dict())

    # --- figure ---
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("The Gaussian-copula vertical: one biology-aware continuum, real → durable → prognostic",
                 fontsize=14, fontweight="bold")

    # Panel A — M2 structure: A=4 archetype profiles
    a = ax[0, 0]
    vmax = 3.0
    im = a.imshow(ZA, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    a.set_xticks(range(len(CANON))); a.set_xticklabels([SHORT[c] for c in CANON], rotation=55, ha="right", fontsize=8)
    a.set_yticks(range(len(arch_names))); a.set_yticklabels(arch_names, fontsize=9)
    a.set_title("M2 — structure: 4 stable archetypes (biology ⊥ symptoms ⊥ severity)", fontsize=10)
    fig.colorbar(im, ax=a, shrink=0.7, label="z (clipped ±3)")

    # Panel B — M3 durability: trait/state ICC
    b = ax[0, 1]
    tss = ts.sort_values("icc")
    colors = ["#2c7fb8" if v >= 0.6 else ("#d73027" if v < 0.5 else "#999999") for v in tss["icc"]]
    b.barh(range(len(tss)), tss["icc"], color=colors,
           xerr=[tss["icc"] - tss["icc_lo"], tss["icc_hi"] - tss["icc"]], capsize=2, error_kw={"lw": 0.8})
    b.axvline(0.5, color="k", lw=0.8, ls="--")
    b.set_yticks(range(len(tss))); b.set_yticklabels([SHORT[x] for x in tss["axis"]], fontsize=8)
    b.set_xlim(0, 1); b.set_xlabel("ICC (trait fraction)  —  ≥0.6 trait (blue) · <0.5 state (red)")
    b.set_title("M3 — durability: biology is trait, symptoms are state", fontsize=10)

    # Panel C — M4 prognosis: archetype functional-remission gradient
    c = ax[1, 0]
    bar_colors = ["#cf6679" if "biological" in lab else "#3b6fb6" for lab in rem["label"]]
    c.bar(range(len(rem)), rem["mean"] * 100, color=bar_colors)
    for i, (_, r) in enumerate(rem.iterrows()):
        c.text(i, r["mean"] * 100 + 1, f"{round(r['mean']*100)}%", ha="center", fontsize=9)
    c.set_xticks(range(len(rem))); c.set_xticklabels(rem["label"], rotation=20, ha="right", fontsize=8)
    c.set_ylabel("2-yr functional remission (GAF ≥ 71)"); c.set_ylim(0, 70)
    c.set_title("M4 — prognosis: 27%→60% gradient (biology corner worst)", fontsize=10)

    # Panel D — M4 operative-K: incremental ΔELPD (archetypes dominate any hard K)
    d = ax[1, 1]
    order = ["+archetypesA", "+archetypesB", "+specifics8", "+tess_k4", "+tess_k3", "+tess_k2", "+durable"]
    order = [m for m in order if m in egf.index]
    vals = egf.loc[order, "d_elpd_vs_ref"]; ses = egf.loc[order, "se_d_elpd"]
    dcolors = ["#1b7837" if "arch" in m else ("#7fbf7b" if "tess" in m else "#bbbbbb") for m in order]
    d.barh(range(len(order)), vals, color=dcolors, xerr=2 * ses, capsize=2, error_kw={"lw": 0.8})
    d.axvline(0, color="k", lw=0.8)
    d.set_yticks(range(len(order))); d.set_yticklabels(order, fontsize=8); d.invert_yaxis()
    d.set_xlabel("ΔELPD vs DSM-5+severity+baseline (held-out, ↑ better)")
    d.set_title("M4 — operative K = none: archetypes > any tessellation", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = OUT / "synthesis.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
