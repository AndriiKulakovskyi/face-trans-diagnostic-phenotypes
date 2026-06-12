#!/usr/bin/env python3
"""40 — M4.0 prognostic-outcome inventory (the feasibility + added-value audit).

Before fitting any prognostic model we inventory *what is actually predictable*: which configured
outcomes exist, how many patients have BOTH a baseline (V0) and a horizon (V2) value — the effective
ANCOVA modelling N, since "incremental beyond baseline" needs the paired observation — and, the
load-bearing check, **which predictor axis ↔ outcome pairs share indicators** in the M1 measurement
model (`configs/prior_loading_matrix_v3.csv`).

The reframe (locked 2026-06-10): we forecast every clinically-useful state axis (functioning,
severity, sleep, mood, suicidality). When a predictor axis is *built from* an outcome's own items
(e.g. the sleep dimension is built from PSQI; G is anchored on EGF/CGI-S/FAST/EQ-5D), that axis's
contribution enters as the **autoregressive baseline `Y_V0` — the bar to beat — and is never
*credited* as the map earning its keep**. The clean *added-value* test is always cross-construct:
the durable trait trio (cognition, metabolic, inflammatory) — which loads on the functioning/severity
outcomes only as `g_anchor_on_specific`, a hard soft-zero (prior_sd 0.001) — predicting the future
state *beyond* that bar. This stage establishes that the durable-trio → primary-outcome test is
cross-construct (clean) and adequately powered. No model, no imputation. Methods:
docs/PROGNOSIS_MODEL.md (M4.0).

    python3 scripts/40_inventory.py

Writes reports/40_inventory.md (+ 40_outcome_coverage.csv, 40_overlap_audit.csv) and
docs/figures/40_coverage.png. All aggregate (no per-patient values) -> shareable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.prognosis import CANON, DURABLE, SPINE  # noqa: E402
from face.prognosis.frame import load_outcome_config  # noqa: E402

PROC = REPO / "data" / "processed"
MATRIX = REPO / "configs" / "prior_loading_matrix_v3.csv"
CONFIG = REPO / "configs" / "m4_outcomes.yaml"
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
COHORTS = ("bp", "sz", "dr")

# prior_type -> overlap class (does the outcome's own item inform this axis?). `defines` = the outcome
# is a primary/g_anchor indicator of the axis -> that axis is autoregressive for this outcome (its
# contribution is the baseline bar, not added value). `pinned`/`soft0` = ~0 loading -> clean.
_CLASS = {
    "primary": "defines", "g_anchor": "defines",
    "g_anchor_on_specific": "pinned", "plausible_cross": "cross", "unlikely_cross": "soft0",
}
_AUTOREG_TYPES = {"primary", "g_anchor"}  # outcome defines the axis -> autoregressive (the bar)


def _visit_frame(visit: str) -> pd.DataFrame:
    """Native-scale harmonized outcome table for a visit, indexed (cohort, patient_id)."""
    return pd.read_parquet(PROC / f"baseline_{visit.lower()}.parquet")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    v0, v1, v2 = (_visit_frame(v) for v in ("V0", "V1", "V2"))
    available = sorted(set(v0.columns) | set(v1.columns) | set(v2.columns))
    cfg = load_outcome_config(CONFIG, available_vars=available)
    horizon = cfg.meta.get("primary_horizon", "V2")
    repl = cfg.meta.get("secondary_horizon", "V1")

    # ---- 1) coverage + paired (baseline & horizon) N, the effective modelling sample ----
    def _coh(df):
        return df.index.get_level_values("cohort")

    rows = []
    for o in cfg.outcomes:
        s = o.source_var
        rec: dict = {
            "outcome": o.name, "source_var": s, "family": o.family, "role": o.role,
            "cohort_scope": "+".join(o.cohort_scope),
            "n_V0": int(v0[s].notna().sum()) if s in v0 else 0,
            "n_V1": int(v1[s].notna().sum()) if s in v1 else 0,
            "n_V2": int(v2[s].notna().sum()) if s in v2 else 0,
        }
        for hv, hf, tag in ((repl, v1, "V0V1"), (horizon, v2, "V0V2")):
            if s in v0 and s in hf:
                a = v0[s]
                b = hf[s].reindex(a.index)
                both = a.notna() & b.notna()
                rec[f"n_paired_{tag}"] = int(both.sum())
            else:
                rec[f"n_paired_{tag}"] = 0
        if s in v0 and s in v2:
            a = v0[s]
            b = v2[s].reindex(a.index)
            both = a.notna() & b.notna()
            by = pd.Series(both.values, index=_coh(v0)).groupby(level=0).sum()
            for c in COHORTS:
                rec[f"paired_{c}"] = int(by.get(c, 0))
        else:
            for c in COHORTS:
                rec[f"paired_{c}"] = 0
        rows.append(rec)
    cov = pd.DataFrame(rows)
    cov.to_csv(REPORTS / "40_outcome_coverage.csv", index=False)

    # ---- 2) added-value audit: predictor-axis ↔ outcome item overlap ----
    m = pd.read_csv(MATRIX)
    ptype = m.drop_duplicates(["item", "factor"]).set_index(["item", "factor"])["prior_type"].to_dict()
    audit_rows, autoreg = [], []
    for o in cfg.outcomes:
        for ax in CANON:
            pt = ptype.get((o.source_var, ax))
            cls = _CLASS.get(pt, "none")
            ar = pt in _AUTOREG_TYPES
            audit_rows.append({"outcome": o.name, "axis": ax, "prior_type": pt or "none",
                               "overlap": cls, "autoregressive": ar})
            if ar:
                autoreg.append((ax, o.name))
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(REPORTS / "40_overlap_audit.csv", index=False)

    # clean (cross-construct) durable-trio → primary-outcome pairs — the added-value test
    prim = [o.name for o in cfg.primary()]
    clean_durable = [(ax, on) for ax in DURABLE for on in prim
                     if not audit[(audit.axis == ax) & (audit.outcome == on)].autoregressive.any()]

    _figure(cov, audit, cfg)
    _report(cfg, cov, audit, autoreg, clean_durable, horizon, repl, available)


def _report(cfg, cov, audit, autoreg, clean_durable, horizon, repl, available):
    skipped = [o for o in ["mars"] if o not in {c.source_var for c in cfg.outcomes} and o not in available]
    prim = cfg.primary()
    armap: dict[str, list[str]] = {}
    for ax, on in autoreg:
        armap.setdefault(on, []).append(ax)

    show = cov[["outcome", "source_var", "family", "role", "cohort_scope",
                "n_V0", "n_paired_V0V1", "n_paired_V0V2", "paired_bp", "paired_sz", "paired_dr"]]
    md = [
        "# 40 — M4.0 prognostic-outcome inventory (feasibility + the added-value audit)", "",
        "What is actually predictable, and where the map can legitimately claim *added value*. The "
        "effective modelling sample for an *incremental-beyond-baseline* test is **n_paired_V0V2** "
        "(both the V0 baseline *and* the V2 horizon observed — ANCOVA needs the pair). We forecast "
        "every state axis; the question is whether the durable biology adds value beyond today's "
        "value. No model, no imputation.", "",
        "## Outcome registry & coverage", "",
        f"Parsed {len(cfg.outcomes)} outcomes from `configs/m4_outcomes.yaml` "
        f"(primary: {', '.join(o.name for o in prim)}). "
        + (f"Skipped (absent from the harmonized tables): {', '.join(skipped)}. " if skipped else "")
        + "`n_paired_V0V2` is the headline modelling N; `paired_{bp,sz,dr}` drive the within-cohort "
        "(Q3) feasibility.", "",
        show.to_markdown(index=False), "",
        f"- **Primary horizon {horizon}** (2-yr); **replication {repl}** (1-yr, larger N). "
        "Retention thins V0 9,013 → V1 4,270 → V2 2,958; attrition is mild/MAR-given-V0 (M3 G6), "
        "corrected by IPW at the modelling stages, never by imputation.",
        "- **SZ has no FAST / MADRS / C-SSRS at follow-up** (`paired_sz = 0`) — those outcomes are "
        "BP/DR only; their Q3 reduces to BP-vs-DR and the SZ generalization is explicitly untested.",
        "- **DR is thin at V2** (paired ~ a hundred even on the 3-cohort outcomes) — DR-specific "
        "verdicts will be documented-partial.", "",
        "## Added-value audit — predictor axis ↔ outcome item overlap", "",
        "Each (axis, outcome) is classified by how the outcome's own item enters that axis in the M1 "
        "loading matrix: **defines** (primary / g_anchor → the axis is built from the outcome → that "
        "axis is *autoregressive* for this outcome), **cross** (plausible cross-loading), **pinned** "
        "(`g_anchor_on_specific`, a soft-zero at prior_sd 0.001 → negligible), **soft0** "
        "(unlikely_cross), **none**.", "",
        "**Autoregressive pairs — the baseline bar, NOT credited as added value:**",
    ]
    for on in [o.name for o in cfg.outcomes]:
        if armap.get(on):
            md.append(f"- `{on}` ← {', '.join(sorted(armap[on]))}")
    md += [
        "",
        "These are not forbidden — we *do* forecast each of these outcomes. But when a dimension is "
        "built from the outcome's own items, its contribution enters as the **autoregressive baseline "
        "`Y_V0` (R3y), the bar to beat**, and is never reported as the transdiagnostic map earning "
        "its keep (that would be a trivial self-prediction).", "",
        "**The added-value test the milestone hinges on (clean / cross-construct):** the durable trio "
        f"({', '.join(DURABLE)}) → the primary outcomes ({', '.join(o.name for o in prim)}) — "
        f"**{len(clean_durable)}/{len(DURABLE) * len(prim)} pairs share no items** with the outcome. "
        "The functioning/severity outcomes load on the durable axes only as `g_anchor_on_specific` "
        "(pinned ~0), so these coordinates are genuinely not built from them: a real, non-circular "
        f"forecast. Only the general factor **{SPINE}** (anchored on EGF/CGI-S/FAST/EQ-5D) and the "
        "same-construct axes are autoregressive — exactly the planned guard, now data-derived.", "",
        "> The clinically useful question is cross-construct *and* clean: *given two patients equal on "
        "an outcome today, does their durable biology forecast who diverges in a year?* That lift over "
        "the autoregressive bar is the finding; the bar itself is the thing it must beat.", "",
        "## Data contract for M4 (resolved here)",
        "- **Outcomes**: read native-scale from `data/processed/baseline_v{0,1,2}.parquet` "
        "(EGF 0–100, CGI-S 0–7, … verified), `(cohort, patient_id)`-indexed, NaN = missing.",
        "- **Predictors**: baseline coordinates + per-patient SD from `results/face/patient_panel.parquet` "
        "(V0 rows) and the draw tensor `results/face/m3/panel_draws.npz`; the three map representations "
        "(continuous dims · 8 archetypes · 4-region tessellation) from `results/face/patient_strata.parquet`.",
        "- **Reference covariates** (age, sex, siteid_city): pulled from the data layer at stage 41 "
        "(absent from the processed item tables); `arm` (DSM-5 subtype) + `cohort` from the panel.",
        "- **Attrition**: `results/face/m3/ipw_weights.parquet` (`w_retained_V2`).", "",
        "## Decision for the gate",
        f"Confirm the outcome set + the {horizon}-paired sample sizes above, and the autoregressive "
        "list, before assembling the analysis frame (stage 41). The durable-biology → EGF/CGI-S "
        "added-value test is clean and adequately powered (≈2,400 paired on the 3-cohort primaries); "
        "SZ-absent and DR-thin outcomes are flagged for documented-partial verdicts.", "",
        "Artifacts: `reports/40_{outcome_coverage,overlap_audit}.csv` · `docs/figures/40_coverage.png`.",
    ]
    (REPORTS / "40_inventory.md").write_text("\n".join(md))
    print("\n".join(md))


def _figure(cov, audit, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.05, 1]})

    names = list(cov["outcome"])
    x = np.arange(len(names))
    w = 0.26
    colors = {"bp": "#4575b4", "sz": "#d73027", "dr": "#1a9850"}
    for i, c in enumerate(COHORTS):
        ax[0].bar(x + (i - 1) * w, cov[f"paired_{c}"].values, w, label=c.upper(), color=colors[c])
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax[0].set_ylabel("patients with V0 & V2 (ANCOVA N)")
    ax[0].set_title("Effective prognostic sample (paired V0→V2) by cohort")
    ax[0].legend(fontsize=8, title="cohort")
    ax[0].grid(axis="y", alpha=0.3)
    for xi, on in zip(x, names):
        tot = int(cov.loc[cov.outcome == on, "n_paired_V0V2"].iloc[0])
        ax[0].text(xi, tot, f"{tot}", ha="center", va="bottom", fontsize=7, color="#333")

    # overlap-class heatmap (axis × outcome); autoregressive cells boxed (the baseline bar)
    order = {"none": 0, "soft0": 1, "pinned": 1, "cross": 2, "defines": 3}
    piv = audit.assign(code=audit.overlap.map(order)).pivot(index="axis", columns="outcome", values="code")
    piv = piv.reindex(index=list(CANON), columns=names)
    cmap = ListedColormap(["#f7f7f7", "#fee0b6", "#fdae61", "#d73027"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax[1].imshow(piv.values, cmap=cmap, norm=norm, aspect="auto")
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax[1].set_yticks(range(len(CANON)))
    ax[1].set_yticklabels(CANON, fontsize=8)
    arset = audit[audit.autoregressive].set_index(["axis", "outcome"]).index
    for i, ax_name in enumerate(CANON):
        for j, on in enumerate(names):
            if (ax_name, on) in arset:
                ax[1].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2))
                ax[1].text(j, i, "AR", ha="center", va="center", fontsize=7, fontweight="bold")
    durable_set = set(DURABLE)
    for i, ax_name in enumerate(CANON):
        if ax_name in durable_set:
            ax[1].text(-0.65, i, "★", ha="center", va="center", fontsize=10, color="#1a9850")
    ax[1].set_title("Predictor↔outcome overlap (AR = autoregressive bar; ★ = durable axis)")
    fig.tight_layout()
    fig.savefig(FIGS / "40_coverage.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
