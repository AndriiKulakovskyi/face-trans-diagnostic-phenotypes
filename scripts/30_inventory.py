#!/usr/bin/env python3
"""30 — M3.0 longitudinal-coverage inventory (the M3 feasibility gate).

Before scoring any follow-up visit we inventory *what is actually there*: how retention thins
V0 -> V1 -> V2, which of the 9 M1 axes keep enough re-administered indicators to be scored, and
which indicators are **re-administered** (fresh values each visit) vs **carried-forward / identical**
(value copied or never changes -> cannot inform within-person change). The dictionary's
`temporal_scope` is hardcoded "current" (schema_gen.py), so the split is derived **empirically**
from the within-patient V0->V1 change rate.

CAUTION (the load-bearing nuance): a high change rate means the item was *re-collected with varying
answers*, NOT that the construct is "state-like". The CTQ childhood-trauma items — a definitionally
*fixed* history — show change rates up to ~0.94 from recall noise alone. So this inventory only
decides **coverage** (is there fresh follow-up data to score an axis?); the genuine **trait vs state**
question is settled in G3 (stage 35), which deconvolves measurement error. No scoring, no imputation.
Methods: docs/TEMPORAL_MODEL.md (M3.0).

    python3 scripts/30_inventory.py

Writes reports/30_inventory.md (+ 30_retention.csv, 30_axis_coverage.csv, 30_indicator_temporal.csv)
and docs/figures/30_{retention,axis_coverage}.png. All aggregate (no per-patient values) -> shareable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.data import build_unified_dataframe, load_variables, to_harmonized_dataset  # noqa: E402
from face.temporal import CANON, VISITS  # noqa: E402
from face.temporal.dropout import retention_table  # noqa: E402

XLSX = REPO / "data" / "face-common-vars.xlsx"
MATRIX = REPO / "configs" / "prior_loading_matrix_v3.csv"
REPORTS = REPO / "reports"
FIGS = REPO / "docs" / "figures"
COHORTS = ("bp", "sz", "dr")
MIN_OBS = 30                       # the engine's per-indicator min-observation guard (matches §06)
CARRIED_THR = 0.02                 # change rate below this = carried-forward / identical V0==V1
HIVAR_THR = 0.20                   # figure reference: "high re-administration variability"


def _harmonized_visit(df, variables, visit, modeled):
    """Per-visit harmonized matrix restricted to modeled indicators (raw, NaN=missing, skip-logic on)."""
    ds = to_harmonized_dataset(df, variables, visit=visit, normalize=False, apply_skip_logic=True)
    items = [it for it in modeled if it in ds.X.columns]
    return ds.X[items].apply(pd.to_numeric, errors="coerce")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    # ---- modeled-indicator metadata (home factor / family / block / section) ----
    m = pd.read_csv(MATRIX)
    home = (m[m.prior_type.isin(["primary", "g_anchor"])].drop_duplicates("item")
            .set_index("item")["factor"])
    meta = (m.drop_duplicates("item").set_index("item")[["likelihood_family", "modeling_block"]].copy())
    meta["home_factor"] = home                      # NaN for cross-loading windows (no home)
    modeled = list(meta.index)
    variables = load_variables(str(XLSX))
    section = {v.canonical_name: getattr(v, "section", "") for v in variables}

    # ---- harmonized long frame (all visits) ----
    df = build_unified_dataframe("data", str(XLSX), readiness=["READY", "PARTIAL"], format="long")

    # ---- 1) retention (all visits for context; V0-V2 is the M3 window) ----
    ret = retention_table(df)
    ret.to_csv(REPORTS / "30_retention.csv", index=False)
    ret_w = ret.pivot(index="visit", columns="cohort", values="n_patients").reindex(
        sorted(ret["visit"].unique(), key=lambda v: int(v[1:])))
    ret_w["total"] = ret_w[list(COHORTS)].sum(axis=1)
    fracv0 = (ret_w / ret_w.loc["V0"]).round(3)

    # ---- 2) per-visit coverage of modeled indicators (V0, V1, V2) ----
    Xv = {v: _harmonized_visit(df, variables, v, modeled) for v in VISITS}
    present = list(Xv["V0"].columns)
    cov = pd.DataFrame(index=present)
    cov["home_factor"] = [meta.loc[it, "home_factor"] for it in present]
    cov["block"] = [meta.loc[it, "modeling_block"] for it in present]
    cov["family"] = [meta.loc[it, "likelihood_family"] for it in present]
    cov["section"] = [section.get(it, "") for it in present]
    for v in VISITS:
        X = Xv[v]
        cov[f"nobs_{v}"] = [int(X[it].notna().sum()) if it in X.columns else 0 for it in present]

    # ---- 3) re-administered vs carried-forward — empirical V0->V1 within-patient change rate ----
    # (re-administration signal ONLY; trait vs state is G3, which removes measurement/recall noise.)
    X0, X1 = Xv["V0"], Xv["V1"]
    common = X0.index.intersection(X1.index)
    chg_rate, n_paired = [], []
    for it in present:
        if it in X1.columns and len(common):
            a, b = X0.loc[common, it], X1.loc[common, it]
            both = a.notna() & b.notna()
            n = int(both.sum())
            changed = (np.abs(a[both].to_numpy() - b[both].to_numpy()) > 1e-9).mean() if n else np.nan
        else:
            n, changed = 0, np.nan
        n_paired.append(n)
        chg_rate.append(round(float(changed), 3) if changed == changed else np.nan)
    cov["n_paired_v0v1"] = n_paired
    cov["change_rate_v0v1"] = chg_rate
    cov["temporal_class"] = np.where(
        cov["n_paired_v0v1"] < MIN_OBS, "untested",
        np.where(cov["change_rate_v0v1"] < CARRIED_THR, "carried", "re-administered"))
    cov.reset_index(names="item").to_csv(REPORTS / "30_indicator_temporal.csv", index=False)

    # ---- 4) per-axis roll-up (axis × visit coverage + re-administration) ----
    axis_rows = []
    for ax in CANON:
        items = cov.index[cov["home_factor"] == ax]
        rec: dict = {"axis": ax, "n_indicators": int(len(items))}
        for v in VISITS:
            rec[f"items_ge30_{v}"] = int((cov.loc[items, f"nobs_{v}"] >= MIN_OBS).sum())
        tc = cov.loc[items, "temporal_class"]
        rec["n_readministered"] = int((tc == "re-administered").sum())
        rec["n_carried"] = int((tc == "carried").sum())
        rec["n_untested"] = int((tc == "untested").sum())
        cr = cov.loc[items, "change_rate_v0v1"].dropna()
        rec["median_change_rate"] = round(float(cr.median()), 3) if len(cr) else float("nan")
        # verdict = follow-up COVERAGE only (trait/state is G3, never decided here)
        trackable = rec["items_ge30_V1"] >= 1 and rec["items_ge30_V2"] >= 1
        if not trackable:
            rec["verdict"] = "coverage-limited"
        elif rec["n_indicators"] <= 2:
            rec["verdict"] = "thin"
        else:
            rec["verdict"] = "trackable"
        axis_rows.append(rec)
    axis = pd.DataFrame(axis_rows)
    axis.to_csv(REPORTS / "30_axis_coverage.csv", index=False)

    win = cov.index[cov["home_factor"].isna()]   # windows (cross-loaders, no home axis)
    _figures(ret_w, fracv0, axis)

    # ---- report ----
    def _icon(v):
        return {"trackable": "✅", "thin": "⚠️", "coverage-limited": "⛔"}.get(v, "")
    md = ["# 30 — M3.0 longitudinal-coverage inventory (V0 → V1 → V2)", "",
          "The M3 feasibility gate: retention, per-axis follow-up coverage, and an **empirical** "
          "re-administered vs carried-forward split (within-patient V0→V1 change rate; the dictionary's "
          "`temporal_scope` is hardcoded `current`, so it cannot be used). No scoring, no imputation.", "",
          "> **Read the change rate as re-administration, not state.** A high rate means the item was "
          "*re-collected with varying answers*, not that the construct moved. The CTQ childhood-trauma "
          "items (a fixed history) reach change rates ~0.9 from recall noise alone. This stage decides "
          "**coverage** only; **trait vs state is G3** (stage 35), which deconvolves measurement error.", "",
          "## Retention (unique patients per visit; V0–V2 is the M3 window)",
          ret_w.loc[list(VISITS)].to_markdown(),
          "", "Fraction of each cohort's V0 roster retained:",
          fracv0.loc[list(VISITS)].to_markdown(), "",
          f"- All three cohorts well-represented at V1/V2 (total {int(ret_w.loc['V1','total'])} / "
          f"{int(ret_w.loc['V2','total'])} vs {int(ret_w.loc['V0','total'])} at V0). Full visit grid "
          "in `reports/30_retention.csv`. Attrition is *characterized* in G6 (stage 31), never filled.", "",
          f"## Per-axis follow-up coverage (modeled indicators with ≥{MIN_OBS} obs)", "",
          "`verdict` = coverage only (✅ trackable: ≥1 indicator ≥30 obs at V1 *and* V2 · ⚠️ thin: ≤2 "
          "indicators · ⛔ coverage-limited). `n_readministered`/`n_carried` = re-administration split "
          "(not trait/state).", ""]
    show = axis.copy()
    show["v"] = show["verdict"].map(_icon)
    show = show[["axis", "n_indicators", "items_ge30_V0", "items_ge30_V1", "items_ge30_V2",
                 "n_readministered", "n_carried", "median_change_rate", "verdict", "v"]]
    md += [show.to_markdown(index=False), "",
           f"- Windows (MADRS/QIDS/STAI cross-loaders, no home axis): {len(win)} item(s) "
           f"[{', '.join(win)}] — inform severity/cognition/sleep via cross-loadings.",
           "- Per-indicator detail (n_obs per visit, change rate, class): "
           "`reports/30_indicator_temporal.csv`.", "",
           "## Feasibility read (gate)",
           "- **Trackable** (fresh follow-up data at V1 *and* V2): "
           + ", ".join(axis.loc[axis.verdict == "trackable", "axis"]) + ".",
           "- **Thin** (≤2 indicators — scored but caveated): "
           + (", ".join(axis.loc[axis.verdict == "thin", "axis"]) or "none") + ".",
           "- **Coverage-limited at follow-up**: "
           + (", ".join(axis.loc[axis.verdict == "coverage-limited", "axis"]) or "none") + ".",
           "- **Carried-forward / identical** indicators (cannot inform change): "
           + (", ".join(cov.index[cov.temporal_class == "carried"]) or "none")
           + " — a single indicator; **no axis is carry-forward**, so every axis is scored from its own "
           "observed cells at each visit (correcting the earlier 'developmental_risk is static' "
           "assumption — its CTQ items are re-administered; G3 will test whether that variation is "
           "genuine state or recall noise).", "",
           "## Decision for the gate",
           "Confirm the V0→V1→V2 window and the per-axis coverage above before building the scoring "
           "substrate (stage 32). All trackable axes are scored per visit; `mania_activation` (2 "
           "indicators) is scored but flagged thin; trait vs state for every axis is deferred to G3.", "",
           "Artifacts: `reports/30_{retention,axis_coverage,indicator_temporal}.csv` · "
           "`docs/figures/30_{retention,axis_coverage}.png`."]
    (REPORTS / "30_inventory.md").write_text("\n".join(md))
    print("\n".join(md))


def _figures(ret_w, fracv0, axis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis = list(VISITS)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for c in COHORTS:
        ax[0].plot(vis, ret_w.loc[vis, c].values, "o-", label=c.upper())
    ax[0].plot(vis, ret_w.loc[vis, "total"].values, "ks--", label="total", lw=2)
    ax[0].set_title("Retention — patients per visit (M3 window)")
    ax[0].set_ylabel("patients"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for c in COHORTS:
        ax[1].plot(vis, (fracv0.loc[vis, c] * 100).values, "o-", label=c.upper())
    ax[1].set_title("Retention — % of cohort V0 roster")
    ax[1].set_ylabel("% retained"); ax[1].set_ylim(0, 105); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "30_retention.png", dpi=130); plt.close(fig)

    # axis coverage heatmap (items >=30 obs per visit) + re-administration intensity bars
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    mat = axis.set_index("axis")[[f"items_ge30_{v}" for v in vis]].reindex(list(CANON))
    im = ax[0].imshow(mat.values, cmap="YlGnBu", aspect="auto")
    ax[0].set_xticks(range(len(vis))); ax[0].set_xticklabels(vis)
    ax[0].set_yticks(range(len(mat))); ax[0].set_yticklabels(mat.index, fontsize=8)
    for i in range(len(mat)):
        for j in range(len(vis)):
            ax[0].text(j, i, int(mat.values[i, j]), ha="center", va="center", fontsize=8)
    ax[0].set_title(f"Indicators with ≥{MIN_OBS} obs per axis × visit")
    fig.colorbar(im, ax=ax[0], shrink=0.8)
    a2 = axis.set_index("axis").reindex(list(CANON))
    ax[1].barh(range(len(a2)), a2["median_change_rate"].values, color="#756bb1")
    ax[1].axvline(HIVAR_THR, color="k", ls="--", lw=1, label=f"high variability ≥ {HIVAR_THR}")
    ax[1].set_yticks(range(len(a2))); ax[1].set_yticklabels(a2.index, fontsize=8)
    ax[1].invert_yaxis(); ax[1].set_xlabel("median V0→V1 change rate (re-administration, ≠ state)")
    ax[1].set_title("Axis re-administration intensity"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "30_axis_coverage.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
