"""§4.5 — parsimonious screening panel: distil the 7 axes into a short item set.

Clinical-translation step (reviewer 2.1). The 7 locked dimensional axes are scored from a
54-domain research battery — too long for routine care. We distil them into **short panels**:
a sparse item->axis map that reconstructs the axis scores from a handful of routinely collected
items, so a clinic could compute approximate dimensional scores in a fraction of the time.

Two panels (the research model is never filled; a PANEL is a deployment surrogate that *does*
mean-impute its few items — a disclosed, separate choice):
  - **Shared panel** — a multi-task elastic-net selects ONE shared item set for all 7 axes
    (row-wise L1); densest support within a <=15-item cap. Parsimony-optimal, but a symptom-
    optimized shared L1 drops items unique to low-variance axes (work-disability, onset).
  - **Per-axis (group-aware) panel** — the top-2 elastic-net items *per axis*, unioned, so every
    axis contributes its defining items (covers work-disability/onset within a similar budget).
Both report a flagged **routine metabolic-panel** add-on (BMI, waist, triglycerides, HDL, glucose,
HbA1c, BP), because the metabolic axis loads on labs no questionnaire can produce.

Validation (leakage-safe):
  1. Reconstruction R^2 per axis with the elastic-net SELECTION re-run inside each CV fold (as in
     20_robustness_cvrefit.py), so selection optimism cannot inflate it.
  2. Head-to-head vs DSM (reuse the §2.9 design) under **repeated 5-fold CV (R=200) with 95%
     intervals** (as in 11_phase5_ci.py), reporting BOTH the panel axes alone (M1-M0) and the
     combined model (M2-M0) — does the cheap panel preserve the dimensions' advantage over DSM
     (QoL) and complement it (functioning)? Full-model reference: QoL +0.038, functioning combined
     +0.034 (§3.4).

Artifacts: results/screening_panel_{items,fidelity,headtohead}.csv + meta.json;
results/reports/screening_panel.html + results/reports/figures/fig7_screening_panel.{png,svg}.
Run:  python3 scripts/22_screening_panel.py
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

import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
from sklearn.linear_model import (  # noqa: E402
    LinearRegression,
    LogisticRegression,
    MultiTaskElasticNet,
    Ridge,
)
from sklearn.model_selection import (  # noqa: E402
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from trans_diag import (  # noqa: E402
    AXIS_NAMES,
    DOMAIN_SECTIONS,
    build_unified_dataframe,
    load_variables,
    to_harmonized_dataset,
)
from trans_diag.domains import BIOLOGY_COMPOSITES, BIOLOGY_SECTIONS  # noqa: E402
from trans_diag.outcomes import OUTCOMES  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "results" / "reports"
DICT = REPO_ROOT / "data" / "face-common-vars.xlsx"
SEED = 0
BUDGETS = [5, 8, 10, 12, 15, 20, 25]      # questionnaire-item budgets to sweep
L1_RATIO = 0.8                             # mostly-L1 elastic net (sparse, collinearity-stable)
ITEM_COVERAGE_FLOOR = 0.20                 # drop items observed in < 20% of patients
HEADLINE_CAP = 15                          # reviewer-requested <=15-item shared panel
K_PER_AXIS = 2                             # per-axis (group-aware) panel: top items per axis
PERAXIS_ALPHA = 0.02                       # dense elastic-net for per-axis coefficient ranking
R_CI = 200                                 # repeated-CV repeats for the head-to-head intervals
# Fixed, flagged routine metabolic-panel add-on (recovers the metabolic axis a questionnaire can't)
LAB_ADDON = [c for c, _ in BIOLOGY_COMPOSITES["metabolic_syndrome"]]


def _standardize_impute(train, test):
    """Z-score on train stats, then mean-impute (->0). Disclosed deployment imputation."""
    mu, sd = train.mean(), train.std(ddof=0).replace(0, 1.0)
    ztr = ((train - mu) / sd).fillna(0.0).to_numpy(float)
    zte = ((test - mu) / sd).fillna(0.0).to_numpy(float)
    return ztr, zte


def _support_path(Xq, Y, n_alpha=60):
    """Sparsity path: for each alpha (sparse->dense), the (alpha, support_size, mask)."""
    path = []
    for alpha in np.logspace(0.4, -2.2, n_alpha):
        m = MultiTaskElasticNet(alpha=alpha, l1_ratio=L1_RATIO, max_iter=5000).fit(Xq, Y)
        mask = (np.abs(m.coef_) > 1e-8).any(axis=0)
        path.append((float(alpha), int(mask.sum()), mask))
    return path


def _cap(path, budget):
    """Densest support not exceeding `budget` items (a true item cap)."""
    cand = [p for p in path if p[1] <= budget]
    return max(cand, key=lambda p: p[1]) if cand else path[0]


def _shared_select(q_cols, Xq, Y, alpha):
    """Items with any nonzero coefficient across the 7 axes (shared L1 panel)."""
    m = MultiTaskElasticNet(alpha=alpha, l1_ratio=L1_RATIO, max_iter=5000).fit(Xq, Y)
    keep = (np.abs(m.coef_) > 1e-8).any(axis=0)
    return [q_cols[j] for j in range(len(q_cols)) if keep[j]]


def _per_axis_select(q_cols, Xq, Y, k=K_PER_AXIS, alpha=PERAXIS_ALPHA):
    """Union of the top-`k` items per axis (by |coef| of a dense elastic-net) — guarantees every
    axis contributes its defining items."""
    m = MultiTaskElasticNet(alpha=alpha, l1_ratio=L1_RATIO, max_iter=5000).fit(Xq, Y)
    sel: list[str] = []
    for a in range(Y.shape[1]):
        for j in np.argsort(-np.abs(m.coef_[a]))[:k]:
            if q_cols[j] not in sel:
                sel.append(q_cols[j])
    return sel


def _ols_reconstruct(Xtr, Ytr, Xte):
    return LinearRegression().fit(Xtr, Ytr).predict(Xte)


def _r2(y, yhat):
    ss = ((y - yhat) ** 2).sum(0)
    tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss / np.where(tot > 0, tot, 1.0)


def _repeated_cv(X, y, kind, R=R_CI):
    """R repeats of shuffled 5-fold CV (Ridge R² / logistic AUC), paired by seed — as in 11."""
    Xs = StandardScaler().fit_transform(X)
    out = np.empty(R)
    for r in range(R):
        if kind == "continuous":
            cv = KFold(5, shuffle=True, random_state=r)
            out[r] = np.mean(cross_val_score(Ridge(alpha=1.0), Xs, y, cv=cv, scoring="r2"))
        else:
            cv = StratifiedKFold(5, shuffle=True, random_state=r)
            out[r] = np.mean(cross_val_score(LogisticRegression(max_iter=2000), Xs, y,
                                             cv=cv, scoring="roc_auc"))
    return out


def _ci(a):
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)

    # ── 1. teacher axes + raw item matrix, aligned ──────────────────────────────
    A = pd.read_parquet(RESULTS_DIR / "dimensional_final_scores.parquet")
    A.index = pd.MultiIndex.from_arrays(
        [A.index.get_level_values("cohort").astype(str),
         A.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    A = A.dropna()
    variables = load_variables(DICT)
    by_section = {v.canonical_name: v.section for v in variables}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", DICT, readiness=["READY", "PARTIAL"],
                                     format="long")
        raw = to_harmonized_dataset(df, variables, visit="V0", sections=DOMAIN_SECTIONS)
    X = raw.X.reindex(A.index)
    X = X.loc[:, X.notna().mean() >= ITEM_COVERAGE_FLOOR].select_dtypes("number")
    lab_cols = [c for c in X.columns if by_section.get(c) in BIOLOGY_SECTIONS]
    q_cols = [c for c in X.columns if c not in lab_cols]
    labs = [c for c in LAB_ADDON if c in X.columns]
    print(f"items: {X.shape[1]} ({len(q_cols)} questionnaire, {len(lab_cols)} lab); "
          f"teacher n={len(A):,}; metabolic-panel add-on = {labs}")

    Y = ((A - A.mean()) / A.std(ddof=0)).to_numpy(float)
    Xq_all, _ = _standardize_impute(X[q_cols], X[q_cols])
    metab_idx = AXIS_NAMES.index("metabolic")

    # ── 2. budget sweep (fidelity-vs-length curve; in-sample, illustrative) ─────
    path = _support_path(Xq_all, Y)
    sweep = []
    for B in BUDGETS:
        _alpha, _supp, mask = _cap(path, B)
        sel = [q_cols[j] for j in range(len(q_cols)) if mask[j]]
        Xsel, _ = _standardize_impute(X[sel], X[sel])
        r2 = _r2(Y, _ols_reconstruct(Xsel, Y, Xsel))
        sweep.append({"budget": len(sel), **{a: round(float(r2[i]), 3) for i, a in enumerate(AXIS_NAMES)}})
        print(f"  cap {B:>2} -> {len(sel):>2} items; in-sample R2 "
              f"depression={r2[0]:.2f} mania={r2[3]:.2f} externalizing={r2[4]:.2f} "
              f"metabolic={r2[metab_idx]:.2f}")
    sweep_df = pd.DataFrame(sweep)

    # ── 3. define the two panels (full sample) ──────────────────────────────────
    headline_alpha, _hsupp, hmask = _cap(path, HEADLINE_CAP)
    shared_sel = [q_cols[j] for j in range(len(q_cols)) if hmask[j]]
    peraxis_sel = _per_axis_select(q_cols, Xq_all, Y)
    print(f"\nshared panel: {len(shared_sel)} items (cap {HEADLINE_CAP}); "
          f"per-axis panel: {len(peraxis_sel)} items (top-{K_PER_AXIS}/axis); "
          f"+ {len(labs)}-value metabolic-panel add-on")

    # ── 4. leakage-safe reconstruction: SELECTION re-run inside each CV fold ─────
    kf = KFold(5, shuffle=True, random_state=SEED)
    acc = {k: [] for k in ("shared_q", "shared_ql", "peraxis_ql")}
    for tr, te in kf.split(X):
        Xq_tr, _ = _standardize_impute(X[q_cols].iloc[tr], X[q_cols].iloc[te])
        s_sel = _shared_select(q_cols, Xq_tr, Y[tr], headline_alpha)
        p_sel = _per_axis_select(q_cols, Xq_tr, Y[tr])
        for key, cols in (("shared_q", s_sel), ("shared_ql", s_sel + labs),
                          ("peraxis_ql", p_sel + labs)):
            Xtr, Xte = _standardize_impute(X[cols].iloc[tr], X[cols].iloc[te])
            acc[key].append(_r2(Y[te], _ols_reconstruct(Xtr, Y[tr], Xte)))
    recon = {k: np.mean(v, axis=0) for k, v in acc.items()}
    print("\nin-fold reconstruction R2 (honest):")
    for i, a in enumerate(AXIS_NAMES):
        print(f"  {a:22s} shared={recon['shared_q'][i]:+.2f}  shared+labs={recon['shared_ql'][i]:+.2f}"
              f"  per-axis+labs={recon['peraxis_ql'][i]:+.2f}")

    # ── 5. full-sample panel surrogate axes for the head-to-head ────────────────
    def panel_axes(cols):
        Xs, _ = _standardize_impute(X[cols], X[cols])
        pred = _ols_reconstruct(Xs, Y, Xs)
        return pd.DataFrame(pred, index=A.index, columns=[f"axis{i+1}" for i in range(len(AXIS_NAMES))])

    pa = {"shared (questionnaire)": panel_axes(shared_sel),
          "per-axis (+ labs)": panel_axes(peraxis_sel + labs)}

    # ── 6. head-to-head vs DSM with repeated-CV 95% intervals (M1 and M2) ───────
    df["pid"] = df["cohort"].str.lower() + "::" + df["usubjid_patients"].astype(str)
    v0 = df[df["visit"] == "V0"].set_index("pid")
    vk = df[df["visit"] == "V1"].set_index("pid")
    base = pd.DataFrame(index=v0.index)
    base["age"] = pd.to_numeric(v0["age"], errors="coerce")
    base["sex"] = pd.to_numeric(v0["sex"], errors="coerce")
    dsm = pd.get_dummies(v0["arm"].astype(str), drop_first=True).astype(float)
    dsm_cols = list(dsm.columns)
    base = base.join(dsm)

    def to_pid(p):
        p = p.reset_index()
        p["pid"] = p["cohort"].astype(str) + "::" + p["patient_id"].astype(str)
        return p.set_index("pid")[[c for c in p.columns if c.startswith("axis")]]

    print(f"\nhead-to-head vs DSM (repeated 5-fold CV, R={R_CI}, 95% CI):")
    h2h = []
    for label, paxes in pa.items():
        P = to_pid(paxes); acols = list(P.columns)
        for name, kind, col, tf in OUTCOMES:
            if col not in df.columns:
                continue
            y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
            yk = pd.to_numeric(vk[col], errors="coerce")
            if tf is not None:
                yk = tf(yk)
            d = base.join(y0).join(P).join(yk.rename("y")).dropna(
                subset=["y", "baseline", "age", "sex"] + acols)
            if (kind == "binary" and (d["y"].nunique() < 2 or d["y"].mean() < 0.02)) or len(d) < 200:
                continue
            yv = d["y"].to_numpy(float); bc = ["baseline", "age", "sex"]
            m0 = _repeated_cv(d[bc + dsm_cols].to_numpy(float), yv, kind)
            m1 = _repeated_cv(d[bc + acols].to_numpy(float), yv, kind)
            m2 = _repeated_cv(d[bc + dsm_cols + acols].to_numpy(float), yv, kind)
            (dd, dl, dh) = _ci(m1 - m0); (cd, cl, ch) = _ci(m2 - m0)
            metric = "R2" if kind == "continuous" else "AUC"
            h2h.append({"panel": label, "outcome": name, "n": len(d), "metric": metric,
                        "DSM": round(float(np.mean(m0)), 3), "axes": round(float(np.mean(m1)), 3),
                        "combined": round(float(np.mean(m2)), 3),
                        "axes_minus_DSM": f"{dd:+.3f} [{dl:+.3f},{dh:+.3f}]",
                        "combined_minus_DSM": f"{cd:+.3f} [{cl:+.3f},{ch:+.3f}]",
                        "axes_minus_DSM_mean": round(dd, 3), "combined_minus_DSM_mean": round(cd, 3),
                        "axes_excludes_0": bool(dl > 0 or dh < 0),
                        "combined_excludes_0": bool(cl > 0 or ch < 0)})
            print(f"  [{label}] {name}: axes−DSM {dd:+.3f}[{dl:+.3f},{dh:+.3f}]  "
                  f"combined−DSM {cd:+.3f}[{cl:+.3f},{ch:+.3f}]")
    h2h_df = pd.DataFrame(h2h)

    # ── 7. persist ──────────────────────────────────────────────────────────────
    items_rows = ([{"item": c, "instrument": by_section.get(c, "?"), "block": "shared"} for c in shared_sel]
                  + [{"item": c, "instrument": by_section.get(c, "?"), "block": "per-axis"} for c in peraxis_sel]
                  + [{"item": c, "instrument": by_section.get(c, "?"), "block": "metabolic-panel"} for c in labs])
    pd.DataFrame(items_rows).to_csv(RESULTS_DIR / "screening_panel_items.csv", index=False)
    sweep_df.to_csv(RESULTS_DIR / "screening_panel_fidelity.csv", index=False)
    h2h_df.to_csv(RESULTS_DIR / "screening_panel_headtohead.csv", index=False)

    def axdict(arr):
        return {a: round(float(arr[i]), 3) for i, a in enumerate(AXIS_NAMES)}

    def h2hpick(panel, outcome, field):
        m = h2h_df[(h2h_df.panel == panel) & (h2h_df.outcome == outcome)]
        return m[field].iloc[0] if len(m) else None

    meta = {"n_teacher": int(len(A)), "n_items_total": int(X.shape[1]),
            "headline_cap": HEADLINE_CAP, "k_per_axis": K_PER_AXIS, "R_ci": R_CI,
            "n_questionnaire_items": len(shared_sel), "n_peraxis_items": len(peraxis_sel),
            "shared_items": shared_sel, "peraxis_items": peraxis_sel, "metabolic_panel_addon": labs,
            "recon_r2_questionnaire": axdict(recon["shared_q"]),
            "recon_r2_questionnaire_plus_labs": axdict(recon["shared_ql"]),
            "recon_r2_peraxis_plus_labs": axdict(recon["peraxis_ql"]),
            "qol_panel_axes_minus_DSM": h2hpick("shared (questionnaire)", "EQ-5D quality of life", "axes_minus_DSM_mean"),
            "qol_panel_axes_minus_DSM_ci": h2hpick("shared (questionnaire)", "EQ-5D quality of life", "axes_minus_DSM"),
            "egf_panel_combined_minus_DSM_ci": h2hpick("shared (questionnaire)", "EGF functioning", "combined_minus_DSM"),
            "qol_peraxis_axes_minus_DSM_ci": h2hpick("per-axis (+ labs)", "EQ-5D quality of life", "axes_minus_DSM"),
            "note": "Item->axis sparse distillation (MultiTaskElasticNet, in-fold selection). Two panels: "
                    "a <=15-item SHARED panel and a group-aware PER-AXIS panel (top-2 items/axis), each + a "
                    "flagged routine metabolic-panel add-on (the metabolic axis is not questionnaire-recoverable). "
                    "Head-to-head under repeated 5-fold CV (R=200, 95% CI), M1 (axes) and M2 (combined). "
                    "Deployment surrogate (mean-imputes its items); the research model stays imputation-free."}
    (RESULTS_DIR / "screening_panel_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    _report(sweep_df, recon, shared_sel, peraxis_sel, labs, h2h_df)
    print("\nWrote results/screening_panel_* + results/reports/screening_panel.html. Done.")
    return 0


def _report(sweep_df, recon, shared_sel, peraxis_sel, labs, h2h_df):
    f1 = go.Figure()
    for a in ("depression_severity", "mania_activation", "externalizing", "metabolic"):
        f1.add_scatter(x=sweep_df["budget"], y=sweep_df[a], mode="lines+markers", name=a)
    f1.update_layout(title="Reconstruction fidelity vs panel length (shared, in-sample)", height=320,
                     xaxis_title="# questionnaire items", yaxis_title="R²", margin=dict(t=44))
    f2 = go.Figure()
    for key, nm in (("shared_q", "shared"), ("shared_ql", "shared+labs"), ("peraxis_ql", "per-axis+labs")):
        f2.add_bar(x=list(AXIS_NAMES), y=list(recon[key]), name=nm)
    f2.update_layout(title="In-fold reconstruction R² per axis (honest)", barmode="group",
                     height=360, yaxis_title="R²", xaxis_tickangle=-30, margin=dict(t=44, b=130))
    f3 = go.Figure()
    for field, nm in (("axes_minus_DSM_mean", "axes − DSM"), ("combined_minus_DSM_mean", "combined − DSM")):
        sub = h2h_df[h2h_df.panel == "shared (questionnaire)"]
        f3.add_bar(x=sub["outcome"], y=sub[field], name=nm)
    f3.add_hline(y=0, line_dash="dash")
    f3.update_layout(title="Shared panel vs DSM on 1-yr outcomes (full: QoL +0.038, func. combined +0.034)",
                     barmode="group", height=320, yaxis_title="Δ vs DSM", margin=dict(t=44))
    try:
        from plotly.subplots import make_subplots
        figdir = REPORTS_DIR / "figures"; figdir.mkdir(parents=True, exist_ok=True)
        g = make_subplots(rows=1, cols=2, subplot_titles=(
            "In-fold reconstruction R² per axis", "Panel vs DSM (Δ, 1-yr outcomes)"))
        for key, nm in (("shared_q", "shared"), ("peraxis_ql", "per-axis+labs")):
            g.add_bar(x=list(AXIS_NAMES), y=list(recon[key]), name=nm, row=1, col=1)
        sub = h2h_df[h2h_df.panel == "shared (questionnaire)"]
        g.add_bar(x=sub["outcome"], y=sub["axes_minus_DSM_mean"], name="axes−DSM", row=1, col=2)
        g.add_bar(x=sub["outcome"], y=sub["combined_minus_DSM_mean"], name="combined−DSM", row=1, col=2)
        g.add_hline(y=0, line_dash="dash", row=1, col=2)
        g.update_layout(title="Figure 7. Parsimonious screening panel", barmode="group",
                        margin=dict(t=70, b=130))
        g.update_xaxes(tickangle=-30)
        for ext in ("png", "svg"):
            g.write_image(str(figdir / f"fig7_screening_panel.{ext}"), width=1150, height=440, scale=2)
        print("  wrote results/reports/figures/fig7_screening_panel.png/.svg")
    except Exception as e:  # kaleido optional
        print(f"  (static fig7 skipped: {e})")

    css = "body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}h1{color:#2b3a55}.c{background:#eef6fb;border-left:4px solid #2b8cbe;padding:10px 14px;margin:12px 0}code{font-size:12px}table{border-collapse:collapse;font-size:12px;margin:10px 0}th,td{border:1px solid #e5e7eb;padding:4px 8px}th{background:#eef2f7}"
    hrows = "".join(
        f"<tr><td>{r.panel}</td><td>{r.outcome}</td><td>{r.axes_minus_DSM}</td>"
        f"<td>{r.combined_minus_DSM}</td></tr>" for r in h2h_df.itertuples())
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Parsimonious screening panel (§4.5)</h1>",
            f"<div class='c'><b>Shared panel</b> ({len(shared_sel)} items): "
            f"<code>{', '.join(shared_sel)}</code>.<br><b>Per-axis panel</b> ({len(peraxis_sel)} items): "
            f"<code>{', '.join(peraxis_sel)}</code>.<br><b>Routine metabolic-panel add-on</b>: "
            f"<code>{', '.join(labs)}</code>.</div>",
            "<table><tr><th>panel</th><th>outcome</th><th>axes−DSM [95% CI]</th>"
            "<th>combined−DSM [95% CI]</th></tr>", hrows, "</table>",
            pio.to_html(f1, include_plotlyjs="cdn", full_html=False),
            pio.to_html(f2, include_plotlyjs=False, full_html=False),
            pio.to_html(f3, include_plotlyjs=False, full_html=False), "</body></html>"]
    (REPORTS_DIR / "screening_panel.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
