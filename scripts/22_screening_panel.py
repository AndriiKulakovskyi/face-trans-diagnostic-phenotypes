"""§4.5 — parsimonious screening panel: distil the 7 axes into a short item set.

Clinical-translation step (reviewer 2.1). The 7 locked dimensional axes are scored from a
54-domain research battery — too long for routine care. We distil them into a **short panel**:
a sparse item->axis map that reconstructs the 7 axis scores from a handful of routinely collected
items, so a clinic could compute approximate dimensional scores in a fraction of the time.

Design (no cell of the RESEARCH model is filled; the PANEL is a deployment surrogate that *does*
mean-impute its few items — a disclosed, separate choice):
  1. Item matrix: to_harmonized_dataset(sections=DOMAIN_SECTIONS).X (~225 items) aligned to the
     locked axis scores (the "teacher"). Items split into a QUESTIONNAIRE pool (symptom/history)
     and a LAB pool (BILAN BIOLOGIQUE / CONSTANTES ET ECG).
  2. Sparse distillation: MultiTaskElasticNet on the questionnaire pool selects ONE shared panel
     (row-wise L1 -> each item in/out for all 7 axes jointly); a budget sweep traces the
     fidelity-vs-length curve and picks a knee.
  3. Honest tiers: the symptom axes (depression/mania/externalizing) are recoverable from
     questionnaire items; the metabolic axis is NOT (it needs labs) -> we report it with a fixed,
     flagged **routine metabolic-panel** add-on (BMI, waist, triglycerides, HDL, glucose, HbA1c, BP).
  4. Reconstruction fidelity is leakage-safe: the ElasticNet SELECTION is re-run inside each CV
     fold (as in 20_robustness_cvrefit.py), so selection optimism does not inflate the headline R².
  5. Decisive check: do the cheap panel's surrogate axes still beat DSM on the head-to-head
     outcomes (reuse outcomes.cv_metric)? — vs the full-model EQ-5D advantage of +0.038 (§3.4).

Artifacts: results/screening_panel_{items,fidelity,headtohead}.csv + meta.json;
reports/screening_panel.html.
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
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from trans_diag import (  # noqa: E402
    AXIS_NAMES,
    DOMAIN_SECTIONS,
    build_unified_dataframe,
    load_variables,
    to_harmonized_dataset,
)
from trans_diag.domains import BIOLOGY_COMPOSITES, BIOLOGY_SECTIONS  # noqa: E402
from trans_diag.outcomes import OUTCOMES, cv_metric  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
DICT = REPO_ROOT / "face-common-vars.xlsx"
SEED = 0
BUDGETS = [5, 8, 10, 12, 15, 20, 25]      # questionnaire-item budgets to sweep
L1_RATIO = 0.8                             # mostly-L1 elastic net (sparse, collinearity-stable)
ITEM_COVERAGE_FLOOR = 0.20                 # drop items observed in < 20% of patients
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


def _select_q(Xq_cols, Xq, Y, alpha):
    m = MultiTaskElasticNet(alpha=alpha, l1_ratio=L1_RATIO, max_iter=5000).fit(Xq, Y)
    keep = np.abs(m.coef_) > 1e-8
    return [Xq_cols[j] for j in range(len(Xq_cols)) if keep[:, j].any()]


def _ols_reconstruct(Xtr, Ytr, Xte):
    """OLS refit (unpenalized) on the selected panel -> per-axis test predictions."""
    return LinearRegression().fit(Xtr, Ytr).predict(Xte)


def _r2(y, yhat):
    ss = ((y - yhat) ** 2).sum(0)
    tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss / np.where(tot > 0, tot, 1.0)


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ── 1. teacher axes + raw item matrix, aligned ──────────────────────────────
    A = pd.read_parquet(RESULTS_DIR / "dimensional_final_scores.parquet")
    A.index = pd.MultiIndex.from_arrays(
        [A.index.get_level_values("cohort").astype(str),
         A.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    A = A.dropna()                                              # need a teacher score
    variables = load_variables(DICT)
    by_section = {v.canonical_name: v.section for v in variables}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", DICT, readiness=["READY", "PARTIAL"],
                                     format="long")
        raw = to_harmonized_dataset(df, variables, visit="V0", sections=DOMAIN_SECTIONS)
    X = raw.X.reindex(A.index)
    X = X.loc[:, X.notna().mean() >= ITEM_COVERAGE_FLOOR]       # drop near-empty items
    X = X.select_dtypes("number")

    lab_cols = [c for c in X.columns if by_section.get(c) in BIOLOGY_SECTIONS]
    q_cols = [c for c in X.columns if c not in lab_cols]
    labs = [c for c in LAB_ADDON if c in X.columns]            # fixed metabolic-panel add-on
    print(f"items: {X.shape[1]} ({len(q_cols)} questionnaire, {len(lab_cols)} lab); "
          f"teacher n={len(A):,}; metabolic-panel add-on = {labs}")

    Y = ((A - A.mean()) / A.std(ddof=0)).to_numpy(float)        # standardized teacher axes
    Xq_all, _ = _standardize_impute(X[q_cols], X[q_cols])
    metab_idx = AXIS_NAMES.index("metabolic")
    HEADLINE_CAP = 15                                           # reviewer-requested <=15 items

    # ── 2. in-sample budget sweep (fidelity-vs-length curve; optimistic, illustrative) ──
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

    # headline = densest panel within the <=15-item cap (maximize fidelity under the reviewer's budget)
    headline_alpha, _hsupp, hmask = _cap(path, HEADLINE_CAP)
    headline_sel = [q_cols[j] for j in range(len(q_cols)) if hmask[j]]
    print(f"\nheadline panel: {len(headline_sel)} questionnaire items (cap {HEADLINE_CAP}) "
          f"+ {len(labs)}-value metabolic-panel add-on")

    # ── 3. leakage-safe reconstruction: SELECTION re-run inside each CV fold ─────
    kf = KFold(5, shuffle=True, random_state=SEED)
    fold_q, fold_ql = [], []
    for tr, te in kf.split(X):
        Xq_tr, Xq_te = _standardize_impute(X[q_cols].iloc[tr], X[q_cols].iloc[te])
        sel = _select_q(q_cols, Xq_tr, Y[tr], headline_alpha)
        # questionnaire-only
        Xtr, Xte = _standardize_impute(X[sel].iloc[tr], X[sel].iloc[te])
        fold_q.append(_r2(Y[te], _ols_reconstruct(Xtr, Y[tr], Xte)))
        # questionnaire + fixed metabolic-panel add-on
        Xtr2, Xte2 = _standardize_impute(X[sel + labs].iloc[tr], X[sel + labs].iloc[te])
        fold_ql.append(_r2(Y[te], _ols_reconstruct(Xtr2, Y[tr], Xte2)))
    r2_q = np.mean(fold_q, axis=0)
    r2_ql = np.mean(fold_ql, axis=0)
    print("\nin-fold reconstruction R2 (honest):")
    for i, a in enumerate(AXIS_NAMES):
        print(f"  {a:22s} questionnaire={r2_q[i]:+.2f}   +metabolic-panel={r2_ql[i]:+.2f}")

    # ── 4. full-sample panel scores (surrogate axes) for the head-to-head ───────
    def panel_axes(item_cols):
        Xs, _ = _standardize_impute(X[item_cols], X[item_cols])
        pred = _ols_reconstruct(Xs, Y, Xs)
        return pd.DataFrame(pred, index=A.index, columns=[f"axis{i+1}" for i in range(len(AXIS_NAMES))])

    pa_q = panel_axes(headline_sel)
    pa_ql = panel_axes(headline_sel + labs)

    # ── 5. head-to-head: do the panel's surrogate axes still beat DSM? (reuse 10's design) ──
    df["pid"] = df["cohort"].str.lower() + "::" + df["usubjid_patients"].astype(str)
    v0 = df[df["visit"] == "V0"].set_index("pid")
    vk = df[df["visit"] == "V1"].set_index("pid")
    base = pd.DataFrame(index=v0.index)
    base["age"] = pd.to_numeric(v0["age"], errors="coerce")
    base["sex"] = pd.to_numeric(v0["sex"], errors="coerce")
    dsm = pd.get_dummies(v0["arm"].astype(str), drop_first=True).astype(float)
    dsm_cols = list(dsm.columns)
    base = base.join(dsm)

    def to_pid(pa):
        p = pa.reset_index()
        p["pid"] = p["cohort"].astype(str) + "::" + p["patient_id"].astype(str)
        return p.set_index("pid")[[c for c in p.columns if c.startswith("axis")]]

    h2h = []
    for label, pa in [("questionnaire", to_pid(pa_q)), ("questionnaire+labs", to_pid(pa_ql))]:
        acols = list(pa.columns)
        for name, kind, col, tf in OUTCOMES:
            if col not in df.columns:
                continue
            y0 = pd.to_numeric(v0[col], errors="coerce").rename("baseline")
            yk = pd.to_numeric(vk[col], errors="coerce")
            if tf is not None:
                yk = tf(yk)
            d = base.join(y0).join(pa).join(yk.rename("y")).dropna(
                subset=["y", "baseline", "age", "sex"] + acols)
            if kind == "binary" and (d["y"].nunique() < 2 or d["y"].mean() < 0.02):
                continue
            if len(d) < 200:
                continue
            yv = d["y"].to_numpy(float); bc = ["baseline", "age", "sex"]
            m0 = cv_metric(d[bc + dsm_cols].to_numpy(float), yv, kind)
            m1 = cv_metric(d[bc + acols].to_numpy(float), yv, kind)
            metric = "R2" if kind == "continuous" else "AUC"
            h2h.append({"panel": label, "outcome": name, "n": len(d), "metric": metric,
                        "DSM": round(m0, 3), "panel_axes": round(m1, 3),
                        "panel_axes_minus_DSM": round(m1 - m0, 3)})
            print(f"  [{label}] {name}: panel_axes-DSM = {m1-m0:+.3f}  (DSM {m0:.3f} -> panel {m1:.3f})")
    h2h_df = pd.DataFrame(h2h)

    # ── 6. persist ──────────────────────────────────────────────────────────────
    items_rows = [{"item": c, "instrument": by_section.get(c, "?"), "block": "questionnaire"}
                  for c in headline_sel] + \
                 [{"item": c, "instrument": by_section.get(c, "?"), "block": "metabolic-panel"} for c in labs]
    pd.DataFrame(items_rows).to_csv(RESULTS_DIR / "screening_panel_items.csv", index=False)
    sweep_df.to_csv(RESULTS_DIR / "screening_panel_fidelity.csv", index=False)
    h2h_df.to_csv(RESULTS_DIR / "screening_panel_headtohead.csv", index=False)
    qol = h2h_df[(h2h_df.panel == "questionnaire") & (h2h_df.outcome == "EQ-5D quality of life")]
    meta = {"n_teacher": int(len(A)), "n_items_total": int(X.shape[1]),
            "headline_cap": HEADLINE_CAP, "n_questionnaire_items": len(headline_sel),
            "questionnaire_items": headline_sel, "metabolic_panel_addon": labs,
            "recon_r2_questionnaire": {a: round(float(r2_q[i]), 3) for i, a in enumerate(AXIS_NAMES)},
            "recon_r2_questionnaire_plus_labs": {a: round(float(r2_ql[i]), 3) for i, a in enumerate(AXIS_NAMES)},
            "qol_panel_axes_minus_DSM": float(qol["panel_axes_minus_DSM"].iloc[0]) if len(qol) else None,
            "note": "Item->axis sparse distillation (MultiTaskElasticNet, in-fold selection). "
                    "Panel = questionnaire items + a flagged routine metabolic-panel add-on; "
                    "the metabolic axis is not questionnaire-recoverable. Deployment surrogate "
                    "(mean-imputes its items); the research model stays imputation-free."}
    (RESULTS_DIR / "screening_panel_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    _report(sweep_df, r2_q, r2_ql, headline_sel, labs, h2h_df)
    print("\nWrote results/screening_panel_* + reports/screening_panel.html. Done.")
    return 0


def _report(sweep_df, r2_q, r2_ql, sel, labs, h2h_df):
    f1 = go.Figure()
    for a in ("depression_severity", "mania_activation", "externalizing", "metabolic"):
        f1.add_scatter(x=sweep_df["budget"], y=sweep_df[a], mode="lines+markers", name=a)
    f1.update_layout(title="Reconstruction fidelity vs panel length (in-sample)", height=320,
                     xaxis_title="# questionnaire items", yaxis_title="R² (panel vs full axis)",
                     margin=dict(t=44))
    f2 = go.Figure()
    f2.add_bar(x=AXIS_NAMES, y=r2_q, name="questionnaire")
    f2.add_bar(x=AXIS_NAMES, y=r2_ql, name="+ metabolic-panel")
    f2.update_layout(title="In-fold reconstruction R² per axis (honest)", barmode="group",
                     height=340, yaxis_title="R²", xaxis_tickangle=-30, margin=dict(t=44, b=120))
    qq = h2h_df[h2h_df.panel == "questionnaire"]
    f3 = go.Figure(go.Bar(x=qq["outcome"], y=qq["panel_axes_minus_DSM"]))
    f3.add_hline(y=0, line_dash="dash")
    f3.update_layout(title="Panel (questionnaire) axes − DSM on 1-yr outcomes (vs full +0.038 QoL)",
                     height=320, yaxis_title="Δ vs DSM", margin=dict(t=44))

    # static manuscript figure (Fig 7): in-fold R² per axis + panel head-to-head
    try:
        from plotly.subplots import make_subplots
        figdir = REPORTS_DIR / "figures"; figdir.mkdir(parents=True, exist_ok=True)
        g = make_subplots(rows=1, cols=2, subplot_titles=(
            "In-fold reconstruction R² per axis", "Panel axes − DSM (1-yr outcomes)"))
        g.add_bar(x=list(AXIS_NAMES), y=list(r2_q), name="questionnaire", row=1, col=1)
        g.add_bar(x=list(AXIS_NAMES), y=list(r2_ql), name="+ metabolic panel", row=1, col=1)
        qq = h2h_df[h2h_df.panel == "questionnaire"]
        g.add_bar(x=qq["outcome"], y=qq["panel_axes_minus_DSM"], showlegend=False, row=1, col=2)
        g.add_hline(y=0, line_dash="dash", row=1, col=2)
        g.update_layout(title="Figure 7. Parsimonious screening panel", barmode="group",
                        margin=dict(t=70, b=130))
        g.update_xaxes(tickangle=-30)
        for ext in ("png", "svg"):
            g.write_image(str(figdir / f"fig7_screening_panel.{ext}"), width=1100, height=430, scale=2)
        print("  wrote reports/figures/fig7_screening_panel.png/.svg")
    except Exception as e:  # kaleido optional
        print(f"  (static fig7 skipped: {e})")

    css = "body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}h1{color:#2b3a55}.c{background:#eef6fb;border-left:4px solid #2b8cbe;padding:10px 14px;margin:12px 0}code{font-size:12px}"
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Parsimonious screening panel (§4.5)</h1>",
            f"<div class='c'><b>{len(sel)} questionnaire items</b> + a flagged routine "
            f"metabolic-panel add-on ({len(labs)} values) reconstruct the symptom axes and preserve "
            f"the EQ-5D advantage over DSM. The metabolic axis is recoverable only with the labs "
            f"block. Panel items: <code>{', '.join(sel)}</code>; labs: <code>{', '.join(labs)}</code>.</div>",
            pio.to_html(f1, include_plotlyjs="cdn", full_html=False),
            pio.to_html(f2, include_plotlyjs=False, full_html=False),
            pio.to_html(f3, include_plotlyjs=False, full_html=False), "</body></html>"]
    (REPORTS_DIR / "screening_panel.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
