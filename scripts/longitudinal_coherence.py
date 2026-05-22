"""Phase 4 — temporal coherence of the V0 trans-diagnostic phenotypes (V0→V4).

Do the five V0 domain-phenotypes persist across annual follow-up visits? We
build the SAME domain scores at every visit, assign each patient-visit to its
nearest V0 phenotype centroid (masked, no imputation), and measure how often a
patient stays in their V0 phenotype.

Pipeline:
  build_unified_dataframe (all visits)
    -> per visit: to_harmonized_dataset(sections=DOMAIN_SECTIONS) -> raw items
    -> pool visits -> build_domain_scores (common scale across visits)
    -> restrict to the 54 V0 clustering domains
    -> residualize on per-visit age + sex (spline + cross-fit)
    -> V0 cluster centroids (from results/cluster_domains_assignments.csv)
    -> masked nearest-centroid assignment for every (patient, visit)
    -> coherence: ARI(V0↔Vk), transition matrices, per-phenotype persistence.

DR is excluded at V3 (attrition cliff: only ~2 patients).

Artifacts (results/ + reports/):
    longitudinal_assignments.csv   patient_uid × visit → phenotype
    longitudinal_coherence.csv     per-visit n_paired / ARI-vs-V0 / % persistence
    longitudinal_transitions.csv   V0→Vk transition counts (long form)
    longitudinal.html              Sankey + transition heatmaps + coherence table

Run:  python3 scripts/longitudinal_coherence.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "archive"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402

from face_common import (  # noqa: E402
    ADMINISTRATIVE_FEATURES,
    COHORT_TO_CODE,
    DOMAIN_SECTIONS,
    build_domain_scores,
    build_unified_dataframe,
    load_variables,
    residualize_features,
    to_harmonized_dataset,
)

DATA_DIR = REPO_ROOT / "data"
DICT_PATH = REPO_ROOT / "face-common-vars.xlsx"
RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"

VISITS = ["V0", "V1", "V2", "V3", "V4"]
MIN_OBS = 10            # min observed domains to assign a patient-visit
SPLINE_DF = 4
CROSS_FIT = 5
RANDOM_STATE = 0
PALETTE = ["#3498db", "#e67e22", "#16a085", "#9b59b6", "#c0392b", "#7f8c8d"]


def _visit_index(frame_visit: pd.DataFrame, visit: str) -> pd.MultiIndex:
    code = frame_visit["cohort"].map(COHORT_TO_CODE)
    pid = frame_visit["usubjid_patients"].astype(str)
    return pd.MultiIndex.from_arrays(
        [code.to_numpy(), pid.to_numpy(), [visit] * len(frame_visit)],
        names=("cohort", "patient_id", "visit"))


# Phenotype assignment is a classifier trained on V0 (domain scores -> V0 labels);
# see main(). A nearest-centroid in domain space cannot reproduce the V0
# *spectral-embedding* clusters (different geometry; self-ARI ≈ 0), so we learn the
# mapping with a NaN-native gradient-boosted classifier instead.


def main() -> int:
    RESULTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    variables = load_variables(DICT_PATH)
    exclude = set(ADMINISTRATIVE_FEATURES) | {
        v.canonical_name for v in variables if v.canonical_name.endswith("_mhoccur")}
    v0_domains = list(pd.read_parquet(RESULTS_DIR / "cluster_domains_scores.parquet").columns)
    v0_assign = pd.read_csv(RESULTS_DIR / "cluster_domains_assignments.csv")
    v0_label = pd.Series(
        v0_assign["cluster"].to_numpy(),
        index=pd.MultiIndex.from_arrays(
            [v0_assign["cohort"].str.lower().to_numpy(),
             v0_assign["usubjid_patients"].astype(str).to_numpy()],
            names=("cohort", "patient_id")), name="v0_label")

    print("Loading frame + building per-visit domain scores...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(DATA_DIR, DICT_PATH, readiness=["READY", "PARTIAL"],
                                     format="long")
        item_blocks, covar_blocks = [], []
        for v in VISITS:
            fv = df[df["visit"] == v]
            if fv.empty:
                continue
            ds = to_harmonized_dataset(df, variables, visit=v, exclude=exclude,
                                       sections=DOMAIN_SECTIONS)
            X = ds.X.copy()
            X.index = pd.MultiIndex.from_arrays(
                [X.index.get_level_values("cohort"),
                 X.index.get_level_values("patient_id"), [v] * len(X)],
                names=("cohort", "patient_id", "visit"))
            item_blocks.append(X)
            cov = fv[["cohort", "usubjid_patients", "age", "sex"]].copy()
            cov.index = _visit_index(fv, v)
            covar_blocks.append(cov[["age", "sex"]])

    items = pd.concat(item_blocks)
    covars = pd.concat(covar_blocks)
    covars = covars[~covars.index.duplicated(keep="first")].reindex(items.index)

    # pooled domain scores (common scale across visits), restricted to V0 domains
    scores, _ = build_domain_scores(items, variables)
    scores = scores.reindex(columns=v0_domains)
    print(f"  pooled patient-visits: {len(scores):,}  domains: {scores.shape[1]}")

    # residualize on per-visit age + sex (nonlinear spline + cross-fit)
    scores_r = residualize_features(scores, covars, spline_df=SPLINE_DF,
                                    cross_fit=CROSS_FIT, random_state=RANDOM_STATE)

    # Train a phenotype classifier on V0 (domain scores -> V0 embedding labels) and
    # apply it to every visit. HistGradientBoosting handles NaN natively (no
    # imputation); 5-fold accuracy on V0 reports the rule's validity.
    v0_rows = scores_r.xs("V0", level="visit")
    yv0 = v0_label.reindex(v0_rows.index)
    train = yv0.notna() & (v0_rows.notna().sum(axis=1) >= MIN_OBS)
    Xtr = v0_rows[train].to_numpy(np.float64)
    ytr = yv0[train].astype(int).to_numpy()
    k = int(len(np.unique(ytr)))
    clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    cv_acc = float(cross_val_score(clf, Xtr, ytr, cv=5).mean())
    clf.fit(Xtr, ytr)
    nobs = scores_r.notna().sum(axis=1).to_numpy()
    lab = np.where(nobs >= MIN_OBS, clf.predict(scores_r.to_numpy(np.float64)), -1)

    assign = pd.DataFrame({
        "cohort": [c.upper() for c in scores_r.index.get_level_values("cohort")],
        "patient_id": scores_r.index.get_level_values("patient_id"),
        "visit": scores_r.index.get_level_values("visit"),
        "phenotype": lab,
    })
    assign["patient_uid"] = assign["cohort"] + "::" + assign["patient_id"]
    assign = assign[assign["phenotype"] >= 0]
    assign.to_csv(RESULTS_DIR / "longitudinal_assignments.csv", index=False)

    print(f"\nV0 phenotype classifier: 5-fold accuracy = {cv_acc:.3f} "
          f"({int(train.sum()):,} V0 patients, k={k}) — validity of the assignment rule")

    # coherence vs V0
    wide = assign.pivot_table(index="patient_uid", columns="visit", values="phenotype",
                              aggfunc="first")
    cohort_of = assign.drop_duplicates("patient_uid").set_index("patient_uid")["cohort"]
    coh_rows, transitions = [], []
    for v in VISITS[1:]:
        if v not in wide.columns:
            continue
        pair = wide[["V0", v]].dropna()
        if v == "V3":                                  # DR attrition cliff
            pair = pair[cohort_of.reindex(pair.index) != "DR"]
        if len(pair) < 20:
            continue
        a0, ak = pair["V0"].astype(int), pair[v].astype(int)
        ari = adjusted_rand_score(a0, ak)
        persist = float((a0.to_numpy() == ak.to_numpy()).mean())
        coh_rows.append({"visit": v, "n_paired": len(pair), "ari_vs_V0": ari,
                         "pct_persist": persist})
        ct = pd.crosstab(a0, ak)
        for i in ct.index:
            for j in ct.columns:
                transitions.append({"to_visit": v, "from": int(i), "to": int(j),
                                    "n": int(ct.loc[i, j])})
        print(f"  V0→{v}: n={len(pair):5d}  ARI={ari:.3f}  persistence={persist:.1%}")
    coherence = pd.DataFrame(coh_rows)
    coherence.to_csv(RESULTS_DIR / "longitudinal_coherence.csv", index=False)
    pd.DataFrame(transitions).to_csv(RESULTS_DIR / "longitudinal_transitions.csv", index=False)

    # per-phenotype persistence (V0→V1, the largest follow-up)
    if "V1" in wide.columns:
        p = wide[["V0", "V1"]].dropna()
        per = (p["V0"] == p["V1"]).groupby(p["V0"]).mean()
        print("\nper-phenotype persistence V0→V1:",
              {int(i): round(float(x), 2) for i, x in per.items()})

    _write_report(coherence, pd.DataFrame(transitions), wide, k, cv_acc)

    meta = {"min_obs": MIN_OBS, "residualize": {"spline_df": SPLINE_DF, "cross_fit": CROSS_FIT},
            "v0_classifier_cv_accuracy": cv_acc, "k": k,
            "n_patient_visits_assigned": int(len(assign)),
            "coherence": coherence.to_dict(orient="records"),
            "dr_excluded_at": "V3", "site_note": "site excluded; ComBat sensitivity deferred (#43)"}
    (RESULTS_DIR / "longitudinal_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("\nWrote results/longitudinal_* and reports/longitudinal.html. Done.")
    return 0


def _write_report(coherence, trans, wide, k, cv_acc):
    def heat(v):
        sub = trans[trans["to_visit"] == v]
        if sub.empty:
            return ""
        ct = sub.pivot_table(index="from", columns="to", values="n", fill_value=0)
        ct = ct.reindex(index=range(k), columns=range(k), fill_value=0)
        row = ct.div(ct.sum(1).replace(0, 1), axis=0)        # row-normalized
        fig = go.Figure(go.Heatmap(
            z=row.to_numpy(), x=[f"C{c}" for c in row.columns], y=[f"C{c}" for c in row.index],
            text=ct.to_numpy(), texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=1,
            colorbar=dict(title="row frac", thickness=12)))
        fig.update_layout(title=f"V0 → {v} transitions (row-normalized; n in cells)",
                          height=360, xaxis_title=f"{v} phenotype", yaxis_title="V0 phenotype",
                          margin=dict(t=46, l=70, r=20, b=46))
        return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                           config={"displayModeBar": False})

    # Sankey V0→V1→V2
    sankey_html = ""
    stages = [s for s in ["V0", "V1", "V2"] if s in wide.columns]
    if len(stages) >= 2:
        nodes = [f"{s}·C{c}" for s in stages for c in range(k)]
        nidx = {n: i for i, n in enumerate(nodes)}
        src, tgt, val = [], [], []
        for a, b in zip(stages, stages[1:]):
            pair = wide[[a, b]].dropna()
            ct = pd.crosstab(pair[a].astype(int), pair[b].astype(int))
            for i in ct.index:
                for j in ct.columns:
                    if ct.loc[i, j] > 0:
                        src.append(nidx[f"{a}·C{int(i)}"]); tgt.append(nidx[f"{b}·C{int(j)}"])
                        val.append(int(ct.loc[i, j]))
        node_color = [PALETTE[c % len(PALETTE)] for _ in stages for c in range(k)]
        fig = go.Figure(go.Sankey(
            node=dict(label=nodes, color=node_color, pad=14, thickness=14),
            link=dict(source=src, target=tgt, value=val)))
        fig.update_layout(title="Phenotype flow V0 → V1 → V2", height=460,
                          margin=dict(t=46, l=10, r=10, b=10))
        sankey_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False,
                                  config={"displayModeBar": False})

    css = ("body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;"
           "padding:0 24px 60px;color:#1f2933;line-height:1.5}h1{color:#2b3a55}"
           "h2{color:#2b3a55;border-bottom:2px solid #2b3a55;padding-bottom:6px;margin-top:34px}"
           "table{border-collapse:collapse;font-size:13px;margin:10px 0}"
           "th,td{border:1px solid #e5e7eb;padding:5px 10px}th{background:#eef2f7}"
           ".callout{border-left:4px solid #16a085;background:#f2fbf6;padding:10px 14px;margin:14px 0}")
    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>FACE longitudinal "
             f"coherence</title><style>{css}</style></head><body>",
             "<h1>FACE — temporal coherence of the V0 phenotypes (V0→V4)</h1>",
             f"<div class='callout'>Assignment rule = a phenotype classifier trained on "
             f"V0 (domain scores → V0 labels); 5-fold V0 accuracy <b>{cv_acc:.3f}</b>. "
             "Applied unchanged to each follow-up visit. DR excluded at V3 (attrition).</div>",
             "<h2>Coherence vs V0</h2>",
             coherence.assign(ari_vs_V0=coherence["ari_vs_V0"].round(3),
                              pct_persist=(coherence["pct_persist"]*100).round(0))
             .to_html(index=False, border=0)]
    if sankey_html:
        parts += ["<h2>Phenotype flow</h2>", sankey_html]
    parts.append("<h2>Transition matrices</h2>")
    for v in ["V1", "V2", "V3", "V4"]:
        h = heat(v)
        if h:
            parts.append(h)
    parts.append("</body></html>")
    (REPORTS_DIR / "longitudinal.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
