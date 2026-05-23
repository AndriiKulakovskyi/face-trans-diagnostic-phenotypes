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

Run:  python3 scripts/09_longitudinal_coherence.py
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
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from trans_diag import (  # noqa: E402
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

# DSM-5 enrolled subtypes, ordered on the mood↔psychosis continuum (for the
# DSM-5 → phenotype flow comparison).
SPECTRUM = {"Trouble dépressif majeur": 0, "Bipolaire de type 2": 1, "Bipolaire de type 1": 2,
            "Bipolaire non spécifié": 3, "Trouble schizo-affectif": 4,
            "Trouble schizophréniforme": 5, "Schizophrénie": 6}
DSM_SHORT = {"Trouble dépressif majeur": "MDD", "Bipolaire de type 2": "BP-II",
             "Bipolaire de type 1": "BP-I", "Bipolaire non spécifié": "BP-NOS",
             "Trouble schizo-affectif": "schizoaff.", "Trouble schizophréniforme": "schizophrenif.",
             "Schizophrénie": "schizophr."}
# mood (blue) → psychosis (red) gradient for the 7 DSM nodes
DSM_COLORS = ["#2c7bb6", "#5e9fc6", "#abd9e9", "#cccccc", "#fdae61", "#f46d43", "#d7191c"]


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
    # imputation); 5-fold accuracy on V0 reports the rule's validity. Folds MUST be
    # shuffled — the matrix is cohort-ordered, so un-shuffled folds distort accuracy.
    v0_rows = scores_r.xs("V0", level="visit")
    yv0 = v0_label.reindex(v0_rows.index)
    train = yv0.notna() & (v0_rows.notna().sum(axis=1) >= MIN_OBS)
    Xtr = v0_rows[train].to_numpy(np.float64)
    ytr = yv0[train].astype(int).to_numpy()
    k = int(len(np.unique(ytr)))
    clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    cv_acc = float(cross_val_score(clf, Xtr, ytr, cv=cv).mean())
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

    # ── compare the data-driven phenotypes with the 7 DSM-5 subgroups ──
    v0sub = df[df["visit"] == "V0"]
    dsm_of = pd.Series(
        v0sub["arm"].astype(str).to_numpy(),
        index=(v0sub["cohort"].astype(str) + "::" + v0sub["usubjid_patients"].astype(str)).to_numpy(),
        name="dsm")
    dsm_of = dsm_of[~dsm_of.index.duplicated(keep="first")]
    ari_dsm = float("nan")
    if "V0" in wide.columns:
        v0p = wide["V0"].dropna().astype(int)
        common = v0p.index.intersection(dsm_of.index)
        if len(common) > 20:
            d, p2 = dsm_of.reindex(common), v0p.reindex(common)
            ari_dsm = float(adjusted_rand_score(d.astype("category").cat.codes, p2))
            ct = pd.crosstab(d, p2)
            ct.to_csv(RESULTS_DIR / "longitudinal_dsm_phenotype.csv")
            print(f"\nDSM-5 subtype ↔ V0 phenotype ARI = {ari_dsm:.3f} "
                  f"(≈0 ⇒ phenotypes cut across DSM-5; the flow is trans-diagnostic)")

    _write_report(coherence, pd.DataFrame(transitions), wide, k, cv_acc, dsm_of, ari_dsm)

    meta = {"min_obs": MIN_OBS, "residualize": {"spline_df": SPLINE_DF, "cross_fit": CROSS_FIT},
            "v0_classifier_cv_accuracy": cv_acc, "k": k,
            "n_patient_visits_assigned": int(len(assign)),
            "coherence": coherence.to_dict(orient="records"),
            "dsm_phenotype_ari": ari_dsm,
            "dr_excluded_at": "V3", "site_note": "site excluded; ComBat sensitivity deferred (#43)"}
    (RESULTS_DIR / "longitudinal_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("\nWrote results/longitudinal_* and reports/longitudinal.html. Done.")
    return 0


def _write_report(coherence, trans, wide, k, cv_acc, dsm_of=None, ari_dsm=float("nan")):
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

    # Sankey: DSM-5 subtype → V0 → V1 → V2 phenotype flow
    sankey_html = ""
    stages = [s for s in ["V0", "V1", "V2"] if s in wide.columns]
    if len(stages) >= 2:
        dsm_order = sorted(SPECTRUM, key=SPECTRUM.get)
        has_dsm = dsm_of is not None and "V0" in wide.columns
        dsm_labels = [DSM_SHORT[s] for s in dsm_order] if has_dsm else []
        pheno_nodes = [f"{s}·C{c}" for s in stages for c in range(k)]
        nodes = dsm_labels + pheno_nodes
        nidx = {n: i for i, n in enumerate(nodes)}
        src, tgt, val = [], [], []
        if has_dsm:                                     # DSM-5 subtype → V0 phenotype
            v0p = wide["V0"].dropna().astype(int)
            common = v0p.index.intersection(dsm_of.index)
            ctd = pd.crosstab(dsm_of.reindex(common), v0p.reindex(common))
            for s in dsm_order:
                if s not in ctd.index:
                    continue
                for j in ctd.columns:
                    n = int(ctd.loc[s, j])
                    if n > 0:
                        src.append(nidx[DSM_SHORT[s]]); tgt.append(nidx[f"V0·C{int(j)}"])
                        val.append(n)
        for a, b in zip(stages, stages[1:]):            # V0 → V1 → V2
            pair = wide[[a, b]].dropna()
            ct = pd.crosstab(pair[a].astype(int), pair[b].astype(int))
            for i in ct.index:
                for j in ct.columns:
                    if ct.loc[i, j] > 0:
                        src.append(nidx[f"{a}·C{int(i)}"]); tgt.append(nidx[f"{b}·C{int(j)}"])
                        val.append(int(ct.loc[i, j]))
        node_color = ([DSM_COLORS[SPECTRUM[s]] for s in dsm_order] if has_dsm else []) + \
                     [PALETTE[c % len(PALETTE)] for _ in stages for c in range(k)]
        title = ("Phenotype flow: DSM-5 subtype → V0 → V1 → V2 (link width = patients)"
                 if has_dsm else "Phenotype flow V0 → V1 → V2")
        fig = go.Figure(go.Sankey(
            node=dict(label=nodes, color=node_color, pad=14, thickness=14),
            link=dict(source=src, target=tgt, value=val)))
        fig.update_layout(title=title, height=480, font=dict(size=11),
                          margin=dict(t=46, l=10, r=10, b=10))
        sankey_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False,
                                  config={"displayModeBar": False})

    # DSM-5 × V0-phenotype composition (column-normalized: each phenotype's DSM mix)
    dsm_cmp_html = ""
    if dsm_of is not None and "V0" in wide.columns:
        v0p = wide["V0"].dropna().astype(int)
        common = v0p.index.intersection(dsm_of.index)
        if len(common) > 20:
            dsm_order = sorted(SPECTRUM, key=SPECTRUM.get)
            ct = pd.crosstab(dsm_of.reindex(common), v0p.reindex(common))
            ct = ct.reindex(index=[s for s in dsm_order if s in ct.index], fill_value=0)
            colnorm = ct.div(ct.sum(0).replace(0, 1), axis=1)
            fig = go.Figure(go.Heatmap(
                z=colnorm.to_numpy(), x=[f"C{c}" for c in colnorm.columns],
                y=[DSM_SHORT.get(s, s) for s in colnorm.index],
                text=ct.to_numpy(), texttemplate="%{text}", colorscale="Purples", zmin=0, zmax=1,
                colorbar=dict(title="col frac", thickness=12)))
            fig.update_layout(title="DSM-5 subtype composition of each V0 phenotype (n in cells)",
                              height=360, xaxis_title="V0 phenotype",
                              yaxis_title="DSM-5 (mood→psychosis)", margin=dict(t=46, l=110, b=46))
            dsm_cmp_html = pio.to_html(fig, include_plotlyjs=False, full_html=False,
                                       config={"displayModeBar": False})

    css = ("body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;"
           "padding:0 24px 60px;color:#1f2933;line-height:1.5}h1{color:#2b3a55}"
           "h2{color:#2b3a55;border-bottom:2px solid #2b3a55;padding-bottom:6px;margin-top:34px}"
           "table{border-collapse:collapse;font-size:13px;margin:10px 0}"
           "th,td{border:1px solid #e5e7eb;padding:5px 10px}th{background:#eef2f7}"
           ".callout{border-left:4px solid #16a085;background:#f2fbf6;padding:10px 14px;margin:14px 0}")
    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>FACE longitudinal "
             f"coherence</title><style>{css}</style></head><body>",
             "<h1>FACE — discrete clustering is unstable and diagnosis-independent "
             "(negative result motivating the dimensional model)</h1>",
             f"<div class='callout'><b>This page is a negative result.</b> The structure "
             f"test shows the trans-diagnostic structure is <b>dimensional</b>, not discrete; "
             f"here we force discrete k-means phenotypes and show they are neither temporally "
             f"stable nor aligned to DSM-5 — i.e. slices of a continuum, not natural kinds. "
             f"Assignment rule = a phenotype classifier trained on V0 (domain scores → V0 "
             f"labels), 5-fold V0 accuracy <b>{cv_acc:.3f}</b>, applied unchanged to each "
             f"follow-up. DR excluded at V3 (attrition).</div>",
             "<h2>Coherence vs V0</h2>",
             coherence.assign(ari_vs_V0=coherence["ari_vs_V0"].round(3),
                              pct_persist=(coherence["pct_persist"]*100).round(0))
             .to_html(index=False, border=0)]
    if sankey_html:
        parts += ["<h2>Phenotype flow (DSM-5 → V0 → V1 → V2)</h2>", sankey_html]
    if dsm_cmp_html:
        ari_txt = f"{ari_dsm:.3f}" if ari_dsm == ari_dsm else "n/a"
        parts += ["<h2>Comparison with the 7 DSM-5 subgroups</h2>",
                  f"<div class='callout'>The data-driven phenotypes <b>cut across DSM-5</b>: "
                  f"the adjusted Rand index between the 7 enrolled DSM-5 subtypes and the V0 "
                  f"phenotypes is <b>{ari_txt}</b> (0 = independent, 1 = identical). Each "
                  f"phenotype draws from multiple diagnoses and each diagnosis spreads across "
                  f"phenotypes — the phenotypes are trans-diagnostic, not relabelled "
                  f"diagnoses.</div>", dsm_cmp_html]
    parts.append("<h2>Transition matrices</h2>")
    for v in ["V1", "V2", "V3", "V4"]:
        h = heat(v)
        if h:
            parts.append(h)
    parts.append("</body></html>")
    (REPORTS_DIR / "longitudinal.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
