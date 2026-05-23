"""Direction-A clustering: trans-diagnostic phenotypes on balanced domain scores.

Pipeline (LABBOOK E7-E8):
  build_unified_dataframe (READY+PARTIAL)
    -> to_harmonized_dataset(sections = symptom + biology, raw)        adapter
    -> build_domain_scores                                              domains.py
    -> coverage floor (>= COVERAGE_FLOOR)                               drop near-empty
    -> residualize_features(spline_df, cross_fit) on age + sex          double-ML
    -> normalize_for_embedding (robust z)
    -> HarmonizedDataset (stub schema) -> MultipartiteSpectral          engine, no imputation
    -> k selection: bootstrap stability + consensus PAC + gap + silhouette
       + independence panel (cluster vs sex / age / cohort)
    -> headline clustering: engine enrichment naming on domain scores

Confound is removed in the feature space (Tier 1), then *verified* (Tier 3) —
we do NOT use ARI-vs-sister as a selection criterion (direction A; low cohort
agreement is expected/desired).

Artifacts (results/):
    cluster_domains_embedding.parquet
    cluster_domains_scores.parquet     per-patient domain scores (residualized)
    cluster_domains_assignments.csv
    cluster_domains_kselect.csv        per-k stability / PAC / gap / independence
    cluster_domains_naming.csv         per-cluster top domains + cohort mix
    cluster_domains_meta.json

Run:  python3 scripts/cluster_domains.py            # auto-pick k
      python3 scripts/cluster_domains.py --k 5
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    silhouette_score,
)

from face_common import (  # noqa: E402
    ADMINISTRATIVE_FEATURES,
    DOMAIN_SECTIONS,
    build_domain_scores,
    build_unified_dataframe,
    load_variables,
    normalize_for_embedding,
    residualize_features,
    to_harmonized_dataset,
)
from face_common.engine import (  # noqa: E402
    FeatureSchema,
    HarmonizedDataset,
    MultipartiteSpectralEmbedding,
    bootstrap_stability,
    compute_cluster_feature_enrichment,
    run_kmeans,
)

DATA_DIR = REPO_ROOT / "data"
DICT_PATH = REPO_ROOT / "face-common-vars.xlsx"
RESULTS_DIR = REPO_ROOT / "results"

COVERAGE_FLOOR = 0.30      # drop domains observed in < 30% of patients
SPLINE_DF = 4              # natural-spline knots for age
CROSS_FIT = 5             # K-fold cross-fitting for residualization
RANDOM_STATE = 0
SILHOUETTE_SAMPLE = 5000
EMBED_CONFIG = dict(
    min_coverage=0.30, min_features_per_partition=3, n_components_per_partition=8,
    k_neighbours=10, include_4way=True, include_mask_columns=True,
    l2_normalize=True, feature_mode="cumulative", partition_weighting="sqrt_info",
)


def _git_rev() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--k", type=int, default=None,
                   help="headline k (default: auto-pick by stability)")
    p.add_argument("--k-sweep", type=int, nargs=2, default=(2, 9),
                   metavar=("LO", "HI"))
    p.add_argument("--readiness", nargs="+", default=["READY", "PARTIAL"])
    p.add_argument("--n-boot", type=int, default=20)
    return p.parse_args()


# ─── independence / stability metrics ────────────────────────────────────────

def cramers_v(labels: np.ndarray, cat: np.ndarray) -> float:
    ct = pd.crosstab(pd.Series(labels), pd.Series(cat)).to_numpy(float)
    if ct.size == 0 or ct.sum() == 0:
        return float("nan")
    chi2 = ((ct - ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / ct.sum()) ** 2
            / (ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / ct.sum())).sum()
    n = ct.sum()
    r, k = ct.shape
    denom = n * (min(r, k) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")


def eta_squared(labels: np.ndarray, y: np.ndarray) -> float:
    """Fraction of variance in continuous y explained by cluster (one-way ANOVA)."""
    m = np.isfinite(y)
    labels, y = labels[m], y[m]
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    ss_between = sum(((y[labels == c].mean() - grand) ** 2) * (labels == c).sum()
                     for c in np.unique(labels))
    return float(ss_between / ss_tot) if ss_tot > 0 else float("nan")


def distance_correlation(x: np.ndarray, y2: np.ndarray, sample_n: int = 1500,
                         seed: int = 0) -> float:
    """Distance correlation (Székely) between 1D x and (possibly multivariate) y2.

    Captures nonlinear/joint dependence; 0 ⇔ independence. Computed on a random
    patient subsample to bound the N×N memory cost.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    if n > sample_n:
        idx = rng.choice(n, size=sample_n, replace=False)
        x, y2 = x[idx], y2[idx]
    x = x.reshape(-1, 1).astype(float)
    y2 = y2.astype(float)
    if y2.ndim == 1:
        y2 = y2.reshape(-1, 1)
    a = np.abs(x - x.T)
    b = np.sqrt(((y2[:, None, :] - y2[None, :, :]) ** 2).sum(-1))
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvarx, dvary = (A * A).mean(), (B * B).mean()
    denom = np.sqrt(dvarx * dvary)
    return float(np.sqrt(max(dcov2, 0) / denom)) if denom > 0 else 0.0


def consensus_pac(arr: np.ndarray, k: int, *, n_boot: int = 20, frac: float = 0.8,
                  sample_n: int = 3000, u=(0.1, 0.9), seed: int = 0) -> float:
    """Proportion of Ambiguous Clustering (Monti consensus). Lower = more stable."""
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    samp = np.sort(rng.choice(n, size=min(sample_n, n), replace=False))
    pos = {v: i for i, v in enumerate(samp)}
    m = len(samp)
    M = np.zeros((m, m)); C = np.zeros((m, m))
    for b in range(n_boot):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        lab = KMeans(n_clusters=k, random_state=b, n_init=5).fit_predict(arr[idx])
        keep = [(pos[v], l) for v, l in zip(idx, lab) if v in pos]
        if len(keep) < 2:
            continue
        ii = np.array([p for p, _ in keep]); ll = np.array([l for _, l in keep])
        same = (ll[:, None] == ll[None, :]).astype(float)
        M[np.ix_(ii, ii)] += same
        C[np.ix_(ii, ii)] += 1
    iu = np.triu_indices(m, k=1)
    c = C[iu]; vals = M[iu][c > 0] / c[c > 0]
    return float(((vals > u[0]) & (vals < u[1])).mean()) if vals.size else float("nan")


def gap_statistic(arr: np.ndarray, k: int, *, n_ref: int = 4, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)

    def wk(X, kk):
        lab = KMeans(n_clusters=kk, random_state=0, n_init=5).fit_predict(X)
        return sum(((X[lab == c] - X[lab == c].mean(0)) ** 2).sum()
                   for c in np.unique(lab))

    logw = np.log(wk(arr, k) + 1e-12)
    lo, hi = arr.min(0), arr.max(0)
    refs = [np.log(wk(rng.uniform(lo, hi, size=arr.shape), k) + 1e-12)
            for _ in range(n_ref)]
    return float(np.mean(refs) - logw)


# ─── domain dataset wrapper ──────────────────────────────────────────────────

def wrap_domain_dataset(scores: pd.DataFrame, metadata: pd.DataFrame,
                        kinds: dict[str, str]) -> HarmonizedDataset:
    """Wrap a domain-score matrix into a HarmonizedDataset with a stub schema."""
    feats = [{
        "id": d, "label_fr": d, "block": kinds.get(d, "symptom"),
        "type": "continuous", "temporal_scope": "current",
        "cohorts": ("bp", "sz", "dr"),
    } for d in scores.columns]
    blocks = [{"id": b, "label_fr": b, "description": f"{b} domains"}
              for b in sorted(set(kinds.get(d, "symptom") for d in scores.columns))]
    schema = FeatureSchema.model_validate(
        {"version": "face_common-domains-0.1", "blocks": blocks, "features": feats})
    fm = pd.DataFrame(
        [{"feature_id": f["id"], "label_fr": f["id"], "block": f["block"],
          "type": "continuous", "temporal_scope": "current", "unit": None,
          "direction": "none", "cohorts": "bp,sz,dr"} for f in feats]
    ).set_index("feature_id")
    return HarmonizedDataset(X=scores, metadata=metadata, feature_metadata=fm,
                             schema=schema)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── 1. raw domain-section matrix → domain scores ────────────────────────
    print(f"Loading frame (readiness={args.readiness})...")
    variables = load_variables(DICT_PATH)
    exclude = set(ADMINISTRATIVE_FEATURES) | {
        v.canonical_name for v in variables if v.canonical_name.endswith("_mhoccur")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(DATA_DIR, DICT_PATH, readiness=args.readiness,
                                     format="long")
        raw = to_harmonized_dataset(df, variables, visit="V0", exclude=exclude,
                                    sections=DOMAIN_SECTIONS)          # symptom+biology, raw
        full = to_harmonized_dataset(df, variables, visit="V0",
                                     exclude=ADMINISTRATIVE_FEATURES)  # for age/sex covariates

    scores, meta = build_domain_scores(raw.X, variables)
    kinds = meta["kind"].to_dict()
    print(f"  domain scores: {scores.shape[1]} ({(meta.kind=='symptom').sum()} symptom, "
          f"{(meta.kind=='biology').sum()} biology)")

    # ── 2. coverage floor ───────────────────────────────────────────────────
    cov = scores.notna().mean()
    keep = cov[cov >= COVERAGE_FLOOR].index
    dropped = sorted(set(scores.columns) - set(keep))
    scores = scores[keep]
    print(f"  coverage floor {COVERAGE_FLOOR:.0%}: kept {len(keep)}, "
          f"dropped {len(dropped)} ({dropped})")

    # ── 3. residualize on age + sex (nonlinear spline + cross-fit) ──────────
    covars = full.X.reindex(scores.index)[["age", "sex"]]
    scores_resid = residualize_features(scores, covars, spline_df=SPLINE_DF,
                                        cross_fit=CROSS_FIT, random_state=RANDOM_STATE)
    scores_resid.to_parquet(RESULTS_DIR / "cluster_domains_scores.parquet")
    Xn = normalize_for_embedding(scores_resid)

    # ── 4. embedding (engine masked-cosine spectral, no imputation) ─────────
    dataset = wrap_domain_dataset(Xn, raw.metadata.loc[scores.index], kinds)
    print("Fitting MultipartiteSpectralEmbedding on domain scores...")
    model = MultipartiteSpectralEmbedding(**EMBED_CONFIG)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = model.fit(dataset).transform().values
    emb.to_parquet(RESULTS_DIR / "cluster_domains_embedding.parquet")
    print(f"  embedding: {emb.shape[0]:,} × {emb.shape[1]} dims, "
          f"{len(model._partitions)} partitions")
    arr = emb.to_numpy(np.float64)

    cohorts = pd.Series(emb.index.get_level_values("cohort"), index=emb.index)
    age = covars.loc[emb.index, "age"].to_numpy(float)
    sex = covars.loc[emb.index, "sex"].to_numpy(float)
    age_t = pd.qcut(pd.Series(age), 3, labels=False, duplicates="drop").to_numpy()

    # ── 5. k selection ──────────────────────────────────────────────────────
    lo, hi = args.k_sweep
    print(f"\nk selection {lo}..{hi} (stability ↑ / PAC ↓ / gap ↑; independence ↓):")
    rows = []
    for k in range(lo, hi + 1):
        labels = run_kmeans(emb, n_clusters=k, random_state=RANDOM_STATE).labels.to_numpy()
        sil = float(silhouette_score(arr, labels,
                    sample_size=min(SILHOUETTE_SAMPLE, len(arr)), random_state=RANDOM_STATE))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stab = bootstrap_stability(emb, n_clusters=k, n_bootstraps=args.n_boot,
                                       random_state=RANDOM_STATE)["mean_ari"]
            pac = consensus_pac(arr, k, n_boot=args.n_boot)
            gap = gap_statistic(arr, k)
        rows.append({
            "k": k, "stability_ari": stab, "pac": pac, "gap": gap, "silhouette": sil,
            "sex_cramersV": cramers_v(labels, sex),
            "age_eta2": eta_squared(labels, age),
            "cohort_ari": float(adjusted_rand_score(cohorts, labels)),
        })
    ksel = pd.DataFrame(rows)
    ksel.to_csv(RESULTS_DIR / "cluster_domains_kselect.csv", index=False)
    print(ksel.round(4).to_string(index=False))

    # auto-pick: most stable (max bootstrap ARI), tie-break smaller k
    k = args.k if args.k else int(ksel.loc[ksel["stability_ari"].idxmax(), "k"])
    print(f"\nHeadline k = {k}" + (" (chosen)" if args.k else " (auto: max stability)"))

    # ── 6. headline clustering + naming ─────────────────────────────────────
    labels = run_kmeans(emb, n_clusters=k, random_state=RANDOM_STATE).labels
    contingency = pd.crosstab(labels, cohorts)
    print("\ncluster × cohort:")
    print(contingency.to_string())

    sexv = cramers_v(labels.to_numpy(), sex)
    aget = float(adjusted_rand_score(age_t, labels)) if np.isfinite(age_t).all() else float("nan")
    dcor_age = distance_correlation(age, pd.get_dummies(labels).to_numpy())
    cohort_ari = float(adjusted_rand_score(cohorts, labels))
    print(f"\nINDEPENDENCE (want ≈0): sex Cramér's V={sexv:.3f}  age dCor={dcor_age:.3f}  "
          f"age-tertile ARI={aget:.3f}  | cohort ARI={cohort_ari:.3f}")

    enr = compute_cluster_feature_enrichment(scores_resid.loc[emb.index], labels,
                                             q_threshold=0.05)
    top = enr.top_per_cluster(8)
    naming_rows = []
    for c in sorted(labels.unique()):
        comp = cohorts[labels == c].value_counts()
        mix = ", ".join(f"{kk.upper()} {int(v/comp.sum()*100)}%" for kk, v in comp.items())
        feats = top[top["cluster"] == c]
        desc = "; ".join(f"{r.feature_id}({'↑' if r.effect_rank_biserial>=0 else '↓'})"
                         for r in feats.itertuples())
        naming_rows.append({"cluster": int(c), "n": int((labels == c).sum()),
                            "cohort_mix": mix, "top_domains": desc})
        print(f"  [{c}] n={int((labels==c).sum()):5d}  {mix}\n        {desc}")
    pd.DataFrame(naming_rows).to_csv(RESULTS_DIR / "cluster_domains_naming.csv", index=False)

    # ── 7. persist ──────────────────────────────────────────────────────────
    assignments = pd.DataFrame({
        "cohort": [c.upper() for c in emb.index.get_level_values("cohort")],
        "usubjid_patients": emb.index.get_level_values("patient_id"),
        "cluster": labels.to_numpy(),
    })
    assignments.insert(0, "patient_uid",
                       assignments["cohort"] + "::" + assignments["usubjid_patients"].astype(str))
    assignments.to_csv(RESULTS_DIR / "cluster_domains_assignments.csv", index=False)

    meta_json = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_rev": _git_rev(),
        "readiness": args.readiness, "coverage_floor": COVERAGE_FLOOR,
        "residualize": {"covariates": ["age", "sex"], "spline_df": SPLINE_DF,
                        "cross_fit": CROSS_FIT},
        "n_domains": int(scores.shape[1]), "dropped_domains": dropped,
        "embedding_dim": int(emb.shape[1]), "headline_k": int(k),
        "cluster_sizes": labels.value_counts().sort_index().to_dict(),
        "independence": {"sex_cramersV": sexv, "age_dcor": dcor_age,
                         "age_tertile_ari": aget, "cohort_ari": cohort_ari},
        "site_note": "site (siteid_city) excluded; ComBat sensitivity analysis deferred (task #43)",
    }
    (RESULTS_DIR / "cluster_domains_meta.json").write_text(
        json.dumps(meta_json, indent=2, ensure_ascii=False, default=str))
    print(f"\nWrote results/cluster_domains_* (kselect, scores, embedding, "
          f"assignments, naming, meta). Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
