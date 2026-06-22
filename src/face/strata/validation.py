"""M2.4 — validation battery (§7): Q1 existence · Q2 not-just-severity · Q3 transdiagnostic · Q4 stable /
not-an-artefact · head-to-head vs DSM-5 (the "better description" test, §1.7). Applied to both soft views
(archetypes = lead; tessellation). Diagnosis/covariates are validation-only, never inputs."""
from __future__ import annotations

import numpy as np


def eta_squared(labels, X):
    """Per-axis η² (between-group SS / total SS) — how much of each coordinate a partition explains."""
    labels = np.asarray(labels)
    X = np.asarray(X, dtype="float64")
    grand = X.mean(0)
    tot = ((X - grand) ** 2).sum(0)
    bet = np.zeros(X.shape[1])
    for g in np.unique(labels):
        m = labels == g
        bet += m.sum() * (X[m].mean(0) - grand) ** 2
    return bet / np.where(tot > 0, tot, 1.0)


def cramers_v(a, b):
    import pandas as pd
    from scipy.stats import chi2_contingency
    ct = pd.crosstab(pd.Series(np.asarray(a)), pd.Series(np.asarray(b))).to_numpy()
    if min(ct.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(ct)[0]
    n = ct.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(np.asarray(a), np.asarray(b)))


def coverage_artifact(nobs, labels, seed=0, n_perm=30):
    """Can the COVERAGE pattern (observed-indicator counts per axis) predict the partition beyond chance?

    Predictive skill ⇒ membership is driven by missingness, not values (an artefact). Plain accuracy is
    weak under class imbalance (P3-06), so we add imbalance-robust metrics — **balanced accuracy**,
    **macro-F1**, **log-loss** — and a **permutation test** (does balanced accuracy beat label-permuted
    nulls?). ``lift`` is retained for backward compatibility."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    rng = np.random.default_rng(seed)
    nobs, labels = np.asarray(nobs), np.asarray(labels)
    n_classes = len(np.unique(labels))

    def _clf():
        return RandomForestClassifier(120, random_state=seed, n_jobs=-1)

    def _bal(y):
        return float(cross_val_score(_clf(), nobs, y, cv=4, scoring="balanced_accuracy").mean())

    acc = float(cross_val_score(_clf(), nobs, labels, cv=4).mean())
    base = float(np.bincount(labels).max() / len(labels))
    bal = _bal(labels)
    f1m = float(cross_val_score(_clf(), nobs, labels, cv=4, scoring="f1_macro").mean())
    ll = float(-cross_val_score(_clf(), nobs, labels, cv=4, scoring="neg_log_loss").mean())
    null = np.array([_bal(rng.permutation(labels)) for _ in range(n_perm)])
    p_perm = float((null >= bal).mean())
    return {"classifier_acc": acc, "majority_baseline": base, "lift": acc - base,
            "balanced_acc": bal, "balanced_chance": 1.0 / n_classes, "macro_f1": f1m, "log_loss": ll,
            "perm_p_value": p_perm}


def tess_seed_stability(X, S, K, seeds=(1, 2, 3)):
    """Re-fit the XD tessellation across seeds; ARI of the MAP partition vs seed 0 (reproducibility)."""
    from face.strata.mixture import xd_em
    base = xd_em(X, S, K, seed=0)["resp"].argmax(1)
    aris = [ari(base, xd_em(X, S, K, seed=s)["resp"].argmax(1)) for s in seeds]
    return {"mean_ari": float(np.mean(aris)), "min_ari": float(np.min(aris))}


def assignment_usefulness(resp, *, tau: float = 0.5):
    """Are the soft regions OPERATIONALLY useful — i.e. are patients actually assignable, or is everyone
    in the mushy middle? (A scheme where 95% of patients are 50/50 is not useful, however "real" it is.)

    ``resp`` is the [N, K] soft membership (XD responsibilities or archetype simplex weights). Reports the
    confident-dominant fraction (max membership > tau), the normalized-entropy distribution, the
    boundary-patient fraction (near-uniform membership), and the effective number of regions actually used
    (perplexity exp(mean entropy)). Gate: PASS if confident-dominant >= 0.50 and median normalized entropy
    <= 0.6; FAIL if confident-dominant < 0.30 (mushy middle); else CONDITIONAL."""
    r = np.clip(np.asarray(resp, dtype="float64"), 1e-12, 1.0)
    N, K = r.shape
    raw_H = -(r * np.log(r)).sum(1)                        # nats
    H = raw_H / np.log(K) if K > 1 else np.zeros(N)        # normalized to [0, 1]
    mx = r.max(1)
    conf = float((mx > tau).mean())
    med_H = float(np.median(H))
    gate = "PASS" if (conf >= 0.5 and med_H <= 0.6) else ("FAIL" if conf < 0.3 else "CONDITIONAL")
    return {"K": int(K), "confident_dominant_frac": conf, "median_norm_entropy": med_H,
            "iqr_norm_entropy": [float(np.quantile(H, 0.25)), float(np.quantile(H, 0.75))],
            "boundary_frac": float((mx < (1.0 / K + 0.05)).mean()),
            "effective_n_regions": float(np.exp(raw_H.mean())), "tau": float(tau), "gate": gate}


def choose_K_operational(X, S, Ks=range(2, 9), seeds=(1, 2, 3), seed=0):
    """Choose the operational number of regions when the cloud is a CONTINUUM (no natural K). K is then a
    granularity choice, not a discovered kind-count: pick the SMALLEST K that keeps confident assignment
    (>=0.5) and stability (seed-ARI >=0.8); report the full sweep + the deliberate choice. Internal-only
    criteria (parsimony + assignment confidence + stability) — no external/predictive validity here."""
    from face.strata.mixture import xd_em
    rows = []
    for K in Ks:
        fit = xd_em(X, S, K, seed=seed)
        au = assignment_usefulness(fit["resp"])
        stab = tess_seed_stability(X, S, K, seeds=seeds)["mean_ari"]
        rows.append({"K": int(K), "bic": float(fit["bic"]),
                     "confident_dominant_frac": au["confident_dominant_frac"],
                     "median_norm_entropy": au["median_norm_entropy"], "seed_ari": float(stab)})
    ok = [r for r in rows if r["confident_dominant_frac"] >= 0.5 and r["seed_ari"] >= 0.8]
    chosen = min(ok, key=lambda r: r["K"]) if ok else max(rows, key=lambda r: r["confident_dominant_frac"])
    return {"sweep": rows, "chosen_K": int(chosen["K"]),
            "rationale": ("smallest K with confident-dominant>=0.5 and seed-ARI>=0.8" if ok
                          else "no K met both gates; fell back to max confident-dominant")}
