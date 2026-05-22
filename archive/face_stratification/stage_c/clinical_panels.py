"""Minimum clinical-feature panel discovery for Stage C clusters.

Given a final consensus clustering and the Stage A harmonized matrix, this
module finds the **minimum clinical-feature panel** for each cluster — the
smallest set of routinely measurable clinical features that most accurately
distinguishes cluster members from non-members.

Why not "biomarker"?
--------------------
In the strict biomedical / regulatory sense (NIH–FDA BEST framework), a
biomarker is a molecular, cellular, histologic, radiographic or physiologic
indicator of a biological process. The panels discovered here are dominated
by **demographics, comorbidity counts, substance-use flags and rating-scale
totals** — clinically *actionable* phenotypic descriptors, not biological
markers. We therefore deliberately use the more honest terminology
``ClinicalFeaturePanel`` / ``discover_clinical_feature_panel``, and reserve
the word "biomarker" for a future extension that would evaluate genuinely
biological inputs (lipids, glucose, QTc, inflammation markers, imaging,
genotype).

The legacy identifiers ``BiomarkerPanel``, ``discover_biomarker_panel`` etc.
are still exported as thin aliases from :mod:`stage_c.biomarkers` for
backward compatibility with notebooks and older scripts.

Design
------
For each cluster we frame the problem as a binary classification
(member vs rest) and:

1. Filter the feature set to a **clinically actionable whitelist** — brief
   and inexpensive to measure in routine care (vitals, basic labs,
   demographics, short rating-scale totals, history booleans).
2. **Remove embedding-input features by default** — the eight universally
   measured features that seeded the Stage A transdiagnostic similarity
   graph and therefore leak directly into the Stage B / B2.5 embedding are
   excluded from the candidate pool unless the caller explicitly opts in.
   This prevents circular "we recover the clusters from features that
   produced them" artefacts.
3. Rank features by **univariate AUC** (one-feature logistic regression)
   computed on the current fold only.
4. Greedily select the top-k features that maximally improve a
   multi-feature logistic regression AUC (forward selection, ≤6 features).
5. Report the panel's **overall AUC**, **cohort-stratified AUC**, and
   **suggested thresholds** (Youden's J per feature).

The result for each cluster is a :class:`ClinicalFeaturePanel` with the full
audit trail, ready to be rendered into a clinical card.

No GPU, no neural networks — logistic regression is the right tool because
clinicians want a transparent, inspectable panel with known coefficients.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Clinically-actionable feature whitelist ───────────────────────────────


_CLINICAL_FEATURE_WHITELIST_BY_TYPE: dict[str, list[str]] = {
    # Biological — cheap, non-invasive, or routinely measured
    "biological": [
        "bio_bmi", "bio_waist_cm", "bio_sbp_mmhg", "bio_dbp_mmhg",
        "bio_hr_bpm", "bio_qtc_ms", "bio_fasting_glucose",
        "bio_total_cholesterol", "bio_hdl_cholesterol", "bio_triglycerides",
    ],
    # Demographics — free, near-100% coverage
    "demographic": [
        "demo_age_years", "demo_sex_male", "demo_education_years_ordinal",
        "demo_marital_partnered", "demo_employed",
    ],
    # Simple clinical instruments — 5-15 items, 5-10 min to administer
    "brief_instruments": [
        "inst_madrs_total", "inst_cgis_total", "inst_ymrs_total",
        "inst_qids_total", "inst_asrm_total", "inst_bdi2_total",
        "inst_shaps_total", "inst_mars_total",
        "inst_psqi_total", "inst_ess_total",
        "inst_stai_ya_total", "inst_bis10_total",
        "inst_hama_total", "inst_lsas_total",
        "inst_eq5d_total", "inst_fast_total",
        "inst_ctq_total",
    ],
    # Behavioral / history — simple boolean / count questions
    "behavioral_history": [
        "sub_tobacco_current", "sub_tobacco_cpd", "sub_alcohol_current",
        "sub_cannabis_current", "sub_use_disorder",
        "cm_n_somatic", "cm_n_psychiatric",
        "sui_ever_ideation", "sui_ever_attempt", "sui_n_attempts",
        "fh_bipolar_any", "fh_suicide_any", "fh_substance_any",
        "fh_n_affected_relatives",
        "psyh_age_first_episode", "psyh_illness_duration_years",
        "psyh_n_hospitalizations_lifetime",
    ],
}


#: The eight universally-measured features that seed the Stage A
#: transdiagnostic similarity graph. Including them in the candidate pool
#: would create a circular "reconstruct-the-embedding-inputs" artefact
#: (observed as inflated AUC on C3 in the unsanitised run). They are
#: therefore excluded from the default whitelist.
EMBEDDING_INPUT_FEATURES: frozenset[str] = frozenset({
    "demo_age_years",
    "demo_sex_male",
    "sub_tobacco_current",
    "sub_alcohol_current",
    "sub_cannabis_current",
    "sub_use_disorder",
    "cm_n_somatic",
    "cm_n_psychiatric",
})


def default_clinical_feature_whitelist(
    *, exclude_embedding_inputs: bool = True
) -> list[str]:
    """Return the flat list of all whitelisted clinical features.

    Parameters
    ----------
    exclude_embedding_inputs:
        If ``True`` (default), removes the eight universally-measured
        features listed in :data:`EMBEDDING_INPUT_FEATURES` that seed the
        Stage A transdiagnostic similarity graph. This is the
        recommended, leakage-safe default for clinical-feature-panel
        discovery: it prevents the panel from trivially recovering cluster
        labels using features that already defined the embedding those
        clusters live in. Set to ``False`` only if you explicitly want the
        legacy leakage-prone behaviour for auditing purposes.
    """
    flat = [
        f
        for group in _CLINICAL_FEATURE_WHITELIST_BY_TYPE.values()
        for f in group
    ]
    if exclude_embedding_inputs:
        flat = [f for f in flat if f not in EMBEDDING_INPUT_FEATURES]
    return flat


# ─── Result types ────────────────────────────────────────────────────────────


@dataclass
class ClinicalFeaturePanel:
    """Discovered minimum clinical-feature panel for one cluster.

    The panel is a fitted logistic-regression classifier on a small set
    (≤6) of clinically-actionable features. It can be applied to new data
    via :meth:`predict_proba` / :meth:`evaluate_auc`, which makes it
    suitable for external validation (held-out splits, bootstrap CI,
    independent cohorts, etc.).

    Despite the ``BiomarkerPanel`` legacy alias, the features selected
    here are **phenotypic clinical descriptors**, not biological
    biomarkers in the regulatory sense. Downstream narratives should use
    the language of "parsimonious clinical discriminator" or "sparse
    phenotypic signature".
    """

    cluster_id: int
    cluster_size: int
    panel_features: list[str]
    panel_thresholds: dict[str, dict[str, float]]  # feature → {threshold, direction, youden_j}
    overall_auc: float
    cohort_stratified_auc: dict[str, float]
    single_feature_aucs: dict[str, float]
    coefficients: dict[str, float]
    intercept: float
    clinical_narrative: str
    whitelist_excludes_embedding_inputs: bool = True
    # Fitted sklearn model for held-out prediction (not saved to JSON)
    _sklearn_model: Any | None = None
    _train_mu: np.ndarray | None = None
    _train_sigma: np.ndarray | None = None
    _train_median_fill: dict[str, float] | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(cluster membership) for new patients.

        Uses the same preprocessing as training: median-imputation (from
        the training-split medians), then standardization (using the
        training mu/sigma). The panel columns must all be present in
        ``X``; extra columns are ignored.
        """
        if self._sklearn_model is None:
            raise RuntimeError(
                "ClinicalFeaturePanel has no fitted sklearn model — it "
                "was likely loaded from JSON and cannot predict on new "
                "data."
            )
        sub = X[self.panel_features].copy()
        if self._train_median_fill is not None:
            for c in self.panel_features:
                sub[c] = sub[c].fillna(self._train_median_fill.get(c, 0.0))
        arr = sub.to_numpy(dtype=np.float64)
        if self._train_mu is not None and self._train_sigma is not None:
            arr = (arr - self._train_mu) / np.where(
                self._train_sigma > 0, self._train_sigma, 1.0
            )
        return self._sklearn_model.predict_proba(arr)[:, 1]

    def evaluate_auc(self, X: pd.DataFrame, y_true: np.ndarray) -> float:
        """Compute held-out AUC on new data. Returns NaN if degenerate."""
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError as exc:
            raise ImportError("evaluate_auc requires scikit-learn.") from exc

        y_arr = np.asarray(y_true).astype(int)
        if len(np.unique(y_arr)) < 2:
            return float("nan")
        try:
            scores = self.predict_proba(X)
        except Exception as exc:  # noqa: BLE001
            logger.warning("predict_proba failed: %s", exc)
            return float("nan")
        return float(roc_auc_score(y_arr, scores))

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "cluster_size": self.cluster_size,
            "panel_features": self.panel_features,
            "panel_thresholds": self.panel_thresholds,
            "overall_auc": self.overall_auc,
            "cohort_stratified_auc": self.cohort_stratified_auc,
            "single_feature_aucs": {
                k: v
                for k, v in sorted(
                    self.single_feature_aucs.items(), key=lambda kv: -kv[1]
                )
            },
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "clinical_narrative": self.clinical_narrative,
            "whitelist_excludes_embedding_inputs": (
                self.whitelist_excludes_embedding_inputs
            ),
        }


# ─── Univariate AUC ──────────────────────────────────────────────────────────


def _univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC for a single continuous feature classifying ``y``.

    Handles NaN via pairwise deletion. Signs are not assumed — we return
    ``max(AUC, 1 - AUC)`` so the caller doesn't need to know the direction.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError("AUCs require scikit-learn.") from exc

    valid = np.isfinite(x)
    if valid.sum() < 30:
        return float("nan")
    x_v = x[valid]
    y_v = y[valid]
    if len(np.unique(y_v)) < 2:
        return float("nan")
    try:
        auc = float(roc_auc_score(y_v, x_v))
    except ValueError:
        return float("nan")
    return max(auc, 1.0 - auc)


def _youden_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, str, float]:
    """Find the best single-feature threshold by Youden's J statistic.

    Returns ``(threshold, direction, youden_j)``. ``direction`` is
    ``">="`` if the cluster is enriched *above* the threshold, ``"<="``
    if below.
    """
    try:
        from sklearn.metrics import roc_curve
    except ImportError as exc:
        raise ImportError("thresholds require scikit-learn.") from exc

    valid = np.isfinite(x)
    if valid.sum() < 30 or len(np.unique(y[valid])) < 2:
        return (float("nan"), ">=", float("nan"))

    best: tuple[float, str, float] = (float("nan"), ">=", -np.inf)
    for sign, direction in [(1.0, ">="), (-1.0, "<=")]:
        try:
            fpr, tpr, thresh = roc_curve(y[valid], sign * x[valid])
        except ValueError:
            continue
        j = tpr - fpr
        idx = int(np.argmax(j))
        if j[idx] > best[2]:
            t = sign * thresh[idx]
            if not np.isfinite(t):
                continue
            best = (float(t), direction, float(j[idx]))
    return best


# ─── Logistic regression + greedy forward selection ────────────────────────


@dataclass
class _LogRegFit:
    """Return type of :func:`_fit_logreg` carrying the model + preprocessing."""

    auc: float
    coef: dict[str, float]
    intercept: float
    model: Any
    mu: np.ndarray
    sigma: np.ndarray
    median_fill: dict[str, float]


def _fit_logreg(X: pd.DataFrame, y: np.ndarray) -> _LogRegFit:
    """Fit a logistic regression with NaN-safe median imputation.

    Median imputation and standardization statistics are computed on the
    *given* ``X`` only — when the caller passes the training fold, this
    keeps preprocessing leakage-free.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError("logistic regression requires scikit-learn.") from exc

    X_imputed = X.copy()
    median_fill: dict[str, float] = {}
    for c in X_imputed.columns:
        med = float(X_imputed[c].median())
        if not np.isfinite(med):
            med = 0.0
        median_fill[c] = med
        X_imputed[c] = X_imputed[c].fillna(med)

    arr = X_imputed.to_numpy(dtype=np.float64)
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    sd_safe = np.where(sigma > 0, sigma, 1.0)
    arr_z = (arr - mu) / sd_safe

    model = LogisticRegression(max_iter=200, class_weight="balanced")
    model.fit(arr_z, y)
    y_pred = model.predict_proba(arr_z)[:, 1]
    auc = float(roc_auc_score(y, y_pred))
    coef = {c: float(v) for c, v in zip(X.columns, model.coef_[0])}
    return _LogRegFit(
        auc=auc,
        coef=coef,
        intercept=float(model.intercept_[0]),
        model=model,
        mu=mu,
        sigma=sd_safe,
        median_fill=median_fill,
    )


def _greedy_forward_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    max_features: int,
    initial: list[str] | None = None,
) -> tuple[list[str], _LogRegFit | None]:
    """Greedy forward selection: at each step add the feature that most
    improves logistic regression AUC, up to ``max_features``.

    Returns ``(selected_features, final_logreg_fit)``.
    """
    remaining = set(X.columns)
    selected: list[str] = list(initial or [])
    for f in selected:
        remaining.discard(f)

    current_auc = -np.inf
    best_fit: _LogRegFit | None = None

    while len(selected) < max_features and remaining:
        best_candidate = None
        best_auc_local = current_auc
        best_fit_local: _LogRegFit | None = None
        for f in list(remaining):
            trial = selected + [f]
            try:
                fit = _fit_logreg(X[trial], y)
            except Exception:  # noqa: BLE001
                continue
            if fit.auc > best_auc_local:
                best_auc_local = fit.auc
                best_candidate = f
                best_fit_local = fit
        if best_candidate is None:
            break
        selected.append(best_candidate)
        remaining.discard(best_candidate)
        current_auc = best_auc_local
        best_fit = best_fit_local

    return selected, best_fit


# ─── Main entry point ────────────────────────────────────────────────────────


#: Minimum number of positives required before a panel is attempted. Set
#: below the legacy threshold of 20 so that small consensus clusters
#: (e.g. a 12-patient boundary group) are still reported rather than
#: silently dropped.
MIN_PANEL_POSITIVES: int = 10


def discover_clinical_feature_panel(
    X: pd.DataFrame,
    cluster_labels: pd.Series,
    cohort_labels: pd.Series,
    target_cluster: int,
    *,
    max_panel_size: int = 6,
    feature_whitelist: list[str] | None = None,
    min_univariate_auc: float = 0.55,
    exclude_embedding_inputs: bool = True,
) -> ClinicalFeaturePanel:
    """Discover a minimum clinical-feature panel for a target cluster.

    Parameters
    ----------
    X:
        Stage A harmonized matrix (may contain NaNs — they are handled via
        median imputation on the given slice of data).
    cluster_labels:
        Series of consensus cluster labels (same index as ``X``).
    cohort_labels:
        Series of DSM cohort labels (same index as ``X``). Used to compute
        per-cohort AUC so we can check cross-cohort validity.
    target_cluster:
        The cluster we're building a panel for (one-vs-rest).
    max_panel_size:
        Maximum number of features in the final panel. Default 6.
    feature_whitelist:
        Restrict selection to these features. If ``None``, uses
        :func:`default_clinical_feature_whitelist` honouring
        ``exclude_embedding_inputs``.
    min_univariate_auc:
        Drop features with univariate AUC below this threshold before
        greedy selection (speeds up the search and filters noise).
    exclude_embedding_inputs:
        When ``feature_whitelist`` is ``None``, whether to exclude the
        eight universally-measured features that seed the Stage A
        transdiagnostic graph. Default ``True`` (leakage-safe).

    Raises
    ------
    ValueError
        If the target cluster has fewer than :data:`MIN_PANEL_POSITIVES`
        members — in that case no panel is attempted and the caller should
        treat the cluster as too small to yield a stable discriminator.
    """
    if feature_whitelist is None:
        feature_whitelist = default_clinical_feature_whitelist(
            exclude_embedding_inputs=exclude_embedding_inputs
        )
    feature_whitelist = [f for f in feature_whitelist if f in X.columns]

    y = (cluster_labels.to_numpy() == target_cluster).astype(int)
    if y.sum() < MIN_PANEL_POSITIVES:
        raise ValueError(
            f"Target cluster {target_cluster} has {int(y.sum())} members "
            f"(< MIN_PANEL_POSITIVES={MIN_PANEL_POSITIVES}); "
            "clinical-feature-panel discovery is not meaningful."
        )

    X_pool = X[feature_whitelist]

    # Univariate AUC — computed on the given X only. When called per-fold
    # from validate_clinical_feature_panel_cv, this means the filter only
    # sees the training fold.
    univariate = {
        f: _univariate_auc(X_pool[f].to_numpy(dtype=np.float64), y)
        for f in feature_whitelist
    }
    univariate_sorted = {
        k: v
        for k, v in sorted(
            univariate.items(),
            key=lambda kv: -(kv[1] if not np.isnan(kv[1]) else 0.0),
        )
    }

    candidates = [
        f for f, a in univariate.items() if not np.isnan(a) and a >= min_univariate_auc
    ]
    if not candidates:
        candidates = [
            k
            for k in univariate_sorted
            if not np.isnan(univariate_sorted[k])
        ][:10]

    X_candidates = X_pool[candidates]
    selected, final_fit = _greedy_forward_selection(
        X_candidates, y, max_features=max_panel_size
    )
    if final_fit is None:
        raise RuntimeError(
            f"Forward selection produced no fit for cluster {target_cluster}."
        )

    overall_auc = final_fit.auc
    coef = final_fit.coef
    intercept = final_fit.intercept

    # Per-cohort AUC — within-cohort refit
    cohort_auc: dict[str, float] = {}
    for cohort in sorted(set(cohort_labels)):
        mask = cohort_labels.to_numpy() == cohort
        if mask.sum() < 30:
            continue
        sub_y = y[mask]
        if sub_y.sum() == 0 or sub_y.sum() == len(sub_y):
            cohort_auc[cohort] = float("nan")
            continue
        try:
            sub_fit = _fit_logreg(X_candidates[selected].loc[mask], sub_y)
            sub_auc = sub_fit.auc
        except Exception:  # noqa: BLE001
            sub_auc = float("nan")
        cohort_auc[cohort] = sub_auc

    # Youden thresholds per selected feature
    thresholds: dict[str, dict[str, float]] = {}
    for f in selected:
        t, direction, j = _youden_threshold(
            X_pool[f].to_numpy(dtype=np.float64), y
        )
        thresholds[f] = {
            "threshold": float(t) if np.isfinite(t) else float("nan"),
            "direction": direction,
            "youden_j": float(j) if np.isfinite(j) else float("nan"),
        }

    narrative_bits = [
        f"Cluster {target_cluster}: {int(y.sum())} patients",
        f"Overall AUC: {overall_auc:.3f}",
    ]
    if cohort_auc:
        ca_bits = ", ".join(
            f"{c.upper()}={v:.2f}" for c, v in cohort_auc.items() if not np.isnan(v)
        )
        narrative_bits.append(f"Per-cohort AUC ({ca_bits})")
    narrative_bits.append(f"Panel ({len(selected)}): " + ", ".join(selected))
    narrative_bits.append(
        "Thresholds: "
        + "; ".join(
            f"{f} {info['direction']} {info['threshold']:.2f} (J={info['youden_j']:.2f})"
            for f, info in thresholds.items()
            if np.isfinite(info["threshold"])
        )
    )
    narrative_bits.append(
        "Sanitised (no embedding inputs)"
        if exclude_embedding_inputs and feature_whitelist is not None
        else "Unsanitised (legacy)"
    )
    narrative = " | ".join(narrative_bits)

    return ClinicalFeaturePanel(
        cluster_id=target_cluster,
        cluster_size=int(y.sum()),
        panel_features=selected,
        panel_thresholds=thresholds,
        overall_auc=overall_auc,
        cohort_stratified_auc=cohort_auc,
        single_feature_aucs=univariate_sorted,
        coefficients=coef,
        intercept=intercept,
        clinical_narrative=narrative,
        whitelist_excludes_embedding_inputs=exclude_embedding_inputs,
        _sklearn_model=final_fit.model,
        _train_mu=final_fit.mu,
        _train_sigma=final_fit.sigma,
        _train_median_fill=final_fit.median_fill,
    )


def discover_all_clinical_feature_panels(
    X: pd.DataFrame,
    cluster_labels: pd.Series,
    cohort_labels: pd.Series,
    *,
    max_panel_size: int = 6,
    feature_whitelist: list[str] | None = None,
    min_univariate_auc: float = 0.55,
    exclude_embedding_inputs: bool = True,
) -> dict[int, ClinicalFeaturePanel]:
    """Discover clinical-feature panels for every cluster in one call.

    Clusters with fewer than :data:`MIN_PANEL_POSITIVES` members are
    logged at WARNING level and skipped — the returned dict therefore
    may be missing cluster ids. The list of skipped clusters is emitted
    so callers can explicitly flag them in downstream reports (fixes
    the earlier silent-drop behaviour for small clusters such as C0).
    """
    out: dict[int, ClinicalFeaturePanel] = {}
    skipped: list[tuple[int, str]] = []
    for cid in sorted(cluster_labels.unique()):
        if cid < 0:
            continue
        try:
            panel = discover_clinical_feature_panel(
                X,
                cluster_labels,
                cohort_labels,
                int(cid),
                max_panel_size=max_panel_size,
                feature_whitelist=feature_whitelist,
                min_univariate_auc=min_univariate_auc,
                exclude_embedding_inputs=exclude_embedding_inputs,
            )
        except ValueError as exc:
            logger.warning(
                "Cluster %d: too small for clinical-feature panel (%s)", cid, exc
            )
            skipped.append((int(cid), str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Clinical-feature panel for cluster %d failed: %s", cid, exc
            )
            skipped.append((int(cid), f"error: {exc}"))
            continue
        out[int(cid)] = panel

    if skipped:
        logger.warning(
            "Clinical-feature panel discovery skipped %d cluster(s): %s",
            len(skipped),
            ", ".join(f"C{cid} ({reason})" for cid, reason in skipped),
        )
    return out


# ─── Held-out validation ─────────────────────────────────────────────────────


@dataclass
class ClinicalFeaturePanelValidationResult:
    """Stratified shuffle-split CV validation result for one target cluster."""

    cluster_id: int
    n_splits: int
    train_auc_mean: float
    train_auc_std: float
    test_auc_mean: float
    test_auc_std: float
    test_auc_by_split: list[float]
    test_auc_by_cohort: dict[str, dict[str, float]]  # cohort → {mean, std, n_splits}
    feature_selection_stability: dict[str, float]  # feature → fraction of splits it was selected
    all_splits: list[dict[str, Any]]
    whitelist_excludes_embedding_inputs: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "n_splits": self.n_splits,
            "train_auc_mean": self.train_auc_mean,
            "train_auc_std": self.train_auc_std,
            "test_auc_mean": self.test_auc_mean,
            "test_auc_std": self.test_auc_std,
            "test_auc_by_split": self.test_auc_by_split,
            "test_auc_by_cohort": self.test_auc_by_cohort,
            "feature_selection_stability": self.feature_selection_stability,
            "generalization_gap": self.train_auc_mean - self.test_auc_mean,
            "whitelist_excludes_embedding_inputs": (
                self.whitelist_excludes_embedding_inputs
            ),
        }


def validate_clinical_feature_panel_cv(
    X: pd.DataFrame,
    cluster_labels: pd.Series,
    cohort_labels: pd.Series,
    target_cluster: int,
    *,
    n_splits: int = 5,
    test_fraction: float = 0.2,
    max_panel_size: int = 6,
    feature_whitelist: list[str] | None = None,
    min_univariate_auc: float = 0.55,
    random_state: int = 0,
    exclude_embedding_inputs: bool = True,
) -> ClinicalFeaturePanelValidationResult:
    """Stratified shuffle-split CV for one target cluster.

    Each split:

    1. Stratifies by the joint ``(target_cluster, cohort)`` stratum so
       both folds have representative cohort + cluster proportions.
    2. Runs the full :func:`discover_clinical_feature_panel` pipeline on
       the **train fold only** (preprocessing, univariate filter, and
       greedy forward selection are all fit on the training data).
    3. Applies the fitted panel to the held-out test fold and computes
       test AUC overall and per cohort using the training-fold median
       imputation and standardisation statistics.
    4. Records which features were selected so we can report
       feature-selection stability across splits.

    Returns a :class:`ClinicalFeaturePanelValidationResult` with scalar
    metrics and the per-split audit trail.
    """
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
    except ImportError as exc:
        raise ImportError(
            "validate_clinical_feature_panel_cv requires scikit-learn."
        ) from exc

    y = (cluster_labels.to_numpy() == target_cluster).astype(int)
    n_positives = int(y.sum())
    n_negatives = int(len(y) - n_positives)
    if n_positives < MIN_PANEL_POSITIVES or n_negatives < MIN_PANEL_POSITIVES:
        raise ValueError(
            f"Cluster {target_cluster} has n+={n_positives}, n-={n_negatives}; "
            f"stratified CV requires ≥ MIN_PANEL_POSITIVES="
            f"{MIN_PANEL_POSITIVES} on each side."
        )

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_fraction,
        random_state=random_state,
    )

    # Joint (y, cohort) stratification so cohorts are balanced in both folds.
    strat_key = np.array(
        [
            f"{int(yi)}_{cohort}"
            for yi, cohort in zip(y, cohort_labels.to_numpy(), strict=True)
        ],
        dtype=object,
    )
    # scikit-learn stratification requires each stratum to have ≥2 samples.
    from collections import Counter

    ctr = Counter(strat_key)
    strat_key = np.array(
        [
            k if ctr[k] >= 2 else f"{k.split('_')[0]}_COLLAPSED"
            for k in strat_key
        ]
    )
    # After collapsing rare (y, cohort) cells, it can still happen that one
    # of the collapsed strata has only 1 member (e.g. a small cluster
    # concentrated in one cohort). StratifiedShuffleSplit refuses in that
    # case. Fall back to plain y-stratification so we still report a panel
    # — losing cross-cohort balance in the split is a far lesser evil
    # than silently dropping the whole cluster.
    ctr2 = Counter(strat_key)
    if any(v < 2 for v in ctr2.values()):
        logger.warning(
            "Cluster %d: joint (y, cohort) stratification degenerate; "
            "falling back to y-only stratification.",
            target_cluster,
        )
        strat_key = y.astype(str)

    split_rows: list[dict[str, Any]] = []
    selection_counts: dict[str, int] = {}

    for split_idx, (train_idx, test_idx) in enumerate(
        splitter.split(X, strat_key)
    ):
        X_train = X.iloc[train_idx]
        y_train = pd.Series(y[train_idx], index=X_train.index)
        cohort_train = cohort_labels.iloc[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y[test_idx]
        cohort_test = cohort_labels.iloc[test_idx]

        cluster_train = pd.Series(
            np.where(y_train.to_numpy() == 1, target_cluster, -1),
            index=X_train.index,
            name="cluster",
            dtype="int64",
        )

        try:
            panel = discover_clinical_feature_panel(
                X_train,
                cluster_train,
                cohort_train,
                target_cluster,
                max_panel_size=max_panel_size,
                feature_whitelist=feature_whitelist,
                min_univariate_auc=min_univariate_auc,
                exclude_embedding_inputs=exclude_embedding_inputs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Split %d failed for cluster %d: %s", split_idx, target_cluster, exc
            )
            continue

        # Test AUC overall
        test_auc = panel.evaluate_auc(X_test, y_test)

        # Per-cohort test AUC
        per_cohort_test: dict[str, float] = {}
        for cohort in sorted(set(cohort_test.to_numpy())):
            mask = cohort_test.to_numpy() == cohort
            if mask.sum() < 15:
                continue
            sub_y = y_test[mask]
            if sub_y.sum() == 0 or sub_y.sum() == len(sub_y):
                continue
            per_cohort_test[str(cohort)] = panel.evaluate_auc(
                X_test.iloc[np.where(mask)[0]], sub_y
            )

        for feat in panel.panel_features:
            selection_counts[feat] = selection_counts.get(feat, 0) + 1

        split_rows.append(
            {
                "split": split_idx,
                "train_auc": panel.overall_auc,
                "test_auc": test_auc,
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "n_positives_train": int(y_train.sum()),
                "n_positives_test": int(y_test.sum()),
                "panel_features": list(panel.panel_features),
                "per_cohort_test_auc": per_cohort_test,
            }
        )

    if not split_rows:
        raise RuntimeError(f"All splits failed for cluster {target_cluster}.")

    train_aucs = np.array([r["train_auc"] for r in split_rows])
    test_aucs = np.array(
        [r["test_auc"] for r in split_rows if np.isfinite(r["test_auc"])]
    )

    per_cohort_agg: dict[str, dict[str, float]] = {}
    all_cohorts: set[str] = set()
    for r in split_rows:
        all_cohorts.update(r["per_cohort_test_auc"].keys())
    for cohort in sorted(all_cohorts):
        vals = [r["per_cohort_test_auc"].get(cohort, np.nan) for r in split_rows]
        vals_finite = np.array([v for v in vals if np.isfinite(v)])
        if vals_finite.size == 0:
            per_cohort_agg[cohort] = {
                "mean": float("nan"),
                "std": float("nan"),
                "n_splits": 0,
            }
        else:
            per_cohort_agg[cohort] = {
                "mean": float(vals_finite.mean()),
                "std": float(vals_finite.std()),
                "n_splits": int(vals_finite.size),
            }

    total_splits = len(split_rows)
    selection_stability = {
        feat: count / total_splits for feat, count in selection_counts.items()
    }

    return ClinicalFeaturePanelValidationResult(
        cluster_id=target_cluster,
        n_splits=total_splits,
        train_auc_mean=float(train_aucs.mean()),
        train_auc_std=float(train_aucs.std()),
        test_auc_mean=float(test_aucs.mean()) if test_aucs.size else float("nan"),
        test_auc_std=float(test_aucs.std()) if test_aucs.size else float("nan"),
        test_auc_by_split=test_aucs.tolist(),
        test_auc_by_cohort=per_cohort_agg,
        feature_selection_stability=selection_stability,
        all_splits=split_rows,
        whitelist_excludes_embedding_inputs=exclude_embedding_inputs,
    )


def validate_all_clinical_feature_panels_cv(
    X: pd.DataFrame,
    cluster_labels: pd.Series,
    cohort_labels: pd.Series,
    *,
    n_splits: int = 5,
    test_fraction: float = 0.2,
    **kwargs: Any,
) -> dict[int, ClinicalFeaturePanelValidationResult]:
    """Run stratified shuffle-split CV validation for every cluster.

    Clusters that are too small (<:data:`MIN_PANEL_POSITIVES` positives
    or negatives) are logged at WARNING level and skipped. The skipped
    list is exposed through the logger so callers can report the
    exclusion explicitly rather than silently.
    """
    out: dict[int, ClinicalFeaturePanelValidationResult] = {}
    skipped: list[tuple[int, str]] = []
    for cid in sorted(cluster_labels.unique()):
        if cid < 0:
            continue
        try:
            out[int(cid)] = validate_clinical_feature_panel_cv(
                X,
                cluster_labels,
                cohort_labels,
                int(cid),
                n_splits=n_splits,
                test_fraction=test_fraction,
                **kwargs,
            )
        except ValueError as exc:
            logger.warning(
                "Validation skipped for cluster %d: %s", cid, exc
            )
            skipped.append((int(cid), str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("Validation failed for cluster %d: %s", cid, exc)
            skipped.append((int(cid), f"error: {exc}"))
            continue
    if skipped:
        logger.warning(
            "Clinical-feature panel validation skipped %d cluster(s): %s",
            len(skipped),
            ", ".join(f"C{cid} ({reason})" for cid, reason in skipped),
        )
    return out
