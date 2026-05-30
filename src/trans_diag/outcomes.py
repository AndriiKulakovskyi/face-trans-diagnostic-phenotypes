"""Shared head-to-head outcome-modelling helpers.

Extracted from the Phase-5 script so the numbered pipeline scripts can reuse the
*exact* same CV metric / incremental-value test without importing one another
(digit-prefixed module names are not importable in Python). Used by the
head-to-head, CI, de-circularization, ComBat and cognition steps.

The transform signature is ``tf(y0, yk)`` (post-audit 2026-05); pass-through
callers via ``apply_outcome_tf`` so legacy ``tf(yk)`` signatures still work.

NOTE on the hospitalization outcome: the column is named ``_lt`` (lifetime) but the
empirical V0 vs V1 distribution makes clear it is **a lifetime tally only at V0**
— at follow-up visits the question becomes an interval count ("since last visit"):
V0 mean = 2.73 (P(>0)=0.81), V1 mean = 0.18 (P(>0)=0.14). So the outcome
``(V1_lt > 0)`` is in practice "any incident hospitalization since V0", with the
V0 lifetime count entering the model as prior-history baseline (NOT a tautological
predictor of V1, despite the shared column name). The transform retains the legacy
``yk -> (yk>0)`` form.
"""
from __future__ import annotations

import numpy as np

RANDOM = 0

# (name, kind, source_col, transform) — the 1-year outcomes.
# `transform`, when set, is called as ``tf(y0, yk)`` and returns the modelled outcome.
# `None` means use yk as-is (continuous outcome).
OUTCOMES = [
    ("EGF functioning", "continuous", "egf", None),
    ("any hospitalization", "binary", "nboccur_hospitalisation_lt",
     lambda y0, yk: (yk > 0).astype(float)),   # at V1, this column is incident-since-V0
    ("EQ-5D quality of life", "continuous", "eq5d", None),
]


def cv_metric(X, y, kind):
    """Shuffled 5-fold CV: Ridge R² (continuous) / logistic AUC (binary).

    NOTE: folds MUST be shuffled — the patient matrix is ordered by cohort
    (BP…SZ…DR), so un-shuffled KFold makes cohort-imbalanced folds and distorts
    the R² (e.g. EGF baseline 0.19 unshuffled vs 0.33 shuffled).
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    if kind == "continuous":
        kf = KFold(5, shuffle=True, random_state=RANDOM)
        return float(np.mean(cross_val_score(Ridge(alpha=1.0), Xs, y, cv=kf, scoring="r2")))
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM)
    return float(np.mean(cross_val_score(LogisticRegression(max_iter=2000), Xs, y,
                                         cv=skf, scoring="roc_auc")))


def added_axes_test(df_xy, base_cols, dsm_cols, axis_cols, y, kind):
    """In-sample p-value for adding the axes to the DSM model (F-test / LRT).
    Returns NaN if the model fails to converge (e.g. rare-subtype separation)."""
    import statsmodels.api as sm
    try:
        X0 = sm.add_constant(df_xy[base_cols + dsm_cols].astype(float))
        X2 = sm.add_constant(df_xy[base_cols + dsm_cols + axis_cols].astype(float))
        if kind == "continuous":
            from scipy.stats import f as fdist
            m0, m2 = sm.OLS(y, X0).fit(), sm.OLS(y, X2).fit()
            dfn, dfd = len(axis_cols), int(m2.df_resid)
            F = ((m0.ssr - m2.ssr) / dfn) / (m2.ssr / dfd)
            return float(1 - fdist.cdf(F, dfn, dfd))
        from scipy.stats import chi2
        m0 = sm.Logit(y, X0).fit(disp=0)
        m2 = sm.Logit(y, X2).fit(disp=0)
        return float(1 - chi2.cdf(2 * (m2.llf - m0.llf), len(axis_cols)))
    except Exception:
        return float("nan")


def axis_betas(d, cols_all, axis_cols, y, kind):
    """Standardized per-axis effects from a regularized model (robust to collinearity)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(d[cols_all].to_numpy(float))
    if kind == "continuous":
        coef = Ridge(alpha=1.0).fit(Xs, y).coef_
    else:
        coef = LogisticRegression(max_iter=2000).fit(Xs, y).coef_[0]
    beta = dict(zip(cols_all, coef, strict=False))
    return {a: float(beta[a]) for a in axis_cols}


def apply_outcome_tf(y0, yk, tf):
    """Apply an outcome transform with the post-audit (y0, yk) signature.

    Supports both ``tf(y0, yk)`` (the current spec — needed for the incident
    hospitalization outcome) and the legacy ``tf(yk)`` callers in case any
    survive. Used as: ``yk = apply_outcome_tf(y0, yk, tf)`` if ``tf is not None``.
    """
    import inspect
    try:
        sig = inspect.signature(tf)
        n_params = len([p for p in sig.parameters.values()
                        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
                        and p.default is inspect.Parameter.empty])
    except (TypeError, ValueError):
        n_params = 2
    return tf(y0, yk) if n_params >= 2 else tf(yk)


COHORT_DUMMY_PREFIX = "cohort_"


def cohort_dummies(cohort_series):
    """Build cohort dummies (2 of 3 levels, drop-first) for fair head-to-head comparators.

    Returns a (DataFrame, list_of_columns) tuple. The original head-to-head
    misspecified M1 by omitting cohort, while M0 always carried arm (a 7-level
    proxy that encodes cohort + subtype). To restore comparator parity the
    fair head-to-head adds these cohort dummies to M1 (and M2 already has arm,
    which subsumes cohort).
    """
    import pandas as pd
    dum = pd.get_dummies(cohort_series.astype(str).str.lower(), drop_first=True).astype(float)
    dum.columns = [f"{COHORT_DUMMY_PREFIX}{c}" for c in dum.columns]
    return dum, list(dum.columns)
