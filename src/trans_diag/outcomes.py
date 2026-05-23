"""Shared head-to-head outcome-modelling helpers.

Extracted from the Phase-5 script so the numbered pipeline scripts can reuse the
*exact* same CV metric / incremental-value test without importing one another
(digit-prefixed module names are not importable in Python). Used by the
head-to-head, CI, de-circularization, ComBat and cognition steps.
"""
from __future__ import annotations

import numpy as np

RANDOM = 0

# (name, kind, source_col, transform) — the 1-year outcomes.
OUTCOMES = [
    ("EGF functioning", "continuous", "egf", None),
    ("any hospitalization", "binary", "nboccur_hospitalisation_lt",
     lambda s: (s > 0).astype(float)),
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
    beta = dict(zip(cols_all, coef))
    return {a: float(beta[a]) for a in axis_cols}
