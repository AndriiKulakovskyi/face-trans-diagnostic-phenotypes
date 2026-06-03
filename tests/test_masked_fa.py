"""Tests for the imputation-free factor analysis (trans_diag.masked_fa).

Validates the math behind the dimensional model's masked estimator (scripts 01–35): the
masked correlation, principal-axis factoring + varimax recover a known 2-factor structure,
and the masked posterior-mean scores track the true factors while leaving sparsely-observed
rows unscored (NaN) rather than imputing them.
"""
import numpy as np
import pandas as pd

from trans_diag.masked_fa import (
    masked_correlation,
    masked_loadings,
    masked_scores,
    nearest_pd,
    paf_loadings,
    varimax,
)


def _synth(n=3000, seed=0):
    """A clean 2-factor generative model: 6 indicators, 3 per factor."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, 2))
    L = np.array([[0.8, 0.0], [0.7, 0.0], [0.6, 0.0],
                  [0.0, 0.8], [0.0, 0.7], [0.0, 0.6]])
    X = F @ L.T + 0.4 * rng.standard_normal((n, 6))
    return pd.DataFrame(X, columns=[f"d{i}" for i in range(6)]), F, L


def _cong(a, b):
    return abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def test_nearest_pd_is_positive_definite_correlation():
    A = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])  # indefinite
    P = nearest_pd(A)
    assert np.all(np.linalg.eigvalsh(P) > 0)          # positive-definite
    assert np.allclose(np.diag(P), 1.0)               # unit diagonal (a correlation matrix)
    assert np.allclose(P, P.T)                        # symmetric


def test_masked_correlation_matches_pandas_on_complete_data():
    X, _, _ = _synth()
    R = masked_correlation(X, min_pair=10)
    assert np.allclose(R, X.corr().to_numpy(), atol=1e-6)


def test_masked_correlation_ignores_missing_cells():
    # corr of two columns must use only co-observed rows, never a filled value
    X, _, _ = _synth(n=500)
    Xm = X.copy()
    Xm.iloc[:250, 0] = np.nan      # half of d0 missing
    R = masked_correlation(Xm, min_pair=10)
    direct = Xm[["d0", "d1"]].dropna().corr().iloc[0, 1]
    assert abs(R[0, 1] - direct) < 0.05


def test_paf_varimax_recovers_known_loadings():
    X, _, L = _synth()
    Lhat = varimax(paf_loadings(masked_correlation(X), 2))
    for j in range(2):                                 # each true factor recovered at >0.9
        assert max(_cong(L[:, j], Lhat[:, k]) for k in range(2)) > 0.9


def test_masked_scores_leave_sparse_rows_nan_and_track_factors():
    X, F, _ = _synth()
    Lhat = masked_loadings(X, 2)
    Xm = X.copy()
    Xm.iloc[0, 1:] = np.nan                            # row 0: 1 observed < k=2 → unscored
    z = (Xm - Xm.mean()) / Xm.std(ddof=0)
    S = masked_scores(z, Lhat)
    assert np.all(np.isnan(S[0]))                      # sparse row not imputed, left NaN
    assert np.all(np.isfinite(S[1:]))                  # all others scored
    fin = np.isfinite(S).all(1)
    c = np.abs(np.corrcoef(S[fin].T, F[fin].T)[:2, 2:])
    assert c.max(axis=1).min() > 0.7                   # each score tracks a true factor


def test_masked_scores_no_imputation_invariance():
    # adding extra missingness to OTHER rows must not change a complete row's score
    X, _, _ = _synth(n=800)
    L = masked_loadings(X, 2)
    z = (X - X.mean()) / X.std(ddof=0)
    s_full = masked_scores(z, L)
    z2 = z.copy()
    z2.iloc[10:20, 0] = np.nan                         # perturb only rows 10–19
    s_pert = masked_scores(z2, L)
    assert np.allclose(s_full[0], s_pert[0], atol=1e-9)   # row 0 unchanged (per-row scoring)
