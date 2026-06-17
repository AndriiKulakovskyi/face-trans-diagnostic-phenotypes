"""M2.2 — measurement-error Gaussian mixture as a SOFT TESSELLATION (§3.2).

Continuum verdict (M2.1) ⇒ this mixture is reported as a **soft tessellation** of the continuum (a discrete
overlay for communication / a decision-region label), **NOT** natural-kind biotypes. It does honour the
project's core principle — propagate M1's per-patient uncertainty: each coordinate carries a known diagonal
measurement-error variance ``S_i``, deconvolved via **Extreme Deconvolution** (Bovy et al. 2011), the EM for

    x_i  ~  Σ_k π_k · Normal(m_k, V_k + S_i).

So the recovered components (m_k, V_k) describe the UNDERLYING cloud free of measurement noise;
prior-dominated coordinates (large S_i) self-down-weight (the no-imputation principle at the coordinate
layer); DR's absent substance cell (wide S_i) contributes ≈nothing. Soft responsibilities r_ik = the
tessellation membership. K is a tessellation granularity (continuum ⇒ no natural K; parsimony ~3–4, M2.1).
"""
from __future__ import annotations

import numpy as np

_LOG2PI = float(np.log(2.0 * np.pi))


def _estep_k(X, S, mu_k, V_k):
    """Per-component: log N(x_i | mu_k, V_k+S_i), plus T_k^{-1} and diff for the M-step.

    ``S`` is the known per-patient measurement error: ``[N, D]`` diagonal variances (default) OR the
    FULL ``[N, D, D]`` covariance (the P2-04 faithfulness arm — uses the coherent-scorer cross-dimension
    uncertainty instead of only the marginal SDs)."""
    N, D = X.shape
    T = np.broadcast_to(V_k, (N, D, D)).copy()
    if S.ndim == 3:                                       # full per-patient covariance S_i [N,D,D]
        T = T + S
    else:                                                 # diagonal noise [N,D]
        idx = np.arange(D)
        T[:, idx, idx] += S                               # T_i = V_k + diag(S_i)
    Tinv = np.linalg.inv(T)
    _, logdet = np.linalg.slogdet(T)
    diff = X - mu_k                                        # [N,D]
    maha = np.einsum("ni,nij,nj->n", diff, Tinv, diff)
    logpdf = -0.5 * (D * _LOG2PI + logdet + maha)
    return logpdf, Tinv, diff


def xd_em(X, S, K, *, n_iter=200, tol=1e-5, reg=1e-4, seed=0):
    """Extreme-Deconvolution EM for a K-component mixture with known per-patient diagonal noise S [N,D].
    Returns dict(pi, mu [K,D], V [K,D,D], resp [N,K], loglik, bic, n_iter)."""
    from sklearn.cluster import KMeans
    X = np.asarray(X, dtype="float64")
    S = np.asarray(S, dtype="float64")
    N, D = X.shape
    km = KMeans(K, n_init=5, random_state=seed).fit(X)
    lab = km.labels_
    pi = np.array([(lab == k).mean() for k in range(K)])
    mu = km.cluster_centers_.copy()
    V = np.stack([np.cov(X[lab == k].T) + reg * np.eye(D) if (lab == k).sum() > D
                  else np.eye(D) for k in range(K)])
    I = np.eye(D)
    prev = -np.inf
    for it in range(n_iter):
        logr = np.zeros((N, K)); Tinv = [None] * K; diff = [None] * K
        for k in range(K):
            lp, Tinv[k], diff[k] = _estep_k(X, S, mu[k], V[k])
            logr[:, k] = np.log(pi[k] + 1e-300) + lp
        m = logr.max(1, keepdims=True)
        ll = float((m[:, 0] + np.log(np.exp(logr - m).sum(1))).sum())
        r = np.exp(logr - m); r /= r.sum(1, keepdims=True)        # [N,K]
        # M-step (deconvolved): b_ik = mu_k + V_k T^{-1}(x-mu_k); B_ik = V_k - V_k T^{-1} V_k
        for k in range(K):
            VkTinv = V[k][None] @ Tinv[k]                          # [N,D,D]
            b = mu[k] + np.einsum("nij,nj->ni", VkTinv, diff[k])   # [N,D]
            B = V[k][None] - VkTinv @ V[k][None]                   # [N,D,D]
            w = r[:, k]; Nk = w.sum() + 1e-12
            pi[k] = Nk / N
            mu[k] = (w[:, None] * b).sum(0) / Nk
            d = b - mu[k]
            V[k] = (w[:, None, None] * (B + np.einsum("ni,nj->nij", d, d))).sum(0) / Nk + reg * I
        if ll - prev < tol * abs(prev):
            break
        prev = ll
    p = (K - 1) + K * D + K * D * (D + 1) // 2
    bic = -2.0 * ll + p * np.log(N)
    return {"pi": pi, "mu": mu, "V": V, "resp": r, "loglik": ll, "bic": float(bic),
            "n_iter": it + 1, "K": K}


def xd_fixed_labels(X, S, labels, *, n_iter=60, reg=1e-4):
    """Deconvolved mixture whose components are FIXED to a given hard partition (e.g. the 7 DSM-5
    subtypes). Iterates the XD M-step with responsibilities pinned to the label one-hot, then scores the
    resulting mixture likelihood. Used for the §1.7 head-to-head: does a free K-component tessellation
    describe the cloud better (lower BIC) than the DSM-5 partition?"""
    X = np.asarray(X, dtype="float64"); S = np.asarray(S, dtype="float64")
    N, D = X.shape
    uniq = list(dict.fromkeys(labels))
    K = len(uniq)
    onehot = np.zeros((N, K))
    for k, u in enumerate(uniq):
        onehot[np.asarray(labels) == u, k] = 1.0
    pi = onehot.mean(0)
    mu = np.stack([X[onehot[:, k] == 1].mean(0) for k in range(K)])
    V = np.stack([np.cov(X[onehot[:, k] == 1].T) + reg * np.eye(D) if (onehot[:, k] == 1).sum() > D
                  else np.eye(D) for k in range(K)])
    I = np.eye(D); ll = -np.inf
    for _ in range(n_iter):
        logcomp = np.zeros((N, K)); Tinv = [None] * K; diff = [None] * K
        for k in range(K):
            lp, Tinv[k], diff[k] = _estep_k(X, S, mu[k], V[k])
            logcomp[:, k] = np.log(pi[k] + 1e-300) + lp
        m = logcomp.max(1, keepdims=True)
        ll = float((m[:, 0] + np.log(np.exp(logcomp - m).sum(1))).sum())
        for k in range(K):                                       # M-step with r FIXED to the labels
            VkTinv = V[k][None] @ Tinv[k]
            b = mu[k] + np.einsum("nij,nj->ni", VkTinv, diff[k])
            B = V[k][None] - VkTinv @ V[k][None]
            w = onehot[:, k]; Nk = w.sum() + 1e-12
            mu[k] = (w[:, None] * b).sum(0) / Nk
            d = b - mu[k]
            V[k] = (w[:, None, None] * (B + np.einsum("ni,nj->nij", d, d))).sum(0) / Nk + reg * I
    p = (K - 1) + K * D + K * D * (D + 1) // 2
    return {"loglik": ll, "bic": float(-2.0 * ll + p * np.log(N)), "K": K}


def bic_sweep(X, S, Ks=range(2, 9), seed=0):
    """XD BIC over K (for the tessellation-granularity report; continuum ⇒ no sharp optimum expected)."""
    out = {}
    for K in Ks:
        out[K] = xd_em(X, S, K, seed=seed)["bic"]
    return out
