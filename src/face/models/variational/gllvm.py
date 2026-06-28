"""Pure-PyTorch kernels for the variational mixed-likelihood GLLVM atlas engine.

This module is intentionally free of any ``face`` / pandas imports: it speaks only in
tensors and numpy arrays so it can be unit-tested on synthetic data without the data
contract.  The orchestration layer (``gllvm_model_oop``) builds the
:class:`LoadingOntology` from the certified prior matrix and feeds tensors here.

The model is the GLLVM equivalent of the certified Bayesian sparse bifactor/ESEM
measurement model, but trained by stochastic variational inference instead of NUTS:

* one general burden axis G (factor index 0) + specific axes;
* an ontology-constrained **linear** decoder ``eta_ij = alpha_j + lambda_j . f_i``;
* positive (softplus) home loadings, signed cross-loadings, structural-zero forbidden cells;
* mixed per-item likelihoods (gaussian / bernoulli / ordinal / count), observed cells only;
* a per-patient variational posterior ``q_i(f_i) = N(mu_i, diag(s_i^2))`` with a
  bifactor-faithful correlated prior ``N(0, Phi)`` (G — and any pinned axis — orthogonal).

Unlike the NUTS engine, every patient keeps an explicit variational latent for every
factor, so there is no continuous/explicit split and no conditional-Gaussian
decomposition; per-item likelihoods attach directly to ``f_i``.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Family = Literal["gaussian", "bernoulli", "ordinal", "count"]

# Families the orchestration layer maps every indicator onto.  ``gaussian`` covers the
# rank-INT'd continuous + promoted block (copula mode) and any log/affine-standardized
# continuous item (native mode); the three discrete families keep their native likelihood.
FAMILIES: tuple[Family, ...] = ("gaussian", "bernoulli", "ordinal", "count")


def inverse_softplus(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Stable inverse of ``softplus`` so we can initialize a softplus-parameterized
    positive quantity at a target value."""
    x = torch.clamp(x, min=eps)
    return torch.log(torch.expm1(x))


@dataclass(frozen=True)
class LoadingOntology:
    """Sparse loading structure consumed by :class:`VariationalGLLVM` — the GLLVM
    analogue of the certified ``LoadingSpec``, but expressed as the masks/tensors the
    nn.Module needs (no pandas).

    * ``free_mask[j, k]``     — the cell is estimated (else a structural exact zero).
    * ``positive_mask[j, k]`` — the cell is sign-anchored positive (softplus); these are
      the home / G-anchor cells that fix each factor's orientation.
    * ``prior_mean`` / ``prior_sd`` — the per-cell ontology prior used by the loading
      penalty (read verbatim from the prior matrix).
    * ``item_family``        — per-item likelihood family in :data:`FAMILIES`.
    * ``ord_n_cat``          — number of ordered categories for each ordinal item.
    * ``kind``               — audit label per free cell (primary / g_anchor / bifactor_G
      / window / cross), carried straight through to the loadings export.
    """

    free_mask: np.ndarray
    positive_mask: np.ndarray
    prior_mean: np.ndarray
    prior_sd: np.ndarray
    item_family: list[str]
    ord_n_cat: dict[int, int]
    kind: dict[tuple[int, int], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        J, K = self.free_mask.shape
        for name in ("positive_mask", "prior_mean", "prior_sd"):
            arr = getattr(self, name)
            if arr.shape != (J, K):
                raise ValueError(f"{name} shape {arr.shape} != free_mask shape {(J, K)}")
        if len(self.item_family) != J:
            raise ValueError(f"item_family length {len(self.item_family)} != J {J}")
        bad = set(self.item_family) - set(FAMILIES)
        if bad:
            raise ValueError(f"unsupported families {bad}; allowed {FAMILIES}")
        # A positive cell must be free; a free positive cell can never also be a structural zero.
        if np.any(self.positive_mask & ~self.free_mask):
            raise ValueError("positive_mask has cells that are not free")

    @property
    def n_items(self) -> int:
        return self.free_mask.shape[0]

    @property
    def n_factors(self) -> int:
        return self.free_mask.shape[1]


class CorrelationPrior(nn.Module):
    """Bifactor-faithful latent prior covariance ``Phi``.

    Mirrors ``BayesianBifactorESEM._build_phi``: the general factor G (index 0) and any
    additional ``orthogonal_indices`` (e.g. substance, whose cross-factor correlations are
    non-identifiable) are pinned to identity rows/columns; the remaining specific block
    carries a learnable correlation via a normalized lower-triangular Cholesky factor.

    The KL divergence ``KL[q_i || N(0, Phi)]`` is computed with the **same full Phi** the
    decoder's generative covariance implies.  The (small, K x K) Cholesky / inverse /
    log-determinant are evaluated on CPU in float64 (MPS lacks robust ``linalg.cholesky``);
    everything per-patient stays elementwise on the model's device.
    """

    def __init__(
        self,
        n_factors: int,
        *,
        orthogonal_indices: tuple[int, ...] = (0,),
        jitter: float = 1e-4,
    ):
        super().__init__()
        self.n_factors = n_factors
        # G (index 0) is always orthogonal in the bifactor model.
        self.orthogonal_indices = tuple(sorted(set(orthogonal_indices) | {0}))
        self.block_indices = [k for k in range(n_factors) if k not in self.orthogonal_indices]
        self.jitter = jitter
        d = len(self.block_indices)
        if d > 0:
            self.raw_lower = nn.Parameter(torch.zeros(d, d))
        else:
            self.register_parameter("raw_lower", None)

    @property
    def _device(self) -> torch.device:
        if self.raw_lower is not None:
            return self.raw_lower.device
        return torch.device("cpu")

    def correlation_block(self) -> torch.Tensor:
        """The ``d x d`` correlation matrix on the free specific block."""
        assert self.raw_lower is not None
        d = self.raw_lower.shape[0]
        lower = torch.tril(self.raw_lower, diagonal=-1)
        diag = torch.diag(F.softplus(torch.diagonal(self.raw_lower)) + 0.5)
        chol = lower + diag
        cov = chol @ chol.T
        scale = torch.sqrt(torch.clamp(torch.diag(cov), min=1e-6))
        corr = cov / (scale[:, None] * scale[None, :])
        eye = torch.eye(d, device=corr.device, dtype=corr.dtype)
        corr = corr + self.jitter * eye
        renorm = torch.sqrt(torch.diag(corr))
        return corr / renorm[:, None] / renorm[None, :]

    def forward(self) -> torch.Tensor:
        """The full ``F x F`` correlation matrix Phi with the pinned axes orthogonal."""
        K = self.n_factors
        if self.raw_lower is None:
            return torch.eye(K, device=self._device)
        block = self.correlation_block()
        phi = torch.eye(K, device=block.device, dtype=block.dtype)
        idx = torch.tensor(self.block_indices, dtype=torch.long, device=block.device)
        phi = phi.clone()
        phi[idx[:, None], idx[None, :]] = block
        return phi

    def kl_diag_gaussian(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Per-patient ``KL[N(mu_i, diag(exp logvar_i)) || N(0, Phi)]``.

        Returns a ``(batch,)`` tensor.  The K x K linear algebra is done on CPU/float64
        (autograd flows back through the device transfer); the per-patient terms are
        elementwise on ``mu``'s device.
        """
        phi = self.forward()
        K = mu.shape[-1]
        # Move to CPU first, THEN upcast (MPS cannot convert a tensor to float64 in place).
        phi_cpu = phi.to("cpu").to(torch.float64)
        eye = torch.eye(K, device="cpu", dtype=torch.float64)
        chol = torch.linalg.cholesky(phi_cpu)
        inv_phi = torch.cholesky_solve(eye, chol)
        logdet_phi = 2.0 * torch.log(torch.diagonal(chol)).sum()
        # Cast back down on CPU, then move to the model's device (float32-safe on MPS).
        inv_phi_dev = inv_phi.to(mu.dtype).to(mu.device)
        diag_inv = torch.diagonal(inv_phi_dev)
        logdet_phi_dev = logdet_phi.to(mu.dtype).to(mu.device)
        var = torch.exp(logvar)
        trace_term = (var * diag_inv).sum(dim=-1)
        quad_term = (mu @ inv_phi_dev * mu).sum(dim=-1)
        logdet_q = logvar.sum(dim=-1)
        return 0.5 * (trace_term + quad_term - K + logdet_phi_dev - logdet_q)

    def kl_lowrank(self, mu: torch.Tensor, logvar: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Per-patient ``KL[N(mu_i, diag(exp logvar_i) + U_i U_iᵀ) || N(0, Phi)]``.

        ``U`` is ``(batch, K, R)`` (the low-rank factor of the posterior covariance).  A
        low-rank ``q`` can represent the cross-factor posterior covariance the mean-field
        diagonal cannot — the mechanism that lets Phi's off-diagonals not be attenuated.

        Uses the matrix-determinant lemma for ``log|Σ|`` and the small ``R x R`` cap matrix;
        the K x K and R x R linear algebra are done on CPU/float64 (MPS-safe).
        """
        phi = self.forward()
        K = mu.shape[-1]
        phi_cpu = phi.to("cpu").to(torch.float64)
        eye = torch.eye(K, device="cpu", dtype=torch.float64)
        chol = torch.linalg.cholesky(phi_cpu)
        inv_phi = torch.cholesky_solve(eye, chol)
        logdet_phi = 2.0 * torch.log(torch.diagonal(chol)).sum()
        inv_dev = inv_phi.to(mu.dtype).to(mu.device)
        logdet_phi_dev = logdet_phi.to(mu.dtype).to(mu.device)
        diag_inv = torch.diagonal(inv_dev)
        d = torch.exp(logvar)  # (B, K)
        # tr(Phi⁻¹ Σ) = tr(Phi⁻¹ D) + tr(Uᵀ Phi⁻¹ U)
        phi_U = torch.einsum("kl,blr->bkr", inv_dev, U)
        tr_low = (U * phi_U).sum(dim=(-2, -1))
        trace_term = (d * diag_inv).sum(-1) + tr_low
        quad_term = (mu @ inv_dev * mu).sum(-1)
        # log|Σ| = log|D| + log|I_R + Uᵀ D⁻¹ U|  (matrix determinant lemma)
        dinv = torch.exp(-logvar)
        R = U.shape[-1]
        cap = torch.einsum("bkr,bk,bks->brs", U, dinv, U)
        cap = cap + torch.eye(R, device=cap.device, dtype=cap.dtype)
        logdet_cap = torch.logdet(cap.to("cpu").to(torch.float64)).to(mu.dtype).to(mu.device)
        logdet_q = logvar.sum(-1) + logdet_cap
        return 0.5 * (trace_term + quad_term - K + logdet_phi_dev - logdet_q)

    def kl_full(self, mu: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
        """Per-patient ``KL[N(mu_i, L_i L_iᵀ) || N(0, Phi)]`` for a full-covariance posterior.

        ``chol`` is ``(batch, K, K)`` lower-triangular with positive diagonal (the Cholesky factor
        of the per-patient posterior covariance).  A full covariance can represent the entire
        cross-factor posterior covariance — the most expressive (and exact, for K small) option
        for closing the Phi off-diagonal attenuation.
        """
        phi = self.forward()
        K = mu.shape[-1]
        phi_cpu = phi.to("cpu").to(torch.float64)
        eye = torch.eye(K, device="cpu", dtype=torch.float64)
        pchol = torch.linalg.cholesky(phi_cpu)
        inv_phi = torch.cholesky_solve(eye, pchol)
        logdet_phi = 2.0 * torch.log(torch.diagonal(pchol)).sum()
        inv_dev = inv_phi.to(mu.dtype).to(mu.device)
        logdet_phi_dev = logdet_phi.to(mu.dtype).to(mu.device)
        llt = chol @ chol.transpose(-1, -2)  # (B, K, K) = Sigma_i
        # tr(Phi⁻¹ Σ) = sum(Phi⁻¹ ⊙ Σ) (both symmetric)
        trace_term = (inv_dev.unsqueeze(0) * llt).sum(dim=(-2, -1))
        quad_term = (mu @ inv_dev * mu).sum(-1)
        logdet_q = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        return 0.5 * (trace_term + quad_term - K + logdet_phi_dev - logdet_q)

    def offdiag_l2_penalty(self, weight: float = 1e-3) -> torch.Tensor:
        """Weak L2 stabilizer on Phi's off-diagonal (``Omega(Phi)``)."""
        phi = self.forward()
        offdiag = phi - torch.diag(torch.diagonal(phi))
        return weight * (offdiag**2).sum()


class VariationalGLLVM(nn.Module):
    """Variational GLLVM: per-patient mean-field ``q(f)`` + ontology-constrained linear
    decoder + mixed per-item likelihoods.

    Global parameters (loadings, intercepts, residual scales, cutpoints, Phi) are point
    (MAP) estimates in this version — loading uncertainty is left to bootstrap / seed
    ensembles, and the NUTS engine remains the uncertainty authority.
    """

    def __init__(
        self,
        n_patients: int,
        ontology: LoadingOntology,
        *,
        orthogonal_indices: tuple[int, ...] = (0,),
        psi_floor: float = 0.05,
        loading_prior_weight: float = 1.0,
        phi_penalty_weight: float = 1e-3,
        sigma_prior_weight: float = 0.0,
        count_alpha_prior_weight: float = 0.0,
        cutpoint_prior_weight: float = 0.0,
        min_logvar: float = -7.0,
        max_logvar: float = 3.0,
        q_mu_init_sd: float = 0.05,
        q_logvar_init: float = -1.5,
        q_rank: int = 0,
        full_cov: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        self.ontology = ontology
        self.N = int(n_patients)
        self.J = ontology.n_items
        self.K = ontology.n_factors
        self.q_rank = int(q_rank)
        self.full_cov = bool(full_cov)
        self.psi_floor = psi_floor
        self.loading_prior_weight = loading_prior_weight
        self.phi_penalty_weight = phi_penalty_weight
        self.sigma_prior_weight = sigma_prior_weight
        self.count_alpha_prior_weight = count_alpha_prior_weight
        self.cutpoint_prior_weight = cutpoint_prior_weight
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.families: list[str] = list(ontology.item_family)

        if seed is not None:
            torch.manual_seed(seed)

        # Per-patient variational parameters (non-amortized mean field).
        self.q_mu = nn.Embedding(self.N, self.K)
        self.q_logvar = nn.Embedding(self.N, self.K)
        nn.init.normal_(self.q_mu.weight, mean=0.0, std=q_mu_init_sd)
        nn.init.constant_(self.q_logvar.weight, q_logvar_init)
        # Posterior covariance: mean-field diagonal (default), low-rank+diagonal (q_rank>0), or a
        # full per-patient Cholesky (full_cov).  full_cov takes precedence over q_rank.
        self.q_U = None
        self.q_chol = None
        if self.full_cov:
            tril = torch.tril_indices(self.K, self.K, offset=0)
            self.register_buffer("_tril_r", tril[0])
            self.register_buffer("_tril_c", tril[1])
            self.q_chol = nn.Embedding(self.N, self.K * (self.K + 1) // 2)
            with torch.no_grad():
                # Init at a small isotropic covariance (diagonal ~ exp(0.5*q_logvar_init), off-diag 0).
                init = torch.zeros(self.K * (self.K + 1) // 2)
                diag_pos = (self._tril_r == self._tril_c).nonzero(as_tuple=True)[0]
                init[diag_pos] = inverse_softplus(torch.exp(torch.tensor(0.5 * q_logvar_init)))
                self.q_chol.weight.copy_(init.unsqueeze(0).expand(self.N, -1))
        elif self.q_rank > 0:
            self.q_U = nn.Embedding(self.N, self.K * self.q_rank)
            nn.init.normal_(self.q_U.weight, mean=0.0, std=0.01)

        # Global decoder parameters.
        self.alpha = nn.Parameter(torch.zeros(self.J))
        self.raw_loading = nn.Parameter(torch.zeros(self.J, self.K))
        self.raw_sigma = nn.Parameter(torch.full((self.J,), inverse_softplus(torch.tensor(0.8)).item()))
        self.raw_count_alpha = nn.Parameter(torch.zeros(self.J))

        self.factor_prior = CorrelationPrior(self.K, orthogonal_indices=orthogonal_indices)

        self.ordinal_cutpoints = nn.ParameterDict()
        self._init_ordinal_cutpoints()

        # Ontology masks / priors as buffers (move with .to(device), saved in state_dict).
        self.register_buffer("loading_free", torch.as_tensor(ontology.free_mask, dtype=torch.bool))
        self.register_buffer("loading_positive", torch.as_tensor(ontology.positive_mask, dtype=torch.bool))
        self.register_buffer("loading_prior_mean", torch.as_tensor(ontology.prior_mean, dtype=torch.float32))
        self.register_buffer("loading_prior_sd", torch.as_tensor(ontology.prior_sd, dtype=torch.float32))
        self._init_loadings_from_prior()

    # ------------------------------------------------------------------ init
    def _init_ordinal_cutpoints(self) -> None:
        for j, family in enumerate(self.families):
            if family != "ordinal":
                continue
            C = int(self.ontology.ord_n_cat.get(j, 2))
            n_cuts = max(C - 1, 1)
            cuts = torch.linspace(-1.5, 1.5, n_cuts)
            raw = torch.zeros(n_cuts)
            raw[0] = cuts[0]
            if n_cuts > 1:
                diffs = torch.clamp(torch.diff(cuts), min=1e-3)
                raw[1:] = inverse_softplus(diffs)
            self.ordinal_cutpoints[str(j)] = nn.Parameter(raw)

    def _init_loadings_from_prior(self) -> None:
        with torch.no_grad():
            mean = self.loading_prior_mean
            positive = self.loading_positive
            free = self.loading_free
            raw = torch.zeros_like(mean)
            pos_free = positive & free
            raw[pos_free] = inverse_softplus(torch.clamp(mean[pos_free], min=0.05))
            signed_free = free & (~positive)
            raw[signed_free] = mean[signed_free]
            self.raw_loading.copy_(raw)

    @torch.no_grad()
    def initialize_from_data(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        """Initialize item intercepts / residual scales / cutpoints from observed marginals.

        ``x`` is (N, J) with arbitrary fill at missing cells; ``mask`` is (N, J) bool.
        """
        alpha = torch.zeros(self.J)
        raw_sigma = self.raw_sigma.detach().clone()
        for j, family in enumerate(self.families):
            obs = mask[:, j]
            if int(obs.sum()) == 0:
                continue
            values = x[obs, j].to(torch.float64)
            if family == "gaussian":
                alpha[j] = values.mean().to(alpha.dtype)
                resid_sd = torch.clamp(values.std(), min=self.psi_floor + 1e-3)
                raw_sigma[j] = inverse_softplus(
                    torch.clamp(resid_sd - self.psi_floor, min=1e-3)
                ).to(raw_sigma.dtype)
            elif family == "bernoulli":
                p = torch.clamp(values.mean(), 1e-3, 1 - 1e-3)
                alpha[j] = torch.logit(p).to(alpha.dtype)
            elif family == "count":
                m = torch.clamp(values.mean(), min=1e-3)
                alpha[j] = torch.log(m).to(alpha.dtype)
            elif family == "ordinal":
                # Intercept is absorbed by the cutpoints; init cutpoints from the empirical CDF.
                alpha[j] = 0.0
                C = int(self.ontology.ord_n_cat.get(j, 2))
                if C >= 2 and str(j) in self.ordinal_cutpoints:
                    counts = torch.bincount(values.long(), minlength=C).to(torch.float64)
                    cum = torch.cumsum(counts, dim=0) / counts.sum()
                    cum = torch.clamp(cum[:-1], 1e-3, 1 - 1e-3)
                    locs = torch.logit(cum)  # cutpoint locations
                    raw = torch.zeros(locs.numel())
                    raw[0] = locs[0]
                    if locs.numel() > 1:
                        diffs = torch.clamp(torch.diff(locs), min=1e-3)
                        raw[1:] = inverse_softplus(diffs)
                    self.ordinal_cutpoints[str(j)].copy_(raw.to(self.ordinal_cutpoints[str(j)].dtype))
        self.alpha.copy_(alpha.to(self.alpha.dtype))
        self.raw_sigma.copy_(raw_sigma)

    # ---------------------------------------------------------------- params
    def loading_matrix(self) -> torch.Tensor:
        """The J x F loading matrix: softplus-positive home cells, raw signed cross cells,
        exact structural zeros everywhere else."""
        lam = torch.zeros_like(self.raw_loading)
        positive_free = self.loading_free & self.loading_positive
        signed_free = self.loading_free & (~self.loading_positive)
        lam = torch.where(positive_free, F.softplus(self.raw_loading) + 1e-5, lam)
        lam = torch.where(signed_free, self.raw_loading, lam)
        return lam

    def sigma(self) -> torch.Tensor:
        """Gaussian residual SDs with the ``psi_floor`` (Heywood guard, matches NUTS)."""
        return self.psi_floor + F.softplus(self.raw_sigma)

    def count_alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_count_alpha) + 1e-3

    def ordered_cutpoints(self, j: int) -> torch.Tensor:
        """Monotone cutpoints for ordinal item ``j`` (first + cumulative positive steps)."""
        raw = self.ordinal_cutpoints[str(j)]
        if raw.numel() == 1:
            return raw
        first = raw[:1]
        increments = F.softplus(raw[1:]) + 1e-3
        return torch.cat([first, first + torch.cumsum(increments, dim=0)])

    def phi(self) -> torch.Tensor:
        return self.factor_prior.forward()

    # ----------------------------------------------------------------- model
    def _cholesky(self, patient_idx: torch.Tensor) -> torch.Tensor:
        """Per-patient lower-triangular Cholesky factor L (B, K, K) with positive diagonal."""
        raw = self.q_chol(patient_idx)  # (B, M), M = K(K+1)/2
        B = raw.shape[0]
        L = raw.new_zeros(B, self.K, self.K)
        L[:, self._tril_r, self._tril_c] = raw
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        new_diag = F.softplus(diag) + 1e-4
        return L - torch.diag_embed(diag) + torch.diag_embed(new_diag)

    def sample_latent(
        self,
        patient_idx: torch.Tensor,
        *,
        n_mc: int = 1,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns ``(f, mu, logvar, cov)`` where ``cov`` is the low-rank factor U, the full
        Cholesky L, or None (mean-field) — the KL dispatches on the model's covariance mode."""
        mu = self.q_mu(patient_idx)
        logvar = torch.clamp(self.q_logvar(patient_idx), self.min_logvar, self.max_logvar)
        cov = None
        if self.full_cov:
            cov = self._cholesky(patient_idx)  # (B, K, K)
        elif self.q_U is not None:
            cov = self.q_U(patient_idx).view(-1, self.K, self.q_rank)  # (B, K, R)
        if deterministic:
            f = mu.unsqueeze(0)
        else:
            if self.full_cov:
                eps = torch.randn((n_mc, mu.shape[0], self.K), device=mu.device, dtype=mu.dtype)
                f = mu.unsqueeze(0) + torch.einsum("bij,sbj->sbi", cov, eps)
            else:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn((n_mc, *mu.shape), device=mu.device, dtype=mu.dtype)
                f = mu.unsqueeze(0) + std.unsqueeze(0) * eps
                if cov is not None:
                    eta = torch.randn((n_mc, mu.shape[0], self.q_rank), device=mu.device, dtype=mu.dtype)
                    f = f + torch.einsum("bkr,sbr->sbk", cov, eta)
        return f, mu, logvar, cov

    def linear_predictor(self, f: torch.Tensor) -> torch.Tensor:
        lam = self.loading_matrix()
        eta = torch.einsum("sbk,jk->sbj", f, lam)
        return eta + self.alpha.view(1, 1, self.J)

    def negative_log_likelihood(
        self,
        eta: torch.Tensor,
        x_batch: torch.Tensor,
        mask_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Observed-cell negative log-likelihood, averaged over the MC latent draws.

        Missing cells contribute nothing — the mask removes them.  There is no imputed,
        zero-filled, or mean-filled likelihood.
        """
        total = eta.new_zeros(())
        S = eta.shape[0]
        for j, family in enumerate(self.families):
            obs = mask_batch[:, j]
            if not torch.any(obs):
                continue
            eta_j = eta[:, obs, j]
            x_j = x_batch[obs, j]
            if family == "gaussian":
                sigma_j = self.sigma()[j]
                resid = (x_j.unsqueeze(0) - eta_j) / sigma_j
                nll = 0.5 * (math.log(2.0 * math.pi) + 2.0 * torch.log(sigma_j) + resid**2)
                total = total + nll.sum() / S
            elif family == "bernoulli":
                target = x_j.unsqueeze(0).expand_as(eta_j)
                nll = F.binary_cross_entropy_with_logits(eta_j, target, reduction="sum")
                total = total + nll / S
            elif family == "ordinal":
                y = x_j.long()
                cuts = self.ordered_cutpoints(j)
                probs = self._ordinal_probabilities(eta_j, cuts)
                probs_y = probs.gather(dim=-1, index=y.view(1, -1, 1).expand(S, -1, 1)).squeeze(-1)
                nll = -torch.log(torch.clamp(probs_y, min=1e-8)).sum()
                total = total + nll / S
            elif family == "count":
                eta_clamped = torch.clamp(eta_j, min=-10.0, max=10.0)
                mu = torch.exp(eta_clamped)
                alpha = self.count_alpha()[j]
                nll = -self._negative_binomial_log_prob(x_j.unsqueeze(0), mu, alpha)
                total = total + nll.sum() / S
        return total

    @staticmethod
    def _ordinal_probabilities(eta: torch.Tensor, cuts: torch.Tensor) -> torch.Tensor:
        cuts = cuts.to(eta.device)
        cdf = torch.sigmoid(cuts.view(1, 1, -1) - eta.unsqueeze(-1))
        first = cdf[..., :1]
        if cuts.numel() > 1:
            middle = cdf[..., 1:] - cdf[..., :-1]
            last = 1.0 - cdf[..., -1:]
            probs = torch.cat([first, middle, last], dim=-1)
        else:
            last = 1.0 - cdf[..., -1:]
            probs = torch.cat([first, last], dim=-1)
        return torch.clamp(probs, min=1e-8, max=1.0)

    @staticmethod
    def _negative_binomial_log_prob(
        y: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        y = torch.clamp(y, min=0.0)
        log_amu = torch.log(alpha + mu)
        return (
            torch.lgamma(y + alpha)
            - torch.lgamma(alpha)
            - torch.lgamma(y + 1.0)
            + alpha * (torch.log(alpha) - log_amu)
            + y * (torch.log(mu) - log_amu)
        )

    # ----------------------------------------------------------------- terms
    def latent_kl(self, mu: torch.Tensor, logvar: torch.Tensor,
                  cov: torch.Tensor | None = None) -> torch.Tensor:
        if self.full_cov:
            return self.factor_prior.kl_full(mu, cov).sum()
        if cov is not None:
            return self.factor_prior.kl_lowrank(mu, logvar, cov).sum()
        return self.factor_prior.kl_diag_gaussian(mu, logvar).sum()

    def loading_prior_penalty(self) -> torch.Tensor:
        lam = self.loading_matrix()
        free = self.loading_free
        z = (lam - self.loading_prior_mean) / torch.clamp(self.loading_prior_sd, min=1e-4)
        return self.loading_prior_weight * 0.5 * (z[free] ** 2).sum()

    def global_prior_penalty(self) -> torch.Tensor:
        """Penalties on shared global parameters (added once per ELBO evaluation, not scaled
        by N/B): the loading ontology prior, the weak Phi off-diagonal stabilizer, and
        (optionally, to track NUTS) HalfNormal/Normal penalties on sigma / count-alpha /
        cutpoints."""
        penalty = self.loading_prior_penalty()
        penalty = penalty + self.factor_prior.offdiag_l2_penalty(self.phi_penalty_weight)
        if self.sigma_prior_weight > 0.0:
            # HalfNormal(1) on (sigma - floor): -log p propto 0.5 * resid^2.
            resid = self.sigma() - self.psi_floor
            penalty = penalty + self.sigma_prior_weight * 0.5 * (resid**2).sum()
        if self.count_alpha_prior_weight > 0.0:
            # HalfNormal(2) on the count dispersion.
            penalty = penalty + self.count_alpha_prior_weight * 0.5 * ((self.count_alpha() / 2.0) ** 2).sum()
        if self.cutpoint_prior_weight > 0.0:
            for j, family in enumerate(self.families):
                if family != "ordinal":
                    continue
                cuts = self.ordered_cutpoints(j)
                loc = torch.linspace(-1.5, 1.5, cuts.numel(), device=cuts.device, dtype=cuts.dtype)
                penalty = penalty + self.cutpoint_prior_weight * 0.5 * (((cuts - loc) / 2.0) ** 2).sum()
        return penalty

    def batch_loss(self, patient_idx: torch.Tensor, *, n_mc: int = 1) -> dict[str, torch.Tensor]:
        """Negative-ELBO for a minibatch.

        ``J = (N/B) (L_obs + KL) + Omega(Lambda) + Omega(Phi)``: the per-patient
        reconstruction + KL are scaled to the full population; the global penalties are
        added once (they are shared parameters, not per-patient), never scaled by N/B.
        """
        x_batch = self._x[patient_idx]
        mask_batch = self._mask[patient_idx]
        f, mu, logvar, U = self.sample_latent(patient_idx, n_mc=n_mc)
        eta = self.linear_predictor(f)
        nll = self.negative_log_likelihood(eta, x_batch, mask_batch)
        kl = self.latent_kl(mu, logvar, U)
        scale = self.N / patient_idx.numel()
        penalty = self.global_prior_penalty()
        loss = scale * (nll + kl) + penalty
        return {"loss": loss, "nll": nll.detach(), "kl": kl.detach(), "penalty": penalty.detach()}

    def attach_data(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        """Register the (already-encoded) data tensors as non-persistent buffers so
        ``batch_loss`` can index them on-device.  ``x`` is (N, J); ``mask`` is (N, J) bool.
        Non-persistent keeps them out of ``state_dict`` (the checkpoint stores parameters,
        not the data)."""
        self.register_buffer("_x", x, persistent=False)
        self.register_buffer("_mask", mask, persistent=False)

    # --------------------------------------------------------------- outputs
    @torch.no_grad()
    def coordinates(self) -> dict[str, np.ndarray]:
        mu = self.q_mu.weight.detach()
        if self.full_cov:
            idx = torch.arange(self.N, device=mu.device)
            chunks = []
            for i in range(0, self.N, 8192):  # chunk to bound the (N,K,K) build
                L = self._cholesky(idx[i : i + 8192])
                chunks.append(torch.sqrt((L**2).sum(-1)))  # diag(LLᵀ) row norms
            sd = torch.cat(chunks, 0)
        else:
            logvar = torch.clamp(self.q_logvar.weight.detach(), self.min_logvar, self.max_logvar)
            var = torch.exp(logvar)
            if self.q_U is not None:
                U = self.q_U.weight.detach().view(self.N, self.K, self.q_rank)
                var = var + (U**2).sum(-1)  # marginal variance = diag(D + UUᵀ)
            sd = torch.sqrt(var)
        return {"mean": mu.cpu().numpy(), "sd": sd.cpu().numpy()}

    @torch.no_grad()
    def loadings(self) -> np.ndarray:
        return self.loading_matrix().detach().cpu().numpy()

    @torch.no_grad()
    def phi_matrix(self) -> np.ndarray:
        return self.factor_prior.forward().detach().cpu().numpy()


@dataclass
class TrainingConfig:
    epochs: int = 4000
    lr: float = 1e-2
    n_mc: int = 1
    batch_size: int | None = None  # None => full batch
    grad_clip_norm: float | None = 5.0
    weight_decay: float = 0.0
    print_every: int = 200
    seed: int = 20260605
    # Early stopping on the (smoothed) -ELBO: stop when the best loss has not improved by
    # more than ``early_stop_rel_tol`` (relative) for ``early_stop_patience`` epochs.  None
    # disables it (runs the full ``epochs`` budget).
    early_stop_patience: int | None = None
    early_stop_rel_tol: float = 1e-4


class GLLVMTrainer:
    """Stochastic variational optimizer for :class:`VariationalGLLVM`.

    AdamW with the per-patient variational embeddings placed in a no-weight-decay group
    (weight decay there would be a spurious second prior on the coordinates, on top of the
    KL's ``N(0, Phi)``).  Each epoch records the loss components, the pre-clip gradient norm,
    and cumulative wall-clock — the SVI diagnostic trace (the analogue of NUTS R-hat/ESS).
    """

    def __init__(
        self,
        model: VariationalGLLVM,
        config: TrainingConfig | None = None,
        *,
        progress: Callable[[dict[str, float]], None] | None = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.history: list[dict[str, float]] = []
        self.progress = progress
        self.stopped_early = False

    def _optimizer(self) -> torch.optim.Optimizer:
        # The per-patient variational embeddings get NO weight decay (decay there would be a
        # spurious second prior on the coordinates, on top of the KL's N(0, Phi)).
        q_params = [self.model.q_mu.weight, self.model.q_logvar.weight]
        for emb in (self.model.q_U, self.model.q_chol):
            if emb is not None:
                q_params.append(emb.weight)
        no_decay = set(q_params)
        decay_params = [p for p in self.model.parameters() if p.requires_grad and p not in no_decay]
        groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": q_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=self.config.lr)

    def _grad_norm(self) -> float:
        sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum().cpu())
        return math.sqrt(sq)

    def fit(self) -> list[dict[str, float]]:
        torch.manual_seed(self.config.seed)
        device = self.model.q_mu.weight.device
        optimizer = self._optimizer()
        N = self.model.N
        bs = self.config.batch_size or N
        clip = self.config.grad_clip_norm
        best = math.inf
        since_improved = 0
        start = time.time()

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            if bs >= N:
                perm = [torch.arange(N, device=device)]
            else:
                shuffled = torch.randperm(N, device=device)
                perm = [shuffled[i : i + bs] for i in range(0, N, bs)]

            agg = {"loss": 0.0, "nll": 0.0, "kl": 0.0, "penalty": 0.0}
            gnorm = 0.0
            for idx in perm:
                optimizer.zero_grad(set_to_none=True)
                out = self.model.batch_loss(idx, n_mc=self.config.n_mc)
                out["loss"].backward()
                if clip is not None:
                    gnorm += float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip))
                else:
                    gnorm += self._grad_norm()
                optimizer.step()
                for key in agg:
                    agg[key] += float(out[key].detach().cpu())
            nb = len(perm)
            record = {
                "epoch": epoch,
                **{k: agg[k] / nb for k in agg},
                "grad_norm": gnorm / nb,
                "elapsed_sec": round(time.time() - start, 2),
            }
            self.history.append(record)
            if self.progress is not None:
                self.progress(record)
            if self.config.print_every and (epoch == 1 or epoch % self.config.print_every == 0):
                print(
                    f"epoch={epoch:05d} loss={record['loss']:.3f} nll={record['nll']:.3f} "
                    f"kl={record['kl']:.3f} penalty={record['penalty']:.3f} "
                    f"|grad|={record['grad_norm']:.2f} t={record['elapsed_sec']:.0f}s",
                    flush=True,
                )

            # Early stopping on relative improvement of the -ELBO.
            patience = self.config.early_stop_patience
            if patience is not None:
                if record["loss"] < best * (1.0 - self.config.early_stop_rel_tol):
                    best = record["loss"]
                    since_improved = 0
                else:
                    since_improved += 1
                    if since_improved >= patience:
                        self.stopped_early = True
                        if self.config.print_every:
                            print(
                                f"epoch={epoch:05d} early stop: no >"
                                f"{self.config.early_stop_rel_tol:.0e} improvement in {patience} epochs",
                                flush=True,
                            )
                        break
        return self.history


# --------------------------------------------------------------------------- testing
def synthetic_gllvm_dataset(
    n: int = 400,
    *,
    seed: int = 0,
    missing_frac: float = 0.2,
) -> dict:
    """Plant a known 3-factor bifactor (G + 2 specifics) with one item of each family and
    return tensors + truth, for synthetic-recovery tests.  Pure numpy/torch — no data
    contract.

    Layout: factors [G, spec1, spec2]; items are grouped 3 per factor-role, with a positive
    home loading + a signed bifactor-G loading, one gaussian / bernoulli / ordinal / count
    item per specific factor.
    """
    rng = np.random.default_rng(seed)
    K = 3
    # Specific block correlation.
    corr = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.3], [0.0, 0.3, 1.0]])
    L = np.linalg.cholesky(corr)
    f = rng.standard_normal((n, K)) @ L.T  # (n, K), G orthogonal to specifics

    # 8 items: 2 gaussian (spec1, spec2), 2 bernoulli, 2 ordinal, 2 count — each home on a
    # specific factor, each also loading on G.
    families = ["gaussian", "gaussian", "bernoulli", "bernoulli", "ordinal", "ordinal", "count", "count"]
    home_factor = [1, 2, 1, 2, 1, 2, 1, 2]
    J = len(families)
    free = np.zeros((J, K), dtype=bool)
    positive = np.zeros((J, K), dtype=bool)
    prior_mean = np.zeros((J, K), dtype=np.float32)
    prior_sd = np.ones((J, K), dtype=np.float32) * 0.25
    kind: dict[tuple[int, int], str] = {}
    Lam_true = np.zeros((J, K), dtype=np.float64)
    ord_n_cat: dict[int, int] = {}
    for j in range(J):
        h = home_factor[j]
        free[j, h] = True
        positive[j, h] = True
        prior_mean[j, h] = 0.6
        prior_sd[j, h] = 0.3
        kind[(j, h)] = "primary"
        Lam_true[j, h] = rng.uniform(0.7, 1.1)
        free[j, 0] = True  # bifactor G
        prior_mean[j, 0] = 0.0
        prior_sd[j, 0] = 0.25
        kind[(j, 0)] = "bifactor_G"
        Lam_true[j, 0] = rng.uniform(0.2, 0.5)
        if families[j] == "ordinal":
            ord_n_cat[j] = 4

    eta = f @ Lam_true.T  # (n, J)
    alpha_true = np.zeros(J)
    x = np.full((n, J), np.nan, dtype=np.float64)
    for j, fam in enumerate(families):
        lin = alpha_true[j] + eta[:, j]
        if fam == "gaussian":
            x[:, j] = lin + 0.4 * rng.standard_normal(n)
        elif fam == "bernoulli":
            p = 1.0 / (1.0 + np.exp(-lin))
            x[:, j] = (rng.random(n) < p).astype(float)
        elif fam == "ordinal":
            C = ord_n_cat[j]
            cuts = np.linspace(-1.2, 1.2, C - 1)
            cdf = 1.0 / (1.0 + np.exp(-(cuts[None, :] - lin[:, None])))
            probs = np.diff(np.concatenate([np.zeros((n, 1)), cdf, np.ones((n, 1))], axis=1), axis=1)
            x[:, j] = np.array([rng.choice(C, p=pr / pr.sum()) for pr in probs], dtype=float)
        elif fam == "count":
            mu = np.exp(np.clip(lin, -4, 4))
            x[:, j] = rng.poisson(mu).astype(float)

    mask = np.ones((n, J), dtype=bool)
    if missing_frac > 0:
        drop = rng.random((n, J)) < missing_frac
        mask[drop] = False
    x_filled = np.nan_to_num(x, nan=0.0)

    ontology = LoadingOntology(
        free_mask=free,
        positive_mask=positive,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        item_family=families,
        ord_n_cat=ord_n_cat,
        kind=kind,
    )
    return {
        "x": torch.as_tensor(x_filled, dtype=torch.float32),
        "mask": torch.as_tensor(mask, dtype=torch.bool),
        "ontology": ontology,
        "f_true": f,
        "Lam_true": Lam_true,
        "families": families,
        "n_factors": K,
    }
