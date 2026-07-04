"""Golden tests for the pure-torch variational GLLVM kernels.

These run on CPU with fixed seeds — no data contract, no MPS — so they are deterministic in
CI.  They cover the load-bearing invariants: synthetic recovery, observed-cell-only loss,
the closed-form KL, ordinal probability normalization, structural-zero loadings, positive
home loadings, and bifactor Phi orthogonality.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from analyses.variational_gllvm.gllvm import (  # noqa: E402
    CorrelationPrior,
    GLLVMTrainer,
    LoadingOntology,
    TrainingConfig,
    VariationalGLLVM,
    synthetic_gllvm_dataset,
)


def _build_model(data, *, orthogonal_indices=(0,), seed=0):
    model = VariationalGLLVM(
        data["x"].shape[0], data["ontology"], orthogonal_indices=orthogonal_indices, seed=seed
    )
    model.initialize_from_data(data["x"], data["mask"])
    model.attach_data(data["x"], data["mask"])
    return model


def test_synthetic_recovery_small_n_fixed_seed():
    torch.manual_seed(0)
    data = synthetic_gllvm_dataset(n=400, seed=1, missing_frac=0.15)
    model = _build_model(data, orthogonal_indices=(0,), seed=0)
    GLLVMTrainer(model, TrainingConfig(epochs=1200, lr=2e-2, print_every=0, seed=0)).fit()

    lam = model.loadings()
    free = data["ontology"].free_mask
    r = np.corrcoef(lam[free], data["Lam_true"][free])[0, 1]
    assert r > 0.9, f"loading recovery r={r:.3f}"

    coords = model.coordinates()["mean"]
    for k in (1, 2):  # the two specific factors
        rk = abs(np.corrcoef(coords[:, k], data["f_true"][:, k])[0, 1])
        assert rk > 0.75, f"coordinate f{k} recovery r={rk:.3f}"


def test_observed_cell_only_loss_ignores_missing():
    data = synthetic_gllvm_dataset(n=120, seed=2, missing_frac=0.3)
    model = _build_model(data)
    idx = torch.arange(model.N)
    f, _mu, _lv, _U = model.sample_latent(idx, deterministic=True)
    eta = model.linear_predictor(f)
    nll0 = model.negative_log_likelihood(eta, data["x"], data["mask"])

    # Corrupt the data ONLY at masked-out (missing) cells; the loss must not change.
    x2 = data["x"].clone()
    missing = ~data["mask"]
    x2[missing] = x2[missing] + 7.0
    nll1 = model.negative_log_likelihood(eta, x2, data["mask"])
    assert torch.allclose(nll0, nll1, atol=1e-5), (float(nll0), float(nll1))


def test_kl_matches_standard_normal_closed_form():
    K = 5
    prior = CorrelationPrior(K, orthogonal_indices=tuple(range(K)))  # Phi = I
    assert prior.raw_lower is None
    torch.manual_seed(3)
    mu = torch.randn(8, K)
    logvar = torch.randn(8, K) * 0.5
    kl = prior.kl_diag_gaussian(mu, logvar)
    var = torch.exp(logvar)
    closed = 0.5 * (var + mu**2 - 1.0 - logvar).sum(dim=-1)
    assert torch.allclose(kl, closed, atol=1e-5)


def test_kl_lowrank_matches_full_covariance():
    """The low-rank KL must equal the brute-force full-covariance KL for Σ = D + UUᵀ."""
    K, R, B = 5, 2, 4
    prior = CorrelationPrior(K, orthogonal_indices=(0, 4))
    torch.manual_seed(0)
    with torch.no_grad():
        prior.raw_lower.normal_(0, 0.5)
    phi = prior.forward().detach().double()
    mu = torch.randn(B, K).double()
    logvar = (torch.randn(B, K) * 0.3).double()
    U = (torch.randn(B, K, R) * 0.4).double()
    phi_inv = torch.linalg.inv(phi)
    logdet_phi = torch.logdet(phi)
    bf = []
    for b in range(B):
        sigma = torch.diag(torch.exp(logvar[b])) + U[b] @ U[b].T
        kl = 0.5 * (torch.trace(phi_inv @ sigma) + mu[b] @ phi_inv @ mu[b] - K
                    + logdet_phi - torch.logdet(sigma))
        bf.append(kl.item())
    got = prior.kl_lowrank(mu, logvar, U).detach()
    assert torch.allclose(got, torch.tensor(bf, dtype=got.dtype), atol=1e-9)


def test_kl_full_matches_full_covariance():
    """The full-covariance KL must equal the brute-force KL for Σ = L Lᵀ."""
    K, B = 5, 4
    prior = CorrelationPrior(K, orthogonal_indices=(0, 4))
    torch.manual_seed(0)
    with torch.no_grad():
        prior.raw_lower.normal_(0, 0.5)
    phi = prior.forward().detach().double()
    mu = torch.randn(B, K).double()
    L = torch.tril(torch.randn(B, K, K)).double()
    for b in range(B):
        L[b].diagonal().copy_(torch.rand(K).double() + 0.3)
    phi_inv = torch.linalg.inv(phi)
    logdet_phi = torch.logdet(phi)
    bf = []
    for b in range(B):
        sigma = L[b] @ L[b].T
        kl = 0.5 * (torch.trace(phi_inv @ sigma) + mu[b] @ phi_inv @ mu[b] - K
                    + logdet_phi - torch.logdet(sigma))
        bf.append(kl.item())
    got = prior.kl_full(mu, L).detach()
    assert torch.allclose(got, torch.tensor(bf, dtype=got.dtype), atol=1e-9)


def test_full_cov_q_trains():
    data = synthetic_gllvm_dataset(n=300, seed=2, missing_frac=0.15)
    model = VariationalGLLVM(data["x"].shape[0], data["ontology"], full_cov=True, seed=0)
    model.attach_data(data["x"], data["mask"])
    GLLVMTrainer(model, TrainingConfig(epochs=600, lr=2e-2, print_every=0, seed=0)).fit()
    lam = model.loadings()
    free = data["ontology"].free_mask
    assert np.corrcoef(lam[free], data["Lam_true"][free])[0, 1] > 0.85
    assert model.coordinates()["sd"].shape == (300, data["n_factors"])


def test_lowrank_q_trains_and_marginal_sd_includes_rank():
    data = synthetic_gllvm_dataset(n=300, seed=2, missing_frac=0.15)
    model = VariationalGLLVM(data["x"].shape[0], data["ontology"], q_rank=2, seed=0)
    model.attach_data(data["x"], data["mask"])
    GLLVMTrainer(model, TrainingConfig(epochs=600, lr=2e-2, print_every=0, seed=0)).fit()
    lam = model.loadings()
    free = data["ontology"].free_mask
    r = np.corrcoef(lam[free], data["Lam_true"][free])[0, 1]
    assert r > 0.85
    # Marginal SD = sqrt(diag(D + UUᵀ)) >= the diagonal-only SD.
    sd = model.coordinates()["sd"]
    diag_sd = np.exp(0.5 * np.clip(model.q_logvar.weight.detach().numpy(), -7, 3))
    assert np.all(sd >= diag_sd - 1e-6)


def test_ordinal_probabilities_sum_to_one():
    eta = torch.linspace(-2, 2, 7).view(1, -1)
    cuts = torch.tensor([-1.0, 0.0, 1.2])  # 3 cutpoints -> 4 categories
    probs = VariationalGLLVM._ordinal_probabilities(eta, cuts)
    assert probs.shape[-1] == cuts.numel() + 1
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert bool((probs >= 0).all())


def test_hard_zero_cells_stay_exactly_zero():
    data = synthetic_gllvm_dataset(n=100, seed=4, missing_frac=0.1)
    # Force a cell that is planted-free to be a structural zero, and verify it stays 0.
    ont = data["ontology"]
    free = ont.free_mask.copy()
    positive = ont.positive_mask.copy()
    free[0, 2] = False
    positive[0, 2] = False
    blocked = LoadingOntology(
        free_mask=free, positive_mask=positive, prior_mean=ont.prior_mean, prior_sd=ont.prior_sd,
        item_family=ont.item_family, ord_n_cat=ont.ord_n_cat, kind=ont.kind,
    )
    model = VariationalGLLVM(data["x"].shape[0], blocked, seed=0)
    model.attach_data(data["x"], data["mask"])
    GLLVMTrainer(model, TrainingConfig(epochs=50, lr=5e-2, print_every=0, seed=0)).fit()
    lam = model.loadings()
    assert lam[0, 2] == 0.0
    assert float(np.abs(lam[~free]).max()) == 0.0


def test_positive_home_loadings_stay_nonnegative():
    data = synthetic_gllvm_dataset(n=120, seed=5, missing_frac=0.1)
    model = _build_model(data)
    GLLVMTrainer(model, TrainingConfig(epochs=80, lr=5e-2, print_every=0, seed=0)).fit()
    lam = model.loadings()
    pos = data["ontology"].positive_mask
    assert float(lam[pos].min()) >= 0.0


def test_phi_pins_orthogonal_axes():
    K = 8
    prior = CorrelationPrior(K, orthogonal_indices=(0, 7))
    torch.manual_seed(6)
    with torch.no_grad():
        prior.raw_lower.normal_()  # random non-trivial correlation block
    phi = prior.forward().detach().numpy()
    for i in (0, 7):
        off = np.abs(np.delete(phi[i], i))
        assert float(off.max()) == 0.0, f"axis {i} not orthogonal"
    # The free specific block carries non-zero correlation.
    assert float(np.abs(phi[1, 2])) > 0.0
    # Phi is a valid correlation matrix (unit diagonal, symmetric PD).
    assert np.allclose(np.diag(phi), 1.0, atol=1e-5)
    assert np.allclose(phi, phi.T, atol=1e-6)
    assert float(np.linalg.eigvalsh(phi).min()) > 0.0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_forward_runs():
    data = synthetic_gllvm_dataset(n=64, seed=7, missing_frac=0.1)
    model = VariationalGLLVM(data["x"].shape[0], data["ontology"], seed=0)
    model.attach_data(data["x"].to("mps"), data["mask"].to("mps"))
    model.to("mps")
    out = model.batch_loss(torch.arange(model.N, device="mps"))
    assert torch.isfinite(out["loss"]).item()
