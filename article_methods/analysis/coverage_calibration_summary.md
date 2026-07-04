# Posterior-coordinate calibration: are the credible intervals honest?

**Track:** FACE-ATLAS methods paper, trust/calibration (Fig 5). **N_sim = 10000, seed = 20260704.**

We test whether the instrument's per-axis credible intervals are calibrated by simulation, the only
setting in which the truth is known. We draw 10,000 synthetic patients with known latent coordinates
theta ~ N(0, Phi) using the **frozen model's own** loadings, intercepts, residual scales, cutpoints and
factor prior; impose the **real FACE missingness** by bootstrapping the full 142-item observed/missing
masks of the 9,013 real patients (mask only, never a clinical value); and project each synthetic patient
through a Fisher-scoring (Gauss-Newton/Laplace) EAP that **reduces exactly to the paper's closed-form
Gaussian projector** on all-Gaussian patterns (verified to 3e-15) and generalises it to the
bernoulli/ordinal/count families.

**The instrument is well-calibrated.** Mean per-axis coverage matches nominal at every level tested:
0.501/0.801/0.899/0.949 at nominal 0.50/0.80/0.90/0.95
(MC SE at 0.95 = 0.0022), and the joint 8-D credible ellipsoid is likewise
honest (0.947 at 0.95). All eight axes individually fall in 0.945-0.953
at the 95% level — within +/-2 SE of nominal — including the bank-limited mania/activation (mean 1.6
home items) and substance (2.7) axes.

**Coverage does not degrade with sparsity.** Stratified by observed home-items on the axis, 95% coverage
is flat from prior-dominated to fully observed: 0.947 (0 items), 0.952
(1-2), 0.949 (3-5), 0.949 (6-10), 0.949 (11+). The
0-item bin recovering ~0.95 is the key correctness check: with no data the projector returns the prior
(mean 0, sd 1), so sampling truth from that same prior must give nominal coverage — it does, confirming the
projection code has no systematic bias. There is no prior-dominated over-coverage and no
over-confidence on thin axes.

**Caveat.** On axes informed mainly by discrete items the posterior SD runs modestly below the residual
RMSE (per-axis sd/rmse ratio 0.90-1.01; lowest suicidality 0.895, mania/substance at parity 1.007/0.996),
because the Laplace-Gaussian posterior is a slightly light-tailed approximation to the true discrete-item
posterior; 95%-level *coverage* is
nonetheless exact because it is set by the interval quantiles rather than the SD. The test uses the frozen
model's Phi (== consolidate/phi.csv); reports/copula_8factor_phi.csv differs by <=0.11 on specific
off-diagonals but is not used, so generation and scoring share one prior by construction.
