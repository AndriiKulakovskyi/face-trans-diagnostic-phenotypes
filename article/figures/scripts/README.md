# Figure-generation scripts

Each `<figure>.py` here is the **verbatim producing code** for the corresponding
`article/figures/<figure>.png`, extracted from its Claude Science artifact lineage
(version_id noted in each file header).

These figures were produced in a shared `face-dev` kernel session in which the fitted
GLLVM model state (`results/face/gllvm_oop/s8_full/model_state.pt`) and derived arrays
(loadings, residual sigmas, likelihood families, posterior coordinates) were loaded once
and reused across cells. Each file is the exact figure cell; run standalone it may need
that shared setup (model load + per-family Fisher-information arrays) reconstructed first.

Reliability/battery figures use **exact per-family Fisher information** (Gaussian λ²/σ²,
Bernoulli λ²p(1−p), graded-response cumulative-logit, NB λ²μr/(r+μ)) — not the λ²/ψ
approximation.
