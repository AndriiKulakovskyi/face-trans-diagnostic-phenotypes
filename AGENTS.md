## Learned User Preferences

- For FACE modeling guidance and audits, prefer step-by-step, line-by-line explanations of the model/MCMC/code flow plus explicit code-theory alignment concerns, framed with precision psychiatry and statistical modeling expertise.

## Learned Workspace Facts

- M1 implementation caveats repeatedly flagged for audits: primary PyMC likelihood has no in-model covariates, `unlikely_cross` cells are hard-zeroed by default, continuous likelihood is Gaussian/log-Gaussian rather than Student-t, final 9D mixed fit is cohort-balanced around N=2000 with full-N projection afterward, and correlated-G biology testing is a continuous-backbone sensitivity model rather than the exact final 9D mixed model.
- `scripts/s5_certify9.py` has a provenance mismatch: comments/manifest mention `hurdle_counts=True`, but `build_mixed()` defaults to plain negative binomial unless `hurdle_counts=True` is explicitly passed; downstream PPC validates the plain-NB path.
- `notebooks/m1_oop_measurement_fit.ipynb` was patched so the setup cell locates the repository root from `notebooks/` and clears stale `face` imports, because another local `face` package can shadow this workspace.
- Parallel OOP measurement work now lives in `src/face/models/bayesian/measurement_model_oop.py`, with `notebooks/run_measurement_model_oop.py`, `notebooks/m1_oop_measurement_fit.ipynb`, and golden tests; smoke defaults are S1-only fast wiring, mixed smoke is opt-in, and the runner caches by model version/stage signature.
- The OOP correlated-specific `Phi` path avoids `pm.LKJCorr` after an 8x8-to-4x4 initializer broadcast failure; it uses a shape-local `Phi_spec_lower` normalized lower-triangle parameterization with regression coverage for sequential S1/S5 builds.
- Latest OOP medium diagnostic run: S1/S2 continuous stages were close but not certified (`rhat` 1.02, no divergences; S2 ESS high), while `s5_9dim_mixed` ran but failed convergence (`rhat` 3.17, ESS 4); do not interpret S5 OOP outputs before stabilization.
