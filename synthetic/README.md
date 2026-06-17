# Synthetic FACE-like data (issue P7-03)

The raw FACE cohort data are confidential, which blocks external reproduction of the measurement engine.
This directory provides a **synthetic** dataset with the same *shape* as the real one — so anyone can run
and certify the engine, and the golden tests can exercise it in CI — without any confidential data.

## What it is

`generate_face_like.py` reads the real **structure** (modelled-indicator set, likelihood families, burden
signs) from `configs/prior_loading_matrix_v3.csv` and draws synthetic values from a **known** bifactor
model:

- general factor **G** held orthogonal to the specific factors (the S1 identification);
- **biology near-⟂G by construction** (planted G-loadings: metabolic 0.08 / inflammatory 0.07 vs
  cognition 0.35 / sleep 0.30) — so a correct engine recovers the headline ordering;
- FACE-like **cohort imbalance** (BP ≫ SZ > DR) and **MCAR missingness** (NaN preserved, never imputed).

It is **not** real data and carries no patient information.

## Use

```bash
python3 synthetic/generate_face_like.py            # -> synthetic/data/{baseline_v0,covariates_v0,site_v0}.parquet + truth.json

# Run the engine on it (the processed-data dir is overridable via FACE_DATA_DIR):
FACE_DATA_DIR=synthetic/data python3 scripts/04_fit.py --stage 1
```

`truth.json` records the planted loadings/σ. `tests/golden/test_synthetic_recovery.py` regenerates the
data, fits the marginalized S1 model, and asserts the home loadings and the biology-⟂-G ordering are
recovered — the end-to-end "reproduce on synthetic data" check.

`synthetic/data/` is regenerable and git-ignored; the generator and this README are tracked.
