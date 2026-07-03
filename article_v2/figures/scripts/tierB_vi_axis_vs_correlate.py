#!/usr/bin/env python3
"""
Tier B (VI) — biology-as-axis vs biology-as-correlate, held-out predictive comparison.

The manuscript's hinge claim (Fig 3): immunometabolic biology is a CO-EQUAL latent axis,
not a downstream correlate of general severity. This turns that into a held-out test using
the SAME variational GLLVM the paper validates against NUTS (Annex H; Tucker >=0.95 on all
8 factors, immunometabolic 0.998).

Two exactly-nested models on the identical N=9,013 x 139-indicator bifactor map:

  AXIS   (current):  full 8-factor ontology. Every item loads on G (overall_severity);
                     the 46 biology cells ALSO load on a dedicated `immunometabolic` axis.
  CORR   (correlate): the SAME ontology with the immunometabolic COLUMN removed — biology
                     items load on general severity G only (no dedicated axis).

Held-out design: mask a random `holdout_frac` of OBSERVED cells (seeded), fit both models
on the remaining cells, score per-cell log predictive density on the held-out cells only.
DeltaELPD = ELPD(axis) - ELPD(corr) on the held-out cells; positive => the biology axis
carries out-of-sample predictive variance that general severity cannot absorb.

Run (repo venv):
  cd article_v2 && OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    ../.venv/bin/python figures/scripts/tierB_vi_axis_vs_correlate.py
Smoke:  ... tierB_vi_axis_vs_correlate.py --smoke
"""
from __future__ import annotations
import argparse, sys, time, json
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "face" / "tierB_vi_axis_vs_correlate"
OUT.mkdir(parents=True, exist_ok=True)

import torch
from face.models.variational.gllvm_model_oop import GLLVMDataset, GLLVMConfig, F8_FIT
from face.models.variational.gllvm import (
    VariationalGLLVM, LoadingOntology, GLLVMTrainer, TrainingConfig,
)


def drop_immuno_column(ont: LoadingOntology, imm_col: int) -> LoadingOntology:
    """Correlate ontology: same as axis but the immunometabolic column is a structural
    zero (biology items load on G only). Returns a NEW LoadingOntology."""
    fm = ont.free_mask.copy();       fm[:, imm_col] = False
    pm = ont.positive_mask.copy();   pm[:, imm_col] = False
    kind = {k: v for k, v in ont.kind.items() if k[1] != imm_col}
    return LoadingOntology(
        free_mask=fm, positive_mask=pm,
        prior_mean=ont.prior_mean.copy(), prior_sd=ont.prior_sd.copy(),
        item_family=list(ont.item_family), ord_n_cat=dict(ont.ord_n_cat), kind=kind,
    )


def make_holdout(mask: torch.Tensor, frac: float, seed: int):
    """Split OBSERVED cells into train/test. Returns (train_mask, test_mask) bool tensors."""
    rng = np.random.default_rng(seed)
    obs = mask.numpy().copy()
    ii, jj = np.where(obs)
    n_hold = int(round(frac * len(ii)))
    sel = rng.choice(len(ii), size=n_hold, replace=False)
    test = np.zeros_like(obs)
    test[ii[sel], jj[sel]] = True
    train = obs & ~test
    return torch.from_numpy(train), torch.from_numpy(test)


def fit_and_score(ont, x, train_mask, test_mask, *, N, epochs, seed, label, n_mc_eval=64):
    """Fit VI GLLVM on train cells; return mean per-cell held-out log predictive density."""
    torch.manual_seed(seed)
    model = VariationalGLLVM(n_patients=N, ontology=ont, seed=seed)
    model.attach_data(x, train_mask)      # fit on TRAIN cells only (test cells masked out)
    tcfg = TrainingConfig(epochs=epochs, lr=1e-2, seed=seed,
                          early_stop_patience=300, early_stop_rel_tol=5e-5)
    trainer = GLLVMTrainer(model, config=tcfg)
    t0 = time.time()
    hist = trainer.fit()
    dt = time.time() - t0
    # Held-out scoring: mean over posterior q(f) draws of per-cell log-lik on TEST cells
    model.eval()
    with torch.no_grad():
        idx = torch.arange(N)
        f, mu, logvar, cov = model.sample_latent(idx, n_mc=n_mc_eval, deterministic=False)
        eta = model.linear_predictor(f)                 # (S, N, J)
        # per-cell log predictive density on test cells (Gaussian/other families via NLL parts)
        lpd = _pointwise_lpd(model, eta, x, test_mask)  # scalar mean over test cells
    n_test = int(test_mask.sum())
    elpd = float(lpd)                                   # SUM of per-cell log pred density
    print(f"  [{label}] fit {dt:.0f}s ({len(hist)} epochs) | held-out ELPD={elpd:.1f} "
          f"over {n_test} cells ({elpd/n_test:+.4f}/cell)", flush=True)
    return {"elpd": elpd, "n_test": n_test, "elpd_per_cell": elpd / n_test,
            "epochs": len(hist), "fit_s": dt}


def _pointwise_lpd(model, eta, x, test_mask):
    """log( (1/S) sum_s p(x_ij | eta_s) ) summed over test cells, families-aware.
    Uses log-mean-exp over MC draws for a proper predictive density (not mean-of-log)."""
    import torch.nn.functional as Fnn
    S = eta.shape[0]
    fam = model.families
    total = eta.new_zeros(())
    tm = test_mask
    for j, family in enumerate(fam):
        obs = tm[:, j]
        if not torch.any(obs):
            continue
        eta_j = eta[:, obs, j]                # (S, nobs)
        x_j = x[obs, j]                       # (nobs,)
        if family == "gaussian":
            sig = model.sigma()[j]
            ll = -0.5 * (np.log(2*np.pi) + 2*torch.log(sig) + ((x_j.unsqueeze(0) - eta_j)/sig)**2)
        elif family == "bernoulli":
            tgt = x_j.unsqueeze(0).expand_as(eta_j)
            ll = -Fnn.binary_cross_entropy_with_logits(eta_j, tgt, reduction="none")
        elif family == "ordinal":
            cuts = model.ordered_cutpoints(j)
            probs = model._ordinal_probabilities(eta_j, cuts)   # (S, nobs, C)
            y = x_j.long().view(1, -1, 1).expand(S, -1, 1)
            ll = torch.log(torch.clamp(probs.gather(-1, y).squeeze(-1), min=1e-8))
        elif family == "count":
            mu = torch.exp(torch.clamp(eta_j, -10, 10))
            alpha = model.count_alpha()[j]
            ll = model._negative_binomial_log_prob(x_j.unsqueeze(0), mu, alpha)
        else:
            continue
        # log-mean-exp over S draws, then sum over cells
        total = total + torch.logsumexp(ll, dim=0).sub(np.log(S)).sum()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260703)
    a = ap.parse_args()
    if a.smoke:
        a.n, a.epochs = 500, 400
    nsub = a.n if a.n > 0 else None

    print(f"Tier B (VI): axis vs correlate | N={'full 9013' if nsub is None else nsub} "
          f"| epochs={a.epochs} holdout={a.holdout}\n", flush=True)

    cfg = GLLVMConfig(smoke=False)
    data = GLLVMDataset(cfg).build(factors=list(F8_FIT), windows=False,
                                   n_subsample=nsub, balanced=(nsub is not None), seed=a.seed)
    N = data.x.shape[0]
    fc = data.factor_cols
    imm = fc.index("immunometabolic")
    print(f"AXIS ontology: {N} patients x {data.ontology.n_items} indicators, "
          f"K={data.ontology.n_factors} factors={fc}", flush=True)
    ont_axis = data.ontology
    ont_corr = drop_immuno_column(ont_axis, imm)
    print(f"CORR ontology: immunometabolic column removed "
          f"({int(ont_axis.free_mask[:, imm].sum())} biology cells -> load on G only), "
          f"K_eff={int((ont_corr.free_mask.sum(0) > 0).sum())}\n", flush=True)

    train_mask, test_mask = make_holdout(data.mask, a.holdout, a.seed)
    print(f"held-out split: {int(train_mask.sum())} train cells / "
          f"{int(test_mask.sum())} test cells\n", flush=True)

    res = {}
    for name, ont in [("axis", ont_axis), ("correlate", ont_corr)]:
        res[name] = fit_and_score(ont, data.x, train_mask, test_mask,
                                  N=N, epochs=a.epochs, seed=a.seed, label=name)

    diff = res["axis"]["elpd"] - res["correlate"]["elpd"]
    res["elpd_diff_axis_minus_correlate"] = diff
    res["holdout_frac"] = a.holdout
    res["N"] = N
    res["n_indicators"] = data.ontology.n_items
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    print(f"\n=== held-out ELPD (log predictive density on {res['axis']['n_test']} held-out cells) ===")
    print(f"  axis      : {res['axis']['elpd']:.1f}  ({res['axis']['elpd_per_cell']:+.4f}/cell)")
    print(f"  correlate : {res['correlate']['elpd']:.1f}  ({res['correlate']['elpd_per_cell']:+.4f}/cell)")
    print(f"\nDeltaELPD (axis - correlate) = {diff:+.1f}  "
          f"({'axis wins' if diff > 0 else 'correlate wins'})", flush=True)
    print(f"written -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
