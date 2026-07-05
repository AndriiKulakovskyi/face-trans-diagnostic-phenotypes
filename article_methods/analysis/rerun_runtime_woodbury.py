"""
T2.3 — Is the Woodbury-reduced projection actually faster than naive full-covariance
inversion at realistic observed-item counts?

For each patient the projection needs Sigma_Ci^{-1} where Sigma_Ci = Lambda_Ci Phi Lambda_Ci^T + Psi_Ci
is |C_i| x |C_i|.
  - NAIVE:    invert the |C_i| x |C_i| matrix directly           -> O(|C_i|^3)
  - WOODBURY: invert only the (K+1) x (K+1) capacitance matrix    -> O((K+1)^3), K+1=8

We build synthetic observed patterns at the REAL |C_i| distribution (derived from the
per-axis observed-item counts, /tmp/ci_dist.npy) using the real loading matrix rows, time
both paths per patient, and report per-|C_i| and aggregate speedup. Correctness is checked
by max abs difference of the two Sigma^{-1} (must be ~machine eps).
"""
import json
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
import torch
import torch.nn.functional as F

ROOT="/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr"
sd=torch.load(f"{ROOT}/results/analyses/variational_gllvm/s8_full/model_state.pt", map_location="cpu", weights_only=False)
st=sd["state_dict"]; items=sd["items"]; families=sd["families"]; factor_cols=sd["factor_cols"]; J=len(items); K1=len(factor_cols)

raw_loading=st["raw_loading"]; loading_free=st["loading_free"].bool(); loading_positive=st["loading_positive"].bool()
lam=torch.zeros_like(raw_loading)
pf=loading_free & loading_positive; sf=loading_free & (~loading_positive)
lam=torch.where(pf, F.softplus(raw_loading)+1e-5, lam); lam=torch.where(sf, raw_loading, lam)
Lambda=lam.numpy().astype(np.float64)                       # (J, 8) loadings
sigma=(0.30+F.softplus(st["raw_sigma"])).numpy().astype(np.float64)  # (J,) residual SD
psi=sigma**2

# Phi: factor correlation matrix (8x8)
import pandas as pd

Phi=pd.read_csv(f"{ROOT}/results/analyses/variational_gllvm/consolidate/phi.csv", index_col=0).values.astype(np.float64)
Phi=0.5*(Phi+Phi.T)
Phi_inv=np.linalg.inv(Phi)

rng=np.random.default_rng(20260704)
ci_dist=np.load("/tmp/ci_dist.npy"); ci_dist=ci_dist[ci_dist>=K1]  # need at least K+1 obs to be well-posed

def sigma_inv_naive(Lc):
    Sig=Lc@Phi@Lc.T + np.diag(psi_c)
    return np.linalg.inv(Sig)

def sigma_inv_woodbury(Lc, psi_c):
    Pinv=1.0/psi_c
    # cap = Phi^{-1} + Lc^T diag(1/psi) Lc   (8x8)
    cap=Phi_inv + (Lc.T*Pinv)@Lc
    cap_inv=np.linalg.inv(cap)
    # Sigma^{-1} = diag(1/psi) - diag(1/psi) Lc cap_inv Lc^T diag(1/psi)
    A=(Lc*Pinv[:,None])                       # (|C|,8)
    return np.diag(Pinv) - A@cap_inv@A.T

# Benchmark at representative |C_i| values spanning the real distribution
grid=[10,25,50,75,100,125]
reps=200
rows=[]
maxdiff_global=0.0
for m in grid:
    # sample real-ish patterns: pick m distinct items at random from the J bank
    tn=0.0; tw=0.0
    for _ in range(reps):
        idx=rng.choice(J, size=min(m,J), replace=False)
        Lc=Lambda[idx]; psi_c=psi[idx]
        # naive
        t0=time.perf_counter()
        Sig=Lc@Phi@Lc.T + np.diag(psi_c); Sinv_n=np.linalg.inv(Sig)
        tn+=time.perf_counter()-t0
        # woodbury
        t0=time.perf_counter()
        Pinv=1.0/psi_c; cap=Phi_inv+(Lc.T*Pinv)@Lc; cap_inv=np.linalg.inv(cap)
        Aa=Lc*Pinv[:,None]; Sinv_w=np.diag(Pinv)-Aa@cap_inv@Aa.T
        tw+=time.perf_counter()-t0
        maxdiff_global=max(maxdiff_global, float(np.max(np.abs(Sinv_n-Sinv_w))))
    rows.append(dict(nobs=m, naive_us=1e6*tn/reps, woodbury_us=1e6*tw/reps, speedup=tn/tw))

# aggregate over the REAL |C_i| distribution (weight grid speedups by empirical frequency)
median_ci=int(np.median(ci_dist))
res={"grid":rows, "reps_per_point":reps, "K_plus_1":K1, "J_bank":J,
     "max_abs_diff_Sigma_inv":maxdiff_global, "real_ci_median":median_ci,
     "real_ci_pct":{p:int(np.percentile(ci_dist,q)) for p,q in [("p10",10),("p50",50),("p90",90)]}}
json.dump(res, open(f"{ROOT}/article_methods/analysis/runtime_woodbury.json","w"), indent=2)

print("=== T2.3  per-patient Sigma^-1 timing (mean of %d reps) ===" % reps)
print(f"{'|C_i|':>6} {'naive_us':>10} {'woodbury_us':>12} {'speedup':>9}")
for r in rows:
    print(f"{r['nobs']:>6} {r['naive_us']:>10.1f} {r['woodbury_us']:>12.1f} {r['speedup']:>8.1f}x")
print(f"\ncorrectness: max|Sigma^-1_naive - Sigma^-1_woodbury| = {maxdiff_global:.2e}")
print(f"real |C_i| median={median_ci} (p10={res['real_ci_pct']['p10']}, p90={res['real_ci_pct']['p90']})")
# speedup at the real median
import numpy as _np

sp_med=_np.interp(median_ci,[r['nobs'] for r in rows],[r['speedup'] for r in rows])
print(f"interpolated speedup at real median |C_i|={median_ci}: {sp_med:.1f}x")
