"""
T0.2 (blocker) — Misspecification / perturbed-DGP calibration stress test.

The paper's calibration figure samples data from the model's OWN parameters and shows the
posterior is calibrated — a reviewer correctly notes that is true by construction (any correctly
coded Bayesian estimator passes). The real question: when the data-generating process VIOLATES
the model's assumptions, does the reported 95% posterior interval degrade GRACEFULLY, or does it
silently break? We generate from four perturbed DGPs of increasing magnitude, score each with the
UNPERTURBED Gaussian EAP (Fisher-scoring, exactly the paper's projection), and map empirical
coverage of the nominal 95% per-axis interval vs perturbation strength.

Perturbations (each a named violation of a model assumption):
  1. CORRELATED RESIDUALS  — inject a shared nuisance factor into residuals (violates local
     independence): resid <- resid + rho * g_extra, rho in {0,.15,.30,.45}.
  2. HEAVY TAILS           — draw residuals from Student-t with df in {inf,8,5,3} (violates
     Gaussianity).
  3. OMITTED FACTOR        — add a 9th latent dimension loading on a random item subset, absent
     from the scoring model (model misspecification): strength s in {0,.25,.5,.75}.
  4. LOADING MISMATCH      — score with loadings perturbed by multiplicative noise
     lam*(1+eps*N(0,1)), eps in {0,.10,.20,.30} (parameter error / transfer to a new sample).

For each: report empirical coverage of the 95% interval (and 50%), averaged over 8 axes,
n=4000 patients per cell, real-ish missingness (keep 50% of items). Grace = coverage stays
near nominal or drifts smoothly & conservatively; break = coverage collapses.
"""
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from scipy import stats

ROOT="/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr"
A=np.load("/tmp/face_arrays.npz",allow_pickle=True)
lam0=A["lam"]; alpha=A["alpha"]; sigma=A["sigma"]; fam=A["fam_int"]; Phi=A["Phi"]
J=lam0.shape[0]; K=8; L=np.linalg.cholesky(Phi+1e-9*np.eye(K)); Phi_inv=np.linalg.inv(Phi); psi0=sigma**2
N=4000; KEEP=0.5; z95=1.959964; z50=0.674490

def gen_eta(n,rng): return rng.standard_normal((n,K))@L.T

def eap_with_var(Yc, mask, lam, psi):
    """Gaussian EAP posterior mean AND per-axis posterior SD via Woodbury capacitance.
    Returns (coord (n,K), sd (n,K))."""
    n=len(Yc); C=np.zeros((n,K)); S=np.zeros((n,K))
    for i in range(n):
        obs=mask[i]
        if obs.sum()<1:
            C[i]=0; S[i]=np.sqrt(np.diag(Phi)); continue
        Lc=lam[obs]; Pinv=1.0/psi[obs]
        cap=Phi_inv+(Lc.T*Pinv)@Lc
        Sig=np.linalg.inv(cap)                       # posterior covariance
        C[i]=Sig@((Lc.T*Pinv)@(Yc[i,obs]-alpha[obs]))
        S[i]=np.sqrt(np.clip(np.diag(Sig),1e-12,None))
    return C,S

def coverage(eta_true,C,S,z):
    lo=C-z*S; hi=C+z*S
    cov=((eta_true>=lo)&(eta_true<=hi)).mean(0)      # per-axis
    return cov.mean()

def base_items(eta,rng):
    lin=eta@lam0.T+alpha[None,:]; return lin

def run_cell(kind,level,seed):
    rng=np.random.default_rng(seed)
    eta=gen_eta(N,rng); lin=base_items(eta,rng)
    lam=lam0.copy(); psi=psi0.copy()
    # residual draw
    if kind=="heavy_tails":
        df={0:np.inf,1:8.0,2:5.0,3:3.0}[level]
        if np.isinf(df): e=rng.standard_normal((N,J))
        else: e=stats.t.rvs(df,size=(N,J),random_state=rng)/np.sqrt(df/(df-2))  # unit variance
        Y=lin+e*sigma[None,:]
    elif kind=="correlated_resid":
        rho=[0,.15,.30,.45][level]
        gextra=rng.standard_normal((N,1))
        e=rng.standard_normal((N,J))+rho*gextra      # shared nuisance across items
        e/=np.sqrt(1+rho**2)                          # keep ~unit variance
        Y=lin+e*sigma[None,:]
    elif kind=="omitted_factor":
        s=[0,.25,.5,.75][level]
        lam9=np.zeros(J); sub=rng.random(J)<0.4; lam9[sub]=rng.standard_normal(sub.sum())
        g9=rng.standard_normal((N,1))
        Y=lin + s*(g9*lam9[None,:]) + rng.standard_normal((N,J))*sigma[None,:]
    elif kind=="loading_mismatch":
        eps=[0,.10,.20,.30][level]
        lam=lam0*(1+eps*rng.standard_normal(lam0.shape))   # SCORING model uses mismatched loadings
        Y=lin+rng.standard_normal((N,J))*sigma[None,:]     # data from TRUE lam0
    mask=rng.random((N,J))<KEEP
    C,S=eap_with_var(Y,mask,lam,psi)
    return coverage(eta,C,S,z95), coverage(eta,C,S,z50)

kinds={"correlated_resid":[0,1,2,3],"heavy_tails":[0,1,2,3],
       "omitted_factor":[0,1,2,3],"loading_mismatch":[0,1,2,3]}
levels_lab={"correlated_resid":["rho=0","rho=.15","rho=.30","rho=.45"],
            "heavy_tails":["df=inf","df=8","df=5","df=3"],
            "omitted_factor":["s=0","s=.25","s=.5","s=.75"],
            "loading_mismatch":["eps=0","eps=.10","eps=.20","eps=.30"]}
out={}
for kind,levels in kinds.items():
    rows=[]
    for lv in levels:
        c95=[]; c50=[]
        for seed in range(3):
            a,b=run_cell(kind,lv,700+lv*10+seed); c95.append(a); c50.append(b)
        rows.append({"level":levels_lab[kind][lv],"cov95":float(np.mean(c95)),"cov50":float(np.mean(c50))})
    out[kind]=rows

json.dump({"nominal":{"cov95":0.95,"cov50":0.50},"N":N,"keep_frac":KEEP,"results":out},
          open(f"{ROOT}/article_methods/analysis/blocker_T02_misspec.json","w"),indent=2)

print("T0.2  MISSPECIFICATION STRESS TEST — empirical coverage of nominal 95% / 50% interval")
print("(data-generating process violates a model assumption; scored with unperturbed Gaussian EAP)")
for kind,rows in out.items():
    print(f"\n  {kind}:")
    print(f"    {'level':>10} {'cov95':>7} {'cov50':>7}")
    for r in rows: print(f"    {r['level']:>10} {r['cov95']:>7.3f} {r['cov50']:>7.3f}")
print("\n  nominal targets: 0.950 / 0.500")
