"""
T0.1 robustness — is the EAP-over-sumscore prognostic gap (simulation B) stable across
random draws and an alternative outcome rule? Repeat full & sparse regimes over 12 seeds.
"""
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT="/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr"
A=np.load("/tmp/face_arrays.npz",allow_pickle=True)
lam=A["lam"]; alpha=A["alpha"]; sigma=A["sigma"]; fam=A["fam_int"]; prim=A["prim"]; Phi=A["Phi"]
J=lam.shape[0]; K=8; L=np.linalg.cholesky(Phi+1e-9*np.eye(K)); Phi_inv=np.linalg.inv(Phi); psi=sigma**2

def gen_eta(n,rng): return rng.standard_normal((n,K))@L.T
def gen_items(eta,rng):
    lin=eta@lam.T+alpha[None,:]; n=len(eta); Y=np.empty((n,J))
    for code in (0,1,2,3):
        m=(fam==code)
        if code==0: Y[:,m]=lin[:,m]+rng.standard_normal((n,m.sum()))*sigma[None,m]
        else: Y[:,m]=lin[:,m]+rng.standard_normal((n,m.sum()))
    return Y
def eap(Y,mask):
    n=len(Y); out=np.zeros((n,K))
    for i in range(n):
        obs=mask[i]
        if obs.sum()<1: continue
        Lc=lam[obs]; Pinv=1.0/psi[obs]; cap=Phi_inv+(Lc.T*Pinv)@Lc
        out[i]=np.linalg.solve(cap,(Lc.T*Pinv)@(Y[i,obs]-alpha[obs]))
    return out
def summ(Y,mask):
    n=len(Y); out=np.zeros((n,K))
    for k in range(K):
        cols=np.where(prim==k)[0]
        if not len(cols): continue
        for i in range(n):
            oc=cols[mask[i,cols]]
            if len(oc): s=np.sign(lam[oc,k]); s[s==0]=1; out[i,k]=np.mean((Y[i,oc]-alpha[oc])*s)
    mu=out.mean(0); sd=out.std(0); sd[sd<1e-8]=1; return (out-mu)/sd

w_true=np.array([-0.9,-0.3,-0.8,-0.2,-0.5,-0.2,0.1,-0.1])
def run(keep_frac,seed,wvec):
    rng=np.random.default_rng(seed)
    et=gen_eta(3000,rng); ee=gen_eta(3000,rng)
    yt=(rng.random(3000)<1/(1+np.exp(-et@wvec))).astype(int)
    ye=(rng.random(3000)<1/(1+np.exp(-ee@wvec))).astype(int)
    mt=rng.random((3000,J))<keep_frac; me=rng.random((3000,J))<keep_frac
    Yt=gen_items(et,rng); Ye=gen_items(ee,rng)
    res={}
    for est,fn in [("EAP",eap),("SUM",summ)]:
        Ct=fn(Yt,mt); Ce=fn(Ye,me)
        lr=LogisticRegression(max_iter=500).fit(Ct,yt)
        res[est]=roc_auc_score(ye,lr.predict_proba(Ce)[:,1])
    return res["EAP"]-res["SUM"]

seeds=range(100,112)
out={}
for reg,kf in [("full",1.0),("sparse",0.25)]:
    gaps=[run(kf,s,w_true) for s in seeds]
    out[reg]={"mean_gap":float(np.mean(gaps)),"sd":float(np.std(gaps)),
              "min":float(np.min(gaps)),"max":float(np.max(gaps)),"n_seeds":len(gaps),
              "frac_EAP_wins":float(np.mean(np.array(gaps)>0))}
# alternative outcome weights
w_alt=np.array([-0.5,-0.6,-0.3,-0.5,-0.4,-0.4,-0.3,-0.2])
gaps_alt=[run(0.25,s,w_alt) for s in seeds]
out["sparse_alt_weights"]={"mean_gap":float(np.mean(gaps_alt)),"sd":float(np.std(gaps_alt)),
                           "frac_EAP_wins":float(np.mean(np.array(gaps_alt)>0))}
json.dump(out,open(f"{ROOT}/article_methods/analysis/blocker_T01_robustness.json","w"),indent=2)
for reg in out:
    r=out[reg]; print(f"{reg:>20}: EAP-SUM gap {r['mean_gap']:+.3f} ± {r['sd']:.3f}  (EAP wins {r['frac_EAP_wins']:.0%} of {out.get('full',{}).get('n_seeds',len(list(seeds)))} seeds)")
