"""
T0.1 (blocker) — Does carrying the posterior (EAP coordinate + uncertainty S_i) buy
downstream prognostic value over a naive point estimate a reader already has on their desk?

Two complementary demonstrations, both on the fitted model:

(A) REAL-DATA SELECTIVE PREDICTION  [headline — uses real remission outcomes]
    The instrument gives every patient a posterior uncertainty (mean per-axis EAP SD).
    A logistic model predicts real remission (egf__remission_V2) from the 8 EAP coordinates.
    Sweeping a confidence threshold (keep the most-certain fraction) shows out-of-sample AUC
    RISE on the confident subset. A sum-score point estimate has NO such uncertainty and cannot
    draw this curve — the value is the triage itself. We contrast against a variance-free
    baseline (keep-by-total-symptom-load) to prove the gain is from *uncertainty*, not severity.

(B) SIMULATION HEAD-TO-HEAD  [known ground truth, isolates the estimator]
    Generate item responses from the fitted GLLVM at controlled sparsity; the true latent
    coordinate drives a known logistic outcome. Compare, at matched sparsity:
      - EAP  : Fisher-scoring posterior mean coordinate (the instrument)
      - SUM  : naive per-axis z-scored mean of observed items (classical test theory)
    on (i) coordinate recovery (corr to truth) and (ii) prognostic AUC. The gap should widen
    as data get sparse, where a point estimate ignores that its inputs are noisier.
"""
import os, json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT="/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr"
AX=['overall_severity','cognition','immunometabolic','sleep','suicidality','developmental_risk','mania_activation','substance']
rng=np.random.default_rng(20260704)

# ============================ (A) REAL-DATA SELECTIVE PREDICTION ============================
d=pd.read_parquet("/tmp/real_prog.parquet")
y=pd.to_numeric(d["egf__remission_V2"],errors="coerce")
have=y.notna().values
Xc=np.column_stack([pd.to_numeric(d[a+"__mean"],errors="coerce").values for a in AX])
SDc=np.column_stack([pd.to_numeric(d[a+"__sd"],errors="coerce").values for a in AX])
Nobs=np.column_stack([pd.to_numeric(d[a+"__n_obs"],errors="coerce").fillna(0).values for a in AX])
X=Xc[have]; SD=SDc[have]; yv=y[have].values.astype(int); NOB=Nobs[have]
# per-patient posterior uncertainty = mean per-axis SD (small = confident)
unc=SD.mean(1)
# variance-free "severity" contrast = overall symptom load (|coords|), to prove the triage
# gain is from UNCERTAINTY not from creaming off easy severe/mild patients
sev=np.abs(X).mean(1)

def oos_pred(Xf,yf,seed=0):
    """5-fold out-of-sample predicted probabilities."""
    p=np.zeros(len(yf)); skf=StratifiedKFold(5,shuffle=True,random_state=seed)
    for tr,te in skf.split(Xf,yf):
        lr=LogisticRegression(max_iter=500,C=1.0).fit(Xf[tr],yf[tr])
        p[te]=lr.predict_proba(Xf[te])[:,1]
    return p

# average OOS probs over repeats for stability
P=np.mean([oos_pred(X,yv,s) for s in range(10)],axis=0)
auc_all=roc_auc_score(yv,P)

fracs=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3]
def retained_auc(order_key, ascending=True):
    idx=np.argsort(order_key)                      # ascending: keep smallest key first
    out=[]
    for f in fracs:
        k=max(50,int(f*len(yv))); keep=idx[:k]
        try: out.append(roc_auc_score(yv[keep],P[keep]))
        except Exception: out.append(np.nan)
    return out

auc_by_unc=retained_auc(unc)        # keep most CONFIDENT (lowest uncertainty)
auc_by_sev=retained_auc(-sev)       # keep most severe (variance-free contrast)
# also random-triage baseline
auc_by_rand=np.mean([[roc_auc_score(*(lambda kk:(yv[kk],P[kk]))(rng.permutation(len(yv))[:max(50,int(f*len(yv)))]))
                      for f in fracs] for _ in range(20)],axis=0)

A_res={
 "endpoint":"egf__remission_V2","n":int(len(yv)),"base_rate":float(yv.mean()),
 "auc_all_patients":float(auc_all),
 "fracs":fracs,
 "auc_retained_by_confidence":[float(x) for x in auc_by_unc],
 "auc_retained_by_severity":[float(x) for x in auc_by_sev],
 "auc_retained_random":[float(x) for x in auc_by_rand],
 "note":"confidence = low mean per-axis posterior SD; a point estimate cannot produce this triage axis at all.",
}

# ============================ (B) SIMULATION HEAD-TO-HEAD ============================
A=np.load("/tmp/face_arrays.npz",allow_pickle=True)
lam=A["lam"]; alpha=A["alpha"]; sigma=A["sigma"]; count_alpha=A["count_alpha"]
fam=A["fam_int"]; cuts=A["cuts_pad"]; ncut=A["ncut"]; prim=A["prim"]; Phi=A["Phi"]
J=lam.shape[0]; K=8
L=np.linalg.cholesky(Phi+1e-9*np.eye(K))
Phi_inv=np.linalg.inv(Phi)
psi=sigma**2

def gen_eta(n): return (rng.standard_normal((n,K))@L.T)   # ~ N(0,Phi)

def gen_items(eta):
    """Sample full item responses given latent eta (n,K). Returns continuous 'y' matrix (n,J)
    on the model's link scale for gaussian/copula; class index for discrete used only via sumscore
    surrogate. For scoring we use the Gaussian/latent responses (the copula continuous block)."""
    n=len(eta); lin=eta@lam.T + alpha[None,:]        # (n,J) linear predictor
    Y=np.full((n,J),np.nan)
    # gaussian: y = lin + noise
    g=(fam==0); Y[:,g]=lin[:,g]+rng.standard_normal((n,g.sum()))*sigma[None,g]
    # bernoulli: latent logit -> we store the model's expected latent (probit-like) as continuous
    b=(fam==1); Y[:,b]=lin[:,b]+rng.standard_normal((n,b.sum()))    # underlying continuous
    # ordinal & count: treat underlying continuous latent + noise (sum-score uses these)
    o=(fam==2); Y[:,o]=lin[:,o]+rng.standard_normal((n,o.sum()))
    c=(fam==3); Y[:,c]=lin[:,c]+rng.standard_normal((n,c.sum()))
    return Y

def eap_coord(Yobs, mask):
    """Gaussian-EAP posterior mean per patient given observed items (Woodbury).
    Treats all observed items as gaussian-linear in eta (the copula continuous approximation),
    which is exactly the paper's projection for the continuous block."""
    n=len(Yobs); out=np.zeros((n,K))
    for i in range(n):
        obs=mask[i]
        if obs.sum()<1: continue
        Lc=lam[obs]; yc=Yobs[i,obs]-alpha[obs]; Pinv=1.0/psi[obs]
        cap=Phi_inv + (Lc.T*Pinv)@Lc
        rhs=(Lc.T*Pinv)@yc
        out[i]=np.linalg.solve(cap,rhs)             # posterior mean (prior mean 0)
    return out

def sumscore_coord(Yobs, mask):
    """Naive per-axis point estimate: z-scored mean of observed items whose primary axis = k."""
    n=len(Yobs); out=np.zeros((n,K))
    for k in range(K):
        cols=np.where(prim==k)[0]
        if len(cols)==0: continue
        for i in range(n):
            oc=cols[mask[i,cols]]
            if len(oc)==0: out[i,k]=0.0
            else:
                # orient each item by loading sign so axis is coherent
                s=np.sign(lam[oc,k]); s[s==0]=1
                out[i,k]=np.mean((Yobs[i,oc]-alpha[oc])*s)
    # standardize columns
    mu=out.mean(0); sd=out.std(0); sd[sd<1e-8]=1
    return (out-mu)/sd

# known outcome rule: remission driven by a fixed weight vector on TRUE latent + noise
w_true=np.array([-0.9,-0.3,-0.8,-0.2,-0.5,-0.2,0.1,-0.1])   # higher burden -> lower remission
def outcome(eta):
    lp=eta@w_true; p=1/(1+np.exp(-lp)); return (rng.random(len(eta))<p).astype(int), p

sparsities=[("full",1.0),("moderate",0.5),("sparse",0.25),("very_sparse",0.12)]
Ntr,Nte=3000,3000
B_rows=[]
for name,keep_frac in sparsities:
    eta_tr=gen_eta(Ntr); eta_te=gen_eta(Nte)
    ytr,_=outcome(eta_tr); yte,pte=outcome(eta_te)
    def mkmask(n): return rng.random((n,J))<keep_frac
    Ytr=gen_items(eta_tr); Yte=gen_items(eta_te); mtr=mkmask(Ntr); mte=mkmask(Nte)
    for est,fn in [("EAP",eap_coord),("SUM",sumscore_coord)]:
        Ctr=fn(Ytr,mtr); Cte=fn(Yte,mte)
        # coordinate recovery (mean per-axis corr to truth, on test)
        recov=np.nanmean([np.corrcoef(Cte[:,k],eta_te[:,k])[0,1] for k in range(K)])
        # prognostic AUC: logistic trained on train coords, eval on test
        lr=LogisticRegression(max_iter=500).fit(Ctr,ytr)
        auc=roc_auc_score(yte,lr.predict_proba(Cte)[:,1])
        B_rows.append(dict(sparsity=name,keep_frac=keep_frac,estimator=est,
                           coord_recovery=float(recov),prognostic_auc=float(auc)))

B_df=pd.DataFrame(B_rows)
# oracle AUC (true eta)
orac={}
for name,kf in sparsities:
    eta_te=gen_eta(Nte); yte,_=outcome(eta_te)
    lr=LogisticRegression(max_iter=500).fit(gen_eta(Ntr),None) if False else None
# oracle on true latent
eta_o=gen_eta(6000); yo,_=outcome(eta_o)
lr_o=LogisticRegression(max_iter=500).fit(eta_o[:3000],yo[:3000])
auc_oracle=roc_auc_score(yo[3000:],lr_o.predict_proba(eta_o[3000:])[:,1])

out={"A_selective_prediction":A_res,
     "B_simulation":B_rows,
     "B_oracle_auc_true_latent":float(auc_oracle),
     "B_true_weights":w_true.tolist(),
     "sim_N_train":Ntr,"sim_N_test":Nte}
json.dump(out,open(f"{ROOT}/article_methods/analysis/blocker_T01_prognosis.json","w"),indent=2)

print("="*68)
print("(A) REAL-DATA SELECTIVE PREDICTION  (endpoint egf__remission_V2, n=%d, base rate %.3f)"%(len(yv),yv.mean()))
print(f"  AUC all patients: {auc_all:.3f}")
print(f"  {'keep frac':>9} {'by CONFIDENCE':>14} {'by severity':>12} {'random':>8}")
for i,f in enumerate(fracs):
    print(f"  {f:>9.0%} {auc_by_unc[i]:>14.3f} {auc_by_sev[i]:>12.3f} {auc_by_rand[i]:>8.3f}")
print("\n"+"="*68)
print("(B) SIMULATION HEAD-TO-HEAD  (oracle AUC on true latent = %.3f)"%auc_oracle)
print(f"  {'sparsity':>12} {'estimator':>9} {'coord_recov':>12} {'prog_AUC':>9}")
for r in B_rows:
    print(f"  {r['sparsity']:>12} {r['estimator']:>9} {r['coord_recovery']:>12.3f} {r['prognostic_auc']:>9.3f}")
# gap summary
print("\n  EAP - SUM prognostic AUC gap by sparsity:")
for name,kf in sparsities:
    e=[r for r in B_rows if r['sparsity']==name and r['estimator']=='EAP'][0]['prognostic_auc']
    s=[r for r in B_rows if r['sparsity']==name and r['estimator']=='SUM'][0]['prognostic_auc']
    print(f"    {name:>12}: EAP {e:.3f}  SUM {s:.3f}  gap {e-s:+.3f}")
