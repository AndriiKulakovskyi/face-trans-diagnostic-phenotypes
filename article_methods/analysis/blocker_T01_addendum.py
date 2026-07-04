"""
T0.1 addendum — the PRINCIPLED real-data uncertainty test.

The paper's abstract claims the per-patient posterior covariance S_i is propagated downstream.
The direct test of whether that carried uncertainty has decision value: for a prognostic model
with linear-predictor weights w, each patient's predicted-log-odds has variance
    v_i = w^T S_i w   (approximated here as sum_k w_k^2 * SD_ik^2, diag of S_i — only per-axis SD is stored).
Triaging by v_i (keep LOW predictive variance = genuinely confident PREDICTIONS, not just broad
coverage) is the correct selective-prediction axis. If OOS AUC rises on the low-v_i subset, the
carried uncertainty demonstrably improves a real decision; if not, we report the null honestly.

Also runs a precision-weighted logistic (down-weight high-v_i patients in the fit) vs unweighted.
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
d=pd.read_parquet("/tmp/real_prog.parquet")
y=pd.to_numeric(d["egf__remission_V2"],errors="coerce"); have=y.notna().values
X=np.column_stack([pd.to_numeric(d[a+"__mean"],errors="coerce").values for a in AX])[have]
SD=np.column_stack([pd.to_numeric(d[a+"__sd"],errors="coerce").values for a in AX])[have]
yv=y[have].values.astype(int)
Xz=(X-X.mean(0))/X.std(0)

skf_seeds=range(10); fracs=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3]
# OOS probs AND per-patient predictive variance v_i using each fold's fitted weights
P=np.zeros(len(yv)); Vpred=np.zeros(len(yv)); cnt=np.zeros(len(yv))
for seed in skf_seeds:
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=seed).split(Xz,yv):
        lr=LogisticRegression(max_iter=500).fit(Xz[tr],yv[tr])
        w=lr.coef_[0]                                   # weights on standardized coords
        P[te]+=lr.predict_proba(Xz[te])[:,1]
        # predictive log-odds variance from carried SD (standardized units): scale SD by column std
        SDz=SD[te]/X.std(0)[None,:]
        Vpred[te]+=(SDz**2*(w**2)[None,:]).sum(1)
        cnt[te]+=1
P/=cnt; Vpred/=cnt
auc_all=roc_auc_score(yv,P)

def retained(order_key):
    idx=np.argsort(order_key); out=[]
    for f in fracs:
        k=max(50,int(f*len(yv))); keep=idx[:k]
        out.append(roc_auc_score(yv[keep],P[keep]))
    return out
auc_by_vpred=retained(Vpred)      # keep low predictive variance (confident PREDICTIONS)

# precision-weighted vs unweighted logistic (does down-weighting noisy patients help OOS AUC?)
def oos_auc(weighted):
    ps=np.zeros(len(yv))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xz,yv):
        if weighted:
            SDz=SD[tr]/X.std(0)[None,:]; vi=(SDz**2).sum(1); sw=1.0/(vi+np.median(vi))
            lr=LogisticRegression(max_iter=500).fit(Xz[tr],yv[tr],sample_weight=sw)
        else:
            lr=LogisticRegression(max_iter=500).fit(Xz[tr],yv[tr])
        ps[te]=lr.predict_proba(Xz[te])[:,1]
    return roc_auc_score(yv,ps)
auc_unw=np.mean([oos_auc(False) for _ in range(1)]); auc_w=np.mean([oos_auc(True) for _ in range(1)])

res={"endpoint":"egf__remission_V2","n":int(len(yv)),"auc_all":float(auc_all),"fracs":fracs,
     "auc_retained_by_predictive_variance":[float(x) for x in auc_by_vpred],
     "precision_weighted_auc":float(auc_w),"unweighted_auc":float(auc_unw)}
json.dump(res,open(f"{ROOT}/article_methods/analysis/blocker_T01_addendum.json","w"),indent=2)
print("PRINCIPLED selective prediction — triage by predictive log-odds variance v_i = sum w_k^2 SD_ik^2")
print(f"  AUC all: {auc_all:.3f}")
print(f"  {'keep':>6} {'by v_i (confident preds)':>26}")
for i,f in enumerate(fracs): print(f"  {f:>6.0%} {auc_by_vpred[i]:>26.3f}")
print(f"\n  precision-weighted logistic AUC: {auc_w:.3f}  vs unweighted: {auc_unw:.3f}  (gap {auc_w-auc_unw:+.3f})")
