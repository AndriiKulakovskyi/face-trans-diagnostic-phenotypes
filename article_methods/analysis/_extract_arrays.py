"""Extract fitted GLLVM arrays -> /tmp/face_arrays.npz (torch here so the
simulation kernel can stay pure-numpy / OMP-free). Also join the real remission
endpoint to the EAP coordinates -> /tmp/real_prog.parquet."""
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT="/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr"
sd=torch.load(f"{ROOT}/results/face/gllvm_oop/s8_full/model_state.pt", map_location="cpu", weights_only=False)
st=sd["state_dict"]; items=sd["items"]; families=sd["families"]; factor_cols=sd["factor_cols"]; J=len(items)

raw_loading=st["raw_loading"]; lf=st["loading_free"].bool(); lp=st["loading_positive"].bool()
lam=torch.zeros_like(raw_loading)
lam=torch.where(lf & lp, F.softplus(raw_loading)+1e-5, lam)
lam=torch.where(lf & (~lp), raw_loading, lam)
lam=lam.numpy().astype(np.float64)                 # (J,8)
alpha=st["alpha"].numpy().astype(np.float64)       # (J,) intercepts
sigma=(0.30+F.softplus(st["raw_sigma"])).numpy().astype(np.float64)  # (J,)
count_alpha=(F.softplus(st["raw_count_alpha"])+1e-3).numpy().astype(np.float64)  # (J,) NB dispersion r

# ordinal cutpoints -> padded array
famcode=np.array({ "gaussian":0,"bernoulli":1,"ordinal":2,"count":3 }.get,dtype=object)
fam_int=np.array([{"gaussian":0,"bernoulli":1,"ordinal":2,"count":3}[f] for f in families],dtype=np.int64)
maxcut=0; cutlist={}
for j in range(J):
    ck=f"ordinal_cutpoints.{j}"
    if ck in st:
        raw=st[ck]
        if raw.numel()==1: cuts=raw
        else:
            first=raw[:1]; inc=F.softplus(raw[1:])+1e-3; cuts=torch.cat([first, first+torch.cumsum(inc,0)])
        cutlist[j]=cuts.numpy().astype(np.float64); maxcut=max(maxcut,len(cutlist[j]))
cuts_pad=np.full((J,maxcut),np.nan); ncut=np.zeros(J,int)
for j,c in cutlist.items(): cuts_pad[j,:len(c)]=c; ncut[j]=len(c)

# primary axis per item = argmax |lambda|
prim=np.argmax(np.abs(lam),axis=1)

Phi=pd.read_csv(f"{ROOT}/results/face/gllvm_oop/consolidate/phi.csv",index_col=0).values.astype(np.float64)
Phi=0.5*(Phi+Phi.T)

np.savez("/tmp/face_arrays.npz", lam=lam, alpha=alpha, sigma=sigma, count_alpha=count_alpha,
         fam_int=fam_int, cuts_pad=cuts_pad, ncut=ncut, prim=prim, Phi=Phi,
         factor_cols=np.array(factor_cols), items=np.array(items))

# ---- join remission endpoint to EAP coordinates ----
AX=list(factor_cols)
cov=pd.read_parquet(f"{ROOT}/results/face/gllvm_oop/consolidate/coordinates.parquet").reset_index()
prog=pd.read_parquet(f"{ROOT}/results/face/prognosis_oop/consolidate/prognosis_patient_risk.parquet")
keep=["cohort","patient_id","egf__remission_V2","cgi_s__remission_V2","arch_dominant_name","arm"]
m=cov.merge(prog[keep], on=["cohort","patient_id"], how="left")
m.to_parquet("/tmp/real_prog.parquet")
print("families:", {f:int((fam_int==c).sum()) for f,c in [("gaussian",0),("bernoulli",1),("ordinal",2),("count",3)]})
print("J",J,"maxcut",maxcut,"prim axis counts:",{AX[k]:int((prim==k).sum()) for k in range(8)})
print("merged shape", m.shape, "remission non-null", int(m["egf__remission_V2"].notna().sum()))
print("saved /tmp/face_arrays.npz and /tmp/real_prog.parquet")
