"""
T0.3 — Does information-ranked item selection beat loading-ranked at matched battery size?

Reuses the EXACT Fisher-information accounting of fig4_voi.py (item_info_on_factor +
greedy prec[axis]+=info, mean_rel=mean(1-1/prec)). The ONLY change is the key by which
the cross-axis item pool is ranked:
  - INFO rule  (the paper's): rank all (item,axis) selections by Fisher information desc.
  - LOADING rule (naive foil): rank by |lambda_{j,axis}| desc.
Both build a shared battery over all 8 axes; we compare mean reliability at matched N
(20/27/35) and report the reliability gap. If loading-ranked ties, Fig 2's thesis is weak;
if it loses, that gap is the demonstrated added value of the information-based rule.
"""
import os, json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, pandas as pd, torch, torch.nn.functional as F

ROOT="/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr"
sd=torch.load(f"{ROOT}/results/face/gllvm_oop/s8_full/model_state.pt", map_location="cpu", weights_only=False)
st=sd["state_dict"]; items=sd["items"]; families=sd["families"]; factor_cols=sd["factor_cols"]; J=len(items)

raw_loading=st["raw_loading"]; loading_free=st["loading_free"].bool(); loading_positive=st["loading_positive"].bool()
lam=torch.zeros_like(raw_loading)
pf=loading_free & loading_positive; sf=loading_free & (~loading_positive)
lam=torch.where(pf, F.softplus(raw_loading)+1e-5, lam)
lam=torch.where(sf, raw_loading, lam)
alpha=st["alpha"]; sigma=0.30+F.softplus(st["raw_sigma"]); count_alpha=F.softplus(st["raw_count_alpha"])+1e-3

def item_info_on_factor(j, fidx):
    ljf=float(lam[j,fidx])
    if abs(ljf)<1e-8: return 0.0
    fam=families[j]; a=float(alpha[j])
    if fam=="gaussian": return ljf**2/float(sigma[j])**2
    if fam=="bernoulli":
        p=1/(1+np.exp(-a)); return ljf**2*p*(1-p)
    if fam=="count":
        mu=np.exp(np.clip(a,-10,10)); r=float(count_alpha[j]); return ljf**2*(mu*r/(r+mu))
    if fam=="ordinal":
        ck=f"ordinal_cutpoints.{j}"
        if ck not in st: return 0.0
        raw=st[ck]
        if raw.numel()==1: cuts=raw
        else:
            first=raw[:1]; inc=F.softplus(raw[1:])+1e-3; cuts=torch.cat([first, first+torch.cumsum(inc,0)])
        cuts=cuts.numpy(); s=1/(1+np.exp(-(cuts-0.0))); cdf=np.concatenate([[0.0],s,[1.0]])
        probs=np.clip(np.diff(cdf),1e-8,1); dsk=-s*(1-s); dsk_full=np.concatenate([[0.0],dsk,[0.0]])
        dprob=np.array([dsk_full[k+1]-dsk_full[k] for k in range(len(probs))])
        return ljf**2*float(np.sum(dprob**2/probs))
    return 0.0

factor_map={f:i for i,f in enumerate(factor_cols)}

# Build the candidate pool: every (item,axis) with non-trivial information, carrying BOTH
# its info and its |loading| so the two rules rank the SAME candidate set.
rows=[]
for f,fidx in factor_map.items():
    for j in range(J):
        info=item_info_on_factor(j,fidx)
        if info>1e-6:
            rows.append(dict(item=items[j], axis=f, info=info, absload=abs(float(lam[j,fidx])), family=families[j]))
pool=pd.DataFrame(rows)

def greedy_curve(pool_sorted):
    prec={a:1.0 for a in factor_cols}; curve=[]
    for k,(_,it) in enumerate(pool_sorted.iterrows(), start=1):
        prec[it.axis]+=it["info"]                       # reliability ALWAYS accrues real info
        curve.append(np.mean([1-1/prec[a] for a in factor_cols]))
    return np.array(curve)

pool_info=pool.sort_values("info",ascending=False).reset_index(drop=True)
pool_load=pool.sort_values("absload",ascending=False).reset_index(drop=True)
curve_info=greedy_curve(pool_info)
curve_load=greedy_curve(pool_load)

def at(curve,n): return float(curve[min(n,len(curve))-1])
sizes=[20,27,35]
res={"rule_info_mean_rel":{str(n):at(curve_info,n) for n in sizes},
     "rule_loading_mean_rel":{str(n):at(curve_load,n) for n in sizes},
     "gap_info_minus_loading":{str(n):at(curve_info,n)-at(curve_load,n) for n in sizes},
     "pool_size":int(len(pool)),
     "n_axes":len(factor_cols)}

# What does the loading rule waste its early picks on? (high-loading but low-info items)
top20_load=pool_load.head(20).copy()
top20_load["info_rank"]=top20_load.apply(lambda r: int((pool_info["item"]==r["item"]).idxmax() if (pool_info["item"]==r["item"]).any() else -1), axis=1)
# fraction of loading-rule's first 27 that are low-information (bottom half of info pool)
med_info=pool["info"].median()
frac_lowinfo_first27=float((pool_load.head(27)["info"]<med_info).mean())
res["loading_rule_frac_first27_below_median_info"]=frac_lowinfo_first27
res["curve_info"]=curve_info[:40].tolist()
res["curve_load"]=curve_load[:40].tolist()

json.dump(res, open(f"{ROOT}/article_methods/analysis/battery_loading_vs_info.json","w"), indent=2)
print("=== T0.3  mean reliability at matched battery size (8 axes) ===")
print(f"{'N':>4} {'info-rule':>10} {'loading-rule':>13} {'gap':>8}")
for n in sizes:
    print(f"{n:>4} {at(curve_info,n):>10.4f} {at(curve_load,n):>13.4f} {at(curve_info,n)-at(curve_load,n):>8.4f}")
print(f"\nloading-rule's first 27 picks below-median info: {frac_lowinfo_first27:.0%}")
print(f"pool size {len(pool)} (item,axis) candidates over {len(factor_cols)} axes")
