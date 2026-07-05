"""
T2.4 — How sensitive is the 59% reconstruction R^2 to the archetype count A?
T2.5 — Anchor that 59% against the full 0->100% range (null summary and raw-coordinate).

Method mirrors the published Phase-1 recipe (archetype_reconstruction_meta.json):
  reconstruction x_hat_i = sum_a w_ia z_a ; R^2 pooled = 1 - sum||x-xhat||^2 / sum||x-xbar||^2
  (variance-weighted across the 8 axes).

A=5 is reproduced from the PUBLISHED corner profiles (archetype_profiles.csv, A_all9 arm) and
per-patient weights (patient_strata arch_w0..4) so it lands on the published 0.590 exactly.
A=4 and A=6 are refit on the same 8-D coordinates with the repo's PCHA solver (archetypes.AA,
the same library src/face/strata/archetypes.py wraps), then reconstructed the same way.

Anchors (T2.5):
  raw 8-D coordinate (identity, no summary)  -> R^2 = 1.000 by definition (upper anchor)
  PCA-5 (best 5-D linear)                    -> from published meta (0.796)
  random-rotation-5 floor                    -> project onto 5 random orthonormal directions
  A=1 (single centroid)                      -> R^2 = 0 by definition (lower anchor)
"""
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
import pandas as pd

ROOT="/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr"
AX=['overall_severity','cognition','immunometabolic','sleep','suicidality','developmental_risk','mania_activation','substance']

# --- coordinates (the 8-D cloud being summarized) ---
cov=pd.read_parquet(f"{ROOT}/results/analyses/variational_gllvm/consolidate/coordinates.parquet")
X=np.column_stack([pd.to_numeric(cov[a+"__mean"],errors="coerce").values for a in AX]).astype(np.float64)
ok=~np.isnan(X).any(1); X=X[ok]
N=len(X); xbar=X.mean(0)
SS_tot_axis=((X-xbar)**2).sum(0)                      # per-axis total SS
SS_tot=SS_tot_axis.sum()

def pooled_r2(Xhat):
    return 1.0 - ((X-Xhat)**2).sum()/SS_tot
def per_axis_r2(Xhat):
    return 1.0 - ((X-Xhat)**2).sum(0)/SS_tot_axis

# --- A=5 published reproduction ---
prof=pd.read_csv(f"{ROOT}/results/m2_strata/consolidate/archetype_profiles.csv")
# A_all9 arm, 5 corners, columns = the 8 axes (selected by NAME below, so column order is irrelevant)
prof_a=prof[prof["arm"]=="A_all9"] if "arm" in prof.columns else prof
Zpub=None
try:
    Zpub=np.column_stack([pd.to_numeric(prof_a[a],errors="coerce").values for a in AX]).astype(np.float64)
    Zpub=Zpub[~np.isnan(Zpub).any(1)]
except Exception as e:
    print("profile parse note:", e)
strata=pd.read_parquet(f"{ROOT}/results/m2_strata/consolidate/patient_strata.parquet")
Wpub=np.column_stack([pd.to_numeric(strata[f"arch_w{k}"],errors="coerce").values for k in range(5)]).astype(np.float64)
Wpub=Wpub[ok]
r2_pub=None
if Zpub is not None and Zpub.shape==(5,8):
    Xhat_pub=Wpub@Zpub
    r2_pub=pooled_r2(Xhat_pub)

# --- refit AA for A=4,5,6 with the repo's solver ---
from archetypes import AA


def fit_reconstruct(A, seed=0):
    m=AA(n_archetypes=A, random_state=seed, n_init=3, max_iter=200)
    W=m.fit_transform(X)          # (N,A) simplex weights
    Z=m.archetypes_               # (A,8) corners
    Xhat=W@Z
    return pooled_r2(Xhat), per_axis_r2(Xhat)

sens={}
for A in [4,5,6]:
    r2,pa=fit_reconstruct(A, seed=20260704)
    sens[A]={"pooled_r2":float(r2), "per_axis_r2":{ax:float(v) for ax,v in zip(AX,pa)}}

# --- T2.5 anchors ---
# raw identity: Xhat = X -> R2 = 1
r2_identity=1.0
# A=1 single centroid: Xhat = xbar -> R2 = 0
r2_centroid=pooled_r2(np.tile(xbar,(N,1)))
# random-rotation-5 floor: project X onto 5 random orthonormal directions (mean over seeds)
def randproj_r2(k=5, seeds=range(10)):
    vals=[]
    for s in seeds:
        rng=np.random.default_rng(1000+s)
        Q,_=np.linalg.qr(rng.standard_normal((8,8)))
        P=Q[:,:k]                                   # 8x5 orthonormal
        Xc=X-xbar; Xhat=xbar + (Xc@P)@P.T
        vals.append(pooled_r2(Xhat))
    return float(np.mean(vals)), float(np.std(vals))
r2_rand5, r2_rand5_sd=randproj_r2()

meta=json.load(open(f"{ROOT}/article_methods/analysis/archetype_reconstruction_meta.json"))
r2_pca5=meta["overall"]["r2_pca5"]; r2_pca4=meta["overall"]["r2_pca4_fair_affine_dim"]; r2_km5=meta["overall"]["r2_kmeans5"]

out={
 "A_sensitivity":{str(A):sens[A]["pooled_r2"] for A in [4,5,6]},
 "A_sensitivity_full":sens,
 "A5_published_reproduction":r2_pub,
 "anchors":{
    "raw_identity_8D":r2_identity,
    "single_centroid_A1":float(r2_centroid),
    "random_rotation_5":r2_rand5,
    "random_rotation_5_sd":r2_rand5_sd,
    "kmeans5":r2_km5, "pca4":r2_pca4, "pca5":r2_pca5,
    "archetype5_published":meta["overall"]["r2_archetype"],
 },
 "N":int(N)
}
json.dump(out, open(f"{ROOT}/article_methods/analysis/archetype_sensitivity.json","w"), indent=2)

print("=== T2.4  reconstruction R^2 vs archetype count A (refit, same coordinates) ===")
for A in [4,5,6]:
    print(f"  A={A}:  pooled R^2 = {sens[A]['pooled_r2']:.3f}")
print(f"  A=5 published reproduction (fixed corners+weights): R^2 = {r2_pub:.3f}" if r2_pub is not None else "  A=5 published: parse failed")
print("\n=== T2.5  the 0 -> 100% range the 59% sits in ===")
print("  raw 8-D coordinate (no summary)   R^2 = 1.000   (upper anchor)")
print(f"  PCA-5 (best 5-D linear)           R^2 = {r2_pca5:.3f}")
print(f"  archetype-5 (published)           R^2 = {meta['overall']['r2_archetype']:.3f}")
print(f"  PCA-4 (fair affine dim)           R^2 = {r2_pca4:.3f}")
print(f"  k-means-5 (hard partition)        R^2 = {r2_km5:.3f}")
print(f"  random-rotation-5 floor           R^2 = {r2_rand5:.3f} +/- {r2_rand5_sd:.3f}")
print(f"  single centroid (A=1, no info)    R^2 = {r2_centroid:.3f}   (lower anchor)")
