"""Explainer: PCA directions vs archetype corners on the 8-axis coordinate cloud.
Regenerates figures/explain_pca_vs_archetype.{png,pdf} from /tmp/pca_vs_arch.npz
(PCA loadings V[5,8], archetype corners Z[5,8], per-component variance frac[8]).
Two heatmaps: (A) PCA loadings as signed axis-mixing contrasts; (B) archetype
corners as nameable patient poles in axis-SD units.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

d = np.load("/tmp/pca_vs_arch.npz", allow_pickle=True)
V, Z, frac = d["V"], d["Z"], d["frac"]
short = list(d["short"])                      # 8 axis short names
pcs   = [f"PC{i+1}\n{frac[i]*100:.0f}%" for i in range(5)]
corners = ["C1\nsleep/mania\npole", "C2\nsevere,\nlow-biology", "C3\nimmuno-\nmetabolic",
           "C4\ndevelop-\nmental", "C5\nlow-burden\npole"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 5.4))

# Panel A — PCA loadings (8 axes x 5 PCs), signed contrasts
A = V.T                                        # 8 x 5
imA = axA.imshow(A, cmap="RdBu_r", norm=TwoSlopeNorm(0, -0.9, 0.9), aspect="auto")
axA.set_xticks(range(5)); axA.set_xticklabels(pcs, fontsize=8)
axA.set_yticks(range(8)); axA.set_yticklabels(short, fontsize=9)
axA.set_title("PCA \u2014 5 directions of maximal variance\n(each column is a signed contrast)", fontsize=9.5)
axA.set_xlabel("principal component (%variance)", fontsize=9)
for i in range(8):
    for j in range(5):
        v=A[i,j]
        if abs(v)>=0.30: axA.text(j,i,f"{v:+.1f}",ha="center",va="center",fontsize=7,
                                  color="white" if abs(v)>0.6 else "#222")
cbA=fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04); cbA.ax.tick_params(labelsize=7)
cbA.set_label("loading (unit-norm direction)", fontsize=8)
axA.text(-0.14,1.06,"a",transform=axA.transAxes,fontsize=13,fontweight="bold")

# Panel B — archetype corners (8 axes x 5 corners), profiles in SD units
B = Z.T                                        # 8 x 5
imB = axB.imshow(B, cmap="RdBu_r", norm=TwoSlopeNorm(0, -3.5, 3.5), aspect="auto")
axB.set_xticks(range(5)); axB.set_xticklabels(corners, fontsize=7.5)
axB.set_yticks(range(8)); axB.set_yticklabels(short, fontsize=9)
axB.set_title("Archetypes \u2014 5 extreme profiles\n(each column is a nameable patient pole)", fontsize=9.5)
axB.set_xlabel("archetype corner (clinical label)", fontsize=9)
for i in range(8):
    for j in range(5):
        v=B[i,j]
        if abs(v)>=1.0: axB.text(j,i,f"{v:+.1f}",ha="center",va="center",fontsize=7,
                                 color="white" if abs(v)>2.2 else "#222")
cbB=fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04); cbB.ax.tick_params(labelsize=7)
cbB.set_label("profile value (axis SD units)", fontsize=8)
axB.text(-0.14,1.06,"b",transform=axB.transAxes,fontsize=13,fontweight="bold")

fig.tight_layout()
BASE="/Users/andriikulakovskyi/Desktop/face-common-bp-sz-dr/article_methods/figures"
fig.savefig(f"{BASE}/explain_pca_vs_archetype.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{BASE}/explain_pca_vs_archetype.pdf", bbox_inches="tight")
plt.close(fig)
print("wrote explain_pca_vs_archetype.png + .pdf")
