# Archetype-simplex reconstruction fidelity — summary

The 5-corner convex blend (each patient x̂ᵢ = Σₐ wᵢₐ zₐ, weights from `arch_w0..4`, corner
profiles from the `A_all9` arm) reconstructs the 8-dim FACE-ATLAS map coordinates with a pooled
**R² = 0.590** (variance-weighted across axes; per-patient cross-axis error ≈ 0.64 SD), i.e. the
five interpretable archetypes retain ~59% of the total coordinate variance in a convex, simplex form.
This sits below the 5-PC linear optimum (**PCA5 R² = 0.796**, a 20.6-pp fidelity gap; the fair
same-affine-dimension baseline PCA4 = 0.680, a 9.1-pp gap) but far above a 5-centroid hard partition
(**k-means5 R² = 0.325**) — the price of interpretability + convexity over a free linear projection
is modest, and the blend vastly outperforms discrete clustering into 5 groups.
Axes are summarized unevenly: **immunometabolic** (R²=0.89) and **overall_severity** (R²=0.86)
reconstruct best because the corners are strongly separated along them, whereas **substance**
(R²≈0, RMSE 0.52 SD) and **cognition** (R²=0.06) are nearly unrecoverable — the five corner profiles
are almost flat on these axes (substance range across corners only ≈ −0.12…+0.11), so the blend
encodes little of their variation.
Counter to the naïve expectation, high-entropy "fog" patients do **not** reconstruct worse:
reconstruction error is weakly *negatively* correlated with archetype-weight entropy
(Pearson r = -0.134, Spearman ρ = -0.150; both p < 1e-30, n=9013) — blended interior
patients lie inside the convex hull of the corners, while patients pinned near a single corner include
the hull-exceeding extremes the corner profile undershoots. Most of the cohort sits at entropy 0.6–1.0
(of max ln5 = 1.609), consistent with a continuum-like population that the 5 corners summarize as a
smooth blend rather than discrete types.
