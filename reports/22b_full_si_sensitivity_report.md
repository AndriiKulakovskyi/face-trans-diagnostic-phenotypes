# 22b — diagonal vs full S_i tessellation sensitivity (P2-04)

The reported tessellation deconvolves the **diagonal** measurement error (marginal SDs). The coherent scorer now exports the **full** per-patient covariance S_i, so we re-run the K=4 XD tessellation under both. If the regions are unchanged, the diagonal approximation is justified.

## Diagonal vs full S_i (K=4)
| metric                                                  | diagonal_S   |      full_S |
|:--------------------------------------------------------|:-------------|------------:|
| BIC                                                     | 199239.9     | 198386      |
| max |component mean shift|                              |              |      0.045  |
| MAP partition ARI (diag vs full)                        |              |      0.914  |
| mean |off-diagonal S_i| (typical cross-dim uncertainty) |              |      0.0181 |

- **Diagonal S_i is a justified approximation:** the MAP partitions agree (ARI 0.914), component means move ≤ 0.045, and the typical off-diagonal coordinate uncertainty is small (0.0181). The reported diagonal-S tessellation stands; the full-S_i arm is available.

- population shares — diagonal [0.31, 0.117, 0.252, 0.321] · full [0.304, 0.117, 0.255, 0.324].
