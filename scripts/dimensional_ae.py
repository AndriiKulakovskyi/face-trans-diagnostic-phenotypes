"""Dimensional trans-diagnostic axes — PyTorch masked autoencoder (the AI companion).

Nonlinear, no-imputation counterpart to scripts/dimensional_axes.py (classical FA).
The autoencoder reconstructs the standardized domain scores with a MASKED loss —
missing entries never contribute, and the observed-mask is fed to the encoder, so
nothing is imputed. The K-dim bottleneck is the learned dimensional representation.

We then ask: do the AE's nonlinear axes agree with the classical factor axes?
(canonical correlations via CCA). High agreement ⇒ the dimensional structure is
robust to the method; the AE may add nonlinear refinement.

Artifacts: results/dimensional_ae_{scores.parquet,meta.json}, reports/dimensional_ae.html.
Run:  python3 scripts/dimensional_ae.py            # K = #classical factors
      python3 scripts/dimensional_ae.py --k 5 --epochs 300
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.cross_decomposition import CCA  # noqa: E402

from face_common import (  # noqa: E402
    ADMINISTRATIVE_FEATURES,
    build_unified_dataframe,
    load_variables,
    to_harmonized_dataset,
)

RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"
SCORES_PATH = RESULTS_DIR / "cluster_domains_scores.parquet"
FA_SCORES = RESULTS_DIR / "dimensional_axes_scores.parquet"
SEED = 0
SPECTRUM = {"Trouble dépressif majeur": 0, "Bipolaire de type 2": 1,
            "Bipolaire de type 1": 2, "Bipolaire non spécifié": 3,
            "Trouble schizo-affectif": 4, "Trouble schizophréniforme": 5,
            "Schizophrénie": 6}


class MaskedAE(nn.Module):
    def __init__(self, d, k, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, k))               # mask-aware input
        self.dec = nn.Sequential(nn.Linear(k, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d))

    def forward(self, x0, mask):
        z = self.enc(torch.cat([x0, mask], dim=1))
        return z, self.dec(z)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--k", type=int, default=None, help="latent dims (default: #classical factors)")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    REPORTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    sc = pd.read_parquet(SCORES_PATH)
    sc.index = pd.MultiIndex.from_arrays(
        [sc.index.get_level_values("cohort").astype(str),
         sc.index.get_level_values("patient_id").astype(str)], names=("cohort", "patient_id"))
    z = (sc - sc.mean()) / sc.std(ddof=0)
    mask = z.notna().to_numpy(np.float32)
    x0 = z.fillna(0.0).to_numpy(np.float32)
    d = x0.shape[1]

    fa = pd.read_parquet(FA_SCORES).reindex(sc.index) if FA_SCORES.exists() else None
    k = args.k or (fa.shape[1] if fa is not None else 5)
    print(f"masked AE: {x0.shape[0]:,} patients × {d} domains → K={k} latent")

    X0 = torch.tensor(x0); M = torch.tensor(mask)
    model = MaskedAE(d, k)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []
    n = len(X0); bs = 512
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, mb = X0[idx], M[idx]
            _, rec = model(xb, mb)
            loss = ((rec - xb) ** 2 * mb).sum() / mb.sum()       # masked MSE, no imputation
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        losses.append(tot / n)
    print(f"  final masked recon MSE = {losses[-1]:.3f}")

    with torch.no_grad():
        latent = model(X0, M)[0].numpy()
    names = [f"ae{i+1}" for i in range(k)]
    pd.DataFrame(latent, columns=names, index=sc.index).to_parquet(
        RESULTS_DIR / "dimensional_ae_scores.parquet")

    # subtype continuum + confound on the AE axes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = build_unified_dataframe(REPO_ROOT / "data", REPO_ROOT / "face-common-vars.xlsx",
                                     readiness=["READY", "PARTIAL"], format="long")
        full = to_harmonized_dataset(df, load_variables(REPO_ROOT / "face-common-vars.xlsx"),
                                     visit="V0", exclude=ADMINISTRATIVE_FEATURES)
    rank = full.metadata.reindex(sc.index)["dsm_diagnosis"].map(SPECTRUM).to_numpy()
    age = full.X.reindex(sc.index)["age"].to_numpy(float)
    sub = pd.DataFrame(latent, columns=names); sub["rank"] = rank
    cent = sub.dropna(subset=["rank"]).groupby("rank").mean()
    cont = [float(spearmanr(cent.index, cent[c]).statistic) for c in names]
    best = int(np.argmax(np.abs(cont)))
    print(f"  mood↔psychosis continuum: best AE axis ae{best+1} |Spearman|={abs(cont[best]):.2f}")
    m = np.isfinite(age)
    conf_age = max(abs(float(np.corrcoef(latent[m, a], age[m])[0, 1])) for a in range(k))
    print(f"  max |corr| AE axis vs age = {conf_age:.3f}")

    # agreement with classical FA axes (canonical correlations)
    can = None
    if fa is not None:
        kc = min(k, fa.shape[1])
        cca = CCA(n_components=kc).fit(latent, fa.to_numpy())
        U, V = cca.transform(latent, fa.to_numpy())
        can = [float(np.corrcoef(U[:, i], V[:, i])[0, 1]) for i in range(kc)]
        print(f"  canonical correlations AE vs classical FA: {[round(c,2) for c in can]} "
              f"(high ⇒ same axes)")

    meta = {"k": k, "epochs": args.epochs, "final_masked_mse": losses[-1],
            "continuum_spearman_per_axis": cont, "mood_axis": f"ae{best+1}",
            "max_confound_corr_age": conf_age,
            "cca_with_classical_fa": can,
            "note": "masked-loss AE → no imputation; mask fed to encoder."}
    (RESULTS_DIR / "dimensional_ae_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    _report(losses, cent, names, cont, best, can)
    print("\nWrote results/dimensional_ae_* + reports/dimensional_ae.html. Done.")
    return 0


def _report(losses, cent, names, cont, best, can):
    spec = {0: "MDD", 1: "BP-II", 2: "BP-I", 3: "BP-NOS", 4: "schizoaff", 5: "schizophrenif", 6: "schizophr"}
    f1 = go.Figure(go.Scatter(y=losses, mode="lines"))
    f1.update_layout(title="Masked-AE training loss", height=300, xaxis_title="epoch",
                     yaxis_title="masked MSE", margin=dict(t=40))
    f2 = go.Figure(go.Scatter(x=[spec.get(int(r), str(r)) for r in cent.index],
                              y=cent[names[best]], mode="lines+markers"))
    f2.update_layout(title=f"DSM subtypes on {names[best]} (|ρ|={abs(cont[best]):.2f})",
                     height=300, xaxis_title="subtype (mood→psychosis)", yaxis_title="AE axis",
                     margin=dict(t=40))
    figs = [pio.to_html(f1, include_plotlyjs="cdn", full_html=False),
            pio.to_html(f2, include_plotlyjs=False, full_html=False)]
    if can is not None:
        f3 = go.Figure(go.Bar(x=[f"cc{i+1}" for i in range(len(can))], y=can))
        f3.update_layout(title="Canonical correlations: AE axes vs classical FA axes",
                         height=300, yaxis_title="correlation", margin=dict(t=40))
        figs.append(pio.to_html(f3, include_plotlyjs=False, full_html=False))
    css = "body{font-family:-apple-system,Segoe UI,sans-serif;margin:0;padding:0 24px 60px}h1{color:#2b3a55}"
    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>",
            "<h1>Trans-diagnostic dimensional axes — masked autoencoder (PyTorch)</h1>"] + figs + ["</body></html>"]
    (REPORTS_DIR / "dimensional_ae.html").write_text("\n".join(html), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
