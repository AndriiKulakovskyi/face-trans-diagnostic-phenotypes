"""V3 Phase A+B+C — measurement-eligibility & data-contract audit.

Turns the 10 candidate dimensions (configs/candidate_dimensions_v3.yaml) into a FACE-grounded
data contract, the GATE for all V3 modeling (docs/V3_PLAN.md, docs/PIPELINE.md §1):

  • per-cohort OBSERVED coverage at V0 for every usable variable (no imputation)
  • likelihood family per variable  (dtype -> Gaussian / lognormal / ordered-logit / Bernoulli / NB)
  • a heuristic missingness taxonomy (structural / clinical-skip / design-or-informative / sporadic)
  • per-dimension eligibility verdict  (core / extension / module / proxy / unsupported)
  • a soft prior loading matrix (primary / cross-loading) for the Bayesian model's priors

Aggregate outputs only (coverage %, never patient rows):
  results/reports/v3_eligibility/dimension_eligibility.md
  results/reports/v3_eligibility/variable_audit.csv
  configs/likelihood_map_v3.yaml
  configs/soft_loading_priors_v3.csv

Run:  python3 scripts/v3_eligibility_audit.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from trans_diag import build_unified_dataframe, load_variables, to_harmonized_dataset  # noqa: E402

COH = ["bp", "sz", "dr"]
DICT = ROOT / "data" / "face-common-vars.xlsx"
CFG = ROOT / "configs" / "candidate_dimensions_v3.yaml"
OUT_REP = ROOT / "results" / "reports" / "v3_eligibility"
OUT_CFG = ROOT / "configs"

# ---- likelihood derivation (auto; review per-variable) -----------------------
SKEWED = {  # right-skewed labs -> lognormal / Student-t after log
    "crp", "trig", "ggt_lbstresc", "alt_lbstresc", "ast_lbstresc", "alp_lbstresc", "bili_lbstresc",
    "prolctn", "gluc", "hba1c", "urate", "iron_lbstresc", "creat_lbstresc",
    "wbc", "neut", "eos", "baso_lbstresc", "mono_lbstresc", "lym_lbstresc", "plat",
}
COUNT = {"nboccur_hospitalisation_lt", "sudose_cigarettes_lt", "isf09a"}
NOMINAL = {"naisstyp", "jobclas", "stprof"}   # categorical but NOT ordered -> review


def likelihood_of(name: str, dtype: str, section: str) -> str:
    d = (dtype or "").lower()
    if "date" in d:
        return "exclude"
    if "binary" in d:
        return "bernoulli"
    if "ordinal" in d:
        return "ordered_logistic"
    if "categor" in d:
        return "categorical_nominal:REVIEW" if name in NOMINAL else "ordered_logistic"
    if "string" in d:
        return "lognormal:parse" if name in SKEWED else "gaussian:parse"
    # numeric / float
    if name in COUNT:
        return "neg_binomial"
    if name in SKEWED or section == "BILAN BIOLOGIQUE":
        return "lognormal" if name in SKEWED else "gaussian"   # most non-skewed labs ~Gaussian
    return "gaussian"


def missingness_of(section: str, design_present: dict, obs: dict) -> str:
    absent = [c for c in COH if not design_present[c]]
    if absent:
        return f"structural(absent:{'/'.join(absent)})"
    if section == "SUICIDE":
        return "clinical_skip"
    present = [obs[c] for c in COH if design_present[c]]
    return "design_or_informative" if (present and min(present) < 0.5) else "sporadic"


def resolve(spec: dict, feats: set, sec_of: dict) -> set:
    if not spec:
        return set()
    out: set = set()
    for n in spec.get("names", []) or []:
        if n in feats:
            out.add(n)
    for s in spec.get("sections", []) or []:
        out |= {f for f in feats if sec_of.get(f) == s}
    rgx = spec.get("name_regex")
    if rgx:
        out |= {f for f in feats if pd.Series([f]).str.contains(rgx, case=False, regex=True).iloc[0]}
    for n in spec.get("exclude", []) or []:
        out.discard(n)
    return out


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text())
    variables = load_variables(str(DICT))
    by_name = {v.canonical_name: v for v in variables}

    df = build_unified_dataframe("data", str(DICT), readiness=["READY", "PARTIAL"], format="long")
    ds = to_harmonized_dataset(df, variables, visit="V0", normalize=False, apply_skip_logic=True)
    X = ds.X
    feats = list(X.columns)
    cc = X.index.get_level_values("cohort").value_counts().to_dict()
    sec_of = {f: (by_name[f].section if f in by_name else None) for f in feats}

    # per-cohort observed coverage
    cov = X.notna().groupby(level="cohort").mean().T.reindex(columns=COH)  # feature x cohort

    covset = set(cfg.get("covariates", []))
    rows = []
    for f in feats:
        v = by_name.get(f)
        dtype = (v.dtype if v else "")
        design = {c: bool(getattr(v, f"{c}_csv_col")) if v else False for c in COH}
        obs = {c: round(float(cov.loc[f, c]) if (f in cov.index and pd.notna(cov.loc[f, c])) else 0.0, 3) for c in COH}
        rows.append({
            "variable": f, "section": sec_of.get(f), "dtype": dtype,
            "likelihood": likelihood_of(f, dtype, sec_of.get(f) or ""),
            "design_cohorts": "".join(c for c in COH if design[c]).upper(),
            "obs_bp": obs["bp"], "obs_sz": obs["sz"], "obs_dr": obs["dr"],
            "missingness": missingness_of(sec_of.get(f) or "", design, obs),
            "role": "covariate" if f in covset else "",
        })
    audit = pd.DataFrame(rows).set_index("variable")

    # ---- per-dimension rollup + verdict --------------------------------------
    featset = set(feats)
    USABLE = 0.30
    dim_rows, prior_rows, dim_members = [], [], {}
    for d in cfg["dimensions"]:
        key, role = d["key"], d["role"]
        members = resolve(d.get("indicators", {}), featset, sec_of)
        dim_members[key] = members
        # per-cohort: usable if >=1 designed indicator observed >= USABLE
        usable = {}
        for c in COH:
            ok = any((audit.loc[m, f"obs_{c}"] >= USABLE) and (c.upper() in audit.loc[m, "design_cohorts"])
                     for m in members) if members else False
            usable[c] = ok
        # median coverage over indicators actually DESIGNED for that cohort (so a 2-cohort
        # peripheral absent in a cohort does not drag the median to 0 — "·" already flags absence)
        def med_cov(c: str) -> float:
            mm = [m for m in members if c.upper() in audit.loc[m, "design_cohorts"]]
            return round(float(audit.loc[mm, f"obs_{c}"].median()), 2) if mm else 0.0
        med = {c: med_cov(c) for c in COH}
        n_ok = sum(usable.values())
        if not members:
            verdict = "unsupported (0 indicators)"
        elif role in ("proxy_only", "unsupported", "covariate", "historical"):
            verdict = f"{role} ({n_ok}-cohort usable)"
        else:
            verdict = {3: "core (3-cohort)", 2: "extension (2-cohort)", 1: "module (1-cohort)", 0: "indicators present but <30% cov"}[n_ok]
        dim_rows.append({
            "dimension": key, "curated_role": role, "n_indicators": len(members),
            "usable_BP": "✓" if usable["bp"] else "·", "usable_SZ": "✓" if usable["sz"] else "·",
            "usable_DR": "✓" if usable["dr"] else "·",
            "median_cov_BP": med["bp"], "median_cov_SZ": med["sz"], "median_cov_DR": med["dr"],
            "data_verdict": verdict,
        })
        # soft priors: primary on own dimension, cross on listed cross_loading dims
        for m in members:
            prior_rows.append({"variable": m, "dimension": key, "status": "primary", "prior_mean": 0.6, "prior_sd": 0.3})
        for xl in d.get("cross_loading", []) or []:
            for m in members:
                prior_rows.append({"variable": m, "dimension": xl, "status": "cross_loading", "prior_mean": 0.0, "prior_sd": 0.25})
    dim = pd.DataFrame(dim_rows)

    # attach dimension membership to the variable audit
    var2dim = {}
    for k, ms in dim_members.items():
        for m in ms:
            var2dim.setdefault(m, []).append(k)
    audit["dimensions"] = [";".join(var2dim.get(f, [])) for f in audit.index]

    # ---- write artifacts -----------------------------------------------------
    OUT_REP.mkdir(parents=True, exist_ok=True)
    audit.sort_values(["section", "variable"]).to_csv(OUT_REP / "variable_audit.csv")
    pd.DataFrame(prior_rows).to_csv(OUT_CFG / "soft_loading_priors_v3.csv", index=False)
    likmap = {f: audit.loc[f, "likelihood"] for f in audit.index if audit.loc[f, "likelihood"] != "exclude"}
    (OUT_CFG / "likelihood_map_v3.yaml").write_text(
        "# Auto-derived from dtype by scripts/v3_eligibility_audit.py — REVIEW per-variable.\n"
        + yaml.safe_dump({"likelihoods": likmap}, sort_keys=True, allow_unicode=True))

    md = ["# V3 dimension eligibility — data-grounded verdict",
          "",
          f"V0 N = {len(X):,} (BP {cc.get('bp', '?')} · SZ {cc.get('sz', '?')} · "
          f"DR {cc.get('dr', '?')}); {len(feats)} usable features. "
          "`usable` = ≥1 designed indicator observed ≥30% in that cohort. Coverage = median observed fraction over indicators.",
          "",
          dim.to_markdown(index=False),
          "",
          "## Likelihood families (auto, review)",
          audit["likelihood"].value_counts().to_frame("n_variables").to_markdown(),
          "",
          "## Missingness taxonomy (heuristic)",
          audit["missingness"].str.replace(r"\(.*\)", "", regex=True).value_counts().to_frame("n_variables").to_markdown(),
          "",
          "Artifacts: `configs/likelihood_map_v3.yaml`, `configs/soft_loading_priors_v3.csv`, "
          "`results/reports/v3_eligibility/variable_audit.csv`.",
          ]
    (OUT_REP / "dimension_eligibility.md").write_text("\n".join(md))

    # ---- console summary -----------------------------------------------------
    print(f"\nV0 N={len(X):,}  features={len(feats)}\n")
    print(dim.to_string(index=False))
    print("\nlikelihood families:", dict(audit["likelihood"].value_counts()))
    print("\nwrote:")
    for p in [OUT_REP / "dimension_eligibility.md", OUT_REP / "variable_audit.csv",
              OUT_CFG / "likelihood_map_v3.yaml", OUT_CFG / "soft_loading_priors_v3.csv"]:
        print("  ", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
