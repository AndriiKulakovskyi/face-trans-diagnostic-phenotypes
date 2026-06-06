"""V3 data layer — harmonization + no-imputation loading of the FACE common-variables dictionary.

Self-contained: the harmonized `Variable` model, the per-variable harmonization rules + sanity bounds,
skip-logic structural-zero decoding, and the V0 observed-data matrix builder. NaN = missing, never
imputed. Reused by the V3 pipeline in `scripts/v3/`.

    from face.data import build_unified_dataframe, load_variables, to_harmonized_dataset
    df = build_unified_dataframe("data", "data/face-common-vars.xlsx",
                                 readiness=["READY", "PARTIAL"], format="long")
    ds = to_harmonized_dataset(df, load_variables("data/face-common-vars.xlsx"), visit="V0")
    # ds.X: MultiIndex[cohort, patient_id] × numeric features (NaN = missing, never imputed)
"""
from .adapter import to_harmonized_dataset
from .harmonized_dataset import HarmonizedDataset
from .loader import build_unified_dataframe
from .variable import Variable, load_variables

__all__ = ["build_unified_dataframe", "load_variables", "Variable",
           "to_harmonized_dataset", "HarmonizedDataset"]
