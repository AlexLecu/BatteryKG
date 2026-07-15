"""Modeling table for the cycle-life prediction experiment.

One row per Severson cell, assembled from the processed artifacts (the same
values the KG carries):
  - early-cycle features (cycles 10-100, leakage-safe)   [severson_features.parquet]
  - parsed charge-policy features + batch                [severson_mit_cycles.parquet]
  - target: log10(cycle_life_nominal)                    [qc.cycle_life_table]
  - policy_group_id: the grouping key for leave-one-policy-group-out CV
  - is_anomalous: QC flag — anomalous cells are excluded from train AND test
    by the experiment (documented in the report)

Run:  python -m src.models.dataset
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED, processed_parquet
from src.ingestion.features import FEATURE_COLUMNS as ALL_EARLY_COLUMNS
from src.ingestion.qc import cycle_life_table

# feature sets used by the models
EARLY_FEATURES = list(ALL_EARLY_COLUMNS)          # all 7 early-cycle scalars
POLICY_FEATURES = ["c_rate_1", "c_rate_2", "soc_transition_pct"]
BASE_FEATURES = EARLY_FEATURES + POLICY_FEATURES + ["batch"]
TARGET = "log10_cycle_life"


def build_dataset() -> pd.DataFrame:
    """Assemble and validate the per-cell modeling table (124 rows)."""
    fpath = PROCESSED / "severson_features.parquet"
    if not fpath.exists():
        raise FileNotFoundError(
            f"{fpath.name} missing — run: python -m src.ingestion.features")
    early = pd.read_parquet(fpath)

    cycles = pd.read_parquet(processed_parquet("severson_mit"))
    meta = (cycles.groupby("cell_id")
                  .agg(batch=("batch", "first"),
                       charge_policy_norm=("charge_policy_norm", "first"),
                       policy_group_id=("policy_group_id", "first"))
                  .reset_index())

    life = cycle_life_table(cycles)[
        ["cell_id", "cycle_life_nominal", "reached_eol_nominal", "is_anomalous"]]

    df = early.merge(meta, on="cell_id", how="inner").merge(life, on="cell_id", how="inner")

    # policy features parsed from the normalized policy string
    from src.kg.features import parse_policy
    feats = df["charge_policy_norm"].map(parse_policy)
    df["c_rate_1"] = [f.c_rate_1 if f else np.nan for f in feats]
    df["c_rate_2"] = [f.c_rate_2 if f else np.nan for f in feats]
    df["soc_transition_pct"] = [f.soc_transition_pct if f else np.nan for f in feats]

    # target
    df[TARGET] = np.log10(df["cycle_life_nominal"].astype(float))

    # --- validation ---------------------------------------------------------
    if len(df) != 124:
        raise ValueError(f"expected 124 cells, got {len(df)}")
    if not df["reached_eol_nominal"].all():
        bad = df.loc[~df["reached_eol_nominal"], "cell_id"].tolist()
        raise ValueError(f"cells without a nominal cycle life: {bad}")
    key_cols = BASE_FEATURES + [TARGET, "policy_group_id"]
    nulls = df[key_cols].isna().sum()
    if nulls.any():
        raise ValueError(f"nulls in modeling columns:\n{nulls[nulls > 0]}")
    if df["cell_id"].duplicated().any():
        raise ValueError("duplicate cell_id rows")

    cols = (["cell_id", "batch", "policy_group_id", "charge_policy_norm",
             "is_anomalous", "cycle_life_nominal", TARGET]
            + EARLY_FEATURES + POLICY_FEATURES)
    return df[cols].sort_values("cell_id").reset_index(drop=True)


if __name__ == "__main__":
    d = build_dataset()
    n_groups = d["policy_group_id"].nunique()
    n_anom = int(d["is_anomalous"].sum())
    kept = d[~d["is_anomalous"]]
    print(f"[dataset] {len(d)} cells, {n_groups} policy groups; "
          f"{n_anom} QC-anomalous (excluded by experiment) -> "
          f"{len(kept)} cells / {kept['policy_group_id'].nunique()} groups usable")
