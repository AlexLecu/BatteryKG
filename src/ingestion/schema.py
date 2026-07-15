"""Common tidy schema shared by every dataset loader.

Every study loader must emit a per-cycle ("long") table conforming to this
schema so downstream KG loading and modelling code can treat all studies
uniformly. One row = one (cell, cycle). Per-cell metadata is denormalised
(repeated on every row of that cell) so a single Parquet is self-contained.

Missing metadata is allowed (nullable), but the four core measurement columns
(cell_id, study, cycle_index, discharge_capacity_ah) must always be present.
"""
from __future__ import annotations

import pandas as pd

# --- Column contract -------------------------------------------------------

# Core per-cycle measurement columns (never null)
CORE_COLUMNS = [
    "cell_id",                 # str  — original study cell/channel id (preserved verbatim)
    "study",                   # str  — dataset tag, e.g. "severson_mit", "sandia"
    "cycle_index",             # int  — 1-based cycle number
    "discharge_capacity_ah",   # float — discharge capacity for that cycle [Ah]
]

# Per-cell metadata columns (nullable; repeated per row)
META_COLUMNS = [
    "chemistry",               # str   — LFP / NCA / NMC / ...
    "nominal_capacity_ah",     # float — datasheet nominal capacity [Ah]
    "temperature_c",           # float — test/ambient temperature [°C]
    "dod_window",              # str   — depth-of-discharge window, e.g. "0-100%", "20-80%"
    "c_rate_charge",           # str   — charge protocol / C-rate (free text; policies can be multistep)
    "c_rate_discharge",        # str   — discharge C-rate
]

ALL_COLUMNS = CORE_COLUMNS + META_COLUMNS

# Pandas dtypes (nullable extension types so metadata can be absent per-cell)
DTYPES = {
    "cell_id": "string",
    "study": "string",
    "cycle_index": "Int64",
    "discharge_capacity_ah": "float64",
    "chemistry": "string",
    "nominal_capacity_ah": "Float64",
    "temperature_c": "Float64",
    "dod_window": "string",
    "c_rate_charge": "string",
    "c_rate_discharge": "string",
}


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, order, and cast a loader's dataframe to the common schema.

    Adds any missing metadata columns as null, enforces column order and
    dtypes, and fails loudly if a core column is missing or contains nulls.
    """
    missing_core = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing_core:
        raise ValueError(f"Loader output missing required core columns: {missing_core}")

    df = df.copy()
    for col in META_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[ALL_COLUMNS].astype(DTYPES)

    for col in CORE_COLUMNS:
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            raise ValueError(f"Core column '{col}' has {n} null value(s); not allowed.")

    return df.sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
