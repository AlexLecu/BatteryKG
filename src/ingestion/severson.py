"""Loader for the MIT/Stanford/Toyota Severson 2019 dataset (Nature Energy).

Reads the three MATLAB v7.3 batch files (2017-05-12, 2017-06-30, 2018-04-12),
reproduces the canonical 124-cell set (exclusions + batch1->batch2
continuation merges from rdbraatz/data-driven-prediction-of-battery-cycle-life),
and emits one tidy per-cycle Parquet conforming to the common schema.

All 124 cells are A123 APR18650M1A (LFP), cycled at 30 degC, discharged at 4C
to 2.0 V; the fast-charge policy varies per cell and is captured in
`c_rate_charge` (from the file's `policy_readable`).

Run:  python -m src.ingestion.severson
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # h5py is only needed for this loader
    import h5py
except ImportError as e:  # pragma: no cover
    raise ImportError("severson loader needs h5py: pip install h5py") from e

from src.config import RAW_SEVERSON, processed_parquet
from src.ingestion import cell_metadata as cm
from src.ingestion.schema import conform

STUDY = "severson_mit"

BATCH_FILES = {
    1: RAW_SEVERSON / "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: RAW_SEVERSON / "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    3: RAW_SEVERSON / "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

# Canonical data-cleaning constants (rdbraatz "Load Data.ipynb"):
# batch1 cells that never reach 80 % within batch1 -> deleted.
EXCLUDE_BATCH1 = {"b1c8", "b1c10", "b1c12", "b1c13", "b1c22"}
# batch3 noisy cells -> deleted.
EXCLUDE_BATCH3 = {"b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"}
# batch1 cells continued in batch2 (3.6C/4C policies): merge, then drop the b2 dup.
CONTINUATIONS = {              # batch1_key -> batch2_key
    "b1c0": "b2c7",
    "b1c1": "b2c8",
    "b1c2": "b2c9",
    "b1c3": "b2c15",
    "b1c4": "b2c16",
}


def _decode_char_array(f: "h5py.File", ref) -> str:
    """Decode a MATLAB char array (uint16 codes) referenced by `ref`."""
    arr = np.array(f[ref]).flatten()
    return "".join(chr(int(c)) for c in arr if int(c) != 0)


def _load_batch(path, batch_num: int) -> dict[str, dict]:
    """Return {cell_key: {cycle, qd, policy}} for one batch file."""
    cells: dict[str, dict] = {}
    with h5py.File(path, "r") as f:
        batch = f["batch"]
        n = batch["summary"].shape[0]
        for i in range(n):
            summary = f[batch["summary"][i, 0]]
            qd = np.array(summary["QDischarge"]).squeeze()
            cyc = np.array(summary["cycle"]).squeeze()
            qd = np.atleast_1d(qd).astype(float)
            cyc = np.atleast_1d(cyc).astype(float)
            try:
                policy = _decode_char_array(f, batch["policy_readable"][i, 0])
            except Exception:
                policy = ""
            cells[f"b{batch_num}c{i}"] = {"cycle": cyc, "qd": qd, "policy": policy}
    return cells


def _clean_and_merge(b1: dict, b2: dict, b3: dict) -> dict[str, dict]:
    """Apply exclusions and batch1->batch2 continuation merges."""
    for k in EXCLUDE_BATCH1:
        b1.pop(k, None)
    for k in EXCLUDE_BATCH3:
        b3.pop(k, None)

    for k1, k2 in CONTINUATIONS.items():
        if k1 in b1 and k2 in b2:
            offset = len(b1[k1]["cycle"])
            b1[k1]["cycle"] = np.hstack([b1[k1]["cycle"], b2[k2]["cycle"] + offset])
            b1[k1]["qd"] = np.hstack([b1[k1]["qd"], b2[k2]["qd"]])
        b2.pop(k2, None)  # drop the batch2 duplicate regardless

    merged = {}
    merged.update(b1)
    merged.update(b2)
    merged.update(b3)
    return merged


def load() -> pd.DataFrame:
    """Load all three batches into the common tidy schema."""
    missing = [p.name for p in BATCH_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Severson batch files: {missing}. "
            "Run: python -m src.ingestion.download severson"
        )

    b1 = _load_batch(BATCH_FILES[1], 1)
    b2 = _load_batch(BATCH_FILES[2], 2)
    b3 = _load_batch(BATCH_FILES[3], 3)
    cells = _clean_and_merge(b1, b2, b3)

    frames = []
    cell = cm.SEVERSON_CELL
    for cell_id, d in cells.items():
        cyc = d["cycle"]
        qd = d["qd"]
        # keep only finite, positive-capacity cycles (drops formation NaNs)
        ok = np.isfinite(cyc) & np.isfinite(qd) & (qd > 0)
        cyc, qd = cyc[ok], qd[ok]
        if cyc.size == 0:
            continue
        frames.append(pd.DataFrame({
            "cell_id": cell_id,
            "study": STUDY,
            "cycle_index": np.round(cyc).astype("int64"),
            "discharge_capacity_ah": qd,
            "chemistry": cell.chemistry,
            "nominal_capacity_ah": cell.nominal_capacity_ah,
            "temperature_c": cm.SEVERSON_TEMPERATURE_C,
            "dod_window": cm.SEVERSON_DOD_WINDOW,
            "c_rate_charge": d["policy"] or pd.NA,
            "c_rate_discharge": cm.SEVERSON_C_RATE_DISCHARGE,
        }))

    df = pd.concat(frames, ignore_index=True)
    # a handful of cells duplicate cycle_index at the merge seam -> keep first
    df = df.drop_duplicates(["cell_id", "cycle_index"], keep="first")
    df = conform(df)
    # study-specific extra column: originating batch (1/2/3) from the cell id.
    # Merge-continued cells keep their batch-1 id -> batch 1 (numBat1/2/3 = 41/43/40).
    df["batch"] = df["cell_id"].str.extract(r"^b(\d)").astype("int64")
    _add_policy_columns(df)
    return df


# Batch-3 policies carry a "-newstructure" data-format tag that is NOT part of
# the charging protocol. Strip it so protocol-identical cells across batches
# fall in the same replicate/cross-validation group.
_NEWSTRUCTURE = "-newstructure"

# Documented source data-quality corrections applied when deriving the canonical
# (normalized) policy string. charge_policy_raw is left faithful to the .mat.
# Each entry's verification evidence is documented inline below.
_POLICY_CORRECTIONS = {
    # b2c12: the .mat stores "4C(31%)-5" — the second-step C-rate is missing its
    # trailing 'C' (unique in the dataset; both policy_readable and the structured
    # 'policy' field '4C_31PER_5' lack it). Neighbours b2c10=3.6C(9%)-5C,
    # b2c11=4C(13%)-5C, b2c13=4C(40%)-6C confirm the two-step "<C1>C(Q%)-<C2>C"
    # family, and 5C is an attested second-step rate in batch 2. Intended policy
    # is 4C(31%)-5C (4C to 31% SOC, then 5C).
    "4C(31%)-5": "4C(31%)-5C",
}


def _add_policy_columns(df: pd.DataFrame) -> None:
    """Add charge_policy_raw / _norm and a stable policy_group_id (CV key)."""
    raw = df["c_rate_charge"].astype("string")
    # normalized = strip the format tag, then apply documented corrections
    norm = raw.str.replace(_NEWSTRUCTURE, "", regex=False).replace(_POLICY_CORRECTIONS)
    # deterministic id per normalized policy: sort unique policies, enumerate
    uniq = sorted(p for p in norm.dropna().unique())
    mapping = {p: f"pg{ i:03d}" for i, p in enumerate(uniq)}
    df["charge_policy_raw"] = raw
    df["charge_policy_norm"] = norm
    df["policy_group_id"] = norm.map(mapping).astype("string")


def build() -> "pd.DataFrame":
    df = load()
    out = processed_parquet(STUDY)
    df.to_parquet(out, index=False)
    print(f"[severson] {df['cell_id'].nunique()} cells, {len(df):,} cycle-rows -> {out}")
    return df


if __name__ == "__main__":
    build()
