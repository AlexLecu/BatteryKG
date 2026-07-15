"""Loader for the Sandia Cell Cycle Testing Data (24-cell short-term set).

24 commercial 18650 cells across 4 chemistries (LCO/LFP/NCA/NMC, 6 each),
short-term cycled across a 5-45 degC temperature sweep. Data ships as Arbin
Excel exports (one workbook per cell x temperature x discharge-mode). We read
the per-cycle `Statistics_*` sheet from each workbook.

Because a physical cell is run across several temperatures (and Reg/Mod
discharge modes), one *capacity-fade series* corresponds to a single
(cell, temperature, mode) workbook. The cell_id therefore encodes all three,
e.g. "NCA_4_25C_Reg"; the physical Sandia cell ("NCA_4") is the leading token.

Run:  python -m src.ingestion.sandia
"""
from __future__ import annotations

import re
import zipfile

import numpy as np
import pandas as pd

from src.config import RAW_SANDIA, processed_parquet
from src.ingestion import cell_metadata as cm
from src.ingestion.schema import conform

STUDY = "sandia"
ZIP_PATH = RAW_SANDIA / "Sandia_Cell_Cycle_Testing_Data.zip"
EXTRACT_DIR = RAW_SANDIA / "Sandia_Cell_Cycle_Testing_Data"

# e.g. NCA_4_25C_Reg.xlsx , NMC_1_35C_Mod.xls
FILE_RE = re.compile(
    r"(?P<chem>LCO|LFP|NCA|NMC)_(?P<num>\d+)_(?P<temp>\d+)C_(?P<mode>Reg|Mod)",
    re.IGNORECASE,
)


def _ensure_extracted() -> None:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("*.xls*")):
        return
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ZIP_PATH.name}. Run: python -m src.ingestion.download sandia"
        )
    print(f"[sandia] extracting {ZIP_PATH.name} ...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_DIR)


def _read_statistics(path) -> pd.DataFrame | None:
    """Read the per-cycle Statistics sheet; return None if unreadable."""
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"[sandia] cannot open {path.name}: {e}")
        return None
    stat_sheets = [s for s in xls.sheet_names if s.startswith("Statistics")]
    if not stat_sheets:
        return None
    df = xls.parse(stat_sheets[0])
    needed = {"Cycle_Index", "Discharge_Capacity(Ah)"}
    if not needed.issubset(df.columns):
        return None
    return df[["Cycle_Index", "Discharge_Capacity(Ah)"]]


def load() -> pd.DataFrame:
    _ensure_extracted()
    files = sorted(EXTRACT_DIR.rglob("*.xls")) + sorted(EXTRACT_DIR.rglob("*.xlsx"))
    files = [p for p in files if not p.name.startswith("~$")]  # skip Excel lock files
    if not files:
        raise FileNotFoundError(f"No workbooks found under {EXTRACT_DIR}")

    # Some series ship as both .xls and .xlsx (the extra copy sometimes lacks a
    # usable Statistics sheet). Collect one frame per series key and, on
    # collision, keep the most complete version — never interleave the two.
    by_series: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for p in files:
        m = FILE_RE.search(p.name)
        if not m:
            skipped.append(p.name)
            continue
        chem = m.group("chem").upper()
        num = int(m.group("num"))
        temp = float(m.group("temp"))
        mode = m.group("mode").capitalize()

        stat = _read_statistics(p)
        if stat is None:
            skipped.append(p.name)
            continue

        cyc = pd.to_numeric(stat["Cycle_Index"], errors="coerce").to_numpy(dtype=float)
        qd = pd.to_numeric(stat["Discharge_Capacity(Ah)"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(cyc) & np.isfinite(qd) & (qd > 0)
        cyc, qd = cyc[ok], qd[ok]
        if cyc.size == 0:
            skipped.append(p.name)
            continue

        commercial = cm.SANDIA_CELLS.get(chem)
        nominal = commercial.nominal_capacity_ah if commercial else np.nan
        cell_id = f"{chem}_{num}_{int(temp)}C_{mode}"

        frame = pd.DataFrame({
            "cell_id": cell_id,
            "study": STUDY,
            "cycle_index": np.round(cyc).astype("int64"),
            "discharge_capacity_ah": qd,
            "chemistry": chem,
            "nominal_capacity_ah": nominal,
            "temperature_c": temp,
            "dod_window": pd.NA,          # short-term temp sweep; DOD not a data field
            "c_rate_charge": pd.NA,
            "c_rate_discharge": pd.NA,
        }).drop_duplicates("cycle_index", keep="first")

        prev = by_series.get(cell_id)
        if prev is None or len(frame) > len(prev):
            by_series[cell_id] = frame

    if skipped:
        print(f"[sandia] skipped {len(skipped)} file(s) (no match / no usable Statistics)")
    df = pd.concat(by_series.values(), ignore_index=True)
    return conform(df)


def build() -> "pd.DataFrame":
    df = load()
    out = processed_parquet(STUDY)
    df.to_parquet(out, index=False)
    n_phys = df["cell_id"].str.extract(r"^([A-Z]+_\d+)")[0].nunique()
    print(f"[sandia] {df['cell_id'].nunique()} series ({n_phys} physical cells), "
          f"{len(df):,} cycle-rows -> {out}")
    return df


if __name__ == "__main__":
    build()
