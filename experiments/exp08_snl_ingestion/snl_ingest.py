"""SNL LFP (Preger et al. 2020, BatteryArchive) -> tidy per-cycle frame.

Data preparation only. This module reads the vendored zip and writes a
schema-conforming Parquet; it never touches the knowledge graph. Graph loading
is a separate, human-gated step (see load_snl.py).

Preprocessing decisions, all consequences of the earlier feasibility report
(outputs/snl_feasibility/SNL_feasibility_report.md):

1. RPT-ONLY CAPACITY SERIES. SNL interleaves Reference Performance Tests —
   full-window (0-100 % DoD) 0.5C discharges — into an ageing schedule whose
   cycling cycles may run at partial DoD (20-80 %, 40-60 %) and up to 3C. A
   partial-DoD cell never discharges fully, so its cycling-cycle capacity is
   not comparable to nameplate: fed to the standard cycle-life routine it
   "reaches" 80 % of nominal at cycle 4. Capacity is therefore read from RPT
   cycles only, which is the same basis the datasheet's capacity claim would
   be measured on and the only basis on which the 30 cells are comparable to
   each other. `cycle_index` is preserved as the ORIGINAL ageing cycle number,
   so a cycle life of 4059 means 4059 ageing cycles, not 4059 RPTs.

2. GAP-SPLIT (>60 s) — NOT APPLICABLE HERE, DELIBERATELY. The feasibility
   report requires splitting discharge segments on >60 s gaps because some
   35 C RPT cycles file two separate discharge events under one Cycle_Index,
   which corrupts Q(V) inversion. That defect lives in the TIMESERIES files
   and only affects voltage-resolved features. This ingestion computes no
   Q(V)/behaviour features (SNL is deliberately not entering the neighbour
   bank), so it reads the per-cycle `cycle_data` tables and never inverts a
   discharge curve. The cycle-level symptom of a merged double discharge — a
   cycle whose capacity exceeds nominal — is screened for explicitly and
   reported rather than silently accepted.

3. IDENTITY IS ASSERTED, NOT EXTRACTED. The CSVs carry no manufacturer or
   model string. The A123 APR18650M1A identity comes from the publication and
   is recorded as such (see stage.py, entity matching).

Run:  python -m experiments.exp08_snl_ingestion.snl_ingest
"""
from __future__ import annotations

import io
import re
import zipfile

import numpy as np
import pandas as pd

from src.config import PROCESSED, ROOT
from src.ingestion.schema import conform

ZIP_PATH = ROOT / "BatteryArchive" / "SNL LFP.zip"
STUDY = "snl_lfp"
OUT_PARQUET = PROCESSED / "snl_lfp_cycles.parquet"

# A123 APR18650M1A, per Preger et al. 2020 ("LFP from A123 Systems
# (Part #APR18650M1A, 1.1 Ah)") and the vendored datasheet claim.
NOMINAL_AH = 1.1
V_FULL_HI, V_FULL_LO = 3.6, 2.0          # full charge/discharge window
RPT_C_RATE = 0.5                          # RPT discharges run at 0.5C
RPT_CURRENT_TOL = 0.70                    # |I| <= 0.70 * nominal counts as <=0.5C-ish
CHEMISTRY = "LFP"

# SNL_18650_LFP_<T>C_<DoD>_<Cchg>-<Cdis>C_<rep>_cycle_data.csv
NAME_RE = re.compile(
    r"SNL_18650_LFP_(?P<temp>\d+)C_(?P<dod>[\d\-]+)_"
    r"(?P<cchg>[\d.]+)-(?P<cdis>[\d.]+)C_(?P<rep>\w+)_cycle_data\.csv$")


def _parse_name(name: str) -> dict | None:
    m = NAME_RE.search(name.rsplit("/", 1)[-1])
    if not m:
        return None
    g = m.groupdict()
    return {"temperature_c": float(g["temp"]),
            "dod_window": f"{g['dod']}%",
            "c_rate_charge": f"{float(g['cchg']):g}C",
            "c_rate_discharge": f"{float(g['cdis']):g}C",
            "replicate": g["rep"]}


def classify_rpt(cyc: pd.DataFrame) -> pd.Series:
    """RPT = full-voltage-window discharge at (about) the 0.5C reference rate."""
    return ((cyc["Max_Voltage (V)"] >= V_FULL_HI - 0.05)
            & (cyc["Min_Voltage (V)"] <= V_FULL_LO + 0.10)
            & (cyc["Min_Current (A)"] >= -RPT_CURRENT_TOL * NOMINAL_AH))


def build_cycles() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tidy per-cycle frame over RPT cycles, per-cell preprocessing audit)."""
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"{ZIP_PATH} missing (BatteryArchive vendored zip)")
    rows, audit = [], []
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = sorted(n for n in z.namelist() if n.endswith("_cycle_data.csv"))
        for name in names:
            meta = _parse_name(name)
            if meta is None:
                continue
            stem = name.rsplit("/", 1)[-1].replace("_cycle_data.csv", "")
            with z.open(name) as fh:
                cyc = pd.read_csv(io.BytesIO(fh.read()))
            cyc = cyc[cyc["Discharge_Capacity (Ah)"] > 0.01].reset_index(drop=True)
            is_rpt = classify_rpt(cyc)
            rpt = cyc[is_rpt]

            # decision 2: cycle-level screen for merged double-discharges
            over = rpt["Discharge_Capacity (Ah)"] > NOMINAL_AH * 1.10
            audit.append({
                "cell_id": stem, "n_cycles_total": int(len(cyc)),
                "n_rpt": int(is_rpt.sum()),
                "n_cycling": int((~is_rpt).sum()),
                "max_ageing_cycle": int(cyc["Cycle_Index"].max()),
                "capacity_over_nominal_flagged": int(over.sum()),
                **meta,
            })
            keep = rpt[~over]
            for _, r in keep.iterrows():
                rows.append({
                    "cell_id": stem, "study": STUDY,
                    "cycle_index": int(r["Cycle_Index"]),
                    "discharge_capacity_ah": float(r["Discharge_Capacity (Ah)"]),
                    "chemistry": CHEMISTRY, "nominal_capacity_ah": NOMINAL_AH,
                    **{k: v for k, v in meta.items() if k != "replicate"},
                })
    df = conform(pd.DataFrame(rows))
    return df, pd.DataFrame(audit)


def build_instances() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-instance frame in the shape src.kg.load.load_dataframe expects.

    SNL varies temperature / DoD / discharge rate at a FIXED 0.5C CC-CV charge,
    so the condition-view policy features (which describe a two-step fast-charge
    protocol) do not apply: c_rate_1/2 and soc_transition_pct are left null and
    no condition-view similarity edges are built. `policy_group_id` groups the
    replicates of one (T, DoD, discharge-rate) cell condition, which is the
    grouping SNL's own design uses.
    """
    from src.ingestion.qc import cycle_life_table
    if not OUT_PARQUET.exists():
        raise FileNotFoundError(f"{OUT_PARQUET} missing — run build_cycles first")
    cycles = pd.read_parquet(OUT_PARQUET)
    life = cycle_life_table(cycles)
    meta = (cycles.groupby("cell_id")
            .agg(temperature_c=("temperature_c", "first"),
                 dod_window=("dod_window", "first"),
                 c_rate_charge=("c_rate_charge", "first"),
                 c_rate_discharge=("c_rate_discharge", "first"))
            .reset_index())
    inst = life.merge(meta, on="cell_id", how="left")
    inst["charge_policy_raw"] = inst["c_rate_charge"] + " CC-CV"
    inst["charge_policy_norm"] = inst["c_rate_charge"]
    inst["policy_group_id"] = (
        "pgSNL_" + inst["temperature_c"].astype(int).astype(str) + "C_"
        + inst["dod_window"].str.replace("%", "", regex=False) + "_"
        + inst["c_rate_discharge"])
    inst["batch"] = pd.NA
    for c in ("c_rate_1", "c_rate_2", "soc_transition_pct"):
        inst[c] = np.nan                       # not a two-step fast-charge protocol
    inst = inst.rename(columns={"cell_id": "study_cell_id", "flags": "qc_flag"})
    return inst, cycles


def main() -> None:
    df, audit = build_cycles()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"[snl_ingest] {df['cell_id'].nunique()} cells, {len(df)} RPT cycle rows "
          f"-> {OUT_PARQUET}")
    print(f"[snl_ingest] ageing cycles spanned: {audit['max_ageing_cycle'].min()}"
          f"-{audit['max_ageing_cycle'].max()}")
    flagged = int(audit["capacity_over_nominal_flagged"].sum())
    print(f"[snl_ingest] cycles dropped by the >110%-of-nominal screen: {flagged}")
    print(audit[["cell_id", "n_rpt", "n_cycling", "max_ageing_cycle",
                 "temperature_c", "dod_window", "c_rate_discharge"]].to_string(index=False))
    return df, audit


if __name__ == "__main__":
    main()
