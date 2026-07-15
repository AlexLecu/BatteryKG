"""Explicit study-cell -> commercial-cell mapping tables.

Per project convention ("keep original study IDs, map to commercial cell via
an explicit table"), all chemistry / nominal-capacity assignments live here so
loaders stay purely mechanical and the provenance of every metadata value is
auditable in one place.

Nominal capacities are manufacturer datasheet values (the "claims" side of the
knowledge graph). Sources are cited inline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommercialCell:
    model: str                 # model designation
    manufacturer: str          # cell manufacturer
    chemistry: str             # LFP / NCA / NMC / LCO
    nominal_capacity_ah: float
    form_factor: str
    source: str                # where the nominal value comes from


# --- Severson/MIT (Nature Energy 2019) -------------------------------------
# All 124 cells are the SAME commercial cell: A123 APR18650M1A (LFP), 1.1 Ah.
# Cycled at 30 degC, fast-charge policies vary per cell, discharge 4C to 2.0 V.
SEVERSON_CELL = CommercialCell(
    model="A123 APR18650M1A",
    manufacturer="A123 Systems",
    chemistry="LFP",
    nominal_capacity_ah=1.1,
    form_factor="18650",
    source="A123 APR18650M1A datasheet (1.1 Ah nominal); Severson et al. 2019, Nat. Energy",
)
SEVERSON_TEMPERATURE_C = 30.0          # forced-convection chamber
SEVERSON_DOD_WINDOW = "0-100%"         # charge to 3.6 V, discharge to 2.0 V
SEVERSON_C_RATE_DISCHARGE = "4C"       # all cells discharged at 4C to 2.0 V


# --- Sandia Cell Cycle Testing Data (24 cells, 4 chemistries) --------------
# Cell model mapping follows Preger et al. 2020 (J. Electrochem. Soc.,
# OSTI 1650174) for LFP/NCA/NMC. The LCO cell model/capacity was NOT confirmed
# from a primary source during ingestion setup -> nominal left as None and
# flagged; confirm against README_Cycle_Data.docx before using LCO nominals.
SANDIA_CELLS: dict[str, CommercialCell | None] = {
    "LFP": CommercialCell(
        model="A123 APR18650M1A",
        manufacturer="A123 Systems",
        chemistry="LFP",
        nominal_capacity_ah=1.1,
        form_factor="18650",
        source="Preger et al. 2020 (OSTI 1650174); A123 APR18650M1A datasheet",
    ),
    "NCA": CommercialCell(
        model="Panasonic NCR18650B",
        manufacturer="Panasonic",
        chemistry="NCA",
        nominal_capacity_ah=3.2,
        form_factor="18650",
        source="Preger et al. 2020 (OSTI 1650174); Panasonic NCR18650B datasheet",
    ),
    "NMC": CommercialCell(
        model="LG Chem 18650HG2",
        manufacturer="LG Chem",
        chemistry="NMC",
        nominal_capacity_ah=3.0,
        form_factor="18650",
        source="Preger et al. 2020 (OSTI 1650174); LG 18650HG2 datasheet",
    ),
    "LCO": CommercialCell(
        model="LCO 18650 (model unconfirmed)",
        manufacturer="unknown",
        chemistry="LCO",
        nominal_capacity_ah=float("nan"),   # UNCONFIRMED — do not trust until verified
        form_factor="18650",
        source="UNVERIFIED — confirm model & capacity in README_Cycle_Data.docx",
    ),
}
