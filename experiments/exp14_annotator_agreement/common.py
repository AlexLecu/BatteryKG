"""Shared loading / normalisation for the second-annotator agreement study.

Reads BOTH annotation sets read-only:
  reference set  data/claims/*.yaml + data/gold/samsung_inr18650_25r_gold.yaml
  comparison set the five filled-in files returned by the second annotator

Nothing here writes to either. No LLM path exists in this experiment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
GOLD_CLAIMS = ROOT / "data" / "claims"
GOLD_HELDOUT = ROOT / "data" / "gold" / "samsung_inr18650_25r_gold.yaml"
DEFAULT_SECOND_SET = ROOT / "data" / "gold" / "second_annotator"
PDF_DIR = ROOT / "data" / "claims" / "datasheets"


def rel_to_root(path: Path) -> str:
    """Repo-relative string for anything written into a published artifact.

    Reports and results.json are released, so they must not carry the absolute
    path of whatever machine produced them. Paths outside the repo are returned
    as-is (the scripts accept an arbitrary directory argument).
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)

# document key -> (reference YAML, second-annotator filename, source PDF, label)
DOCUMENTS = [
    ("a123", GOLD_CLAIMS / "a123_apr18650m1a.yaml",
     "a123_apr18650m1a_annotation.yaml", "a123_apr18650m1a.pdf",
     "A123 APR18650M1A"),
    ("lg", GOLD_CLAIMS / "lg_18650hg2.yaml",
     "lg_inr18650hg2_annotation.yaml", "lg_inr18650hg2.pdf",
     "LG Chem 18650HG2"),
    ("panasonic_mkt", GOLD_CLAIMS / "panasonic_ncr18650b.yaml",
     "panasonic_ncr18650b_annotation.yaml", "panasonic_ncr18650b.pdf",
     "Panasonic NCR18650B (marketing sheet, image-only)"),
    ("panasonic_full", GOLD_CLAIMS / "panasonic_ncr18650b_full_spec.yaml",
     "panasonic_ncr18650b_full_spec_annotation.yaml",
     "panasonic_ncr18650b_full_spec_sanyo.pdf",
     "Panasonic NCR18650B (SANYO full specification)"),
    ("samsung", GOLD_HELDOUT,
     "samsung_inr18650_25r_annotation.yaml", "samsung_inr18650_25r.pdf",
     "Samsung INR18650-25R (held out)"),
]

# The 27-property vocabulary shipped to the second annotator (pre-Samsung).
SHIPPED_VOCAB = {
    "nominal_capacity_ah", "rated_capacity_ah", "minimum_capacity_ah",
    "nominal_voltage_v", "charge_voltage_v", "max_charge_voltage_v",
    "discharge_cutoff_v", "std_charge_current_a", "std_charge_time_h",
    "max_charge_current_a", "fast_charge_current_a", "std_discharge_current_a",
    "fast_discharge_current_a", "max_cont_discharge_a", "cycle_life_cycles",
    "cycle_life_retention_pct", "capacity_retention_at_temp_pct",
    "storage_capacity_remaining_pct", "storage_capacity_recovery_pct", "mass_g",
    "charge_temp_range_c", "discharge_temp_range_c", "operating_temp_range_c",
    "storage_temp_range_c", "internal_impedance_mohm",
    "gravimetric_energy_density_wh_kg", "volumetric_energy_density_wh_l",
}

# unit -> physical class. Cross-unit comparison inside a class is handled by
# src.agents.evaluation.values_match (mAh->Ah, mA->A, mV->V); the class guard
# is what stops "3.0 Ah" matching "3.0 V" in the property-agnostic view.
_UNIT_CLASS = {
    "ah": "capacity", "mah": "capacity",
    "v": "voltage", "mv": "voltage",
    "a": "current", "ma": "current",
    "pct": "pct", "%": "pct",
    "degc": "temperature", "c": "temperature", "°c": "temperature",
    "cycles": "cycles",
    "g": "mass",
    "mohm": "impedance", "mΩ": "impedance", "ohm": "impedance",
    "wh/kg": "energy_grav", "wh/l": "energy_vol",
    "min": "time_min", "h": "time_h",
    "mm": "length",
    "days": "duration", "months": "duration",
    "w": "power", "wh": "energy",
}

ALLOWED_UNITS = set(_UNIT_CLASS)


def unit_class(unit) -> str:
    return _UNIT_CLASS.get(str(unit or "").strip().lower(), f"other:{unit}")


@dataclass
class Claim:
    doc: str
    idx: int                       # position in its own file, 1-based
    property: str
    value: object
    unit: object
    page: object
    cond_text: str
    cond_parsed: dict = field(default_factory=dict)
    notes: str = ""
    source: str = ""               # "reference" | "second"

    @property
    def unit_class(self) -> str:
        return unit_class(self.unit)

    @property
    def is_unspecified(self) -> bool:
        return str(self.cond_text).strip().lower() == "unspecified"

    def stratum(self) -> str:
        """Where the claim sits in the document — for the clustering summary."""
        p = self.property
        if p.startswith("rel_") or "relative_" in p or p == "capacity_retention_at_temp_pct":
            return "characteristic grid"
        if "graph" in (self.notes or "").lower() or "graph" in (self.cond_text or "").lower():
            return "read from graph"
        return "specification row"

    def show_value(self) -> str:
        v = self.value
        if isinstance(v, list):
            return "[" + ", ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x)
                                   for x in v) + "]"
        return f"{v:g}" if isinstance(v, (int, float)) else str(v)

    def label(self) -> str:
        return f"{self.property} = {self.show_value()} {self.unit} (p.{self.page})"


def load_claims(path: Path, doc: str, source: str) -> tuple[list[Claim], dict]:
    """Returns (claims, raw document) — raw kept for omissions / header checks."""
    raw = yaml.safe_load(path.read_text())
    claims = []
    for i, c in enumerate(raw.get("claims") or [], start=1):
        cond = c.get("stated_conditions") or {}
        parsed = {k: v for k, v in cond.items() if k != "text"}
        claims.append(Claim(
            doc=doc, idx=i, property=c.get("property"), value=c.get("value"),
            unit=c.get("unit"), page=c.get("page"),
            cond_text=str(cond.get("text", "")), cond_parsed=parsed,
            notes=str(c.get("notes") or ""), source=source))
    return claims, raw


def pdf_page_count(pdf: Path) -> int | None:
    """Page count without a PDF dependency: the /Type/Pages Count of the root."""
    if not pdf.exists():
        return None
    blob = pdf.read_bytes()
    counts = [int(m) for m in re.findall(rb"/Type\s*/Pages[^>]*?/Count\s+(\d+)", blob)]
    if counts:
        return max(counts)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", blob))
    return pages or None


def document_set(second_dir: Path = DEFAULT_SECOND_SET):
    """Yield (key, label, ref_claims, ref_raw, sec_claims, sec_raw, pdf)."""
    for key, ref_path, sec_name, pdf_name, label in DOCUMENTS:
        sec_path = second_dir / sec_name
        if not sec_path.exists():
            raise FileNotFoundError(f"second-annotator file missing: {sec_path}")
        ref_claims, ref_raw = load_claims(ref_path, key, "reference")
        sec_claims, sec_raw = load_claims(sec_path, key, "second")
        yield dict(key=key, label=label, ref_path=ref_path, sec_path=sec_path,
                   ref_claims=ref_claims, ref_raw=ref_raw,
                   sec_claims=sec_claims, sec_raw=sec_raw,
                   pdf=PDF_DIR / pdf_name)
