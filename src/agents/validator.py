"""Deterministic Validator stage for LLM-extracted claims (no LLM inside).

Applied in order, each stage either passes a claim through (possibly repaired /
normalized) or rejects it with a reason code:

  1. schema gate        — strict conformance; the ONLY repair is mechanical
                          unnesting (a nested list value becomes one claim per
                          inner value). Reasons: missing_field, malformed_value.
  2. value-in-source    — every numeric value must appear in the extracted
                          document text (normalized-number match: "1,000" ==
                          1000; magnitude-guarded x1000 for unit prefixes like
                          3.35 Ah vs "3350 mAh"). Silent unit conversions are
                          rejected by construction. Reason: value_not_in_source.
  3. unit sanity        — per-property physical plausibility bounds (table
                          below). Reason: implausible.
  4. range/tolerance    — a range claim [lo, hi] whose source text reads
     normalizer           "X ± Y" (with X=(lo+hi)/2, Y=(hi-lo)/2) is normalized
                          to the scalar/tolerance convention the gold uses.
  5. consensus          — across N extraction runs, keep claims appearing in
                          >= MIN_RUNS runs (property + value tolerance match);
                          per-claim consensus_level is recorded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.agents.evaluation import _as_floats, _close, _scale, values_match

MIN_RUNS = 2

# --- stage 3: plausibility bounds ------------------------------------------------
# (lo, hi) in the property's canonical unit (claim values are unit-scaled via
# evaluation._scale before checking, so mAh/mA/mV inputs are handled).
PLAUSIBILITY_BOUNDS: dict[str, tuple[float, float]] = {
    # capacities [Ah] — 18650-class cells
    "nominal_capacity_ah": (0.1, 10), "minimum_capacity_ah": (0.1, 10),
    "rated_capacity_ah": (0.1, 10), "initial_capacity_ah": (0.1, 10),
    # voltages [V]
    "nominal_voltage_v": (0, 5), "charge_voltage_v": (0, 5),
    "max_charge_voltage_v": (0, 5), "discharge_cutoff_v": (0, 5),
    "charge_cutoff_v": (0, 5),
    # currents [A]
    "std_charge_current_a": (0.01, 100), "fast_charge_current_a": (0.01, 100),
    "max_charge_current_a": (0.01, 100), "std_discharge_current_a": (0.01, 100),
    "fast_discharge_current_a": (0.01, 100), "max_cont_discharge_a": (0.01, 100),
    "max_pulse_discharge_a": (0.01, 200),
    # C-rates
    "max_cont_discharge_c": (0.05, 100),
    # cycle life
    "cycle_life_cycles": (1, 20000),
    # percentages
    "cycle_life_retention_pct": (0, 100), "storage_capacity_remaining_pct": (0, 100),
    "storage_capacity_recovery_pct": (0, 100), "capacity_retention_at_temp_pct": (0, 100),
    # mass [g] — 18650-class
    "mass_g": (10, 100),
    # temperatures [degC]
    "charge_temp_range_c": (-60, 100), "discharge_temp_range_c": (-60, 100),
    "storage_temp_range_c": (-60, 100), "operating_temp_range_c": (-60, 100),
    # impedance [mOhm]
    "internal_impedance_mohm": (0.1, 1000),
    # energy density
    "volumetric_energy_density_wh_l": (50, 2000),
    "gravimetric_energy_density_wh_kg": (30, 500),
    # times
    "std_charge_time_h": (0.05, 50),
}


@dataclass
class Rejection:
    doc: str
    run: str
    property: str
    value: object
    reason: str
    detail: str = ""


@dataclass
class ValidationResult:
    accepted: list[dict] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    normalized: int = 0          # claims whose range form was tolerance-normalized


# --- stage 1: schema gate ---------------------------------------------------------
def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def schema_gate(claim: dict) -> tuple[list[dict], str | None]:
    """Return (claims, reason). Repair = mechanical unnesting only."""
    if not isinstance(claim, dict) or "property" not in claim or "value" not in claim:
        return [], "missing_field"
    v = claim["value"]
    if _is_num(v):
        return [claim], None
    if isinstance(v, list):
        if v and all(_is_num(x) for x in v) and len(v) == 2:
            return [claim], None                        # legitimate [lo, hi] range
        if v and all(isinstance(x, list) and len(x) == 2 and all(_is_num(y) for y in x)
                     for x in v):
            # nested list of ranges -> one claim per inner range (mechanical)
            out = []
            for inner in v:
                c = dict(claim)
                c["value"] = list(inner)
                c["validator_note"] = "unnested"
                out.append(c)
            return out, None
        if v and all(_is_num(x) for x in v) and len(v) > 2:
            # flat list of scalars -> one claim per scalar (mechanical)
            out = []
            for x in v:
                c = dict(claim)
                c["value"] = x
                c["validator_note"] = "unnested"
                out.append(c)
            return out, None
    return [], "malformed_value"


# --- stage 2: value-in-source -------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _doc_numbers(doc_text: str) -> list[float]:
    # normalize thousands separators ("1,000" -> "1000") and Unicode minus
    cleaned = doc_text.replace("−", "-")
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", cleaned)
    return [float(m) for m in _NUM_RE.findall(cleaned)]


def value_in_source(claim: dict, doc_nums: list[float]) -> bool:
    """Number-presence check on ABSOLUTE values (sign handling in datasheet
    text is unreliable: range dashes, Unicode minus, '+' prefixes), with
    magnitude-guarded x1000 unit-prefix rescaling."""
    vals = _as_floats(claim.get("value"))
    if vals is None:
        return False
    abs_doc = [abs(d) for d in doc_nums]

    def found(x: float) -> bool:
        candidates = [x]
        if x < 100:
            candidates.append(x * 1000)     # Ah -> mAh style prefix rescale
        if x >= 1000:
            candidates.append(x / 1000)
        return any(_close(c, d) for c in candidates for d in abs_doc)

    return all(found(abs(v)) for v in vals)


# --- stage 3: unit sanity -------------------------------------------------------------
def unit_sane(claim: dict) -> bool:
    bounds = PLAUSIBILITY_BOUNDS.get(claim.get("property"))
    if bounds is None:
        return True                          # unknown property: no bounds defined
    vals = _as_floats(claim.get("value"))
    if vals is None:
        return False
    lo, hi = bounds
    scaled = [_scale(v, claim.get("unit")) for v in vals]
    return all(lo <= v <= hi for v in scaled)


# --- stage 4: range/tolerance normalizer -----------------------------------------------
_PM_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:±|\+/-|\+-)\s*(\d+(?:\.\d+)?)")


def normalize_tolerance(claim: dict, doc_text: str) -> bool:
    """If a [lo, hi] range claim corresponds to an 'X ± Y' in the source,
    rewrite it to the scalar X (the gold convention). Returns True if changed."""
    v = claim.get("value")
    if not (isinstance(v, list) and len(v) == 2 and all(_is_num(x) for x in v)):
        return False
    lo, hi = sorted(float(x) for x in v)
    center, half = (lo + hi) / 2, (hi - lo) / 2
    for m in _PM_RE.finditer(doc_text):
        x, y = float(m.group(1)), float(m.group(2))
        if _close(center, x) and _close(half, y, rel=0.05):
            claim["value"] = x
            claim["validator_note"] = (claim.get("validator_note", "") +
                                       " tolerance_normalized").strip()
            return True
    return False


# --- stages 1-4 pipeline over one run's claims --------------------------------------------
def validate_claims(claims: list[dict], doc_text: str, doc: str = "",
                    run: str = "") -> ValidationResult:
    res = ValidationResult()
    doc_nums = _doc_numbers(doc_text)
    for raw in claims:
        repaired, reason = schema_gate(raw)
        if reason:
            res.rejected.append(Rejection(doc, run, str(raw.get("property")),
                                          raw.get("value"), reason))
            continue
        for c in repaired:
            c = dict(c)
            if not value_in_source(c, doc_nums):
                res.rejected.append(Rejection(doc, run, c["property"], c["value"],
                                              "value_not_in_source"))
                continue
            if not unit_sane(c):
                lo, hi = PLAUSIBILITY_BOUNDS[c["property"]]
                res.rejected.append(Rejection(doc, run, c["property"], c["value"],
                                              "implausible", f"bounds [{lo}, {hi}]"))
                continue
            if normalize_tolerance(c, doc_text):
                res.normalized += 1
            res.accepted.append(c)
    return res


# --- stage 5: multi-run consensus ------------------------------------------------------------
def consensus(runs: list[list[dict]], min_runs: int = MIN_RUNS) -> list[dict]:
    """Keep claims appearing in >= min_runs runs (property + value-tolerance
    match). Representative = earliest run's instance; consensus_level recorded."""
    entries = [(ri, c) for ri, run in enumerate(runs) for c in run]
    used = [False] * len(entries)
    kept: list[dict] = []
    for i, (ri, ci) in enumerate(entries):
        if used[i]:
            continue
        cluster_runs = {ri}
        used[i] = True
        for j in range(i + 1, len(entries)):
            if used[j]:
                continue
            rj, cj = entries[j]
            if rj in cluster_runs:
                continue                    # one instance per run
            if (cj.get("property") == ci.get("property")
                    and values_match(ci.get("value"), ci.get("unit"),
                                     cj.get("value"), cj.get("unit"))):
                cluster_runs.add(rj)
                used[j] = True
        if len(cluster_runs) >= min_runs:
            rep = dict(ci)
            rep["consensus_level"] = len(cluster_runs)
            kept.append(rep)
    return kept
