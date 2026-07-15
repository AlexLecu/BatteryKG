"""LLM claim extraction from datasheet text (Experiment 02).

extract_claims(pdf_text, model_config) prompts the model to emit claims in
exactly the project claim schema (JSON form of data/claims/README.md), with
explicit instructions to (a) extract ONLY what the document states and
(b) mark unstated conditions as "unspecified" rather than guessing.

Parse failures are retried up to 3 times, feeding the parse error back to the
model. Every raw response is logged under outputs/extraction_raw/.

NOTE: LLM-extracted claims are NOT loaded into the KG — the KG carries the
hand-labeled gold standard only, until extraction quality (experiment 02)
justifies an automated pipeline with a Validator stage.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.agents.llm_client import ModelConfig, complete
from src.config import OUTPUTS

RAW_DIR = OUTPUTS / "extraction_raw"
MAX_PARSE_RETRIES = 3

# Property vocabulary shown to the model (same one the gold standard uses).
PROPERTY_VOCAB = [
    "nominal_capacity_ah", "minimum_capacity_ah", "rated_capacity_ah",
    "nominal_voltage_v", "charge_voltage_v", "max_charge_voltage_v",
    "discharge_cutoff_v", "std_charge_current_a", "fast_charge_current_a",
    "max_charge_current_a", "std_charge_time_h", "std_discharge_current_a",
    "fast_discharge_current_a", "max_cont_discharge_a", "max_cont_discharge_c",
    "cycle_life_cycles", "cycle_life_retention_pct", "mass_g",
    "charge_temp_range_c", "discharge_temp_range_c", "storage_temp_range_c",
    "operating_temp_range_c", "internal_impedance_mohm",
    "volumetric_energy_density_wh_l", "gravimetric_energy_density_wh_kg",
    "storage_capacity_remaining_pct", "storage_capacity_recovery_pct",
    "capacity_retention_at_temp_pct",
]

SYSTEM_PROMPT = f"""You extract quantitative manufacturer claims from battery datasheets.

Output ONLY a JSON array (no prose, no markdown fences). Each element:
{{
  "property": "<snake_case id — use the vocabulary below when it fits, else a new snake_case name>",
  "value": <number, or [low, high] for a range>,
  "unit": "<Ah|mAh|V|A|mA|C|cycles|pct|degC|g|h|min|mOhm|Wh/l|Wh/kg|...>",
  "page": <integer — the PAGE marker the claim appears under>,
  "stated_conditions": {{
    "text": "<the condition wording from the document, or \\"unspecified\\" if the document states no conditions for this claim>",
    // optional parsed fields ONLY when the document states them:
    // "temperature_c": <num>, "charge_current_a": <num>, "discharge_current_a": <num>,
    // "charge_c_rate": <num>, "discharge_c_rate": <num>, "dod_pct": <num>,
    // "end_condition": "<text>"
  }}
}}

Property vocabulary: {", ".join(PROPERTY_VOCAB)}

Hard rules:
1. Extract ONLY what the document explicitly states. Never infer, convert from a
   graph you cannot see, or fill in typical values.
2. If a claim's conditions are not stated in the document, set
   stated_conditions.text to "unspecified". Do NOT guess conditions.
3. Every quantitative specification is a claim (capacity, voltage, currents,
   cycle life, temperature ranges, mass, impedance, energy density, ...).
   Qualitative statements (e.g. "no leakage") are not.
4. Use the page number from the "=== PAGE n ===" markers.
5. Prefer base units already used in the document; ranges as [low, high]."""


def _parse_json_claims(raw: str) -> list[dict]:
    """Parse the model output into a list of claim dicts (raises on failure)."""
    text = raw.strip()
    # tolerate accidental markdown fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array found in response")
    claims = json.loads(text[start:end + 1])
    if not isinstance(claims, list):
        raise ValueError("top-level JSON is not an array")
    for i, c in enumerate(claims):
        if not isinstance(c, dict):
            raise ValueError(f"claim {i} is not an object")
        for key in ("property", "value"):
            if key not in c:
                raise ValueError(f"claim {i} missing required key '{key}'")
        if not isinstance(c.get("stated_conditions"), dict):
            c["stated_conditions"] = {"text": "unspecified"}
        c["stated_conditions"].setdefault("text", "unspecified")
    return claims


def extract_claims(pdf_text: str, model_config: ModelConfig,
                   doc_name: str = "document", run_tag: str = "run1") -> list[dict]:
    """Extract claims from one document's text. Returns a list of claim dicts."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    user = f"Datasheet text:\n\n{pdf_text}"
    last_err: Exception | None = None
    for attempt in range(1, MAX_PARSE_RETRIES + 1):
        raw = complete(SYSTEM_PROMPT, user, model_config)
        (RAW_DIR / f"{doc_name}_{run_tag}_attempt{attempt}.txt").write_text(raw)
        try:
            return _parse_json_claims(raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            print(f"[extractor] {doc_name} {run_tag}: parse failure "
                  f"(attempt {attempt}/{MAX_PARSE_RETRIES}): {e}")
            # feed the parse error back for the next attempt
            user = (f"Datasheet text:\n\n{pdf_text}\n\n"
                    f"Your previous output could not be parsed ({e}). "
                    "Return ONLY a valid JSON array of claim objects.")
    raise RuntimeError(f"{doc_name}: unparseable after {MAX_PARSE_RETRIES} attempts: {last_err}")
