"""Conditions-focused second LLM pass (experiment 02c).

The value pipeline (consensus + Validator) is frozen; this stage runs one
focused call per ACCEPTED claim: given (property, value, unit) and only the
claim's source page text, extract the exact stated test/measurement
conditions. Three runs are merged by FIELD-LEVEL majority (ties -> the field
is dropped, i.e. 'unspecified' — the conservative choice), and every numeric
condition must pass the same value-in-source check as claim values
(condition_not_in_source -> field dropped + logged).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.agents.evaluation import _close, _numbers_in
from src.agents.llm_client import ModelConfig, complete
from src.agents.validator import _doc_numbers
from src.config import OUTPUTS

COND_RAW_DIR = OUTPUTS / "extraction_raw" / "conditions"
MIN_RUNS = 2
SHORT_PAGE_CHARS = 500          # short pages get one page of surrounding context

# condition fields the model may emit (numeric unless noted)
NUMERIC_FIELDS = [
    "temperature_c", "charge_current_a", "charge_voltage_v", "charge_c_rate",
    "charge_time_min", "discharge_current_a", "discharge_c_rate",
    "discharge_cutoff_v", "dod_pct",
]
TEXT_FIELDS = ["charge_mode", "end_condition", "text"]

SYSTEM_PROMPT = f"""You extract the test/measurement conditions a battery datasheet states for ONE specific claim.

Output ONLY a JSON object. Optional fields — include a field ONLY when the page
text explicitly states it FOR THIS CLAIM:
  numeric: {", ".join(NUMERIC_FIELDS)}
  textual: charge_mode (e.g. "CC-CV"), end_condition (e.g. "capacity >= 60% of nominal")
  text: a short verbatim quote of the stated conditions, or "unspecified" if the
        document states no conditions for this claim.

Hard rules:
1. Extract the exact conditions stated for this specific claim: temperature,
   charge protocol (current, voltage, mode, duration), discharge protocol,
   depth of discharge, end-of-life threshold.
2. Quote only what is stated. For anything not stated, OMIT the field.
   If nothing is stated, output {{"text": "unspecified"}}.
3. Never infer typical values. Every number you output must appear in the
   provided page text."""


# --- page context -----------------------------------------------------------
def split_pages(doc_text: str) -> dict[int, str]:
    parts = re.split(r"=== PAGE (\d+) ===", doc_text)
    pages: dict[int, str] = {}
    for i in range(1, len(parts) - 1, 2):
        pages[int(parts[i])] = parts[i + 1]
    return pages


def page_context(pages: dict[int, str], page: int | None) -> str:
    """The claim's page text; short pages get +/-1 page of context; unknown
    page numbers fall back to the whole document."""
    if page is None or page not in pages:
        return "\n".join(pages[k] for k in sorted(pages))
    txt = pages[page]
    if len("".join(txt.split())) < SHORT_PAGE_CHARS:
        parts = [pages[p] for p in (page - 1, page, page + 1) if p in pages]
        txt = "\n".join(parts)
    return txt


# --- single focused call ---------------------------------------------------------
def _claim_slug(claim: dict) -> str:
    v = claim.get("value")
    v = "-".join(str(x) for x in v) if isinstance(v, list) else str(v)
    return re.sub(r"[^A-Za-z0-9._-]", "_", f"{claim.get('property')}_{v}")[:80]


def _parse_json_object(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in response")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


def extract_conditions(claim: dict, page_text: str, config: ModelConfig,
                       doc: str, run: int, use_cache: bool = True) -> dict:
    """One focused conditions call for one accepted claim (cached per run)."""
    COND_RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = COND_RAW_DIR / f"{doc}_{_claim_slug(claim)}_run{run}.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())

    user = (f"Claim: property={claim.get('property')}, value={claim.get('value')}, "
            f"unit={claim.get('unit')} (stated on this page)\n\nPage text:\n{page_text}")
    raw = complete(SYSTEM_PROMPT, user, config, max_retries=8, backoff=8.0)
    try:
        cond = _parse_json_object(raw)
    except (ValueError, json.JSONDecodeError):
        cond = {"text": "unspecified", "_parse_failed": True}
    cache.write_text(json.dumps(cond, indent=1))
    return cond


# --- field-level consensus ------------------------------------------------------------
def _field_equal(a, b) -> bool:
    na, nb = _numbers_in(a), _numbers_in(b)
    if na and nb and len(na) == len(nb):
        return all(_close(x, y) for x, y in zip(sorted(na), sorted(nb)))
    return str(a).strip().casefold() == str(b).strip().casefold()


def conditions_consensus(runs: list[dict], min_runs: int = MIN_RUNS) -> dict:
    """Field-level majority across runs; no majority -> field dropped
    (conservative 'unspecified')."""
    fields = {k for r in runs for k in r
              if k != "text" and not k.startswith("_")}
    out: dict = {}
    for f in sorted(fields):
        vals = [r[f] for r in runs
                if f in r and str(r[f]).strip().casefold() not in ("", "unspecified", "none", "null")]
        best, best_n = None, 0
        for v in vals:
            n = sum(_field_equal(v, u) for u in vals)
            if n > best_n:
                best, best_n = v, n
        if best_n >= min_runs:
            out[f] = best
    texts = [r.get("text") for r in runs
             if str(r.get("text", "")).strip().casefold() not in ("", "unspecified")]
    out["text"] = texts[0] if texts and out else "unspecified"
    return out


# --- condition-side validation ------------------------------------------------------------
@dataclass
class ConditionRejection:
    doc: str
    claim_property: str
    field: str
    value: object
    reason: str = "condition_not_in_source"


def validate_conditions(cond: dict, page_text: str, doc: str = "",
                        claim_property: str = "") -> tuple[dict, list[ConditionRejection]]:
    """Numeric condition values must appear in the source page (same matcher
    discipline as claim values: abs-matching, Unicode minus, thousand
    separators, magnitude-guarded x1000 rescale). Failures -> field dropped."""
    doc_nums = [abs(d) for d in _doc_numbers(page_text)]

    def in_source(x: float) -> bool:
        cands = [abs(x)]
        if abs(x) < 100:
            cands.append(abs(x) * 1000)
        if abs(x) >= 1000:
            cands.append(abs(x) / 1000)
        return any(_close(c, d) for c in cands for d in doc_nums)

    kept: dict = {}
    rejections: list[ConditionRejection] = []
    for k, v in cond.items():
        if k.startswith("_"):
            continue
        nums = _numbers_in(v) if k != "text" else []
        if nums and not all(in_source(n) for n in nums):
            rejections.append(ConditionRejection(doc, claim_property, k, v))
            continue
        kept[k] = v
    if not any(k for k in kept if k != "text"):
        kept["text"] = "unspecified"
    return kept, rejections
