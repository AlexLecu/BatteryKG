"""Experiment 12 — row-level table grounding pilot (MAKES LLM CALLS).

Reviewer 2 (round 2, comment 2.5): row-level table grounding was identified as
future work but never implemented. This is the pilot that turns the identified
fix into a measured one, on the same held-out document as experiment 11.

What changes relative to the frozen pipeline: ONLY the extraction context. The
frozen pipeline hands the model 86,750 characters of layout-mode text in one
call. Here `table_grid.py` deterministically isolates each table region and the
model sees ONE of them per call — one grid, or one specification row — with the
frozen page text alongside it for conditions.

What does NOT change:
  * the prompt — `extractor.SYSTEM_PROMPT` is imported verbatim and a
    table-grounding addendum is appended (the addendum is the whole diff);
  * the schema — `extractor._parse_json_claims` remains the final schema gate
    for both unit kinds;
  * the Validator — `validate_claims` and `consensus` are imported unchanged,
    so every value still has to pass value-in-source against the FROZEN text
    snapshot, not against the PDF this experiment re-read;
  * the model configuration — Llama-3.3-70B via Groq, temperature 0, seed 42,
    3 runs, >= 2/3 consensus.

Grid cells are named by `table_grid.compose_property_name` from the model's
(family, axis_kind, axis_value); specification rows are named by the model from
the frozen PROPERTY_VOCAB with no naming rule at all. See README §"Convention
leakage" for what that discloses and what it does not.

Every response is cached under outputs/extraction_raw/tables/. Re-running is
free: a cache hit makes zero calls. Nothing under experiments/exp11_heldout_
samsung/ and nothing under src/agents/ is written by this module.

Run:  python -m experiments.exp12_table_grounding.run_grounding
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path

from src.agents import llm_client
from src.agents.conditions_extractor import split_pages
from src.agents.extractor import RAW_DIR, SYSTEM_PROMPT, _parse_json_claims
from src.agents.llm_client import GROQ_LLAMA33, ModelConfig, complete
from src.agents.pdf_text import TEXT_DIR
from src.agents.validator import consensus, validate_claims
from experiments.exp12_table_grounding.table_grid import (
    DOC,
    OUT_DIR,
    TABLES_JSON,
    compose_property_name,
)

TABLE_RAW_DIR = RAW_DIR / "tables"
PREDICTIONS_JSON = OUT_DIR / "predictions_table.json"
RUNS_JSON = OUT_DIR / "runs_raw.json"

N_RUNS = 3
MODEL: ModelConfig = GROQ_LLAMA33
PACE_S = 0.3
MAX_PARSE_RETRIES = 3


# --- prompt: frozen SYSTEM_PROMPT + a per-kind addendum ----------------------
GRID_ADDENDUM = """

TABLE-GROUNDING ADDENDUM — this call.

You are shown ONE table region isolated from the datasheet, plus the text of the
page(s) it sits on. Extract claims from the TABLE REGION only. Use the page text
solely to fill stated_conditions.

The region is a data grid: an axis row naming the condition each column was
measured at, and a row of measured values. Emit ONE claim per data column of the
value row. A grid may also contain a row that is NOT the measured quantity (for
example a cut-off current row) and a column that is NOT data (a fixed condition
stated alongside the axis) — emit nothing for those.

For a grid cell do NOT emit "property". Emit instead:
  "family": one of
      rel_discharge_capacity — capacity measured on DISCHARGE, as a percentage
                               of a reference capacity
      rel_charge_capacity    — capacity resulting from the CHARGE condition the
                               column names, as a percentage of a reference
      rel_capacity           — capacity under a charge PROTOCOL (standard vs
                               rapid), as a percentage of a reference
  "axis_kind": one of temperature_c | current_a | charge_mode
  "axis_value": the column's own axis value — a number for temperature_c and
      current_a, or "std" | "rapid" for charge_mode
and then "value", "unit", "page", "stated_conditions" exactly as specified above.
"""

SPEC_ROW_ADDENDUM = """

TABLE-GROUNDING ADDENDUM — this call.

You are shown ONE row of the specification table, plus the text of the page it
sits on. Extract claims from THAT ROW only. Use the page text solely to fill
stated_conditions.

One row may state MORE THAN ONE claim — a height and a diameter, three storage
temperature ranges, a standard and a rapid charging time. Emit one claim per
stated value; do not merge them into a single claim. Emit "property" as
specified above, using the property vocabulary when it fits.
"""

ADDENDA = {"grid": GRID_ADDENDUM, "spec_row": SPEC_ROW_ADDENDUM}


def unit_context(pages: dict[int, str], unit: dict) -> str:
    """The frozen page text the unit spans — the conditions context."""
    return "\n".join(pages[p] for p in unit["pages"] if p in pages)


def user_message(unit: dict, context: str) -> str:
    span = ", ".join(str(p) for p in unit["pages"])
    return (f"TABLE REGION (page {span}):\n{unit['render']}\n\n"
            f"PAGE TEXT (context for conditions only):\n\n{context}")


# --- parsing: compose grid names, then the frozen schema gate ----------------
def _json_array(raw: str) -> list:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array found in response")
    objs = json.loads(text[start:end + 1])
    if not isinstance(objs, list):
        raise ValueError("top-level JSON is not an array")
    return objs


def parse_unit_response(raw: str, kind: str) -> list[dict]:
    """Grid claims get their property composed deterministically; both kinds
    then pass through the frozen `_parse_json_claims` schema gate."""
    if kind == "spec_row":
        return _parse_json_claims(raw)

    named = []
    for i, o in enumerate(_json_array(raw)):
        if not isinstance(o, dict):
            raise ValueError(f"claim {i} is not an object")
        for key in ("family", "axis_kind", "axis_value", "value"):
            if key not in o:
                raise ValueError(f"claim {i} missing required key '{key}'")
        c = {k: v for k, v in o.items()
             if k not in ("family", "axis_kind", "axis_value", "property")}
        c["property"] = compose_property_name(
            o["family"], o["axis_kind"], o["axis_value"])
        c["grid_axis"] = {"family": o["family"], "axis_kind": o["axis_kind"],
                          "axis_value": o["axis_value"]}
        named.append(c)
    return _parse_json_claims(json.dumps(named))


# --- one cached call per unit per run ----------------------------------------
def extract_unit(unit: dict, context: str, run: int,
                 use_cache: bool = True) -> list[dict]:
    TABLE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = TABLE_RAW_DIR / f"{DOC}_{unit['unit_id']}_run{run}.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())

    system = SYSTEM_PROMPT + ADDENDA[unit["kind"]]
    user = user_message(unit, context)
    last_err: Exception | None = None
    for attempt in range(1, MAX_PARSE_RETRIES + 1):
        raw = complete(system, user, MODEL)
        (TABLE_RAW_DIR / f"{DOC}_{unit['unit_id']}_run{run}_attempt{attempt}.txt"
         ).write_text(raw)
        try:
            claims = parse_unit_response(raw, unit["kind"])
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            print(f"[exp12] {unit['unit_id']} run{run}: parse failure "
                  f"(attempt {attempt}/{MAX_PARSE_RETRIES}): {e}")
            user = (f"{user_message(unit, context)}\n\n"
                    f"Your previous output could not be used ({e}). "
                    "Return ONLY a valid JSON array of claim objects.")
            continue
        for c in claims:
            c["source_unit"] = unit["unit_id"]
            c["unit_kind"] = unit["kind"]
        cache.write_text(json.dumps(claims, indent=1, ensure_ascii=False))
        return claims
    raise RuntimeError(
        f"{unit['unit_id']} run{run}: unparseable after "
        f"{MAX_PARSE_RETRIES} attempts: {last_err}")


def main() -> dict:
    llm_client.reset_usage()
    if not TABLES_JSON.exists():
        raise FileNotFoundError(
            f"{TABLES_JSON} missing — run "
            "`python -m experiments.exp12_table_grounding.table_grid` first.")
    tables = json.loads(TABLES_JSON.read_text())
    units = tables["units"]
    text = (TEXT_DIR / f"{DOC}.txt").read_text()
    pages = split_pages(text)
    print(f"[exp12] {len(units)} extraction units "
          f"({tables['counts']['grid']} grid + {tables['counts']['spec_row']} spec_row), "
          f"{N_RUNS} runs")

    raw_runs: list[list[dict]] = []
    for run in range(1, N_RUNS + 1):
        run_claims: list[dict] = []
        for unit in units:
            cache = TABLE_RAW_DIR / f"{DOC}_{unit['unit_id']}_run{run}.json"
            fresh = not cache.exists()
            claims = extract_unit(unit, unit_context(pages, unit), run)
            run_claims.extend(claims)
            if fresh:
                time.sleep(PACE_S)
        raw_runs.append(run_claims)
        print(f"[exp12] run{run}: {len(run_claims)} raw claims from {len(units)} units")

    validated, val_reports = [], []
    for r, cl in enumerate(raw_runs, 1):
        vr = validate_claims(cl, text, doc=DOC, run=f"run{r}")
        validated.append(vr.accepted)
        val_reports.append({"run": f"run{r}", "in": len(cl),
                            "accepted": len(vr.accepted),
                            "normalized": vr.normalized,
                            "rejected": [asdict(x) for x in vr.rejected]})
        print(f"[exp12] run{r}: validator {len(cl)} -> {len(vr.accepted)} "
              f"({len(vr.rejected)} rejected, {vr.normalized} tolerance-normalized)")

    final = consensus(validated)
    print(f"[exp12] consensus (>= 2/3): {len(final)} claims")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_JSON.write_text(json.dumps(final, indent=1, ensure_ascii=False))
    RUNS_JSON.write_text(json.dumps({
        "raw_runs": raw_runs,
        "raw_run_sizes": [len(r) for r in raw_runs],
        "validator": val_reports,
        "consensus_only": consensus(raw_runs),
        "model": {"provider": MODEL.provider, "model": MODEL.model,
                  "temperature": MODEL.temperature, "seed": MODEL.seed},
        "n_runs": N_RUNS,
        "usage": dict(llm_client.USAGE),
    }, indent=1, ensure_ascii=False))
    print(f"[exp12] usage: {dict(llm_client.USAGE)}")
    print(f"[exp12] wrote {PREDICTIONS_JSON}")

    from experiments.exp12_table_grounding import score
    return score.main()


if __name__ == "__main__":
    main()
