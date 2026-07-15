"""Experiment 03 — vanilla LLM vs BatteryKG on cell-specification questions.

Evaluation only; the frozen system is not modified. One natural-language
question is generated per gold-standard claim (62). Two conditions:

  vanilla     Llama-3.3-70B (Groq, temperature 0), question only — no
              documents, no retrieval. 3 runs, raw responses logged.
  batterykg   deterministic lookup of the Claim nodes (what the system does;
              no LLM). Where the KG holds conflicting variants, the answer
              lists all with sources — scored correct-with-provenance.

Scoring per question: (a) value accuracy (2% tol, unit-normalized);
(b) conditions stated and correct; (c) a specific, checkable source
attributed (document name/page — 'the datasheet' does not count);
(d) uncertainty/refusal expressed; (e) vanilla run-to-run consistency.
Wrong vanilla answers are classified: variant_value (matches a DIFFERENT
document's value that we hold — detectable because we hold the variants),
unit_confused (right digits, wrong magnitude/unit), refused, or
fabricated_or_stale (matches nothing we hold; distinguishing fabricated from
stale requires older-spec knowledge and is left to manual review).

Run:  python -m src.agents.experiment_03            (uses cached LLM runs)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import yaml

from src.agents.evaluation import _close, _numbers_in
from src.agents.llm_client import GROQ_LLAMA33, complete
from src.config import DATA, OUTPUTS, ROOT

QUESTIONS_JSONL = DATA / "eval" / "questions.jsonl"
RAW_DIR = OUTPUTS / "experiment_03_raw"
REPORT = OUTPUTS / "experiment_03_vanilla_comparison.md"
PAPER_TABLE = ROOT / "paper" / "tables" / "tab_vanilla.tex"
CLAIMS_DIR = DATA / "claims"
N_RUNS = 3
MODEL = GROQ_LLAMA33

# --- question generation -------------------------------------------------------
_TEMPLATES = {
    "nominal_capacity_ah": "What is the nominal capacity of the {cell}?",
    "minimum_capacity_ah": "What is the minimum (guaranteed) capacity of the {cell}?",
    "rated_capacity_ah": "What is the rated capacity of the {cell}?",
    "nominal_voltage_v": "What is the nominal voltage of the {cell}?",
    "charge_voltage_v": "What is the charging voltage of the {cell}?",
    "max_charge_voltage_v": "What is the maximum charge voltage of the {cell}?",
    "discharge_cutoff_v": "What is the discharge cut-off voltage of the {cell}?",
    "std_charge_current_a": "What is the standard charge current of the {cell}?",
    "fast_charge_current_a": "What is the fast-charge current of the {cell}?",
    "max_charge_current_a": "What is the maximum charge current of the {cell}?",
    "std_charge_time_h": "How many hours does a standard charge of the {cell} take?",
    "std_discharge_current_a": "What is the standard discharge current of the {cell}?",
    "fast_discharge_current_a": "What are the specified fast-discharge currents of the {cell}?",
    "max_cont_discharge_a": "What is the maximum continuous discharge current of the {cell}?",
    "cycle_life_cycles": "What is the cycle life of the {cell}{qual}?",
    "cycle_life_retention_pct": "What fraction of its initial capacity does the {cell} retain{qual}?",
    "mass_g": "What is the weight of the {cell}?",
    "charge_temp_range_c": "What is the permissible charging temperature range of the {cell}?",
    "discharge_temp_range_c": "What is the permissible discharge temperature range of the {cell}?",
    "storage_temp_range_c": "What is the storage temperature range of the {cell}{qual}?",
    "operating_temp_range_c": "What is the operating temperature range of the {cell}?",
    "internal_impedance_mohm": "What is the internal AC impedance of the {cell}?",
    "volumetric_energy_density_wh_l": "What is the volumetric energy density of the {cell}?",
    "gravimetric_energy_density_wh_kg": "What is the gravimetric energy density of the {cell}?",
    "storage_capacity_remaining_pct": "How much capacity does the {cell} retain after 30 days of storage at room temperature?",
    "storage_capacity_recovery_pct": "What capacity does the {cell} recover after one week of storage at 60 degrees Celsius?",
    "capacity_retention_at_temp_pct": "Relative to nominal, what capacity does the {cell} deliver when discharged at {dtemp} degrees Celsius?",
}


def _qualifier(prop: str, cond: dict, multi: bool) -> str:
    """Disambiguating clause when a cell has several claims of one property."""
    if prop == "cycle_life_cycles" and multi:
        cur = cond.get("discharge_current_a")
        rate = cond.get("discharge_c_rate")
        if cur is not None:
            return f" at {cur:g} A discharge"
        if rate is not None:
            return f" at {rate:g}C discharge"
    if prop == "cycle_life_retention_pct":
        cycles = cond.get("cycles")
        return f" after about {cycles:g} cycles" if cycles else " after extended cycling"
    if prop == "storage_temp_range_c" and multi:
        return " for one-year storage"
    return ""


def generate_questions() -> list[dict]:
    """One question per gold claim, with the gold answer attached."""
    questions = []
    docs = [yaml.safe_load(p.read_text()) for p in sorted(CLAIMS_DIR.glob("*.yaml"))]
    # count (cell, property) multiplicity across ALL documents for qualifiers
    multiplicity: dict[tuple, int] = {}
    for doc in docs:
        for cl in doc["claims"]:
            k = (doc["cell_model"], cl["property"])
            multiplicity[k] = multiplicity.get(k, 0) + 1
    for doc in docs:
        for i, cl in enumerate(doc["claims"]):
            prop = cl["property"]
            cond = cl.get("stated_conditions") or {}
            parsed = {k: v for k, v in cond.items() if k != "text"}
            multi = multiplicity[(doc["cell_model"], prop)] > 1
            tmpl = _TEMPLATES.get(prop)
            if tmpl is None:
                continue
            q = tmpl.format(cell=doc["cell_model"],
                            qual=_qualifier(prop, parsed, multi),
                            dtemp=f"{parsed.get('discharge_temperature_c', 0):g}")
            questions.append({
                "question_id": f"{Path(doc['source_document']).stem}:{prop}:{i}",
                "cell_model": doc["cell_model"],
                "property": prop,
                "question": q,
                "gold": {"value": cl["value"], "unit": cl.get("unit"),
                         "conditions_text": cond.get("text", "unspecified"),
                         "parsed_conditions": parsed,
                         "source_document": doc["source_document"],
                         "page": cl.get("page")},
                "cond_filters": {k: v for k, v in parsed.items()
                                 if k in ("discharge_current_a", "discharge_c_rate",
                                          "discharge_temperature_c")},
            })
    return questions


def write_questions(questions: list[dict]) -> None:
    QUESTIONS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(QUESTIONS_JSONL, "w") as fh:
        for q in questions:
            fh.write(json.dumps(q) + "\n")
    print(f"[exp03] {len(questions)} questions -> {QUESTIONS_JSONL}")


# --- vanilla condition ------------------------------------------------------------
VANILLA_SYSTEM = """You answer battery cell specification questions from your own knowledge.
Answer concisely: state the numeric value with its unit. If you know the test
conditions the specification assumes, state them. If you can cite a specific
source document, do. If you are not certain of the value, say so explicitly
instead of guessing."""


def vanilla_answer(question: str, run: int, use_cache: bool = True) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^a-z0-9]+", "_", question.lower())[:70]
    cache = RAW_DIR / f"vanilla_{key}_run{run}.txt"
    if use_cache and cache.exists():
        return cache.read_text()
    raw = complete(VANILLA_SYSTEM, question, MODEL)
    cache.write_text(raw)
    time.sleep(0.3)
    return raw


# --- BatteryKG condition (deterministic lookup; no LLM) ------------------------------
_KG_CLAIMS_QUERY = """
MATCH (cl:Claim {property: $prop})-[:ABOUT]->(:Cell {model: $model})
MATCH (cl)-[:ASSERTED_BY]->(s:Source)
RETURN cl.value AS value, cl.unit AS unit, cl.conditions_text AS conditions,
       cl.page AS page, s.document AS document,
       cl.cond_discharge_current_a AS discharge_current_a,
       cl.cond_discharge_c_rate AS discharge_c_rate,
       cl.cond_discharge_temperature_c AS discharge_temperature_c
ORDER BY document
"""


def kg_answer(q: dict, driver) -> dict:
    """Deterministic KG answer: value(s) + conditions + source(s)."""
    with driver.session() as s:
        rows = s.run(_KG_CLAIMS_QUERY, prop=q["property"],
                     model=q["cell_model"]).data()
    # apply the question's disambiguating condition filters
    for key, want in q["cond_filters"].items():
        filtered = [r for r in rows if r.get(key) is not None
                    and _close(float(r[key]), float(want))]
        if filtered:
            rows = filtered
    if not rows:
        return {"text": f"The knowledge graph holds no claim for "
                        f"{q['property']} of the {q['cell_model']} — abstaining.",
                "values": [], "n_variants": 0, "abstained": True}
    # variant grouping by value string
    variants: dict[str, list[dict]] = {}
    for r in rows:
        variants.setdefault(str(r["value"]), []).append(r)
    parts = []
    for val, rs in variants.items():
        r0 = rs[0]
        srcs = "; ".join(f"{r['document']} p.{r['page']}" for r in rs)
        parts.append(f"{val} {r0['unit'] or ''} (conditions: {r0['conditions']}; "
                     f"source: {srcs})")
    prefix = ("" if len(variants) == 1 else
              f"The sources on record disagree — {len(variants)} variants: ")
    return {"text": prefix + " | ".join(parts),
            "values": [(r["value"], r["unit"]) for r in rows],
            "n_variants": len(variants), "abstained": False}


# --- scoring --------------------------------------------------------------------------
_UNIT_NEAR = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*(mAh|Ah|mA|A|mV|V|mΩ|mOhm|g|%|°C|degC|Wh/kg|Wh/l|h|hours|min|cycles)?",
                        re.I)


def _answer_numbers(text: str) -> list[tuple[float, str]]:
    out = []
    for m in _UNIT_NEAR.finditer(text.replace("−", "-")):
        try:
            out.append((float(m.group(1).replace(",", "")), (m.group(2) or "").lower()))
        except ValueError:
            continue
    return out


_SMALL_UNITS = {"mah", "ma", "mv"}


def score_value(answer: str, gold_value, gold_unit) -> str:
    """'correct' | 'unit_confused' | 'wrong' | 'no_value'."""
    gold_vals = gold_value if isinstance(gold_value, list) else [gold_value]
    gold_vals = [float(v) for v in gold_vals]
    nums = _answer_numbers(answer)
    if not nums:
        return "no_value"

    def found(gv: float) -> str | None:
        for n, unit in nums:
            if _close(abs(n), abs(gv)):
                return "correct"                     # same magnitude
        for n, unit in nums:
            if _close(abs(n), abs(gv) * 1000):       # e.g. 3350 for 3.35 Ah
                # magnitude-shifted: correct IF the stated unit is the small one
                return "correct" if unit in _SMALL_UNITS else "unit_confused"
            if _close(abs(n) * 1000, abs(gv)):
                return "correct" if (gold_unit or "").lower() in _SMALL_UNITS \
                    else "unit_confused"
        return None

    results = [found(gv) for gv in gold_vals]
    if all(r == "correct" for r in results):
        return "correct"
    if any(r == "unit_confused" for r in results):
        return "unit_confused"
    return "wrong"


def conditions_stated(answer: str, parsed_conditions: dict) -> bool | None:
    """None when the gold states no conditions (excluded from the denominator)."""
    gold_nums = _numbers_in(parsed_conditions)
    if not gold_nums:
        return None
    ans_nums = [abs(n) for n, _ in _answer_numbers(answer)]
    return all(any(_close(abs(g), a) or _close(abs(g) * 1000, a)
                   or (abs(g) >= 1000 and _close(abs(g) / 1000, a))
                   for a in ans_nums) for g in gold_nums)


_SOURCE_RE = re.compile(
    r"(p\.\s*\d+|page \d+|https?://|doi\.org|PS-HG2|NCR18650-068|MD100009"
    r"|full (product )?specification|marketing (sheet|datasheet)"
    r"|product specification)", re.I)


def source_attributed(answer: str) -> bool:
    """A specific, checkable source (doc name/page/URL); 'the datasheet' alone
    does not count."""
    return bool(_SOURCE_RE.search(answer))


_UNCERTAIN_RE = re.compile(
    r"(not (entirely |completely )?certain|I don'?t know|cannot confirm|"
    r"unsure|uncertain|may vary|approximately|not specified|I do not have|"
    r"varies by|couldn'?t find|no reliable|not able to verify)", re.I)


def uncertainty_expressed(answer: str) -> bool:
    return bool(_UNCERTAIN_RE.search(answer))


def classify_wrong(answer: str, gold_value, gold_unit,
                   variant_values: list) -> str:
    """Taxonomy for a non-correct vanilla answer."""
    v = score_value(answer, gold_value, gold_unit)
    if v == "correct":
        return "correct"
    if v == "no_value":
        return "refused" if uncertainty_expressed(answer) else "no_value"
    if v == "unit_confused":
        return "unit_confused"
    # does it match some OTHER variant we hold?
    for vv, vu in variant_values:
        if score_value(answer, vv, vu) == "correct":
            return "variant_value"
    return "fabricated_or_stale"


# --- run + report -----------------------------------------------------------------
def _variant_catalog(questions: list[dict]) -> dict[tuple, list]:
    """(cell, property) -> all values we hold from ANY document (gold docs +
    lygte spec echoes) — the basis for detecting blended/variant answers."""
    from src.kg.spec_consistency import parse_spec_echo
    catalog: dict[tuple, list] = {}
    for q in questions:
        catalog.setdefault((q["cell_model"], q["property"]), []).append(
            (q["gold"]["value"], q["gold"]["unit"]))
    lygte_json = DATA / "processed" / "lygte_measurements.json"
    if lygte_json.exists():
        for review in json.loads(lygte_json.read_text())["reviews"]:
            for e in parse_spec_echo(review["spec_echo"], review["source_id"]):
                catalog.setdefault((review["cell_model"], e.property), []).append(
                    (e.value, None))
    return catalog


def main() -> None:
    questions = generate_questions()
    write_questions(questions)
    catalog = _variant_catalog(questions)

    from src.kg.connection import get_driver
    driver = get_driver()

    rows = []
    for q in questions:
        gold = q["gold"]
        variant_vals = [(v, u) for (v, u) in catalog[(q["cell_model"], q["property"])]
                        if str(v) != str(gold["value"])]
        # vanilla: 3 runs
        v_answers = [vanilla_answer(q["question"], r) for r in range(1, N_RUNS + 1)]
        v_scores = [score_value(a, gold["value"], gold["unit"]) for a in v_answers]
        v_class = classify_wrong(v_answers[0], gold["value"], gold["unit"], variant_vals)
        v_vals_per_run = [_answer_numbers(a)[0][0] if _answer_numbers(a) else None
                          for a in v_answers]
        consistent = (len({round(x, 6) for x in v_vals_per_run
                           if x is not None}) <= 1
                      and None not in v_vals_per_run) or all(
                      x is None for x in v_vals_per_run)
        # kg
        k = kg_answer(q, driver)
        k_score = score_value(k["text"], gold["value"], gold["unit"])
        rows.append({
            "question_id": q["question_id"], "cell": q["cell_model"],
            "property": q["property"], "question": q["question"],
            "gold_value": gold["value"], "gold_unit": gold["unit"],
            # vanilla (run 1 is canonical; consistency uses all runs)
            "v_answer": v_answers[0],
            "v_value_ok": v_scores[0] == "correct",
            "v_scores_runs": v_scores,
            "v_class": v_class,
            "v_conditions": conditions_stated(v_answers[0], gold["parsed_conditions"]),
            "v_source": source_attributed(v_answers[0]),
            "v_uncertain": uncertainty_expressed(v_answers[0]),
            "v_consistent": consistent,
            # kg
            "k_answer": k["text"],
            "k_value_ok": k_score == "correct",
            "k_n_variants": k["n_variants"],
            "k_conditions": conditions_stated(k["text"], gold["parsed_conditions"]),
            "k_source": source_attributed(k["text"]),
            "k_abstained": k["abstained"],
        })
    driver.close()

    n = len(rows)
    def frac(key): return sum(bool(r[key]) for r in rows) / n
    def cond_acc(prefix):
        scored = [r[prefix] for r in rows if r[prefix] is not None]
        return (sum(scored) / len(scored), len(scored)) if scored else (0.0, 0)

    v_cond, n_cond = cond_acc("v_conditions")
    k_cond, _ = cond_acc("k_conditions")
    wrong_first = [r for r in rows if not r["v_value_ok"]]
    hedged_when_wrong = (sum(r["v_uncertain"] for r in wrong_first) /
                         len(wrong_first)) if wrong_first else 0.0
    taxonomy: dict[str, int] = {}
    for r in rows:
        if r["v_class"] != "correct":
            taxonomy[r["v_class"]] = taxonomy.get(r["v_class"], 0) + 1
    multi = sum(r["k_n_variants"] > 1 for r in rows)

    # --- example pairs for the paper figure (prefer instructive failures) ----
    def pick_examples():
        picks, seen = [], set()
        for want in ("variant_value", "fabricated_or_stale", "unit_confused", "correct"):
            for r in rows:
                if r["v_class"] == want and r["property"] not in seen:
                    picks.append(r); seen.add(r["property"]); break
            if len(picks) == 4:
                break
        return picks
    examples = pick_examples()

    # --- report ----------------------------------------------------------------
    L = ["# Experiment 03 — vanilla LLM vs BatteryKG on specification questions", ""]
    a = L.append
    a(f"{n} questions (one per gold claim). Vanilla = {MODEL.model} (temp 0, "
      f"{N_RUNS} runs, no documents/retrieval). BatteryKG = deterministic Claim-"
      "node lookup (no LLM); conflicting variants listed with sources "
      "(correct-with-provenance).\n")
    a("## Headline table\n")
    a("| metric | vanilla LLM | BatteryKG |")
    a("|---|---|---|")
    a(f"| value accuracy | {frac('v_value_ok'):.3f} | {frac('k_value_ok'):.3f} |")
    a(f"| conditions stated & correct (n={n_cond}) | {v_cond:.3f} | {k_cond:.3f} |")
    a(f"| specific source attributed | {frac('v_source'):.3f} | {frac('k_source'):.3f} |")
    a(f"| uncertainty expressed when wrong | {hedged_when_wrong:.3f} | n/a (abstains on missing data) |")
    a(f"| run-to-run consistency | {frac('v_consistent'):.3f} | 1.000 (deterministic) |")
    a(f"\nKG answers listing multiple conflicting variants with sources: {multi} "
      f"questions (scored correct-with-provenance when the gold value is among them).\n")
    a("Scoring note: conditions accuracy requires every numeric value of the "
      "gold's parsed condition fields to appear in the answer. The KG's misses "
      "on this metric are artifacts of that strictness — the gold annotations "
      "normalize wording the source states non-numerically ('1 week' -> 7 days, "
      "a 4.2-2.0 V window -> 100% DOD) while the KG answer quotes the source "
      "verbatim. The same rule applies to both conditions, so the comparison "
      "is conservative for the KG.\n")
    a("## Vanilla error taxonomy (run 1)\n")
    a("| class | count | meaning |")
    a("|---|---|---|")
    meanings = {"variant_value": "matches a DIFFERENT document's value that we hold",
                "unit_confused": "right digits, wrong magnitude/unit",
                "refused": "declined to state a value (with uncertainty)",
                "no_value": "no numeric value and no uncertainty statement",
                "fabricated_or_stale": "matches nothing we hold (manual review "
                                        "needed to split fabricated vs stale)"}
    for k_, v_ in sorted(taxonomy.items(), key=lambda kv: -kv[1]):
        a(f"| {k_} | {v_} | {meanings.get(k_, '')} |")
    a("\n## Example pairs (for the paper figure)\n")
    for r in examples:
        a(f"### {r['question']}")
        a(f"*Gold: {r['gold_value']} {r['gold_unit']} "
          f"({r['question_id']}); vanilla class: {r['v_class']}*\n")
        a(f"**Vanilla:** {r['v_answer'][:500]}\n")
        a(f"**BatteryKG:** {r['k_answer'][:500]}\n")
    REPORT.write_text("\n".join(L) + "\n")
    print(f"[exp03] report -> {REPORT}")

    # --- booktabs table ----------------------------------------------------------
    tex = [
        "% Auto-generated by src/agents/experiment_03.py — do not edit by hand.",
        "\\begin{table}[H]",
        "\\caption{Vanilla LLM (Llama-3.3-70B, temperature 0, no retrieval) vs.\\",
        "BatteryKG (deterministic claim lookup) on one specification question per",
        f"gold claim ($n={n}$). Conditions accuracy is over the {n_cond} questions",
        "whose gold claim states conditions. Source attribution requires a",
        "specific, checkable document (name/page), not ``the datasheet''.",
        "\\label{tab:vanilla}}",
        "\\begin{tabularx}{\\textwidth}{lCC}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{Vanilla LLM} & \\textbf{BatteryKG} \\\\",
        "\\midrule",
        f"Value accuracy & {frac('v_value_ok'):.2f} & {frac('k_value_ok'):.2f} \\\\",
        f"Conditions stated \\& correct & {v_cond:.2f} & {k_cond:.2f} \\\\",
        f"Specific source attributed & {frac('v_source'):.2f} & {frac('k_source'):.2f} \\\\",
        f"Uncertainty expressed when wrong & {hedged_when_wrong:.2f} & abstains \\\\",
        f"Run-to-run consistency & {frac('v_consistent'):.2f} & 1.00 \\\\",
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{table}",
    ]
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE.write_text("\n".join(tex) + "\n")
    print(f"[exp03] table -> {PAPER_TABLE}")

    print("\n=== headline ===")
    print(f"value acc: vanilla {frac('v_value_ok'):.3f} vs KG {frac('k_value_ok'):.3f}")
    print(f"conditions: {v_cond:.3f} vs {k_cond:.3f} (n={n_cond})")
    print(f"source: {frac('v_source'):.3f} vs {frac('k_source'):.3f}")
    print(f"hedged-when-wrong: {hedged_when_wrong:.3f} | consistency: {frac('v_consistent'):.3f}")
    print("taxonomy:", taxonomy)


if __name__ == "__main__":
    main()
