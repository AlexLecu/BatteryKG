"""Experiment 13 — scoring only (NO LLM PATH).

Scores the OCR pilot with the existing matcher (`src.agents.evaluation`):
property exact + value within 2 %, each gold claim matched at most once, greedy
in document order — the same rule as experiments 11 and 12.

THREE DENOMINATORS, because the 14 gold claims are not homogeneous:

  all_14          every gold claim
  spec_table_13   claims from the page-1 specification table — the fair test of
                  OCR on a datasheet's structured content
  graph_1         `discharge_cutoff_v`, whose gold `notes` record it as read
                  from a graph annotation rather than the spec table

The graph subset is derived from the gold YAML's `notes` field (claims whose
notes mention "graph"), not hardcoded — `load_gold()` drops `notes`, so the YAML
is re-read here for that one purpose.

Every false negative is assigned to one bucket of an error taxonomy:

  ocr_value_lost      the gold value is not recoverable from the OCR snapshot at
                      all (Validator's own value_in_source says no)
  validator_rejected  a run produced the claim; the Validator rejected it
  consensus_dropped   it survived the Validator in fewer than 2 of 3 runs
  wrong_value         the property was extracted with a different value (the
                      OCR-corruption signature when that value is in the OCR text)
  not_extracted       no run ever proposed the property

Run:  python -m experiments.exp13_ocr_panasonic.score
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.agents.evaluation import (
    DocEvaluation,
    GoldClaim,
    evaluate_document,
    load_gold,
    values_match,
)
from src.agents.validator import _doc_numbers, value_in_source
from src.config import DATA, ROOT
from experiments.exp13_ocr_panasonic.ocr import OCR_TEXT, OUT_DIR
from experiments.exp13_ocr_panasonic.run_ocr_extraction import (
    GOLD_DOC,
    PREDICTIONS_JSON,
    RUNS_JSON,
)

GOLD_YAML = DATA / "claims" / f"{GOLD_DOC}.yaml"
RESULTS_JSON = OUT_DIR / "results.json"


def _load(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — {what}. This module re-scores cached artifacts "
            "only and must never re-run extraction.")
    return json.loads(path.read_text())


# --- subsets -----------------------------------------------------------------
def graph_claim_keys() -> set[tuple[str, str]]:
    """(property, str(value)) of gold claims whose notes mark them graph-read."""
    doc = yaml.safe_load(GOLD_YAML.read_text())
    return {(c["property"], str(c["value"])) for c in doc["claims"]
            if "graph" in (c.get("notes") or "").lower()}


def _key(g: GoldClaim) -> tuple[str, str]:
    return (g.property, str(g.value))


# --- subset recall -----------------------------------------------------------
def subset_recall(subset: list[GoldClaim], preds: list[dict]) -> dict:
    used = [False] * len(preds)
    hits, matched = 0, []
    for g in subset:
        for i, p in enumerate(preds):
            if used[i] or p.get("property") != g.property:
                continue
            if values_match(g.value, g.unit, p.get("value"), p.get("unit")):
                used[i] = True
                hits += 1
                matched.append(f"{g.property}={g.value}")
                break
    return {"n": len(subset), "hits": hits,
            "recall": round(hits / len(subset), 4) if subset else 0.0,
            "matched": matched,
            "missed": [f"{g.property}={g.value}" for g in subset
                       if f"{g.property}={g.value}" not in matched]}


# --- error taxonomy ----------------------------------------------------------
def classify_fn(g: GoldClaim, raw_runs: list[list[dict]], val_reports: list[dict],
                final: list[dict], ocr_nums: list[float]) -> tuple[str, str]:
    if not value_in_source({"value": g.value, "unit": g.unit}, ocr_nums):
        return "ocr_value_lost", "gold value absent from the OCR snapshot"

    in_runs = [r for r, run in enumerate(raw_runs, 1)
               if any(c.get("property") == g.property
                      and values_match(g.value, g.unit, c.get("value"), c.get("unit"))
                      for c in run)]
    rejections = [(rep["run"], x) for rep in val_reports for x in rep["rejected"]
                  if x["property"] == g.property]
    if in_runs and rejections:
        reasons = sorted({x["reason"] for _, x in rejections})
        return "validator_rejected", (f"proposed in runs {in_runs}, "
                                      f"Validator rejected: {', '.join(reasons)}")
    if in_runs:
        return "consensus_dropped", f"proposed in runs {in_runs} only (< 2/3)"

    same_prop = [c for run in raw_runs for c in run
                 if c.get("property") == g.property]
    if same_prop:
        vals = sorted({str(c.get("value")) for c in same_prop})
        return "wrong_value", f"property extracted with value(s) {vals}"
    return "not_extracted", "property never proposed in any run"


def main() -> dict:
    text = OCR_TEXT.read_text()
    ocr_nums = _doc_numbers(text)
    gold = [g for g in load_gold() if g.doc == GOLD_DOC]
    final = _load(PREDICTIONS_JSON, "run `python -m experiments."
                                    "exp13_ocr_panasonic.run_ocr_extraction` first")
    runs = _load(RUNS_JSON, "the raw runs are needed for the error taxonomy")
    raw_runs, val_reports = runs["raw_runs"], runs["validator"]

    gkeys = graph_claim_keys()
    spec_gold = [g for g in gold if _key(g) not in gkeys]
    graph_gold = [g for g in gold if _key(g) in gkeys]

    ev: DocEvaluation = evaluate_document(GOLD_DOC, gold, final, text)
    metrics = {
        "gold": len(gold), "pred": len(final),
        "tp": ev.tp, "fp": ev.fp, "fn": ev.fn,
        "precision": round(ev.precision, 4), "recall": round(ev.recall, 4),
        "f1": round(ev.f1, 4), "hallucinations": ev.hallucinations,
        "errors": ev.errors,
    }

    denominators = {
        "all_14": subset_recall(gold, final),
        "spec_table_13": subset_recall(spec_gold, final),
        "graph_1": subset_recall(graph_gold, final),
    }

    taxonomy: dict[str, list[str]] = {}
    for g in gold:
        if any(p.get("property") == g.property
               and values_match(g.value, g.unit, p.get("value"), p.get("unit"))
               for p in final):
            continue
        bucket, detail = classify_fn(g, raw_runs, val_reports, final, ocr_nums)
        taxonomy.setdefault(bucket, []).append(
            f"{g.property}={g.value} {g.unit or ''} (page {g.page}) — {detail}")

    ocr_recoverable = sum(
        value_in_source({"value": g.value, "unit": g.unit}, ocr_nums) for g in gold)

    payload = {
        "document": GOLD_DOC,
        "framing": "exploratory pilot — one document, OCR text, frozen pipeline "
                   "unchanged; NOT integrated into the extraction pipeline",
        "snapshot": str(OCR_TEXT.relative_to(ROOT)),
        "snapshot_chars": len(text),
        "model": runs.get("model"),
        "n_runs": runs.get("n_runs"),
        "raw_run_sizes": runs.get("raw_run_sizes"),
        "usage": runs.get("usage"),
        "value_in_source_recoverable": f"{ocr_recoverable}/{len(gold)}",
        "value_metrics": metrics,
        "recall_by_denominator": denominators,
        "error_taxonomy": taxonomy,
        "validator": val_reports,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, indent=1, ensure_ascii=False))

    print(f"\n[exp13] gold {len(gold)}, predictions {len(final)}")
    print(f"[exp13] P={ev.precision:.3f} R={ev.recall:.3f} F1={ev.f1:.3f} "
          f"hallucinations={ev.hallucinations}")
    print(f"[exp13] gold values present in OCR snapshot: {ocr_recoverable}/{len(gold)}")
    for name, d in denominators.items():
        print(f"[exp13] recall {name:14s} {d['hits']}/{d['n']} = {d['recall']:.3f}")
    print("\n[exp13] error taxonomy:")
    for bucket, items in taxonomy.items():
        print(f"  {bucket} ({len(items)}):")
        for it in items:
            print(f"    - {it}")
    print(f"\n[exp13] wrote {RESULTS_JSON}")
    return payload


if __name__ == "__main__":
    main()
