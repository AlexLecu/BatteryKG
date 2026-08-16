"""Experiment 12 — scoring only (NO LLM PATH).

Scores the table-grounding pilot against the SAME 41-claim gold and the SAME
matching rules as experiment 11: `src.agents.evaluation` — property exact +
value within 2 % (unit-scaled), each gold claim matched at most once, greedy in
document order. Nothing here can reach an LLM: predictions come from the cached
artifacts and a missing one is a hard error, never a new call.

Reported:
  * frozen pipeline (exp11's 16 claims, re-scored here so the baseline and the
    pilot come out of the same code path)
  * the table stage alone
  * merged = frozen U table stage (the headline)
  * recall on the 17 table-grid misses, EXACT NAME (the paper metric) and
    VALUE-LEVEL (property-agnostic, unit-class-guarded — the honest lower bound
    on grid reading, independent of the naming convention)
  * the same two views on the 9 missed specification rows
  * delta against the frozen pipeline's recall

The two error buckets are taken from exp11's own `analysis.json` catalog, by
exact FN-line prefix, so this experiment cannot quietly redefine which claims
counted as table-grid misses.

Run:  python -m experiments.exp12_table_grounding.score
"""
from __future__ import annotations

import json
from pathlib import Path

from src.agents.evaluation import (
    DocEvaluation,
    GoldClaim,
    evaluate_document,
    values_match,
)
from src.agents.pdf_text import TEXT_DIR
from src.config import ROOT
from experiments.exp11_heldout_samsung.run_heldout import load_heldout_gold
from experiments.exp12_table_grounding.table_grid import DOC, OUT_DIR
from experiments.exp12_table_grounding.run_grounding import PREDICTIONS_JSON, RUNS_JSON

EXP11_DIR = ROOT / "experiments" / "exp11_heldout_samsung"
FROZEN_PREDICTIONS = EXP11_DIR / "predictions_final.json"
EXP11_ANALYSIS = EXP11_DIR / "analysis.json"
RESULTS_JSON = OUT_DIR / "results.json"

# unit classes for the property-agnostic diagnostic: a value-level match must at
# least agree on what kind of quantity it is
_UNIT_CLASS = {
    "pct": "pct", "%": "pct",
    "degc": "degc", "c": "degc", "℃": "degc", "°c": "degc",
    "a": "current", "ma": "current",
    "v": "volt", "mv": "volt",
    "ah": "capacity", "mah": "capacity",
    "min": "time", "h": "time",
    "mm": "length", "g": "mass", "mohm": "impedance", "cycles": "cycles",
}


def _unit_class(unit: str | None) -> str:
    u = (unit or "").strip().lower()
    return _UNIT_CLASS.get(u, u)


def _load(path: Path, what: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — {what}. This module re-scores cached artifacts "
            "only and must never re-run extraction.")
    return json.loads(path.read_text())


# --- error buckets, taken from exp11's catalog by exact FN-line prefix --------
def _fn_prefix(g: GoldClaim) -> str:
    return f"FN   [{DOC}] {g.property}={g.value} {g.unit or ''}:"


def bucket(gold: list[GoldClaim], catalog_lines: list[str]) -> list[GoldClaim]:
    return [g for g in gold
            if any(line.startswith(_fn_prefix(g)) for line in catalog_lines)]


# --- subset recall -----------------------------------------------------------
def _greedy_recall(subset: list[GoldClaim], preds: list[dict],
                   exact_property: bool) -> tuple[int, list[str]]:
    used = [False] * len(preds)
    hits, matched = 0, []
    for g in subset:
        for i, p in enumerate(preds):
            if used[i]:
                continue
            if exact_property and p.get("property") != g.property:
                continue
            if (not exact_property
                    and _unit_class(g.unit) != _unit_class(p.get("unit"))):
                continue
            if values_match(g.value, g.unit, p.get("value"), p.get("unit")):
                used[i] = True
                hits += 1
                matched.append(f"{g.property}={g.value} <- {p.get('property')}"
                               f"={p.get('value')} [{p.get('source_unit', '-')}]")
                break
    return hits, matched


def _missed(subset: list[GoldClaim], matched: list[str]) -> list[str]:
    got = {m.split(" <- ")[0] for m in matched}
    return [f"{g.property}={g.value}" for g in subset
            if f"{g.property}={g.value}" not in got]


# --- merge -------------------------------------------------------------------
def merge_predictions(frozen: list[dict], table: list[dict]) -> tuple[list[dict], list[str]]:
    """Frozen predictions plus the table stage's, dropping table claims that
    restate a frozen one (same property, value within tolerance)."""
    merged, dropped = list(frozen), []
    for p in table:
        dup = any(f.get("property") == p.get("property")
                  and values_match(f.get("value"), f.get("unit"),
                                   p.get("value"), p.get("unit"))
                  for f in frozen)
        if dup:
            dropped.append(f"{p.get('property')}={p.get('value')}")
        else:
            merged.append(p)
    return merged, dropped


def _row(name: str, ev: DocEvaluation, n_pred: int, n_gold: int) -> dict:
    return {"config": name, "gold": n_gold, "pred": n_pred,
            "tp": ev.tp, "fp": ev.fp, "fn": ev.fn,
            "precision": round(ev.precision, 4), "recall": round(ev.recall, 4),
            "f1": round(ev.f1, 4), "hallucinations": ev.hallucinations,
            "errors": ev.errors}


def main() -> dict:
    text = (TEXT_DIR / f"{DOC}.txt").read_text()
    gold = load_heldout_gold()
    frozen = _load(FROZEN_PREDICTIONS, "exp11's frozen predictions are required "
                                       "as the baseline and merge partner")
    table = _load(PREDICTIONS_JSON, "run `python -m experiments."
                                    "exp12_table_grounding.run_grounding` first")
    analysis = json.loads(EXP11_ANALYSIS.read_text())
    runs_meta = json.loads(RUNS_JSON.read_text()) if RUNS_JSON.exists() else {}

    grid_gold = bucket(gold, analysis["catalog"]["fn_missed_table_cells"])
    spec_gold = bucket(gold, analysis["catalog"]["fn_missed_spec_rows"])
    assert len(grid_gold) == 17, f"expected 17 table-grid misses, got {len(grid_gold)}"
    assert len(spec_gold) == 9, f"expected 9 spec-row misses, got {len(spec_gold)}"

    merged, dropped_dups = merge_predictions(frozen, table)

    configs = {
        "frozen_pipeline_exp11": frozen,
        "table_stage_only": table,
        "merged": merged,
    }
    metrics = {name: _row(name, evaluate_document(DOC, gold, preds, text),
                          len(preds), len(gold))
               for name, preds in configs.items()}

    # subset recalls; the value-level view is confined to the unit kind that is
    # supposed to recover that bucket, so a coincidental number elsewhere in the
    # prediction set cannot absorb a gold claim
    grid_preds = [p for p in table if p.get("unit_kind") == "grid"]
    spec_preds = [p for p in table if p.get("unit_kind") == "spec_row"]

    def subset_block(subset, exact_pool, value_pool) -> dict:
        ex_hits, ex_matched = _greedy_recall(subset, exact_pool, True)
        va_hits, va_matched = _greedy_recall(subset, value_pool, False)
        return {
            "n": len(subset),
            "exact_name": {"hits": ex_hits, "recall": round(ex_hits / len(subset), 4),
                           "matched": ex_matched,
                           "missed": _missed(subset, ex_matched)},
            "value_level": {"hits": va_hits, "recall": round(va_hits / len(subset), 4),
                            "matched": va_matched,
                            "missed": _missed(subset, va_matched)},
        }

    buckets = {
        "table_grid_misses": subset_block(grid_gold, merged, grid_preds),
        "spec_row_misses": subset_block(spec_gold, merged, spec_preds),
    }

    # which family the model picked per grid region (the family-choice finding)
    family_choices: dict[str, dict] = {}
    for p in table:
        if p.get("unit_kind") != "grid":
            continue
        u = p.get("source_unit", "-")
        fam = (p.get("grid_axis") or {}).get("family")
        entry = family_choices.setdefault(u, {"families": {}, "properties": []})
        entry["families"][fam] = entry["families"].get(fam, 0) + 1
        entry["properties"].append(f"{p.get('property')}={p.get('value')}")

    base_r = metrics["frozen_pipeline_exp11"]["recall"]
    delta = {
        "recall": round(metrics["merged"]["recall"] - base_r, 4),
        "precision": round(metrics["merged"]["precision"]
                           - metrics["frozen_pipeline_exp11"]["precision"], 4),
        "f1": round(metrics["merged"]["f1"] - metrics["frozen_pipeline_exp11"]["f1"], 4),
        "baseline_recall": base_r,
    }

    payload = {
        "document": DOC,
        "gold_claims": len(gold),
        "model": runs_meta.get("model"),
        "n_runs": runs_meta.get("n_runs"),
        "raw_run_sizes": runs_meta.get("raw_run_sizes"),
        "validator": runs_meta.get("validator"),
        "usage": runs_meta.get("usage"),
        "merge": {"frozen": len(frozen), "table_stage": len(table),
                  "merged": len(merged), "dropped_duplicates": dropped_dups},
        "value_metrics": metrics,
        "buckets": buckets,
        "grid_family_choices": family_choices,
        "delta_vs_frozen": delta,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, indent=1, ensure_ascii=False))

    # --- response-letter table ---
    print(f"\n{'config':24s} {'pred':>5s} {'TP':>4s} {'FP':>4s} {'FN':>4s} "
          f"{'P':>7s} {'R':>7s} {'F1':>7s} {'hall':>5s}")
    for name, m in metrics.items():
        print(f"{name:24s} {m['pred']:5d} {m['tp']:4d} {m['fp']:4d} {m['fn']:4d} "
              f"{m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} "
              f"{m['hallucinations']:5d}")
    for key, b in buckets.items():
        print(f"\n{key} (n={b['n']}): exact-name {b['exact_name']['hits']}/{b['n']} "
              f"= {b['exact_name']['recall']:.3f}   value-level "
              f"{b['value_level']['hits']}/{b['n']} = {b['value_level']['recall']:.3f}")
        if b["exact_name"]["missed"]:
            print(f"  still missed (exact): {', '.join(b['exact_name']['missed'])}")
    print(f"\ndelta vs frozen: recall {base_r:.3f} -> "
          f"{metrics['merged']['recall']:.3f} ({delta['recall']:+.3f}), "
          f"F1 {delta['f1']:+.3f}, precision {delta['precision']:+.3f}")
    print(f"[exp12] wrote {RESULTS_JSON}")
    return payload


if __name__ == "__main__":
    main()
