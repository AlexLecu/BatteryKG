"""Experiment 11 — post-hoc error analysis of the held-out run.

run_heldout.py produces the PAPER metric (exact property name + value within
2 %). This script does not change that metric; it only splits the resulting
errors into interpretable buckets, and reports one clearly-labelled DIAGNOSTIC
variant so the paper can say how much of the loss is vocabulary alignment and
how much is genuine extraction failure.

The alias map below is written by hand AFTER seeing the errors. It is an
analysis aid, never an input to extraction or to the headline numbers.

Run:  python -m experiments.exp11_heldout_samsung.analyze
"""
from __future__ import annotations

import json

from src.agents.evaluation import evaluate_document, values_match
from src.config import ROOT
from experiments.exp11_heldout_samsung.run_heldout import (
    DOC, OUT_DIR, TEXT_DIR, load_heldout_gold,
)

# gold property -> the name the model actually used for the same physical spec.
#
# Since gold v3 the gold itself carries canonical vocabulary names, so this map
# no longer absorbs gold-side divergence. What remains is model-side: cases where
# a canonical name existed and the model picked a DIFFERENT canonical name.
# Post-hoc, for the diagnostic scoring only.
ALIASES = {
    # §7.11 is a high-temperature storage RECOVERY test (cf. lg_18650hg2 4.3.2);
    # the model labelled it as capacity REMAINING after storage (cf. 4.3.1).
    "storage_capacity_recovery_pct": "storage_capacity_remaining_pct",
}


def main() -> dict:
    text = (TEXT_DIR / f"{DOC}.txt").read_text()
    gold = load_heldout_gold()
    preds = json.loads((OUT_DIR / "predictions_final.json").read_text())
    results = json.loads((OUT_DIR / "results.json").read_text())

    # --- diagnostic A: alias-normalised property names -----------------------
    alias_gold = []
    for g in gold:
        g2 = type(g)(**{**g.__dict__})
        g2.property = ALIASES.get(g.property, g.property)
        g2.matched = False
        alias_gold.append(g2)
    ev_alias = evaluate_document(DOC, alias_gold, [dict(p) for p in preds], text)

    # --- diagnostic B: property-agnostic (value-only) upper bound ------------
    used = [False] * len(preds)
    value_tp = 0
    for g in gold:
        for i, p in enumerate(preds):
            if used[i]:
                continue
            if values_match(g.value, g.unit, p.get("value"), p.get("unit")):
                used[i] = True
                value_tp += 1
                break

    # --- buckets over the paper-metric error catalog -------------------------
    errs = results["value_metrics"]["consensus_validator"]["errors"]
    renamed_gold = set(ALIASES)
    fn_lines = [e for e in errs if e.startswith("FN")]
    fn_rename = [e for e in fn_lines
                 if any(f"] {gp}=" in e for gp in renamed_gold)]
    fn_missed = [e for e in fn_lines if e not in fn_rename]
    fp_lines = [e for e in errs if e.startswith("FP")]
    aliased_pred_names = set(ALIASES.values())
    fp_rename = [e for e in fp_lines
                 if any(f"] {pp}=" in e for pp in aliased_pred_names)]
    fp_true = [e for e in fp_lines if e not in fp_rename]

    # table-cell claims the model never attempted (7.6-7.9 rate/temperature
    # tables) vs. ordinary spec rows it missed
    fn_table = [e for e in fn_missed if "] rel_" in e]
    fn_spec = [e for e in fn_missed if "] rel_" not in e]

    # --- what the gold revision resolved -------------------------------------
    # baseline = the gold version immediately before the current one
    prev = results.get("value_metrics_gold_v2") or results.get("value_metrics_gold_v1")
    resolved = None
    if prev:
        p1 = prev["consensus_validator"]
        p2 = results["value_metrics"]["consensus_validator"]
        prev_fp = {e for e in p1["errors"] if e.startswith("FP")}
        prev_fn = {e for e in p1["errors"] if e.startswith("FN")}
        now_fp = {e for e in p2["errors"] if e.startswith("FP")}
        now_fn = {e for e in p2["errors"] if e.startswith("FN")}
        resolved = {
            "note": results.get("gold_revision_note"),
            "baseline": "v2" if results.get("value_metrics_gold_v2") else "v1",
            "gold_diff": (results.get("gold_diff_v2_v3")
                          or results.get("gold_diff_v1_v2", {})).get("counts"),
            "fp_resolved": sorted(prev_fp - now_fp),
            "fn_resolved": sorted(prev_fn - now_fn),
            "fp_new": sorted(now_fp - prev_fp),
            "fn_new": sorted(now_fn - prev_fn),
            "delta": {"tp": p2["tp"] - p1["tp"], "fp": p2["fp"] - p1["fp"],
                      "fn": p2["fn"] - p1["fn"],
                      "f1": round(p2["f1"] - p1["f1"], 4)},
        }

    out = {
        "gold_version": results.get("gold_version", "v1"),
        "gold_revision_note": results.get("gold_revision_note"),
        "resolved_by_gold_revision": resolved,
        "paper_metric": {k: {m: v for m, v in row.items() if m != "errors"}
                         for k, row in results["value_metrics"].items()},
        "diagnostic_alias_normalised": {
            "note": "post-hoc property-name alias map; NOT the paper metric",
            "aliases": ALIASES,
            "tp": ev_alias.tp, "fp": ev_alias.fp, "fn": ev_alias.fn,
            "precision": round(ev_alias.precision, 4),
            "recall": round(ev_alias.recall, 4),
            "f1": round(ev_alias.f1, 4),
            "hallucinations": ev_alias.hallucinations,
        },
        "diagnostic_value_only": {
            "note": "property-agnostic value match; recall upper bound only",
            "tp": value_tp, "gold": len(gold), "pred": len(preds),
            "recall": round(value_tp / len(gold), 4),
            "precision": round(value_tp / len(preds), 4),
        },
        "error_buckets": {
            "fn_total": len(fn_lines),
            "fn_property_rename": len(fn_rename),
            "fn_missed_table_cells": len(fn_table),
            "fn_missed_spec_rows": len(fn_spec),
            "fp_total": len(fp_lines),
            "fp_property_rename": len(fp_rename),
            "fp_genuine_over_extraction": len(fp_true),
            "hallucinations": results["value_metrics"]["consensus_validator"]["hallucinations"],
        },
        "catalog": {
            "fn_property_rename": fn_rename,
            "fn_missed_table_cells": fn_table,
            "fn_missed_spec_rows": fn_spec,
            "fp_property_rename": fp_rename,
            "fp_genuine_over_extraction": fp_true,
            "conditions": results["conditions"]["errors"],
        },
    }
    (OUT_DIR / "analysis.json").write_text(json.dumps(out, indent=1))
    b = out["error_buckets"]
    print(f"[exp11] FN {b['fn_total']} = {b['fn_property_rename']} rename "
          f"+ {b['fn_missed_table_cells']} table cells + {b['fn_missed_spec_rows']} spec rows")
    print(f"[exp11] FP {b['fp_total']} = {b['fp_property_rename']} rename "
          f"+ {b['fp_genuine_over_extraction']} over-extraction; "
          f"{b['hallucinations']} hallucinations")
    d = out["diagnostic_alias_normalised"]
    print(f"[exp11] alias-normalised diagnostic: P={d['precision']:.3f} "
          f"R={d['recall']:.3f} F1={d['f1']:.3f}")
    print(f"[exp11] value-only recall upper bound: "
          f"{out['diagnostic_value_only']['recall']:.3f}")
    return out


if __name__ == "__main__":
    main()
