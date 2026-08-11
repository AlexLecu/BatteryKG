"""Experiment 11 — re-scoring only, against the revised gold (v2).

The gold annotation was revised AFTER the blind extraction, to align it with the
conventions of the four existing gold files. The predictions are NOT re-derived:
they are read from the frozen artifacts of the original blind run. Nothing in
this module can reach an LLM — the extraction runs are loaded from the on-disk
cache and a missing cache file is a hard error, never a new call.

    raw run i          <- outputs/extraction_raw/<doc>_run<i>_parsed.json
    consensus+validator<- predictions_final.json (frozen, conditions attached)

Also emits the gold v1 -> v2 diff, and replays the v1 score as an integrity
check that gold_v1_snapshot.json faithfully represents what was scored before.

Run:  python -m experiments.exp11_heldout_samsung.rescore
"""
from __future__ import annotations

import json

from src.agents.evaluation import (
    GoldClaim,
    condition_over_extracted,
    conditions_correct,
    evaluate_document,
    match_pairs,
    values_match,
)
from src.agents.validator import consensus, validate_claims
from src.agents.extractor import RAW_DIR
from src.agents.pdf_text import TEXT_DIR
from experiments.exp11_heldout_samsung.run_heldout import (
    DOC, N_RUNS, OUT_DIR, load_heldout_gold,
)

V1_SNAPSHOT = OUT_DIR / "gold_v1_snapshot.json"
V2_SNAPSHOT = OUT_DIR / "gold_v2_snapshot.json"

REVISION_NOTE = ("gold revised for annotation-convention alignment after blind "
                 "extraction; predictions unchanged")
REVISION_NOTE_V3 = ("gold property names aligned to the canonical vocabulary after "
                    "blind extraction; names only — values, pages and conditions "
                    "untouched; predictions unchanged")


# --- frozen artifacts (no LLM path) ------------------------------------------
def cached_raw_runs() -> list[list[dict]]:
    runs = []
    for r in range(1, N_RUNS + 1):
        cache = RAW_DIR / f"{DOC}_run{r}_parsed.json"
        if not cache.exists():
            raise FileNotFoundError(
                f"{cache} missing — this script re-scores only and must never "
                "re-run extraction. Restore the cached run or re-run the blind "
                "pipeline deliberately via run_heldout.py.")
        runs.append(json.loads(cache.read_text()))
    return runs


def load_snapshot_gold(path) -> list[GoldClaim]:
    d = json.loads(path.read_text())
    return [GoldClaim(doc=DOC, cell_model="Samsung INR18650-25R",
                      property=c["property"], value=c["value"], unit=c.get("unit"),
                      page=c.get("page"), conditions_text=c["conditions_text"],
                      parsed_conditions=c["parsed"])
            for c in d["claims"]]


# --- gold diff ---------------------------------------------------------------
def _key(g: GoldClaim) -> tuple:
    v = tuple(g.value) if isinstance(g.value, list) else (g.value,)
    return (g.property, v, g.unit, g.page)


def gold_diff(v1: list[GoldClaim], v2: list[GoldClaim]) -> dict:
    k1 = {_key(g): g for g in v1}
    k2 = {_key(g): g for g in v2}
    added = [k for k in k2 if k not in k1]
    removed = [k for k in k1 if k not in k2]

    # a removed+added pair with the same value/unit/page is a rename
    renames = []
    for a in list(added):
        for r in list(removed):
            if a[1:] == r[1:] and a[0] != r[0]:
                renames.append({"from": r[0], "to": a[0],
                                "value": list(a[1]) if len(a[1]) > 1 else a[1][0],
                                "unit": a[2], "page": a[3]})
                added.remove(a)
                removed.remove(r)
                break

    # conditions changed on claims present in both
    cond_changed = []
    for k in k2:
        if k in k1 and k1[k].parsed_conditions != k2[k].parsed_conditions:
            cond_changed.append({"property": k[0],
                                 "v1": k1[k].parsed_conditions,
                                 "v2": k2[k].parsed_conditions})

    def fmt(keys):
        return [{"property": k[0],
                 "value": list(k[1]) if len(k[1]) > 1 else k[1][0],
                 "unit": k[2], "page": k[3]} for k in keys]

    return {
        "n_v1": len(v1), "n_v2": len(v2),
        "added": fmt(added), "removed": fmt(removed),
        "renamed": renames, "conditions_changed": cond_changed,
        "counts": {"added": len(added), "removed": len(removed),
                   "renamed": len(renames), "conditions_changed": len(cond_changed),
                   "unchanged": len(set(k1) & set(k2)) - len(cond_changed)},
    }


# --- scoring -----------------------------------------------------------------
def score_configs(gold: list[GoldClaim], cfg_preds: dict, text: str) -> dict:
    out = {}
    for name, preds in cfg_preds.items():
        for g in gold:
            g.matched = False
        ev = evaluate_document(DOC, gold, [dict(p) for p in preds], text)
        out[name] = {
            "config": name, "gold": len(gold), "pred": len(preds),
            "tp": ev.tp, "fp": ev.fp, "fn": ev.fn,
            "precision": round(ev.precision, 4), "recall": round(ev.recall, 4),
            "f1": round(ev.f1, 4),
            "conditions_acc": round(ev.conditions_acc, 4),
            "hallucinations": ev.hallucinations,
            "errors": ev.errors,
        }
    return out


def score_conditions(gold: list[GoldClaim], final: list[dict]) -> dict:
    pairs = match_pairs(gold, [dict(p) for p in final], DOC)
    # match_pairs returns copies; re-attach by identity of (property, value)
    lookup = {(p.get("property"), json.dumps(p.get("value"))): p for p in final}

    def pred_for(p):
        return lookup.get((p.get("property"), json.dumps(p.get("value"))), p)

    n = len(pairs)
    res = {"matched_pairs": n}
    for tag, field in (("single_pass", "stated_conditions"), ("second_pass", "conditions")):
        ok = sum(conditions_correct(g, pred_for(p).get(field) or {}) for g, p in pairs)
        over = sum(condition_over_extracted(g, pred_for(p).get(field) or {})
                   for g, p in pairs)
        res[tag] = {"correct": ok, "accuracy": round(ok / n, 4) if n else 0.0,
                    "over_extracted": over,
                    "over_extraction_rate": round(over / n, 4) if n else 0.0}
    res["errors"] = [
        f"COND [{g.property}={g.value}] gold '{g.conditions_text[:80]}' "
        f"| parsed {g.parsed_conditions} | pred2 "
        f"{ {k: v for k, v in (pred_for(p).get('conditions') or {}).items() if k != 'text'} }"
        for g, p in pairs
        if not conditions_correct(g, pred_for(p).get("conditions") or {})
    ]
    return res


def source_dependent_inputs() -> list[str]:
    """Which source-dependent inputs are missing, as human-readable reasons.

    Neither ships in the public release: the text snapshot is derived verbatim
    from a copyrighted datasheet, and the cached raw runs quote it back. Both
    are regenerable — the datasheet URL and retrieval date are in
    data/README.md, and `python -m src.agents.pdf_text` rebuilds the snapshot.
    """
    missing = []
    if not (TEXT_DIR / f"{DOC}.txt").exists():
        missing.append(f"text snapshot {TEXT_DIR/f'{DOC}.txt'} (rebuild: download the "
                       "datasheet listed in data/README.md, then "
                       "`python -m src.agents.pdf_text`)")
    if any(not (RAW_DIR / f"{DOC}_run{r}_parsed.json").exists()
           for r in range(1, N_RUNS + 1)):
        missing.append(f"cached blind-run extractions {RAW_DIR}/{DOC}_run*_parsed.json")
    return missing


def _limited(gold, final: list[dict], missing: list[str]) -> dict:
    """Score the frozen predictions when the source text is unavailable.

    What survives: the gold-vs-prediction match, so TP/FP/FN, precision, recall,
    F1 and condition accuracy are exact — none of them reads the document.

    What does NOT: every check that asks "does this value appear in the source".
    That is the hallucination count (a false positive is only a hallucination if
    its value is absent from the text) and the Validator/consensus replay. They
    are reported as None rather than as a number, because scoring them without
    the text would label every false positive a hallucination and contradict the
    headline result of 0.

    Deliberately does not write results.json: the shipped file records the full
    blind run and must not be overwritten with a partial re-score.
    """
    print("[exp11] source text unavailable — scoring the frozen predictions only.")
    for m in missing:
        print(f"[exp11]   missing: {m}")

    row = score_configs(gold, {"consensus_validator": final}, "")["consensus_validator"]
    row["hallucinations"] = None
    row["errors"] = [e for e in row["errors"] if not e.startswith(("HALL", "FP"))]
    results = {"consensus_validator": row}
    conditions = score_conditions(gold, final)

    print(f"[exp11-limited] consensus_validator  n={row['pred']:3d} TP={row['tp']:2d} "
          f"FP={row['fp']:2d} FN={row['fn']:2d} P={row['precision']:.3f} "
          f"R={row['recall']:.3f} F1={row['f1']:.3f} cond={row['conditions_acc']:.3f} "
          "hall=unavailable")
    print("[exp11-limited] hallucination count, Validator replay and gold-version "
          "replays need the document text; see experiments/exp11_heldout_samsung/"
          "README.md for the numbers of the full run.")

    return {"gold_version": "v3", "gold_claims": len(gold),
            "value_metrics": results, "conditions": conditions,
            "source_checks": "unavailable", "source_checks_missing": missing,
            "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "note": "limited re-scoring; no document text available"}}


def main() -> dict:
    gold_v3 = load_heldout_gold()                      # current file
    gold_v1 = load_snapshot_gold(V1_SNAPSHOT)
    gold_v2 = load_snapshot_gold(V2_SNAPSHOT)
    final = json.loads((OUT_DIR / "predictions_final.json").read_text())

    missing = source_dependent_inputs()
    if missing:
        return _limited(gold_v3, final, missing)

    text = (TEXT_DIR / f"{DOC}.txt").read_text()
    raw_runs = cached_raw_runs()
    validated = [validate_claims(cl, text, doc=DOC, run=f"run{r}").accepted
                 for r, cl in enumerate(raw_runs, 1)]
    cfg = {
        "raw_single_run": raw_runs[0],
        "consensus_only": consensus(raw_runs),
        "consensus_validator": consensus(validated),
    }

    # integrity: the re-derived consensus+validator set must equal the frozen one
    derived = sorted((c["property"], json.dumps(c["value"])) for c in cfg["consensus_validator"])
    frozen = sorted((c["property"], json.dumps(c["value"])) for c in final)
    if derived != frozen:
        raise AssertionError("frozen predictions differ from the cached pipeline output")
    cfg["consensus_validator"] = final

    # integrity: replay the archived scores of every earlier gold version
    prev = json.loads((OUT_DIR / "results.json").read_text())
    archived = {"v1": prev.get("value_metrics_gold_v1"),
                "v2": prev.get("value_metrics_gold_v2") or prev.get("value_metrics")}
    replay = {}
    for tag, gold_old in (("v1", gold_v1), ("v2", gold_v2)):
        arch = (archived.get(tag) or {}).get("consensus_validator")
        now = score_configs(gold_old, {"consensus_validator": final},
                            text)["consensus_validator"]
        replay[tag] = bool(arch) and all(
            now[k] == arch[k] for k in ("tp", "fp", "fn", "hallucinations")
        ) and now["errors"] == arch["errors"]
        archived[tag] = {"consensus_validator": now} if not arch else archived[tag]

    results = score_configs(gold_v3, cfg, text)
    conditions = score_conditions(gold_v3, final)
    diff_v1_v2 = prev.get("gold_diff_v1_v2") or gold_diff(gold_v1, gold_v2)
    diff = gold_diff(gold_v2, gold_v3)

    for name, row in results.items():
        print(f"[exp11-v3] {name:22s} n={row['pred']:3d} TP={row['tp']:2d} "
              f"FP={row['fp']:2d} FN={row['fn']:2d} P={row['precision']:.3f} "
              f"R={row['recall']:.3f} F1={row['f1']:.3f} "
              f"cond={row['conditions_acc']:.3f} hall={row['hallucinations']}")
    print(f"[exp11-v3] conditions: single {conditions['single_pass']['correct']}/"
          f"{conditions['matched_pairs']} (over {conditions['single_pass']['over_extracted']}), "
          f"second {conditions['second_pass']['correct']}/{conditions['matched_pairs']} "
          f"(over {conditions['second_pass']['over_extracted']})")
    print(f"[exp11-v3] gold diff v2->v3: {diff['counts']}")
    print(f"[exp11-v3] replay integrity: "
          + ", ".join(f"{k}={'OK' if v else 'MISMATCH'}" for k, v in replay.items()))

    payload = dict(prev)
    payload.update({
        "gold_version": "v3",
        "gold_revision_note": REVISION_NOTE_V3,
        "gold_revision_note_v2": REVISION_NOTE,
        "gold_claims": len(gold_v3),
        "gold_diff_v1_v2": diff_v1_v2,
        "gold_diff_v2_v3": diff,
        "value_metrics": results,
        "conditions": {**conditions,
                       "condition_field_rejections":
                           prev["conditions"]["condition_field_rejections"]},
        "value_metrics_gold_v1": archived["v1"],
        "value_metrics_gold_v2": archived["v2"],
        # Archive the v2-gold conditions score ONCE. `prev["conditions"]` is the
        # previous file's block, which is the v2 score only on the first re-run;
        # afterwards it is this script's own v3 output, so re-reading it every
        # time would quietly overwrite the archive with v3 numbers and lose the
        # comparison the field exists for. Re-running must be a no-op.
        "conditions_gold_v2": prev.get("conditions_gold_v2",
                                       {k: v for k, v in prev["conditions"].items()}),
        "replay_integrity_ok": replay,
        "usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                  "note": "re-scoring only; predictions read from frozen artifacts"},
        "usage_blind_run": prev["usage"],
    })
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=1))
    print(f"[exp11-v3] wrote {OUT_DIR/'results.json'}")
    return payload


if __name__ == "__main__":
    main()
