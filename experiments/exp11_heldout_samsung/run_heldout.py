"""Experiment 11 — held-out evaluation of the extraction pipeline.

Document: Samsung SDI INR18650-25R "Specification of product" (Rev 1.0, 2014),
frozen text snapshot at data/claims/extracted_text/samsung_inr18650_25r.txt.
This document was NEVER used to develop the prompt, the property vocabulary,
the Validator bounds or the conditions pass — it is a true held-out test.

The gold annotation (data/gold/samsung_inr18650_25r_gold.yaml) is hand-written
and is read ONLY here, at scoring time. It is never shown to the model.

Pipeline (identical to src/agents/pipeline.py, paper configuration):
    3 extraction runs (Llama-3.3-70B / Groq, temperature 0, seed 42, the frozen
    schema-locked prompt) -> deterministic Validator -> >=2/3 consensus
    -> conditions-focused second pass (3 runs, field majority, value-in-source)

Scored configurations:
    (a) raw single run          — run 1, no Validator, no consensus
    (b) consensus only          — >=2/3 over the RAW runs, no Validator
    (c) consensus + Validator   — the paper configuration

Metrics are exactly src.agents.evaluation: property exact + value within 2 %
(unit-scaled), each gold claim matched at most once (greedy, document order);
conditions scored separately on matched pairs; a predicted value absent from
the source text is a hallucination.

Run:  python -m experiments.exp11_heldout_samsung.run_heldout
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from src.agents import llm_client
from src.agents.conditions_extractor import (
    conditions_consensus,
    extract_conditions,
    page_context,
    split_pages,
    validate_conditions,
)
from src.agents.evaluation import (
    GoldClaim,
    condition_over_extracted,
    conditions_correct,
    evaluate_document,
    match_pairs,
)
from src.agents.extractor import RAW_DIR, extract_claims
from src.agents.llm_client import GROQ_LLAMA33, ModelConfig
from src.agents.pdf_text import TEXT_DIR
from src.agents.validator import consensus, validate_claims
from src.config import DATA, ROOT

DOC = "samsung_inr18650_25r"
GOLD_YAML = DATA / "gold" / f"{DOC}_gold.yaml"
OUT_DIR = ROOT / "experiments" / "exp11_heldout_samsung"

N_RUNS = 3
MODEL: ModelConfig = GROQ_LLAMA33
PACE_S = 0.3


# --- gold (read at scoring time only) ----------------------------------------
def load_heldout_gold() -> list[GoldClaim]:
    doc_yaml = yaml.safe_load(GOLD_YAML.read_text())
    out = []
    for cl in doc_yaml["claims"]:
        cond = cl.get("stated_conditions") or {}
        out.append(GoldClaim(
            doc=DOC, cell_model=doc_yaml["cell_model"],
            property=cl["property"], value=cl["value"], unit=cl.get("unit"),
            page=cl.get("page"), conditions_text=cond.get("text", "unspecified"),
            parsed_conditions={k: v for k, v in cond.items() if k != "text"}))
    return out


# --- extraction --------------------------------------------------------------
def extraction_run(text: str, run: int) -> list[dict]:
    cache = RAW_DIR / f"{DOC}_run{run}_parsed.json"
    if cache.exists():
        return json.loads(cache.read_text())
    claims = extract_claims(text, MODEL, doc_name=DOC, run_tag=f"run{run}")
    cache.write_text(json.dumps(claims, indent=1))
    return claims


def _ev_row(name: str, ev, n_gold: int, n_pred: int) -> dict:
    return {
        "config": name, "gold": n_gold, "pred": n_pred,
        "tp": ev.tp, "fp": ev.fp, "fn": ev.fn,
        "precision": round(ev.precision, 4), "recall": round(ev.recall, 4),
        "f1": round(ev.f1, 4),
        "conditions_acc": round(ev.conditions_acc, 4),
        "hallucinations": ev.hallucinations,
        "errors": ev.errors,
    }


def main() -> dict:
    llm_client.reset_usage()
    text = (TEXT_DIR / f"{DOC}.txt").read_text()
    gold = load_heldout_gold()
    n_gold = len(gold)
    print(f"[exp11] gold claims: {n_gold}; snapshot {len(text):,} chars")

    raw_runs = [extraction_run(text, r) for r in range(1, N_RUNS + 1)]
    for r, cl in enumerate(raw_runs, 1):
        print(f"[exp11] run{r}: {len(cl)} raw claims")

    validated = []
    val_reports = []
    for r, cl in enumerate(raw_runs, 1):
        vr = validate_claims(cl, text, doc=DOC, run=f"run{r}")
        validated.append(vr.accepted)
        val_reports.append({
            "run": f"run{r}", "in": len(cl), "accepted": len(vr.accepted),
            "normalized": vr.normalized,
            "rejected": [asdict(x) for x in vr.rejected],
        })
        print(f"[exp11] run{r}: validator {len(cl)} -> {len(vr.accepted)} "
              f"({len(vr.rejected)} rejected, {vr.normalized} tolerance-normalized)")

    cfg_preds = {
        "raw_single_run": raw_runs[0],
        "consensus_only": consensus(raw_runs),
        "consensus_validator": consensus(validated),
    }

    results = {}
    for name, preds in cfg_preds.items():
        ev = evaluate_document(DOC, gold, preds, text)
        results[name] = _ev_row(name, ev, n_gold, len(preds))
        print(f"[exp11] {name:22s} n={len(preds):3d} P={ev.precision:.3f} "
              f"R={ev.recall:.3f} F1={ev.f1:.3f} cond={ev.conditions_acc:.3f} "
              f"hall={ev.hallucinations}")

    # --- conditions-focused second pass on the paper configuration ------------
    final = cfg_preds["consensus_validator"]
    pages = split_pages(text)
    cond_rejections = []
    for claim in final:
        ctx = page_context(pages, claim.get("page"))
        runs = []
        for r in range(1, N_RUNS + 1):
            runs.append(extract_conditions(claim, ctx, MODEL, DOC, r))
            time.sleep(PACE_S)
        merged = conditions_consensus(runs)
        validated_cond, rej = validate_conditions(
            merged, ctx, doc=DOC, claim_property=claim.get("property"))
        cond_rejections.extend(asdict(x) for x in rej)
        claim["conditions"] = validated_cond

    pairs = match_pairs(gold, final, DOC)
    single_ok = sum(conditions_correct(g, p.get("stated_conditions") or {})
                    for g, p in pairs)
    second_ok = sum(conditions_correct(g, p.get("conditions") or {})
                    for g, p in pairs)
    single_over = sum(condition_over_extracted(g, p.get("stated_conditions") or {})
                      for g, p in pairs)
    second_over = sum(condition_over_extracted(g, p.get("conditions") or {})
                      for g, p in pairs)
    n_pairs = len(pairs)
    cond_errors = [
        f"COND [{g.property}={g.value}] gold '{g.conditions_text[:80]}' "
        f"| parsed {g.parsed_conditions} | pred2 "
        f"{ {k: v for k, v in (p.get('conditions') or {}).items() if k != 'text'} }"
        for g, p in pairs
        if not conditions_correct(g, p.get("conditions") or {})
    ]
    conditions_block = {
        "matched_pairs": n_pairs,
        "single_pass": {
            "correct": single_ok,
            "accuracy": round(single_ok / n_pairs, 4) if n_pairs else 0.0,
            "over_extracted": single_over,
            "over_extraction_rate": round(single_over / n_pairs, 4) if n_pairs else 0.0,
        },
        "second_pass": {
            "correct": second_ok,
            "accuracy": round(second_ok / n_pairs, 4) if n_pairs else 0.0,
            "over_extracted": second_over,
            "over_extraction_rate": round(second_over / n_pairs, 4) if n_pairs else 0.0,
        },
        "condition_field_rejections": cond_rejections,
        "errors": cond_errors,
    }
    print(f"[exp11] conditions: single-pass {single_ok}/{n_pairs}, "
          f"second-pass {second_ok}/{n_pairs}, "
          f"over-extraction {second_over}/{n_pairs}")

    payload = {
        "document": DOC,
        "snapshot_chars": len(text),
        "gold_claims": n_gold,
        "model": {"provider": MODEL.provider, "model": MODEL.model,
                  "temperature": MODEL.temperature, "seed": MODEL.seed},
        "n_runs": N_RUNS,
        "raw_run_sizes": [len(r) for r in raw_runs],
        "validator": val_reports,
        "value_metrics": results,
        "conditions": conditions_block,
        "usage": dict(llm_client.USAGE),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=1))
    (OUT_DIR / "predictions_final.json").write_text(json.dumps(final, indent=1))
    print(f"[exp11] wrote {OUT_DIR/'results.json'}")
    return payload


if __name__ == "__main__":
    main()
