"""Experiment 02 — LLM claim extraction vs the hand-labeled gold standard.

Three extraction runs (temperature 0, fixed seed) over the text-bearing
datasheets, evaluated per document and overall: precision / recall / F1,
conditions accuracy on matched claims, hallucination count, and run-to-run
variance (extraction determinism is itself a finding).

The image-only Panasonic marketing sheet (no text layer) is excluded from
extraction and reported as such — its 14 gold claims would need OCR/vision.

LLM-extracted claims are NOT loaded into the KG (gold standard only) until
these numbers justify an automated pipeline with a Validator stage.

Run:  python -m src.agents.experiment_02
"""
from __future__ import annotations

import json
from itertools import combinations

import pandas as pd

from src.agents.evaluation import combine, evaluate_document, load_gold
from src.agents.extractor import RAW_DIR, extract_claims
from src.agents.llm_client import GROQ_LLAMA33, ModelConfig
from src.agents.pdf_text import TEXT_DIR, build_all
from src.config import OUTPUTS

N_RUNS = 3
MODEL: ModelConfig = GROQ_LLAMA33


def _load_or_extract(doc: str, text: str, run: int) -> list[dict]:
    """Cache parsed extractions per (doc, run) so re-running the script after a
    failure does not re-spend API calls. Delete outputs/extraction_raw/ to force."""
    cache = RAW_DIR / f"{doc}_run{run}_parsed.json"
    if cache.exists():
        return json.loads(cache.read_text())
    claims = extract_claims(text, MODEL, doc_name=doc, run_tag=f"run{run}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(claims, indent=1))
    return claims


def _claim_keys(preds_by_doc: dict[str, list[dict]]) -> set[tuple]:
    keys = set()
    for doc, preds in preds_by_doc.items():
        for p in preds:
            v = p.get("value")
            v = tuple(v) if isinstance(v, list) else v
            keys.add((doc, p.get("property"), str(v)))
    return keys


def main() -> None:
    stats = build_all()                      # refresh extracted text snapshots
    gold = load_gold()

    docs = sorted({g.doc for g in gold})
    text_docs = [d for d in docs if stats.get(d, {}).get("has_text")]
    skipped_docs = [d for d in docs if d not in text_docs]
    texts = {d: (TEXT_DIR / f"{d}.txt").read_text() for d in text_docs}
    n_gold_text = sum(g.doc in text_docs for g in gold)
    n_gold_skip = len(gold) - n_gold_text

    print(f"[exp02] {len(gold)} gold claims; {n_gold_text} on text-bearing docs "
          f"{text_docs}; {n_gold_skip} excluded (image-only: {skipped_docs})")

    # --- three runs ---------------------------------------------------------
    runs: list[dict] = []                    # per run: evals + predictions
    for r in range(1, N_RUNS + 1):
        preds_by_doc = {d: _load_or_extract(d, texts[d], r) for d in text_docs}
        evals = [evaluate_document(d, gold, preds_by_doc[d], texts[d])
                 for d in text_docs]
        total = combine(evals)
        runs.append({"run": r, "evals": evals, "total": total,
                     "keys": _claim_keys(preds_by_doc),
                     "n_pred": sum(len(v) for v in preds_by_doc.values())})
        print(f"[exp02] run {r}: P={total.precision:.3f} R={total.recall:.3f} "
              f"F1={total.f1:.3f} cond={total.conditions_acc:.3f} "
              f"hall={total.hallucinations} (n_pred={runs[-1]['n_pred']})")

    # --- run-to-run variance ---------------------------------------------------
    var_rows = [{"run": x["run"], "n_pred": x["n_pred"],
                 "precision": x["total"].precision, "recall": x["total"].recall,
                 "f1": x["total"].f1, "conditions_acc": x["total"].conditions_acc,
                 "hallucinations": x["total"].hallucinations} for x in runs]
    var_df = pd.DataFrame(var_rows)
    jaccards = [len(a["keys"] & b["keys"]) / max(len(a["keys"] | b["keys"]), 1)
                for a, b in combinations(runs, 2)]

    # --- report (run 1 = canonical tables/error catalog) -------------------------
    r1 = runs[0]
    lines: list[str] = []
    a = lines.append
    a("# Experiment 02 — LLM claim extraction vs manual gold standard\n")
    a(f"**Model:** `{MODEL.model}` via {MODEL.provider}, temperature 0, seed "
      f"{MODEL.seed}, {N_RUNS} runs. **Gold:** {len(gold)} hand-labeled claims; "
      f"**{n_gold_text} evaluable** (text-bearing PDFs: {', '.join(text_docs)}).")
    a(f"**Excluded:** {skipped_docs} — image-only PDF (no text layer), its "
      f"{n_gold_skip} gold claims need OCR/vision extraction; recorded as a finding, "
      "not counted against the model.")
    a("\nMatching: document + property (exact) + value (2% tol, unit-scaled); "
      "conditions scored separately on matches (gold-'unspecified' requires the "
      "prediction not to invent conditions). Hallucination = unmatched prediction "
      "whose value appears nowhere in the source text. Note: hallucination-flagged "
      "items include silent unit conversions (value absent in the document's units) "
      "and schema violations (nested values) — per-item diagnosis in the catalog.\n")

    a("## Results (run 1)\n")
    a("| document | gold | pred | TP | FP | FN | P | R | F1 | cond acc | halluc |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in r1["evals"]:
        n_gold_doc = sum(g.doc == e.doc for g in gold)
        a(f"| {e.doc} | {n_gold_doc} | {e.tp + e.fp} | {e.tp} | {e.fp} | {e.fn} "
          f"| {e.precision:.3f} | {e.recall:.3f} | {e.f1:.3f} "
          f"| {e.conditions_acc:.3f} | {e.hallucinations} |")
    t = r1["total"]
    a(f"| **OVERALL** | {n_gold_text} | {t.tp + t.fp} | {t.tp} | {t.fp} | {t.fn} "
      f"| **{t.precision:.3f}** | **{t.recall:.3f}** | **{t.f1:.3f}** "
      f"| **{t.conditions_acc:.3f}** | **{t.hallucinations}** |")

    a("\n## Run-to-run variance (temperature 0, fixed seed)\n")
    a("| run | n_pred | P | R | F1 | cond acc | halluc |")
    a("|---|---|---|---|---|---|---|")
    for _, r in var_df.iterrows():
        a(f"| {int(r['run'])} | {int(r['n_pred'])} | {r['precision']:.3f} "
          f"| {r['recall']:.3f} | {r['f1']:.3f} | {r['conditions_acc']:.3f} "
          f"| {int(r['hallucinations'])} |")
    a(f"\nF1 spread across runs: {var_df['f1'].max() - var_df['f1'].min():.3f} "
      f"(std {var_df['f1'].std(ddof=0):.4f}). "
      f"Pairwise Jaccard of predicted claim sets: "
      f"{', '.join(f'{j:.3f}' for j in jaccards)} "
      f"(1.0 = byte-identical runs).")

    a("\n## Error catalog (run 1)\n")
    a("`FN` missed gold claim · `FP` spurious extraction (value exists in text) · "
      "`HALL` hallucination (value not in text) · `COND` matched claim, wrong "
      "conditions\n")
    a("```")
    for line in sorted(r1["total"].errors):
        a(line)
    a("```")
    a("\n*LLM-extracted claims are NOT loaded into the KG; the KG carries the "
      "manual gold standard only, pending a decision based on these numbers.*")

    report = OUTPUTS / "experiment_02_extraction.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"[exp02] report -> {report}")


if __name__ == "__main__":
    main()
