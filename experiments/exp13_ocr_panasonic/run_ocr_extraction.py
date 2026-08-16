"""Experiment 13 — the FROZEN extraction pipeline, run on OCR text (LLM CALLS).

Exploratory pilot, not a pipeline integration. It answers one question: fed OCR
text from a document that has no text layer at all, how much does the existing
pipeline recover?

Nothing about the pipeline changes. `extractor.SYSTEM_PROMPT` and
`extract_claims` are imported and called exactly as `exp11/run_heldout.py` calls
them — same schema-locked prompt, same 3 runs, temperature 0, seed 42, same
`validate_claims` and `consensus`. The ONLY difference is the text the pipeline
is handed: `panasonic_ncr18650b_ocr.txt` instead of a pdfplumber snapshot.

VALUE-IN-SOURCE RUNS AGAINST THE OCR SNAPSHOT. That is the honest configuration
— the OCR text is the only "source" this document has — and it means an OCR
error in a value can cause a rejection. Measuring that is part of the point:
rejections attributable to OCR damage are a finding about OCR quality, not a
pipeline failure. (Before any LLM call, all 14 gold values were confirmed
present in the snapshot under the Validator's own `value_in_source`, so no gold
claim is blocked by that gate; see README §3.)

Caches are keyed on the OCR document name, so no existing artifact under
outputs/extraction_raw/ is read or written.

Run:  python -m experiments.exp13_ocr_panasonic.run_ocr_extraction
"""
from __future__ import annotations

import json
from dataclasses import asdict

from src.agents import llm_client
from src.agents.extractor import RAW_DIR, extract_claims
from src.agents.llm_client import GROQ_LLAMA33, ModelConfig
from src.agents.validator import consensus, validate_claims
from experiments.exp13_ocr_panasonic.ocr import OCR_TEXT, OUT_DIR

GOLD_DOC = "panasonic_ncr18650b"          # how the gold identifies the document
CACHE_DOC = "panasonic_ncr18650b_ocr"     # cache key, so nothing collides
PREDICTIONS_JSON = OUT_DIR / "predictions_final.json"
RUNS_JSON = OUT_DIR / "runs_raw.json"

N_RUNS = 3
MODEL: ModelConfig = GROQ_LLAMA33


def extraction_run(text: str, run: int) -> list[dict]:
    """One frozen-pipeline extraction run, cached (identical to exp11's)."""
    cache = RAW_DIR / f"{CACHE_DOC}_run{run}_parsed.json"
    if cache.exists():
        return json.loads(cache.read_text())
    claims = extract_claims(text, MODEL, doc_name=CACHE_DOC, run_tag=f"run{run}")
    cache.write_text(json.dumps(claims, indent=1))
    return claims


def main() -> dict:
    llm_client.reset_usage()
    if not OCR_TEXT.exists():
        raise FileNotFoundError(
            f"{OCR_TEXT} missing — run "
            "`python -m experiments.exp13_ocr_panasonic.ocr` first.")
    text = OCR_TEXT.read_text()
    print(f"[exp13] OCR snapshot: {len(text):,} chars")

    raw_runs = [extraction_run(text, r) for r in range(1, N_RUNS + 1)]
    for r, cl in enumerate(raw_runs, 1):
        print(f"[exp13] run{r}: {len(cl)} raw claims")

    validated, val_reports = [], []
    for r, cl in enumerate(raw_runs, 1):
        vr = validate_claims(cl, text, doc=GOLD_DOC, run=f"run{r}")
        validated.append(vr.accepted)
        val_reports.append({"run": f"run{r}", "in": len(cl),
                            "accepted": len(vr.accepted),
                            "normalized": vr.normalized,
                            "rejected": [asdict(x) for x in vr.rejected]})
        print(f"[exp13] run{r}: validator {len(cl)} -> {len(vr.accepted)} "
              f"({len(vr.rejected)} rejected, {vr.normalized} tolerance-normalized)")

    final = consensus(validated)
    print(f"[exp13] consensus (>= 2/3): {len(final)} claims")

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
        "snapshot_chars": len(text),
        "usage": dict(llm_client.USAGE),
    }, indent=1, ensure_ascii=False))
    print(f"[exp13] usage: {dict(llm_client.USAGE)}")

    from experiments.exp13_ocr_panasonic import score
    return score.main()


if __name__ == "__main__":
    main()
