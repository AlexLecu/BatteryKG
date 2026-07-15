"""Experiment 02b — validated extraction pipeline vs raw extraction.

Pipelines compared against the gold standard (same matcher/metrics as 02):
  A. raw single-run          (run 1, as in experiment 02)
  B. consensus-only          (>=2-of-3 runs agreement, no validation rules)
  C. consensus + Validator   (per-run schema/value-in-source/plausibility/
                              tolerance-normalizer, then consensus)

Also reports the rejected-claim log with reason codes and the false-rejection
rate (rejected claims that would have matched gold), and conditions accuracy
before/after (the Validator cannot repair dropped conditions — LLM-side
weakness, flagged honestly).

Uses the three cached extraction runs from experiment 02 (same inputs -> the
comparison isolates the pipeline's effect). Appends '## 02b' to
outputs/experiment_02_extraction.md.

Run:  python -m src.agents.experiment_02b
"""
from __future__ import annotations

import json

from src.agents.evaluation import combine, evaluate_document, load_gold, values_match
from src.agents.extractor import RAW_DIR
from src.agents.pdf_text import TEXT_DIR, build_all
from src.agents.validator import MIN_RUNS, consensus, validate_claims
from src.config import OUTPUTS

N_RUNS = 3


def _cached_run(doc: str, run: int) -> list[dict]:
    cache = RAW_DIR / f"{doc}_run{run}_parsed.json"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache.name} missing — run experiment 02 first (python -m src.agents.experiment_02)")
    return json.loads(cache.read_text())


def _evaluate_pipeline(name: str, preds_by_doc: dict[str, list[dict]],
                       gold, texts) -> dict:
    evals = [evaluate_document(d, gold, preds_by_doc[d], texts[d])
             for d in sorted(preds_by_doc)]
    total = combine(evals)
    return {"name": name, "evals": evals, "total": total,
            "n_pred": sum(len(v) for v in preds_by_doc.values())}


def main() -> None:
    stats = build_all()
    gold = load_gold()
    text_docs = sorted(d for d in {g.doc for g in gold} if stats.get(d, {}).get("has_text"))
    texts = {d: (TEXT_DIR / f"{d}.txt").read_text() for d in text_docs}
    runs_by_doc = {d: [_cached_run(d, r) for r in range(1, N_RUNS + 1)] for d in text_docs}

    # --- pipeline A: raw single-run (run 1) ----------------------------------
    pipe_a = _evaluate_pipeline(
        "raw single-run", {d: runs_by_doc[d][0] for d in text_docs}, gold, texts)

    # --- pipeline B: consensus only ------------------------------------------
    pipe_b = _evaluate_pipeline(
        f"consensus-only (>={MIN_RUNS}/3)",
        {d: consensus(runs_by_doc[d]) for d in text_docs}, gold, texts)

    # --- pipeline C: Validator per run, then consensus ------------------------
    validated_runs: dict[str, list[list[dict]]] = {}
    rejections = []
    n_normalized = 0
    for d in text_docs:
        validated_runs[d] = []
        for r, claims in enumerate(runs_by_doc[d], start=1):
            res = validate_claims(claims, texts[d], doc=d, run=f"run{r}")
            validated_runs[d].append(res.accepted)
            rejections.extend(res.rejected)
            n_normalized += res.normalized
    pipe_c = _evaluate_pipeline(
        "consensus + Validator",
        {d: consensus(validated_runs[d]) for d in text_docs}, gold, texts)

    # --- false-rejection analysis ----------------------------------------------
    false_rejections = []
    for rej in rejections:
        for g in gold:
            if (g.doc == rej.doc and g.property == rej.property
                    and values_match(g.value, g.unit, rej.value, None)):
                false_rejections.append(rej)
                break
    fr_rate = len(false_rejections) / len(rejections) if rejections else 0.0

    # --- report ------------------------------------------------------------------
    lines = ["", "## 02b: Validated pipeline", ""]
    a = lines.append
    a("Deterministic Validator (schema gate with mechanical unnesting; "
      "value-in-source with number normalization; per-property plausibility "
      "bounds; 'X ± Y' tolerance normalization) applied per run, then "
      f">= {MIN_RUNS}-of-{N_RUNS} consensus. Same cached extractions and metrics "
      "as experiment 02.\n")
    a("| pipeline | n_pred | TP | FP | FN | P | R | F1 | cond acc | halluc |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for p in (pipe_a, pipe_b, pipe_c):
        t = p["total"]
        a(f"| {p['name']} | {p['n_pred']} | {t.tp} | {t.fp} | {t.fn} "
          f"| {t.precision:.3f} | {t.recall:.3f} | {t.f1:.3f} "
          f"| {t.conditions_acc:.3f} | {t.hallucinations} |")

    a("\n### Rejected-claim log (Validator, all runs)\n")
    a(f"{len(rejections)} rejections across {N_RUNS} runs x {len(text_docs)} docs; "
      f"{n_normalized} range claims tolerance-normalized (not rejections). "
      f"**False-rejection rate: {len(false_rejections)}/{len(rejections)} "
      f"({fr_rate:.1%})** — rejected claims that would have matched gold.\n")
    a("```")
    for r in sorted(rejections, key=lambda x: (x.doc, x.run, x.property)):
        a(f"{r.reason:20s} [{r.doc} {r.run}] {r.property}={r.value} {r.detail}")
    a("```")
    if false_rejections:
        a("\nFalse rejections (correct claims lost):")
        for r in false_rejections:
            a(f"- [{r.doc} {r.run}] {r.property}={r.value} ({r.reason})")

    ca, cc = pipe_a["total"].conditions_acc, pipe_c["total"].conditions_acc
    a("\n### Conditions accuracy before/after\n")
    a(f"raw {ca:.3f} -> validated {cc:.3f}. The Validator checks values, not "
      "conditions — dropped charge-protocol conditions (the main LLM weakness "
      "from 02) remain uncorrected by design. Fixing that requires an LLM-side "
      "change (conditions-focused prompting or a second extraction pass), not "
      "more validation.")

    report = OUTPUTS / "experiment_02_extraction.md"
    text = report.read_text().split("\n## 02b")[0].rstrip() + "\n"
    report.write_text(text + "\n".join(lines) + "\n")
    print(f"[exp02b] appended -> {report}")

    print("\n=== comparison ===")
    for p in (pipe_a, pipe_b, pipe_c):
        t = p["total"]
        print(f"{p['name']:28s} n={p['n_pred']:3d}  P={t.precision:.3f} "
              f"R={t.recall:.3f} F1={t.f1:.3f} cond={t.conditions_acc:.3f} "
              f"hall={t.hallucinations}")
    print(f"rejections={len(rejections)}  false-rejections={len(false_rejections)} "
          f"({fr_rate:.1%})  tolerance-normalized={n_normalized}")


if __name__ == "__main__":
    main()
