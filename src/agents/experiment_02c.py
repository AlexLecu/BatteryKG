"""Experiment 02c — conditions-focused extraction, evaluated on gold conditions.

Configurations compared on the gold claims matched by the frozen value pipeline:
  (a) single-pass conditions  — stated_conditions from the original extraction
  (b) focused second pass     — one conditions call per claim, run 1 only
  (c) focused + validation +  — 3 runs, field-level majority consensus,
      consensus                 condition-side value-in-source validation

Metrics: claim-level conditions accuracy (same metric as 02/02b), field-level
accuracy per condition group (temperature / charge / discharge / DoD / EOL),
over-extraction rate (conditions invented where gold states none), per-document
breakdown, and LLM cost with/without the stage.

Requires cached extraction + conditions runs (python -m src.agents.pipeline
--with-conditions-pass). Appends '## 02c' to experiment_02_extraction.md.

Run:  python -m src.agents.experiment_02c
"""
from __future__ import annotations

import json

from src.agents.conditions_extractor import (
    COND_RAW_DIR,
    _claim_slug,
    conditions_consensus,
    page_context,
    split_pages,
    validate_conditions,
)
from src.agents.evaluation import (
    _numbers_in,
    _close,
    condition_over_extracted,
    conditions_correct,
    load_gold,
    match_pairs,
)
from src.agents.extractor import RAW_DIR, SYSTEM_PROMPT as VALUE_PROMPT
from src.agents.pdf_text import TEXT_DIR, build_all
from src.agents.validator import consensus, validate_claims
from src.config import OUTPUTS

N_RUNS = 3

# condition-field groups for field-level scoring (numbers-based, so it applies
# uniformly to all three configurations regardless of key naming)
FIELD_GROUPS = {
    "temperature": {"temperature_c", "charge_temperature_c", "discharge_temperature_c"},
    "charge": {"charge_current_a", "charge_voltage_v", "charge_c_rate",
               "charge_time_min", "end_current_ma", "charge_mode"},
    "discharge": {"discharge_current_a", "discharge_c_rate", "discharge_cutoff_v"},
    "dod": {"dod_pct"},
    "eol": {"end_condition", "cycles"},
}


def _group_numbers(fields: dict, group: set[str]) -> list[float]:
    return [n for k, v in fields.items() if k in group for n in _numbers_in(v)]


def _pred_all_numbers(cond: dict) -> list[float]:
    return [n for k, v in cond.items()
            if k != "text" and not str(k).startswith("_") for n in _numbers_in(v)]


def field_level_scores(pairs_with_conds) -> dict[str, tuple[int, int]]:
    """{group: (correct, total)} over gold claims that state that group."""
    scores = {g: [0, 0] for g in FIELD_GROUPS}
    for gold, cond in pairs_with_conds:
        pred_nums = _pred_all_numbers(cond) + _numbers_in(cond.get("text", ""))
        for gname, gfields in FIELD_GROUPS.items():
            gold_nums = _group_numbers(gold.parsed_conditions, gfields)
            if not gold_nums:
                continue
            scores[gname][1] += 1
            if all(any(_close(gn, pn) for pn in pred_nums) for gn in gold_nums):
                scores[gname][0] += 1
    return {g: (c, t) for g, (c, t) in scores.items()}


def _config_metrics(pairs_with_conds) -> dict:
    n = len(pairs_with_conds)
    claim_ok = sum(conditions_correct(g, c) for g, c in pairs_with_conds)
    over = sum(condition_over_extracted(g, c) for g, c in pairs_with_conds)
    per_doc: dict[str, list[int]] = {}
    for g, c in pairs_with_conds:
        d = per_doc.setdefault(g.doc, [0, 0])
        d[1] += 1
        d[0] += int(conditions_correct(g, c))
    return {"n": n, "claim_acc": claim_ok / n if n else 0.0,
            "over_rate": over / n if n else 0.0,
            "fields": field_level_scores(pairs_with_conds),
            "per_doc": {d: (c, t) for d, (c, t) in per_doc.items()}}


def _estimate_tokens_values(texts: dict[str, str]) -> int:
    """chars/4 estimate for the 9 value-extraction calls from cached artifacts."""
    total = 0
    for d, text in texts.items():
        for r in range(1, N_RUNS + 1):
            raw = RAW_DIR / f"{d}_run{r}_attempt1.txt"
            completion = len(raw.read_text()) if raw.exists() else 0
            total += (len(VALUE_PROMPT) + len(text) + completion) // 4
    return total


def main() -> None:
    stats = build_all()
    gold = load_gold()
    docs = sorted(d for d in {g.doc for g in gold} if stats.get(d, {}).get("has_text"))
    texts = {d: (TEXT_DIR / f"{d}.txt").read_text() for d in docs}

    # frozen value pipeline (cached) -> accepted claims + gold pairs
    pairs = []               # (gold, accepted_claim, doc)
    for d in docs:
        runs = [json.loads((RAW_DIR / f"{d}_run{r}_parsed.json").read_text())
                for r in range(1, N_RUNS + 1)]
        validated = [validate_claims(c, texts[d], doc=d, run=f"run{r}").accepted
                     for r, c in enumerate(runs, start=1)]
        accepted = consensus(validated)
        pairs += [(g, p, d) for g, p in match_pairs(gold, accepted, d)]
    print(f"[exp02c] {len(pairs)} gold-matched accepted claims")

    # (a) single-pass conditions carried by the accepted claims
    cfg_a = [(g, p.get("stated_conditions") or {}) for g, p, _ in pairs]

    # (b) focused pass, run 1 only; (c) focused + consensus + validation
    cfg_b, cfg_c, missing = [], [], 0
    for g, p, d in pairs:
        pages = split_pages(texts[d])
        ctx = page_context(pages, p.get("page"))
        runs = []
        for r in range(1, N_RUNS + 1):
            cache = COND_RAW_DIR / f"{d}_{_claim_slug(p)}_run{r}.json"
            if cache.exists():
                runs.append(json.loads(cache.read_text()))
        if not runs:
            missing += 1
            continue
        cfg_b.append((g, runs[0]))
        merged = conditions_consensus(runs)
        validated, _rej = validate_conditions(merged, ctx, doc=d,
                                              claim_property=p.get("property"))
        cfg_c.append((g, validated))
    if missing:
        print(f"[exp02c] WARNING: {missing} claims without cached conditions "
              "(run: python -m src.agents.pipeline --with-conditions-pass)")

    m_a, m_b, m_c = (_config_metrics(c) for c in (cfg_a, cfg_b, cfg_c))

    # --- cost accounting ---------------------------------------------------------
    est_values = _estimate_tokens_values(texts)
    cond_files = list(COND_RAW_DIR.glob("*_run*.json"))
    n_cond_calls = len(cond_files)
    measured = {}
    up = RAW_DIR / "pipeline_usage_with_conditions.json"
    if up.exists():
        measured = json.loads(up.read_text())

    # --- report --------------------------------------------------------------------
    lines = ["", "## 02c: Conditions-focused extraction", ""]
    a = lines.append
    a("Second, conditions-focused LLM pass for accepted claims only (value "
      "pipeline frozen): one call per claim with just the claim's source page, "
      "3 runs, field-level majority consensus (ties -> unspecified), and "
      "condition-side value-in-source validation.\n")

    a("| configuration | n | claim-level cond acc | over-extraction rate |")
    a("|---|---|---|---|")
    for name, m in [("(a) single-pass extractor", m_a),
                    ("(b) focused second pass (1 run)", m_b),
                    ("(c) focused + consensus + validation", m_c)]:
        a(f"| {name} | {m['n']} | {m['claim_acc']:.3f} | {m['over_rate']:.3f} |")

    a("\n### Field-level accuracy (gold claims stating that field group)\n")
    a("| group | (a) single-pass | (b) focused | (c) focused+val+cons |")
    a("|---|---|---|---|")
    for gname in FIELD_GROUPS:
        row = [gname]
        for m in (m_a, m_b, m_c):
            c, t = m["fields"][gname]
            row.append(f"{c}/{t}" + (f" ({c / t:.2f})" if t else " (–)"))
        a("| " + " | ".join(row) + " |")

    a("\n### Per-document claim-level conditions accuracy\n")
    a("| document | (a) | (b) | (c) |")
    a("|---|---|---|---|")
    for d in docs:
        cells = [d]
        for m in (m_a, m_b, m_c):
            c, t = m["per_doc"].get(d, (0, 0))
            cells.append(f"{c}/{t}" + (f" ({c / t:.2f})" if t else ""))
        a("| " + " | ".join(cells) + " |")

    a("\n### Cost (3-document pipeline)\n")
    a("| stage | LLM calls | tokens |")
    a("|---|---|---|")
    a(f"| values only (3 docs x 3 runs) | 9 | ~{est_values:,} (est., chars/4 from cached artifacts) |")
    meas_note = (f"{measured.get('prompt_tokens', 0) + measured.get('completion_tokens', 0):,} measured"
                 if measured else "n/a")
    a(f"| + conditions pass | 9 + {n_cond_calls} = {9 + n_cond_calls} | values est. + {meas_note} |")

    a("\nThe conditions stage multiplies call count by "
      f"~{(9 + n_cond_calls) / 9:.0f}x — the price of per-claim focus.")

    a("\n### Reading the over-extraction rate\n")
    a("The detector is unit-blind (a prediction of 0.6 A is grounded by a quote "
      "saying '600mA'). Manual review of the flagged cases shows two distinct "
      "phenomena rather than free invention: (i) **cross-reference resolution** — "
      "the model attaches protocols from sections the claim's row explicitly "
      "references (e.g. LG 4.3.x tests 'charged per 4.1.1, discharged per 4.1.2'), "
      "which is arguably *richer* than the gold quote and correct in substance; "
      "(ii) **neighbour-row misattribution** — conditions of an adjacent spec row "
      "attached to the wrong claim (e.g. the standard-charge protocol attached to "
      "'Max. Charge Voltage'). Only (ii) is a genuine error; condition-side "
      "value-in-source validation cannot catch it by construction, because the "
      "numbers ARE on the page — they just belong to a different claim. "
      "Distinguishing (i) from (ii) needs row/cell-level grounding, not more "
      "validation.")
    a("\nAlso honestly noted: the EOL-threshold field group scored 0/0 — the "
      "cycle-life claims carrying EOL conditions were largely missed by the "
      "frozen VALUE pipeline (e.g. LG's two cycle-life claims merged into a "
      "range), so their conditions never reached this stage. Improving cycle-life "
      "claim extraction remains the prerequisite for condition-complete "
      "cycle-life claims.")

    report = OUTPUTS / "experiment_02_extraction.md"
    text = report.read_text().split("\n## 02c")[0].rstrip() + "\n"
    report.write_text(text + "\n".join(lines) + "\n")
    print(f"[exp02c] appended -> {report}")

    print("\n=== comparison ===")
    for name, m in [("(a) single-pass", m_a), ("(b) focused 1-run", m_b),
                    ("(c) focused+cons+val", m_c)]:
        print(f"{name:22s} n={m['n']}  claim_acc={m['claim_acc']:.3f}  "
              f"over_extraction={m['over_rate']:.3f}")
    print("field-level (c):", {g: f"{c}/{t}" for g, (c, t) in m_c["fields"].items()})


if __name__ == "__main__":
    main()
