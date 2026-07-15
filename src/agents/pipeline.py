"""End-to-end extraction pipeline (frozen value stage + optional conditions).

    text extraction -> 3 extraction runs -> Validator per run -> consensus
    [-- --with-conditions-pass --> focused conditions pass per accepted claim
       (3 runs, field-level consensus, condition-side validation)]

Cost-accounted: reports total LLM calls and tokens with/without the stage.
Extraction runs and conditions calls are cached under outputs/extraction_raw/,
so a re-run only spends on cache misses.

Run:  python -m src.agents.pipeline [--with-conditions-pass]
"""
from __future__ import annotations

import json
import sys
import time

from src.agents import llm_client
from src.agents.conditions_extractor import (
    conditions_consensus,
    extract_conditions,
    page_context,
    split_pages,
    validate_conditions,
)
from src.agents.extractor import RAW_DIR, extract_claims
from src.agents.llm_client import GROQ_LLAMA33, ModelConfig
from src.agents.pdf_text import TEXT_DIR, build_all
from src.agents.validator import consensus, validate_claims

N_RUNS = 3
MODEL: ModelConfig = GROQ_LLAMA33
PACE_S = 0.3                     # gentle pacing between per-claim calls


def _extraction_run(doc: str, text: str, run: int) -> list[dict]:
    cache = RAW_DIR / f"{doc}_run{run}_parsed.json"
    if cache.exists():
        return json.loads(cache.read_text())
    claims = extract_claims(text, MODEL, doc_name=doc, run_tag=f"run{run}")
    cache.write_text(json.dumps(claims, indent=1))
    return claims


def run_extraction_pipeline(with_conditions_pass: bool = False,
                            use_cache: bool = True) -> dict:
    """Returns {'claims': {doc: [accepted claims]}, 'usage': {...},
    'condition_rejections': [...]}. Accepted claims carry `conditions`
    (validated, consensus) when the conditions pass is enabled."""
    llm_client.reset_usage()
    stats = build_all()
    docs = [d for d, s in stats.items() if s.get("has_text")]
    texts = {d: (TEXT_DIR / f"{d}.txt").read_text() for d in docs}

    accepted: dict[str, list[dict]] = {}
    for d in docs:
        validated_runs = []
        for r in range(1, N_RUNS + 1):
            claims = _extraction_run(d, texts[d], r)
            validated_runs.append(validate_claims(claims, texts[d], doc=d,
                                                  run=f"run{r}").accepted)
        accepted[d] = consensus(validated_runs)

    cond_rejections = []
    if with_conditions_pass:
        for d in docs:
            pages = split_pages(texts[d])
            for claim in accepted[d]:
                ctx = page_context(pages, claim.get("page"))
                runs = []
                for r in range(1, N_RUNS + 1):
                    runs.append(extract_conditions(claim, ctx, MODEL, d, r,
                                                   use_cache=use_cache))
                    time.sleep(PACE_S)
                merged = conditions_consensus(runs)
                validated, rej = validate_conditions(
                    merged, ctx, doc=d, claim_property=claim.get("property"))
                cond_rejections.extend(rej)
                claim["conditions"] = validated

    n_claims = sum(len(v) for v in accepted.values())
    usage = dict(llm_client.USAGE)
    print(f"[pipeline] {n_claims} accepted claims across {len(docs)} docs "
          f"(conditions pass: {with_conditions_pass})")
    print(f"[pipeline] LLM usage this run: {usage['calls']} calls, "
          f"{usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} "
          f"completion tokens (cached steps cost 0)")
    # persist measured usage of fresh (non-cached) runs for cost reporting
    if usage["calls"]:
        tag = "with_conditions" if with_conditions_pass else "values_only"
        (RAW_DIR / f"pipeline_usage_{tag}.json").write_text(json.dumps(usage))
    return {"claims": accepted, "usage": usage,
            "condition_rejections": cond_rejections}


if __name__ == "__main__":
    run_extraction_pipeline("--with-conditions-pass" in sys.argv[1:])
