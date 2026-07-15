"""Promote a staged CandidateSource to a real Source — a HUMAN action.

Promotion is deliberately manual and requires the explicit --confirm flag; the
Literature Monitor never promotes anything itself, regardless of its triage
verdict.

Usage:
    python -m src.agents.promote <candidate_id> --confirm
    python -m src.agents.promote --list          # show pending candidates
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from src.agents.literature_monitor import CANDIDATES_JSONL
from src.kg.connection import get_driver

_PROMOTE_CYPHER = """
MATCH (c:CandidateSource {candidate_id: $cid})
SET c.status = 'promoted', c.promoted_at = $now
MERGE (s:Source {source_id: $source_id})
  SET s.type = 'literature', s.citation = c.title, s.url = $url,
      s.year = c.year, s.promoted_from = $cid, s.retrieved = c.retrieved
RETURN c.title AS title
"""


def promote(candidate_id: str, confirm: bool) -> str:
    """Promote one candidate. Raises without explicit confirmation."""
    if not confirm:
        raise SystemExit(
            "Refusing to promote without --confirm. Promotion creates a real "
            "Source node; review the candidate first:\n"
            f"  python -m src.agents.promote {candidate_id} --confirm")
    # find in the canonical store
    rec = None
    if CANDIDATES_JSONL.exists():
        for line in CANDIDATES_JSONL.read_text().splitlines():
            r = json.loads(line)
            if r["candidate_id"] == candidate_id:
                rec = r
                break
    if rec is None:
        raise SystemExit(f"candidate '{candidate_id}' not found in "
                         f"{CANDIDATES_JSONL}")
    url = (f"https://doi.org/{rec['doi']}" if rec.get("doi")
           else f"https://arxiv.org/abs/{rec['arxiv_id']}" if rec.get("arxiv_id")
           else "")
    source_id = "lit_" + candidate_id.replace(":", "_").replace("/", "_")
    driver = get_driver()
    try:
        with driver.session() as s:
            row = s.run(_PROMOTE_CYPHER, cid=candidate_id, source_id=source_id,
                        url=url,
                        now=datetime.now(timezone.utc).isoformat(timespec="seconds")
                        ).single()
            if row is None:
                raise SystemExit(f"candidate '{candidate_id}' not staged in the KG "
                                 "(run the monitor first)")
    finally:
        driver.close()
    # reflect the status in the canonical JSONL as well
    lines = CANDIDATES_JSONL.read_text().splitlines()
    out = []
    for line in lines:
        r = json.loads(line)
        if r["candidate_id"] == candidate_id:
            r["status"] = "promoted"
        out.append(json.dumps(r))
    CANDIDATES_JSONL.write_text("\n".join(out) + "\n")
    print(f"[promote] '{rec['title'][:70]}' -> Source {source_id}")
    return source_id


def list_pending() -> None:
    if not CANDIDATES_JSONL.exists():
        print("no candidates staged yet — run: "
              "python -m src.agents.literature_monitor run")
        return
    for line in CANDIDATES_JSONL.read_text().splitlines():
        r = json.loads(line)
        if r.get("status") == "pending_review":
            print(f"{r['candidate_id']}\n  [{r.get('triage_verdict', '?'):6s}] "
                  f"{r['title'][:80]} ({r['venue']}, {r['year']})")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--list":
        list_pending()
    else:
        promote(args[0], confirm="--confirm" in args)
