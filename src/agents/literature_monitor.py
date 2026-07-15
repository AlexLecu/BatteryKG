"""Literature Monitor — staged discovery of new sources. NOT an auto-ingester.

Pipeline per run:
  1. Query Semantic Scholar (no key; paced, exponential backoff on 429) and
     the arXiv API for the watch targets (cell model aliases + dataset terms),
     publication year >= 2024.
  2. Stage new hits as CandidateSource records: appended to
     data/monitor/candidates.jsonl AND mirrored as (:CandidateSource
     {status: 'pending_review'}) nodes — a label deliberately separate from
     the real Source nodes. Re-runs dedup on DOI/arXiv id/title hash.
  3. Cheap LLM triage per NEW candidate (one Groq call): does the paper
     plausibly contain or link to cycling data or specifications for the
     matched target? yes/maybe/no + one-line rationale, stored on the record.
  4. NO auto-promotion, regardless of verdict. Promotion to a real Source is
     a human action: python -m src.agents.promote <candidate_id> --confirm
  5. Every run appends a summary to data/monitor/run_log.jsonl.

Schedulable but not scheduled — suitable cron line (documented, not installed):
  0 6 * * 1  cd /path/to/BatteryKG && .venv/bin/python -m src.agents.literature_monitor run

Run:  python -m src.agents.literature_monitor run
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

from src.agents.llm_client import GROQ_LLAMA33, ModelConfig, complete
from src.config import DATA

MONITOR_DIR = DATA / "monitor"
CANDIDATES_JSONL = MONITOR_DIR / "candidates.jsonl"
RUN_LOG_JSONL = MONITOR_DIR / "run_log.jsonl"
MIN_YEAR = 2024
S2_PACE_S = 1.5
ARXIV_PACE_S = 3.0
USER_AGENT = ("BatteryKG-research/1.0 (academic research; contact: "
              "lecu.alex@yahoo.com)")

WATCH_TARGETS = [
    {"target": "A123 APR18650M1A", "queries": ["APR18650M1A"]},
    {"target": "Panasonic NCR18650B", "queries": ["NCR18650B"]},
    {"target": "LG Chem 18650HG2", "queries": ["18650HG2", "LG HG2 18650"]},
    {"target": "datasets", "queries": ["battery cycling dataset",
                                       "capacity fade dataset",
                                       "18650 degradation data"]},
]

TRIAGE_MODEL: ModelConfig = GROQ_LLAMA33


# --- API result parsing (pure; unit-tested on fixtures) ------------------------
def dedup_key(doi: str | None, arxiv_id: str | None, title: str) -> str:
    if doi:
        return f"doi:{doi.lower()}"
    if arxiv_id:
        return f"arxiv:{arxiv_id.lower()}"
    norm = re.sub(r"[^a-z0-9]+", "", title.lower())
    return f"title:{hashlib.sha1(norm.encode()).hexdigest()[:16]}"


def parse_s2_response(payload: dict, target: str, query: str) -> list[dict]:
    out = []
    for p in payload.get("data") or []:
        year = p.get("year")
        if year is None or year < MIN_YEAR:
            continue
        ext = p.get("externalIds") or {}
        doi = ext.get("DOI")
        arxiv_id = ext.get("ArXiv")
        title = p.get("title") or ""
        if not title:
            continue
        out.append({
            "candidate_id": dedup_key(doi, arxiv_id, title),
            "title": title,
            "authors": [a.get("name") for a in (p.get("authors") or [])][:12],
            "venue": p.get("venue") or "",
            "year": year,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "abstract": (p.get("abstract") or "")[:1500],
            "matched_target": target,
            "matched_query": query,
            "api": "semantic_scholar",
        })
    return out


_ATOM = "{http://www.w3.org/2005/Atom}"


def parse_arxiv_response(xml_text: str, target: str, query: str) -> list[dict]:
    out = []
    root = ET.fromstring(xml_text)
    for entry in root.findall(f"{_ATOM}entry"):
        title = (entry.findtext(f"{_ATOM}title") or "").strip().replace("\n", " ")
        published = entry.findtext(f"{_ATOM}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        if not title or year is None or year < MIN_YEAR:
            continue
        raw_id = entry.findtext(f"{_ATOM}id") or ""
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
        out.append({
            "candidate_id": dedup_key(None, arxiv_id, title),
            "title": title,
            "authors": [a.findtext(f"{_ATOM}name") for a in
                        entry.findall(f"{_ATOM}author")][:12],
            "venue": "arXiv",
            "year": year,
            "doi": None,
            "arxiv_id": arxiv_id,
            "abstract": (entry.findtext(f"{_ATOM}summary") or "").strip()[:1500],
            "matched_target": target,
            "matched_query": query,
            "api": "arxiv",
        })
    return out


# --- API fetching (paced + backoff) -----------------------------------------------
def _get_with_backoff(url: str, params: dict, max_retries: int = 5,
                      extra_headers: dict | None = None) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, **(extra_headers or {})}
    for attempt in range(1, max_retries + 1):
        r = requests.get(url, params=params, headers=headers, timeout=45)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        wait = 2 ** attempt
        print(f"[monitor] 429 rate-limited; backing off {wait}s "
              f"(attempt {attempt}/{max_retries})")
        time.sleep(wait)
    raise RuntimeError(f"rate-limited after {max_retries} retries: {url}")


def search_semantic_scholar(query: str, target: str) -> list[dict]:
    import os
    headers = {}
    if os.environ.get("S2_API_KEY"):        # optional free key -> own rate pool
        headers["x-api-key"] = os.environ["S2_API_KEY"]
    r = _get_with_backoff(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {"query": query, "year": f"{MIN_YEAR}-", "limit": 15,
         "fields": "title,authors,venue,year,externalIds,abstract"},
        extra_headers=headers)
    return parse_s2_response(r.json(), target, query)


def search_arxiv(query: str, target: str) -> list[dict]:
    r = _get_with_backoff(
        "https://export.arxiv.org/api/query",
        {"search_query": f'all:"{query}"', "start": 0, "max_results": 15,
         "sortBy": "submittedDate", "sortOrder": "descending"})
    return parse_arxiv_response(r.text, target, query)


# --- staging (file is canonical; KG mirror is best-effort) ---------------------------
def load_existing_keys(jsonl_path=CANDIDATES_JSONL) -> set[str]:
    if not jsonl_path.exists():
        return set()
    return {json.loads(line)["candidate_id"]
            for line in jsonl_path.read_text().splitlines() if line.strip()}


def stage_candidates(candidates: list[dict], jsonl_path=CANDIDATES_JSONL) -> list[dict]:
    """Append candidates that are NEW (dedup on candidate_id). Returns the new ones."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_keys(jsonl_path)
    new, seen_this_run = [], set()
    for c in candidates:
        cid = c["candidate_id"]
        if cid in existing or cid in seen_this_run:
            continue
        seen_this_run.add(cid)
        rec = {**c, "status": "pending_review",
               "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        new.append(rec)
    with open(jsonl_path, "a") as fh:
        for rec in new:
            fh.write(json.dumps(rec) + "\n")
    return new


_KG_STAGE_CYPHER = """
UNWIND $rows AS row
MERGE (c:CandidateSource {candidate_id: row.candidate_id})
  SET c.title = row.title, c.authors = row.authors, c.venue = row.venue,
      c.year = row.year, c.doi = row.doi, c.arxiv_id = row.arxiv_id,
      c.abstract = row.abstract, c.matched_target = row.matched_target,
      c.matched_query = row.matched_query, c.api = row.api,
      c.retrieved = row.retrieved,
      c.status = coalesce(c.status, 'pending_review'),
      c.triage_verdict = row.triage_verdict,
      c.triage_rationale = row.triage_rationale
"""


def kg_stage(candidates: list[dict]) -> bool:
    """Mirror candidates into the KG (best-effort; JSONL stays canonical)."""
    if not candidates:
        return True
    try:
        from src.kg.connection import get_driver
        driver = get_driver(max_retries=1)
        with driver.session() as s:
            s.run(_KG_STAGE_CYPHER, rows=candidates)
        driver.close()
        return True
    except Exception as e:                       # noqa: BLE001
        print(f"[monitor] WARNING: KG staging skipped ({type(e).__name__}: {e}); "
              "candidates.jsonl remains the canonical store")
        return False


# --- triage --------------------------------------------------------------------------
TRIAGE_SYSTEM = """You triage papers for a battery knowledge graph.
Answer in EXACTLY this format (one line):
<yes|maybe|no>: <one-line rationale>
'yes' = the paper plausibly CONTAINS or LINKS TO cycling data or manufacturer
specifications for the named cell/target. 'maybe' = unclear from the abstract.
'no' = unrelated or no data."""


def parse_triage(raw: str) -> tuple[str, str]:
    m = re.match(r"\s*(yes|maybe|no)\s*[:\-–]\s*(.+)", raw.strip(),
                 re.I | re.S)
    if not m:
        return "unparsed", raw.strip()[:200]
    return m.group(1).lower(), " ".join(m.group(2).split())[:300]


def triage_candidate(c: dict, model: ModelConfig = TRIAGE_MODEL) -> tuple[str, str]:
    user = (f"Target: {c['matched_target']}\n"
            f"Title: {c['title']}\n"
            f"Venue: {c['venue']} ({c['year']})\n"
            f"Abstract: {c['abstract'] or '(no abstract available)'}")
    raw = complete(TRIAGE_SYSTEM, user, model)
    return parse_triage(raw)


# --- run -----------------------------------------------------------------------------
S2_CIRCUIT_BREAKER = 2      # consecutive rate-limit exhaustions -> stop hitting S2


def run() -> dict:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    all_hits: list[dict] = []
    queries_issued = []
    s2_failures = 0
    for wt in WATCH_TARGETS:
        for q in wt["queries"]:
            # circuit breaker: the anonymous S2 pool exhausts quickly; once it
            # is clearly dead, stop hammering it for the rest of the run.
            # (Fix: request a free S2 API key and set S2_API_KEY — see README.)
            if s2_failures >= S2_CIRCUIT_BREAKER:
                queries_issued.append({"api": "semantic_scholar", "query": q,
                                       "skipped": "circuit breaker (rate limits)"})
            else:
                try:
                    hits = search_semantic_scholar(q, wt["target"])
                    queries_issued.append({"api": "semantic_scholar", "query": q,
                                           "hits": len(hits)})
                    all_hits.extend(hits)
                    s2_failures = 0
                except Exception as e:           # noqa: BLE001
                    queries_issued.append({"api": "semantic_scholar", "query": q,
                                           "error": str(e)[:120]})
                    s2_failures += 1
                time.sleep(S2_PACE_S)
            try:
                hits = search_arxiv(q, wt["target"])
                queries_issued.append({"api": "arxiv", "query": q, "hits": len(hits)})
                all_hits.extend(hits)
            except Exception as e:               # noqa: BLE001
                queries_issued.append({"api": "arxiv", "query": q,
                                       "error": str(e)[:120]})
            time.sleep(ARXIV_PACE_S)

    new = stage_candidates(all_hits)
    print(f"[monitor] {len(all_hits)} hits -> {len(new)} new candidates")

    # triage only the NEW candidates (one cheap call each)
    triage_counts: dict[str, int] = {}
    for c in new:
        try:
            verdict, rationale = triage_candidate(c)
        except Exception as e:                   # noqa: BLE001
            verdict, rationale = "triage_failed", f"{type(e).__name__}: {e}"[:200]
        c["triage_verdict"] = verdict
        c["triage_rationale"] = rationale
        triage_counts[verdict] = triage_counts.get(verdict, 0) + 1

    # rewrite JSONL records for the new candidates with their triage results
    if new:
        lines = CANDIDATES_JSONL.read_text().splitlines()
        by_id = {c["candidate_id"]: c for c in new}
        out = []
        for line in lines:
            rec = json.loads(line)
            out.append(json.dumps(by_id.get(rec["candidate_id"], rec)))
        CANDIDATES_JSONL.write_text("\n".join(out) + "\n")

    kg_ok = kg_stage(new)

    summary = {"started": started,
               "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "queries": queries_issued,
               "total_hits": len(all_hits),
               "new_candidates": len(new),
               "triage_counts": triage_counts,
               "kg_staged": kg_ok}
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG_JSONL, "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(f"[monitor] run logged: {summary['new_candidates']} new, "
          f"triage {triage_counts}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run()
    else:
        print(__doc__)
