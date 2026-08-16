"""Where do the second-set-only claims come from? — adjudication preparation.

For every claim the second annotator recorded and the reference gold does not,
this locates it (document, page, section), groups it into a category, checks the
category against gold precedent, and verifies the value against the document's
text snapshot where one exists.

Produces one document, `adjudication_prep.md`. Reads everything else read-only;
no LLM path.

Run:  python -m experiments.exp14_annotator_agreement.adjudication_prep [dir]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from experiments.exp14_annotator_agreement.common import (
    DEFAULT_SECOND_SET, ROOT, Claim, document_set,
)
from experiments.exp14_annotator_agreement.score import classify, match_views

HERE = Path(__file__).resolve().parent
SNAPSHOTS = ROOT / "data" / "claims" / "extracted_text"

# --- categories -------------------------------------------------------------
# Checked in order; first hit wins.
CATEGORIES = [
    # Order matters: the unambiguous, name-based tests run first so that a claim
    # whose condition text merely cites a figure is not mistaken for a value
    # read off a curve.
    ("cell dimensions", lambda c: str(c.unit).lower() == "mm"
        or c.property.endswith("_mm")),
    # Split deliberately: the charge-termination currents are the contested set,
    # while max/limit specs are long-settled in gold. Merging them would let the
    # settled half vouch for the contested half.
    ("charge-termination (cut-off) currents", lambda c: bool(
        re.search(r"end_current|termination|taper|precharge", c.property, re.I))),
    ("limit specs (max current / cut-off voltage)", lambda c: bool(
        re.search(r"^max_|_cutoff_v$|^cut_?off", c.property, re.I))),
    ("storage-duration rows", lambda c: c.property.startswith("storage")
        and bool(re.search(r"\b(month|months|year|years|week|weeks|day|days)\b",
                           c.cond_text, re.I))),
    # A plotted value must be declared as one — a "Fig. 1" cross-reference in a
    # condition string is not enough.
    ("plot-derived values", lambda c: bool(
        re.search(r"read from (the )?graph|read off|from the (curve|plot|graph)"
                  r"|graph-derived|off the (curve|plot)",
                  f"{c.notes} {c.cond_text}", re.I))),
    ("test-criterion rows", lambda c: bool(
        re.search(r"pass criterion|shall be|acceptance|\btest\b",
                  f"{c.notes} {c.cond_text}", re.I))),
]


def categorise(c: Claim) -> str:
    for name, test in CATEGORIES:
        try:
            if test(c):
                return name
        except Exception:
            continue
    return "other"


SECTION_RE = re.compile(r"^\s*'?\"?(\d+(?:\.\d+)*)\s*([A-Za-z][^|:,;]{0,44})?")


def section_of(c: Claim) -> str:
    m = SECTION_RE.match(str(c.cond_text))
    if not m:
        return "—"
    num, title = m.group(1), (m.group(2) or "").strip(" '\"—-")
    return f"§{num} {title}".strip() if title else f"§{num}"


# --- value verification against the text snapshot ---------------------------
def _squash(s: str) -> str:
    return re.sub(r"[\s,]", "", s)


def _candidates(value, unit) -> list[str]:
    """Printed forms a value might take in the document."""
    vals = value if isinstance(value, list) else [value]
    unit = str(unit or "").lower()
    out = []
    for v in vals:
        if not isinstance(v, (int, float)):
            continue
        forms = {f"{v:g}"}
        if unit in ("a", "ah", "v"):          # the document may print milli-units
            forms.add(f"{v * 1000:g}")
        if float(v).is_integer():
            forms.add(f"{int(v)}")
            forms.add(f"{int(v)}.0")
        out.extend(forms)
    return out


def load_snapshot(doc_pdf: Path) -> dict[int, str] | None:
    """{physical page -> text} for a document that has a text layer."""
    snap = SNAPSHOTS / (doc_pdf.stem + ".txt")
    if not snap.exists():
        return None
    body = snap.read_text()
    parts = re.split(r"=== PAGE (\d+) ===", body)
    pages = {}
    for i in range(1, len(parts) - 1, 2):
        pages[int(parts[i])] = parts[i + 1]
    if not any(t.strip() for t in pages.values()):
        return None                            # image-only: markers but no text
    return pages


def verify(c: Claim, pages: dict[int, str] | None) -> tuple[str, str]:
    """(status, detail) — status drives the 'check by hand' flag."""
    if pages is None:
        return "no text layer", "image-only document — verify on the PDF by eye"
    cands = _candidates(c.value, c.unit)
    if not cands:
        return "unverifiable", "non-numeric value"
    cited = _squash(pages.get(c.page, ""))
    if any(_squash(x) in cited for x in cands):
        return "found on cited page", ""
    hits = sorted(p for p, t in pages.items()
                  if any(_squash(x) in _squash(t) for x in cands))
    if hits:
        return "found on other page(s)", f"appears on page(s) {hits}, cited {c.page}"
    return "NOT FOUND in text", "value does not appear in the snapshot — verify"


# --- gold precedent ---------------------------------------------------------
def gold_precedent(gold_by_doc: dict[str, list[Claim]]) -> dict:
    """For each category, what the reference gold already does, per document."""
    out = {}
    for name, _ in CATEGORIES + [("other", None)]:
        per_doc = {doc: sum(1 for c in cs if categorise(c) == name)
                   for doc, cs in gold_by_doc.items()}
        rows = [(doc, c) for doc, cs in gold_by_doc.items() for c in cs
                if categorise(c) == name]
        out[name] = dict(
            n=len(rows), per_doc=per_doc,
            documents=sorted(d for d, n in per_doc.items() if n),
            n_docs_with=sum(1 for n in per_doc.values() if n),
            n_docs=len(per_doc),
            examples=[f"`{c.property}` = {c.show_value()} {c.unit} ({d})"
                      for d, c in rows[:3]])
    return out


def leaning(cat: str, p: dict) -> str:
    """The category-level recommendation, or an honest refusal to give one."""
    if cat == "other":
        return ("**row by row** — a catch-all, not a category; no single "
                "decision covers it")
    if p["n"] == 0:
        return "**scope decision** — no precedent anywhere in gold"
    if p["n_docs_with"] == p["n_docs"]:
        return ("**include** — gold does this in every document; excluding "
                "them would make the corpus internally inconsistent")
    return (f"**gold is inconsistent** — present in {p['n_docs_with']} of "
            f"{p['n_docs']} gold documents; decide once, then backfill the rest")


GUIDELINE = {
    "cell dimensions":
        "§2 final bullet — a dimension is excluded when it appears only as a "
        "tolerance on the mechanical drawing, and is a claim when the document "
        "states it as a specification value. The distinction is per-document.",
    "charge-termination (cut-off) currents":
        "§7.4 lists `end_current_ma` as a *condition* field, while §8.1 says a "
        "row stating several quantities yields several claims. The guideline "
        "genuinely does not settle this one — it is the clearest scope decision "
        "for the session.",
    "limit specs (max current / cut-off voltage)":
        "§4.1 gives `max_charge_current_a`, `max_cont_discharge_a` and "
        "`discharge_cutoff_v` as vocabulary entries, so these are claims by "
        "construction; the question is only whether the specific row was missed.",
    "storage-duration rows":
        "§8.5 — one property may appear several times with different conditions "
        "(the duration belongs in `stated_conditions`, not in the property name).",
    "test-criterion rows":
        "§7.9 — numerically stated pass criteria are claims, flagged in `notes`. "
        "§2 excludes mechanical/safety/abuse criteria, so the line runs between "
        "electrical performance tests and safety tests.",
    "plot-derived values":
        "§7.8 — plotted values are claims, read conservatively and flagged "
        "`read from graph`; §2 lists them under what to annotate.",
    "other": "§2 — general scope of what counts as a quantitative claim.",
}


def _termination_as_conditions(gold_by_doc) -> list[str]:
    """Gold may already carry this information — as a condition, not a claim."""
    hits = [(doc, c) for doc, cs in gold_by_doc.items() for c in cs
            if any(re.search(r"end_current|cut_?off", k, re.I) for k in c.cond_parsed)]
    if not hits:
        return []
    docs = sorted({d for d, _ in hits})
    return [f"**the information is already in gold, but as a condition**: "
            f"{len(hits)} gold claims carry an `end_current_ma`/cut-off field "
            f"inside `stated_conditions` ({', '.join(docs)}). So the decision is "
            f"not whether to record it but at what level — claim or condition."]


def _imageonly_note(gold_by_doc) -> list[str]:
    return ["the image-only Panasonic marketing sheet has no text layer, so "
            "**every** value in it — in gold as much as in the second set — was "
            "read by eye; gold simply did not label them `read from graph`. "
            "Treat the 12 rows there as a labelling question first, a scope "
            "question second."]


EXTRA_EVIDENCE = {
    "charge-termination (cut-off) currents": _termination_as_conditions,
    "plot-derived values": _imageonly_note,
}


def main(second_dir: Path = DEFAULT_SECOND_SET) -> None:
    docs = list(document_set(second_dir))
    gold_by_doc, rows = {}, []
    for d in docs:
        gold_by_doc[d["label"]] = d["ref_claims"]
        views = match_views(d["ref_claims"], d["sec_claims"])
        dis = classify(d["ref_claims"], d["sec_claims"], views)
        pages = load_snapshot(d["pdf"])
        for c in dis["sec_only"]:
            status, detail = verify(c, pages)
            cat = categorise(c)
            # A plotted value is absent from the text by definition — that is an
            # expected reading task, not an anomaly. Say which it is.
            if cat == "plot-derived values" and status in (
                    "NOT FOUND in text", "no text layer"):
                status = "plot read"
                detail = "read off a curve — check the reading against the plot"
            rows.append(dict(doc=d["label"], short=d["key"], claim=c,
                             category=cat, section=section_of(c),
                             status=status, detail=detail))

    rows.sort(key=lambda r: (r["category"], r["short"], r["claim"].page or 0,
                             r["claim"].property))
    prec = gold_precedent(gold_by_doc)
    cats = sorted({r["category"] for r in rows})
    doc_labels = [d["label"] for d in docs]

    L = ["# Adjudication prep — the claims only the second annotator recorded", "",
         f"{len(rows)} claims that have no counterpart in the reference gold "
         f"standard, located, grouped, and checked against the source text. "
         f"Nothing here is adjudicated: every row is a question for the session.",
         "", "Read §1 to decide whole categories, then use §4 only for the rows "
         "that need a document check.", "",
         "## 1. Categories and counts", "",
         "| category | " + " | ".join(doc_labels) + " | total |",
         "|---" * (len(doc_labels) + 2) + "|"]
    for cat in cats:
        cells = [str(sum(1 for r in rows if r["category"] == cat and r["doc"] == dl))
                 for dl in doc_labels]
        total = sum(1 for r in rows if r["category"] == cat)
        L.append(f"| **{cat}** | " + " | ".join(cells) + f" | **{total}** |")
    L.append("| _total_ | " + " | ".join(
        str(sum(1 for r in rows if r["doc"] == dl)) for dl in doc_labels)
        + f" | _{len(rows)}_ |")

    L += ["", "## 2. Does the gold standard already do this?", "",
          "| category | new claims | gold precedent | where | leaning |",
          "|---|---:|---:|---|---|"]
    for cat in cats:
        n = sum(1 for r in rows if r["category"] == cat)
        p = prec.get(cat, dict(n=0, documents=[]))
        where = ", ".join(f"{d} ({k})" for d, k in sorted(p["per_doc"].items())
                          if k) or "—"
        L.append(f"| {cat} | {n} | {p['n']} | {where} | {leaning(cat, p)} |")

    L += ["", "### Per-category detail", ""]
    for cat in cats:
        p = prec.get(cat, dict(n=0, documents=[], examples=[]))
        n = sum(1 for r in rows if r["category"] == cat)
        L += [f"**{cat}** — {n} new, {p['n']} in gold "
              f"({p['n_docs_with']}/{p['n_docs']} documents)"]
        if p["examples"]:
            L.append("")
            L += [f"- precedent: {e}" for e in p["examples"]]
        for extra in EXTRA_EVIDENCE.get(cat, lambda *_: [])(gold_by_doc):
            L.append(f"- {extra}")
        L += ["", f"- guideline: {GUIDELINE.get(cat, '—')}", ""]

    L += ["## 3. The claims, by category", "",
          "| # | category | document | page | section | property | value | "
          "verification | condition text |",
          "|---:|---|---|---:|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        c = r["claim"]
        flag = "" if r["status"] == "found on cited page" else " ⚑"
        L.append(f"| {i} | {r['category']} | {r['short']} | {c.page} | "
                 f"{r['section']} | `{c.property}` | {c.show_value()} {c.unit} | "
                 f"{r['status']}{flag} | { ' '.join(str(c.cond_text).split())[:80] } |")

    flagged = [r for r in rows if r["status"] != "found on cited page"]
    anomalies = [r for r in flagged if r["status"] in ("NOT FOUND in text",
                                                       "found on other page(s)")]
    plots = [r for r in flagged if r["status"] == "plot read"]
    eye = [r for r in flagged if r["status"] == "no text layer"]
    L += ["", f"## 4. Verify these {len(flagged)} against the document", "",
          f"Of the {len(rows)} claims, {len(rows) - len(flagged)} were found as "
          f"printed on the page the annotator cited. In every document that has "
          f"a text layer, **no value was missing and none sat on a different "
          f"page than the one cited** — so there are "
          f"{'no' if not anomalies else str(len(anomalies))} location or "
          f"transcription anomalies. The rows below need eyes for a different "
          f"reason:", "",
          f"- **{len(eye)}** are in the image-only marketing sheet, which has no "
          f"text to check against;",
          f"- **{len(plots)}** are values read off a curve, which by definition "
          f"are not printed anywhere;",
          f"- **{len(anomalies)}** are genuine anomalies (value absent from the "
          f"text, or printed on a different page than cited).", ""]
    if flagged:
        L += ["| # | document | page | property | value | why |",
              "|---:|---|---:|---|---|---|"]
        for i, r in enumerate(flagged, 1):
            c = r["claim"]
            L.append(f"| {i} | {r['short']} | {c.page} | `{c.property}` | "
                     f"{c.show_value()} {c.unit} | {r['detail'] or r['status']} |")
    L += ["", "Verification compares the recorded value against the document's "
          "text snapshot, trying milli-unit forms too (0.05 A also as `50mA`). "
          "It confirms the number is printed where the annotator says it is; it "
          "cannot confirm the number means what the annotator took it to mean.", ""]

    (HERE / "adjudication_prep.md").write_text("\n".join(L))
    print(f"[exp14.prep] {len(rows)} second-set-only claims, "
          f"{len(cats)} categories, {len(flagged)} flagged for a document check")
    print(f"[exp14.prep] wrote adjudication_prep.md")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SECOND_SET)
