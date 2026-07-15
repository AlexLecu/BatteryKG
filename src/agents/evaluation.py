"""Evaluation of LLM-extracted claims against the hand-labeled gold standard.

Matching rule: a predicted claim matches a gold claim when document, property
(exact) and value (within tolerance, unit-scaled) agree; each gold claim can be
matched by at most one prediction (greedy, document order). Conditions are
scored separately on matched pairs. Unmatched predictions are checked against
the source text: a predicted value that does not appear anywhere in the
document is counted as a HALLUCINATION (worse than an ordinary false positive).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.config import DATA

REL_TOL = 0.02          # value match tolerance (2 %)

# unit -> multiplier to its base unit (for cross-unit value comparison)
_UNIT_SCALE = {"mah": 1e-3, "ma": 1e-3, "mv": 1e-3}


def _scale(value: float, unit: str | None) -> float:
    return value * _UNIT_SCALE.get((unit or "").strip().lower(), 1.0)


def _as_floats(v) -> list[float] | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return [float(v)]
    if isinstance(v, (list, tuple)) and v and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        return [float(x) for x in v]
    return None


def _close(a: float, b: float, rel: float = REL_TOL) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-9)


def values_match(gold_value, gold_unit, pred_value, pred_unit) -> bool:
    g, p = _as_floats(gold_value), _as_floats(pred_value)
    if g is None or p is None or len(g) != len(p):
        return False
    gs = sorted(_scale(x, gold_unit) for x in g)
    ps = sorted(_scale(x, pred_unit) for x in p)
    return all(_close(a, b) for a, b in zip(gs, ps))


# --- gold loading ------------------------------------------------------------
@dataclass
class GoldClaim:
    doc: str                    # source document stem
    cell_model: str
    property: str
    value: object
    unit: str | None
    page: int | None
    conditions_text: str
    parsed_conditions: dict = field(default_factory=dict)
    matched: bool = False


def load_gold(claims_dir: Path = DATA / "claims") -> list[GoldClaim]:
    out = []
    for path in sorted(claims_dir.glob("*.yaml")):
        doc_yaml = yaml.safe_load(path.read_text())
        stem = Path(doc_yaml["source_document"]).stem
        for cl in doc_yaml["claims"]:
            cond = cl.get("stated_conditions") or {}
            out.append(GoldClaim(
                doc=stem, cell_model=doc_yaml["cell_model"],
                property=cl["property"], value=cl["value"], unit=cl.get("unit"),
                page=cl.get("page"), conditions_text=cond.get("text", "unspecified"),
                parsed_conditions={k: v for k, v in cond.items() if k != "text"}))
    return out


# --- conditions scoring --------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers_in(obj) -> list[float]:
    nums: list[float] = []
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        nums.append(float(obj))
    elif isinstance(obj, str):
        nums.extend(float(m) for m in _NUM_RE.findall(obj))
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            nums.extend(_numbers_in(x))
    elif isinstance(obj, dict):
        for v in obj.values():
            nums.extend(_numbers_in(v))
    return nums


def conditions_correct(gold: GoldClaim, pred_conditions: dict) -> bool:
    """Matched-pair condition score.

    Gold `conditions_text` is a verbatim source quote (traceability); genuine
    test conditions live in the gold's PARSED fields. Scoring:

    - gold has parsed condition fields -> every numeric value in them must
      appear among the prediction's condition numbers (robust to key naming;
      tolerance REL_TOL).
    - gold has NO parsed fields (the document states no conditions for this
      claim; the quote merely names the spec row) -> the prediction is correct
      as long as it does not INVENT numeric conditions absent from the gold
      quote — saying "unspecified" is the right answer, echoing the quote is
      also fine.
    """
    pred_nums = _numbers_in(pred_conditions)
    gold_nums = _numbers_in(gold.parsed_conditions)
    if gold_nums:
        return all(any(_close(gn, pn) for pn in pred_nums) for gn in gold_nums)
    # no parsed gold conditions: penalise only invented numbers
    quote_nums = _numbers_in(gold.conditions_text)
    return all(any(_close(pn, qn) for qn in quote_nums) for pn in pred_nums)


def condition_over_extracted(gold: GoldClaim, cond: dict) -> bool:
    """True when the predicted conditions assert numeric values with no basis
    in the gold claim's conditions (neither its parsed fields nor its verbatim
    source quote) — i.e. conditions were invented where none are stated.

    Magnitude-guarded x1000 rescaling makes the check unit-blind (a prediction
    of 0.6 A is grounded by a quote saying '600mA')."""
    allowed = [abs(a) for a in
               _numbers_in(gold.parsed_conditions) + _numbers_in(gold.conditions_text)]
    pred_nums = [n for k, v in cond.items()
                 if k != "text" and not str(k).startswith("_")
                 for n in _numbers_in(v)]

    def grounded(p: float) -> bool:
        p = abs(p)
        cands = [p]
        if p < 100:
            cands.append(p * 1000)
        if p >= 1000:
            cands.append(p / 1000)
        return any(_close(c, a) for c in cands for a in allowed)

    return any(not grounded(p) for p in pred_nums)


# --- hallucination check ----------------------------------------------------------
def value_in_document(value, unit, doc_text: str) -> bool:
    """Is the predicted value present anywhere in the source text (allowing
    x1000 unit rescale, e.g. 3.35 Ah appearing as 3350 mAh)?"""
    vals = _as_floats(value)
    if vals is None:
        return False
    # absolute values both sides (sign handling in datasheet text is unreliable:
    # range dashes, Unicode minus, '+' prefixes)
    doc_nums = [abs(float(m)) for m in _NUM_RE.findall(doc_text.replace("−", "-"))]

    def found(x: float) -> bool:
        # magnitude-guarded rescaling: x1000 only for small values (Ah -> mAh
        # direction), /1000 only for large ones — otherwise e.g. 999 would
        # "match" a stray '1' via /1000 within tolerance.
        candidates = [x]
        if x < 100:
            candidates.append(x * 1000)
        if x >= 1000:
            candidates.append(x / 1000)
        return any(_close(c, d) for c in candidates for d in doc_nums)

    return all(found(abs(v)) for v in vals)


# --- matching + metrics -------------------------------------------------------------
@dataclass
class DocEvaluation:
    doc: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    conditions_ok: int = 0
    hallucinations: int = 0
    errors: list[str] = field(default_factory=list)   # error catalog lines

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def conditions_acc(self) -> float:
        return self.conditions_ok / self.tp if self.tp else 0.0


def match_pairs(gold: list[GoldClaim], preds: list[dict],
                doc: str) -> list[tuple[GoldClaim, dict]]:
    """Greedy 1:1 (gold, prediction) pairs for one document — same rule as
    evaluate_document (property exact + value within tolerance)."""
    pairs, used = [], [False] * len(preds)
    for g in (g for g in gold if g.doc == doc):
        for i, p in enumerate(preds):
            if used[i] or p.get("property") != g.property:
                continue
            if values_match(g.value, g.unit, p.get("value"), p.get("unit")):
                used[i] = True
                pairs.append((g, p))
                break
    return pairs


def evaluate_document(doc: str, gold: list[GoldClaim], preds: list[dict],
                      doc_text: str) -> DocEvaluation:
    ev = DocEvaluation(doc=doc)
    gold_doc = [g for g in gold if g.doc == doc]
    for g in gold_doc:
        g.matched = False
    used = [False] * len(preds)

    # greedy 1:1 matching: property exact + value within tolerance
    for g in gold_doc:
        for i, p in enumerate(preds):
            if used[i] or p.get("property") != g.property:
                continue
            if values_match(g.value, g.unit, p.get("value"), p.get("unit")):
                used[i] = True
                g.matched = True
                ev.tp += 1
                if conditions_correct(g, p.get("stated_conditions") or {}):
                    ev.conditions_ok += 1
                else:
                    ev.errors.append(
                        f"COND [{doc}] {g.property}={g.value}: conditions wrong — "
                        f"gold '{g.conditions_text[:70]}...' vs pred "
                        f"'{str((p.get('stated_conditions') or {}).get('text'))[:70]}'")
                break

    # false negatives with diagnosis
    for g in gold_doc:
        if g.matched:
            continue
        ev.fn += 1
        same_prop = [p for i, p in enumerate(preds) if p.get("property") == g.property]
        vals_elsewhere = [p for i, p in enumerate(preds) if not used[i]
                          and values_match(g.value, g.unit, p.get("value"), p.get("unit"))]
        if same_prop:
            diag = (f"property predicted but value differs "
                    f"(gold {g.value} {g.unit}; pred {[p.get('value') for p in same_prop]})")
        elif vals_elsewhere:
            diag = (f"value found under different property "
                    f"({[p.get('property') for p in vals_elsewhere]})")
        else:
            diag = "not extracted at all"
        ev.errors.append(f"FN   [{doc}] {g.property}={g.value} {g.unit or ''}: {diag}")

    # false positives + hallucination check
    for i, p in enumerate(preds):
        if used[i]:
            continue
        ev.fp += 1
        if value_in_document(p.get("value"), p.get("unit"), doc_text):
            ev.errors.append(
                f"FP   [{doc}] {p.get('property')}={p.get('value')} {p.get('unit') or ''}: "
                "value exists in document but is not a gold claim "
                "(over-extraction or wrong property name)")
        else:
            ev.hallucinations += 1
            ev.errors.append(
                f"HALL [{doc}] {p.get('property')}={p.get('value')} {p.get('unit') or ''}: "
                "value NOT FOUND anywhere in the document text")
    return ev


def combine(evals: list[DocEvaluation]) -> DocEvaluation:
    total = DocEvaluation(doc="OVERALL")
    for e in evals:
        total.tp += e.tp
        total.fp += e.fp
        total.fn += e.fn
        total.conditions_ok += e.conditions_ok
        total.hallucinations += e.hallucinations
        total.errors.extend(e.errors)
    return total
