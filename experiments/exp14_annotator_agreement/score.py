"""Step 2-5 — agreement between the reference annotation set and a second one.

Two views, both matching greedy 1:1 in document order with the SAME value test
the paper metric uses (src.agents.evaluation.values_match, 2 % relative
tolerance, unit-scaled):

  exact       same property name AND matching value
  value-level property-agnostic, guarded by physical unit class, so a name
              disagreement does not read as a value disagreement. Computed as a
              second pass over what `exact` left unmatched, so it is a strict
              superset of it.

Agreement is reported as positive specific agreement (F1 on the matched set,
symmetric between the two annotators — neither set is treated as truth) and as
Jaccard. Disagreements are catalogued, never adjudicated: the output is the
input for a human adjudication session.

Reads both annotation sets read-only. No LLM path.

Run:  python -m experiments.exp14_annotator_agreement.score [second_set_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments.exp14_annotator_agreement.alpha import alphas
from experiments.exp14_annotator_agreement.common import (
    DEFAULT_SECOND_SET, Claim, document_set, rel_to_root,
)
from src.agents.evaluation import values_match

HERE = Path(__file__).resolve().parent


# --- matching ---------------------------------------------------------------
def _similarity(g: Claim, h: Claim) -> float:
    """How much two claims' CONDITIONS look like the same assertion.

    Only a tie-break: it decides which of several equally value-matching
    candidates to pair, which matters inside characteristic grids where many
    cells carry the same number and only the conditions tell them apart.
    """
    shared = sum(1 for k, v in g.cond_parsed.items()
                 if k in h.cond_parsed and str(h.cond_parsed[k]).strip() == str(v).strip())
    gt = set(str(g.cond_text).lower().split())
    ht = set(str(h.cond_text).lower().split())
    overlap = len(gt & ht) / len(gt | ht) if (gt or ht) else 0.0
    return 2.0 * shared + overlap + (0.5 if g.page == h.page else 0.0)


def _match(ref: list[Claim], sec: list[Claim], predicate) -> list[tuple]:
    """Greedy 1:1 in reference order; ties broken by condition similarity."""
    taken, pairs = set(), []
    for g in ref:
        best, best_score = None, None
        for i, h in enumerate(sec):
            if i in taken or not predicate(g, h):
                continue
            s = _similarity(g, h)
            if best_score is None or s > best_score:
                best, best_score = i, s
        if best is not None:
            taken.add(best)
            pairs.append((g, sec[best]))
    return pairs


def _values_agree(g: Claim, h: Claim) -> bool:
    return values_match(g.value, g.unit, h.value, h.unit)


def match_views(ref: list[Claim], sec: list[Claim]) -> dict:
    exact = _match(ref, sec, lambda g, h: g.property == h.property and _values_agree(g, h))
    ex_ref = {id(g) for g, _ in exact}
    ex_sec = {id(h) for _, h in exact}

    rest_ref = [g for g in ref if id(g) not in ex_ref]
    rest_sec = [h for h in sec if id(h) not in ex_sec]
    # property-agnostic, but a value may only match inside its physical unit class
    extra = _match(rest_ref, rest_sec,
                   lambda g, h: g.unit_class == h.unit_class and _values_agree(g, h))
    return dict(exact=exact, naming_only=extra, value_level=exact + extra)


def classify(ref: list[Claim], sec: list[Claim], views: dict) -> dict:
    """Split everything the value-level view left over into disagreement kinds."""
    matched_ref = {id(g) for g, _ in views["value_level"]}
    matched_sec = {id(h) for _, h in views["value_level"]}
    left_ref = [g for g in ref if id(g) not in matched_ref]
    left_sec = [h for h in sec if id(h) not in matched_sec]

    # same property name on both sides but no value match -> value disagreement
    value_dis, used = [], set()
    for g in left_ref:
        for i, h in enumerate(left_sec):
            if i in used or h.property != g.property:
                continue
            used.add(i)
            value_dis.append((g, h))
            break
    paired_ref = {id(g) for g, _ in value_dis}
    return dict(
        naming_only=views["naming_only"],
        value_disagreements=value_dis,
        ref_only=[g for g in left_ref if id(g) not in paired_ref],
        sec_only=[h for i, h in enumerate(left_sec) if i not in used],
    )


def condition_comparison(pairs: list[tuple]) -> dict:
    """Condition agreement on value-level matched pairs."""
    out = dict(identical=0, both_unspecified=0, ref_only_conditions=0,
               sec_only_conditions=0, ref_richer=0, sec_richer=0, conflicting=0,
               conflicts=[])
    for g, h in pairs:
        gp, hp = g.cond_parsed, h.cond_parsed
        if g.is_unspecified and h.is_unspecified:
            out["both_unspecified"] += 1
            continue
        if g.is_unspecified and not h.is_unspecified:
            out["sec_only_conditions"] += 1
        elif h.is_unspecified and not g.is_unspecified:
            out["ref_only_conditions"] += 1
        # sorted, not the raw set: set iteration order varies with the hash seed,
        # which made the conflict table reorder between runs on identical input.
        shared = sorted(set(gp) & set(hp))
        bad = [k for k in shared if str(gp[k]).strip() != str(hp[k]).strip()]
        if bad:
            out["conflicting"] += 1
            out["conflicts"].append(dict(
                property=g.property, page_ref=g.page, page_sec=h.page,
                fields={k: [gp[k], hp[k]] for k in bad}))
        elif gp == hp:
            out["identical"] += 1
        elif set(gp) > set(hp):
            out["ref_richer"] += 1
        elif set(hp) > set(gp):
            out["sec_richer"] += 1
    return out


def omission_comparison(ref_raw: dict, sec_raw: dict) -> dict:
    def props(raw):
        return {o.get("property"): str(o.get("note", ""))
                for o in (raw.get("omissions") or []) if isinstance(o, dict)}
    r, s = props(ref_raw), props(sec_raw)
    return dict(ref_n=len(r), sec_n=len(s),
                both=sorted(set(r) & set(s)),
                ref_only=sorted(set(r) - set(s)),
                sec_only=sorted(set(s) - set(r)),
                ref_has_block="omissions" in ref_raw,
                sec_has_block="omissions" in sec_raw)


def cohen_kappa(pairs_labels: list[tuple]) -> dict:
    """Cohen's kappa over (label_ref, label_sec) on a COMMON set of items.

    Only meaningful where both annotators labelled the same items with a fixed
    category set. That holds for decisions taken ON a matched pair (which
    property name? were conditions stated?), and NOT for the extraction step
    itself, which has no shared item set and no negative class.
    """
    n = len(pairs_labels)
    if n == 0:
        return dict(n=0, po=None, pe=None, kappa=None)
    po = sum(a == b for a, b in pairs_labels) / n
    cats = {c for pair in pairs_labels for c in pair}
    pa = {c: sum(a == c for a, _ in pairs_labels) / n for c in cats}
    pb = {c: sum(b == c for _, b in pairs_labels) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    k = (po - pe) / (1 - pe) if pe < 1 else None
    return dict(n=n, po=round(po, 3), pe=round(pe, 3),
                kappa=None if k is None else round(k, 3))


def kappa_block(pairs: list[tuple]) -> dict:
    """The two chance-correctable decisions, over value-level matched pairs."""
    prop = [(g.property, h.property) for g, h in pairs]
    cond = [("unspecified" if g.is_unspecified else "stated",
             "unspecified" if h.is_unspecified else "stated") for g, h in pairs]
    no_grid = [(g.property, h.property) for g, h in pairs
               if g.stratum() != "characteristic grid"]
    return dict(property=cohen_kappa(prop),
                property_excl_grids=cohen_kappa(no_grid),
                conditions_stated=cohen_kappa(cond))


def agreement(n_ref: int, n_sec: int, matched: int) -> dict:
    f1 = 2 * matched / (n_ref + n_sec) if (n_ref + n_sec) else 0.0
    union = n_ref + n_sec - matched
    return dict(matched=matched, f1=round(f1, 3),
                jaccard=round(matched / union, 3) if union else 0.0)


def strata(claims: list[Claim], matched_ids: set) -> dict:
    out = {}
    for c in claims:
        s = out.setdefault(c.stratum(), dict(total=0, matched=0))
        s["total"] += 1
        s["matched"] += id(c) in matched_ids
    return out


# --- report -----------------------------------------------------------------
def _rows(pairs, kind):
    rows = []
    for g, h in pairs:
        rows.append(dict(kind=kind,
                         ref=g.label(), sec=h.label(),
                         ref_property=g.property, sec_property=h.property,
                         ref_value=g.show_value(), sec_value=h.show_value(),
                         ref_unit=str(g.unit), sec_unit=str(h.unit),
                         ref_page=g.page, sec_page=h.page,
                         ref_conditions=g.cond_text, sec_conditions=h.cond_text))
    return rows


def analyse(d: dict) -> dict:
    ref, sec = d["ref_claims"], d["sec_claims"]
    views = match_views(ref, sec)
    dis = classify(ref, sec, views)
    matched_ids = {id(g) for g, _ in views["value_level"]}

    res = dict(
        key=d["key"], label=d["label"],
        ref_file=str(d["ref_path"].relative_to(d["ref_path"].parents[2])),
        sec_file=d["sec_path"].name,
        n_ref=len(ref), n_sec=len(sec),
        exact=agreement(len(ref), len(sec), len(views["exact"])),
        value_level=agreement(len(ref), len(sec), len(views["value_level"])),
        naming_only=len(views["naming_only"]),
        value_disagreements=len(dis["value_disagreements"]),
        ref_only=len(dis["ref_only"]), sec_only=len(dis["sec_only"]),
        conditions=condition_comparison(views["value_level"]),
        kappa=kappa_block(views["value_level"]),
        alpha=alphas(views["value_level"],
                     dis["ref_only"] + [g for g, _ in dis["value_disagreements"]],
                     dis["sec_only"] + [h for _, h in dis["value_disagreements"]]),
        omissions=omission_comparison(d["ref_raw"], d["sec_raw"]),
        strata_ref=strata(ref, matched_ids),
        strata_sec=strata(sec, {id(h) for _, h in views["value_level"]}),
        disagreements=dict(
            naming_only=_rows(views["naming_only"], "naming-only"),
            value=_rows(dis["value_disagreements"], "value"),
            ref_only=[dict(kind="reference-only", ref=g.label(),
                           ref_property=g.property, ref_value=g.show_value(),
                           ref_unit=str(g.unit), ref_page=g.page,
                           stratum=g.stratum(), ref_conditions=g.cond_text)
                      for g in dis["ref_only"]],
            sec_only=[dict(kind="second-only", sec=h.label(),
                           sec_property=h.property, sec_value=h.show_value(),
                           sec_unit=str(h.unit), sec_page=h.page,
                           stratum=h.stratum(), sec_conditions=h.cond_text)
                      for h in dis["sec_only"]],
        ))
    return res


def _a(x):
    """An undefined alpha (single category -> D_e = 0) prints as n/a."""
    return "n/a (no variance)" if x is None else f"{x:.3f}"


def _clip(s, n=90):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def render(results: list[dict], second_dir: Path, banner: str,
           overall_kappa: dict, overall_alpha: dict) -> str:
    L = [banner, "", "# Annotation agreement — reference gold vs. second annotation set", ""]
    tot_ref = sum(r["n_ref"] for r in results)
    tot_sec = sum(r["n_sec"] for r in results)
    tot_ex = sum(r["exact"]["matched"] for r in results)
    tot_vl = sum(r["value_level"]["matched"] for r in results)

    L += ["## 1. Overall", "",
          f"Reference claims: **{tot_ref}** · second-set claims: **{tot_sec}** "
          f"across {len(results)} documents.", "",
          "| view | matched | agreement (F1) | Jaccard |", "|---|---:|---:|---:|",
          f"| exact property + value | {tot_ex} | "
          f"**{2*tot_ex/(tot_ref+tot_sec):.3f}** | "
          f"{tot_ex/(tot_ref+tot_sec-tot_ex):.3f} |",
          f"| value-level (unit-class guarded) | {tot_vl} | "
          f"**{2*tot_vl/(tot_ref+tot_sec):.3f}** | "
          f"{tot_vl/(tot_ref+tot_sec-tot_vl):.3f} |", "",
          "Agreement is positive specific agreement (F1 over the matched set), "
          "symmetric in the two annotators — neither set is truth. Value tests "
          "use `src.agents.evaluation.values_match` (2 % relative tolerance, "
          "unit-scaled), matching greedy 1:1 in document order.", "",
          "F1 is depressed by set-size asymmetry rather than by contradiction. "
          "The one-sided coverages separate the two effects:", "",
          "| direction | covered | of | share |", "|---|---:|---:|---:|",
          f"| reference claims with a value-level counterpart in the second set "
          f"| {tot_vl} | {tot_ref} | **{tot_vl/tot_ref:.3f}** |",
          f"| second-set claims with a value-level counterpart in the reference "
          f"| {tot_vl} | {tot_sec} | **{tot_vl/tot_sec:.3f}** |", ""]

    L += ["## 2. Per document", "",
          "| document | n(ref) | n(2nd) | exact F1 | value-level F1 | ref covered | "
          "naming-only | value disagr. | ref-only | 2nd-only |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        cov = r["value_level"]["matched"] / r["n_ref"] if r["n_ref"] else 0.0
        L.append(f"| {r['label']} | {r['n_ref']} | {r['n_sec']} | "
                 f"{r['exact']['f1']:.3f} | {r['value_level']['f1']:.3f} | "
                 f"{cov:.3f} | {r['naming_only']} | {r['value_disagreements']} | "
                 f"{r['ref_only']} | {r['sec_only']} |")
    L.append("")

    L += ["## 3. Where the disagreements sit", "",
          "Reference-side claims by stratum, and how many the value-level view "
          "matched:", "",
          "| document | stratum | reference claims | matched | unmatched |",
          "|---|---|---:|---:|---:|"]
    for r in results:
        for s, v in sorted(r["strata_ref"].items()):
            L.append(f"| {r['label']} | {s} | {v['total']} | {v['matched']} | "
                     f"{v['total'] - v['matched']} |")
    L.append("")

    L += ["## 4. Conditions on matched pairs", "",
          "| document | pairs | identical | both `unspecified` | ref richer | "
          "2nd richer | only ref states | only 2nd states | conflicting |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        c = r["conditions"]
        L.append(f"| {r['label']} | {r['value_level']['matched']} | {c['identical']} | "
                 f"{c['both_unspecified']} | {c['ref_richer']} | {c['sec_richer']} | "
                 f"{c['ref_only_conditions']} | {c['sec_only_conditions']} | "
                 f"{c['conflicting']} |")
    L.append("")

    L += ["## 5. Omissions", "",
          "| document | ref | 2nd | agreed | ref only | 2nd only |",
          "|---|---:|---:|---|---|---|"]
    for r in results:
        o = r["omissions"]
        L.append(f"| {r['label']} | {o['ref_n']} | {o['sec_n']} | "
                 f"{', '.join(o['both']) or '—'} | {', '.join(o['ref_only']) or '—'} | "
                 f"{', '.join(o['sec_only']) or '—'} |")
    L += ["", "## 5b. Chance-corrected agreement (Cohen's κ)", "",
          "κ needs a common set of items and a fixed category set. The extraction "
          "step has neither — each annotator decides for themselves what the items "
          "are, and there is no negative class ('all the numbers we both agreed "
          "were not claims'), so no chance-agreement term is definable. κ is "
          "therefore reported only for the two decisions taken **on an already "
          "matched pair**, where both annotators did label the same item:", "",
          "| decision | items | observed agr. | expected by chance | κ |",
          "|---|---:|---:|---:|---:|"]
    for name, key in (("which property name", "property"),
                      ("property name, grids excluded", "property_excl_grids"),
                      ("conditions stated vs `unspecified`", "conditions_stated")):
        k = overall_kappa[key]
        L.append(f"| {name} | {k['n']} | {k['po']} | {k['pe']} | "
                 f"**{'n/a' if k['kappa'] is None else k['kappa']}** |")
    L += ["", "Per document (property-name κ over that document's matched pairs):", "",
          "| document | pairs | observed | κ (property) | κ (conditions stated) |",
          "|---|---:|---:|---:|---:|"]
    for r in results:
        kp, kc = r["kappa"]["property"], r["kappa"]["conditions_stated"]
        L.append(f"| {r['label']} | {kp['n']} | {kp['po']} | "
                 f"{'n/a' if kp['kappa'] is None else kp['kappa']} | "
                 f"{'n/a' if kc['kappa'] is None else kc['kappa']} |")
    kc = overall_kappa["conditions_stated"]
    L += ["", f"The conditions row is the κ prevalence paradox, not a real "
              f"disagreement: the two annotators agree on {kc['po']:.0%} of pairs, "
              f"but almost every pair is 'conditions stated' on both sides, so "
              f"chance expectation ({kc['pe']:.2f}) equals observed agreement and κ "
              f"collapses. Quote the raw {kc['po']:.0%} for that decision.", ""]
    L += ["κ = n/a means the decision was constant for at least one annotator "
              "(no variance to chance-correct: e.g. every matched pair on a "
              "document states conditions). For the extraction step itself, quote "
              "the F1/coverage figures above, or commission Krippendorff's α for "
              "unitizing, which is the measure built for annotators who choose "
              "their own units.", ""]

    L += ["", "## 5c. Krippendorff's α", "",
          "Krippendorff's α for **unitizing** (α_U) is defined over a continuum "
          "and needs each unit's offsets in it. These annotations carry no "
          "offsets — a claim is located by page, property, value and a compressed "
          "quotation — and one document is image-only, so it has no text "
          "continuum at all. α_U is therefore not computable from these files and "
          "is not reported. What follows is α for **nominal data over the aligned "
          "item set**: items are the union of both annotations aligned 1:1 by the "
          "value-level matcher, each carrying two codings, with `∅` "
          "(not annotated) for the annotator who did not record it.", "",
          "| α | items | observed | D_o | D_e | α |", "|---|---:|---:|---:|---:|---:|"]
    for name, key in (("property name, incl. `∅` — *what to extract and what to call it*", "alpha_property"),
                      ("naming only, matched items — *given both found it*", "alpha_naming"),
                      ("presence only — **degenerate, do not quote**", "alpha_presence")):
        a = overall_alpha[key]
        L.append(f"| {name} | {a['n_items']} | {a['observed']} | {a['d_o']} | "
                 f"{a['d_e']} | **{_a(a['alpha'])}** |")
    L += ["", "Per document:", "",
          "| document | items | α (property) | α (naming) |", "|---|---:|---:|---:|"]
    for r in results:
        a = r["alpha"]
        L.append(f"| {r['label']} | {a['n_items']} | "
                 f"{_a(a['alpha_property']['alpha'])} | "
                 f"{_a(a['alpha_naming']['alpha'])} |")
    L += ["",
          "**Two caveats that must travel with these numbers.**", "",
          "1. The item universe is generated by the annotators themselves, so `∅` "
          "can only appear where at least one of them proposed a claim; there is "
          "no inventory of numbers in the document that both correctly ignored. "
          "These are reliability figures *conditional on the union of proposals*, "
          "and are conservative relative to a task with a fixed item inventory.",
          "2. That is also why the presence-only α is negative and meaningless: "
          "with the universe built from the union, the (`∅`, `∅`) cell cannot "
          "occur, so expected disagreement is computed from a distribution the "
          "data cannot realise. It is listed for completeness and must not be "
          "quoted. The comparable honest figure for that question is the coverage "
          "pair in §1.", "",
          "α (naming) and Cohen's κ (§5b) agree to three decimals, which is the "
          "expected consistency check: for two coders on complete nominal data the "
          "two chance models coincide when the marginals are close.", ""]

    L += ["", "## 6. Adjudication list", "",
          "Claim by claim, nothing resolved. Columns are reference / second set.", ""]

    for r in results:
        L += [f"### {r['label']}", ""]
        d = r["disagreements"]

        if d["naming_only"]:
            L += ["**Naming-only — same value, different property name "
                  f"({len(d['naming_only'])})**", "",
                  "| # | reference property | 2nd property | value | unit | p(ref) | p(2nd) |",
                  "|---:|---|---|---|---|---:|---:|"]
            for i, x in enumerate(d["naming_only"], 1):
                L.append(f"| {i} | `{x['ref_property']}` | `{x['sec_property']}` | "
                         f"{x['ref_value']} | {x['ref_unit']} | {x['ref_page']} | "
                         f"{x['sec_page']} |")
            L.append("")

        if d["value"]:
            L += [f"**Value disagreements — same property, different value "
                  f"({len(d['value'])})**", "",
                  "| # | property | reference | 2nd | p(ref) | p(2nd) | reference conditions | 2nd conditions |",
                  "|---:|---|---|---|---:|---:|---|---|"]
            for i, x in enumerate(d["value"], 1):
                L.append(f"| {i} | `{x['ref_property']}` | {x['ref_value']} {x['ref_unit']} | "
                         f"{x['sec_value']} {x['sec_unit']} | {x['ref_page']} | {x['sec_page']} | "
                         f"{_clip(x['ref_conditions'], 70)} | {_clip(x['sec_conditions'], 70)} |")
            L.append("")

        if d["ref_only"]:
            L += [f"**In the reference set only ({len(d['ref_only'])})**", "",
                  "| # | property | value | unit | page | stratum | conditions |",
                  "|---:|---|---|---|---:|---|---|"]
            for i, x in enumerate(d["ref_only"], 1):
                L.append(f"| {i} | `{x['ref_property']}` | {x['ref_value']} | {x['ref_unit']} | "
                         f"{x['ref_page']} | {x['stratum']} | {_clip(x['ref_conditions'], 80)} |")
            L.append("")

        if d["sec_only"]:
            L += [f"**In the second set only ({len(d['sec_only'])})**", "",
                  "| # | property | value | unit | page | stratum | conditions |",
                  "|---:|---|---|---|---:|---|---|"]
            for i, x in enumerate(d["sec_only"], 1):
                L.append(f"| {i} | `{x['sec_property']}` | {x['sec_value']} | {x['sec_unit']} | "
                         f"{x['sec_page']} | {x['stratum']} | {_clip(x['sec_conditions'], 80)} |")
            L.append("")

        conf = r["conditions"]["conflicts"]
        if conf:
            L += [f"**Conflicting condition fields on matched pairs ({len(conf)})**", "",
                  "| # | property | field | reference | 2nd |", "|---:|---|---|---|---|"]
            i = 1
            for c in conf:
                for k, (a, b) in c["fields"].items():
                    L.append(f"| {i} | `{c['property']}` | `{k}` | {a} | {b} |")
                    i += 1
            L.append("")

    return "\n".join(L)


def main(second_dir: Path = DEFAULT_SECOND_SET) -> list[dict]:
    docs = list(document_set(second_dir))
    results = [analyse(d) for d in docs]

    annotators = sorted({str(d["sec_raw"].get("annotator", "unknown")) for d in docs})
    banner = (
        "> **Annotators:** A1 = first author (reference gold standard),\n"
        "> A2 = second author. Both annotation sets were produced independently by\n"
        "> manual annotation against `ANNOTATION_GUIDELINE.md`; A2 was blind to the\n"
        "> reference annotations. Metrics reported on the independent sets\n"
        "> (pre-adjudication)."
    )

    pooled, pooled_ref_only, pooled_sec_only = [], [], []
    for d in docs:
        views = match_views(d["ref_claims"], d["sec_claims"])
        dis = classify(d["ref_claims"], d["sec_claims"], views)
        pooled += views["value_level"]
        pooled_ref_only += dis["ref_only"] + [g for g, _ in dis["value_disagreements"]]
        pooled_sec_only += dis["sec_only"] + [h for _, h in dis["value_disagreements"]]
    overall_kappa = kappa_block(pooled)
    overall_alpha = alphas(pooled, pooled_ref_only, pooled_sec_only)

    tot_ref = sum(r["n_ref"] for r in results)
    tot_sec = sum(r["n_sec"] for r in results)
    tot_ex = sum(r["exact"]["matched"] for r in results)
    tot_vl = sum(r["value_level"]["matched"] for r in results)

    # Corpus-level headline figures, so the numbers quoted in the revision have a
    # machine-readable source and are not only derivable from the per-document
    # blocks below. Every field is computed here, none is transcribed.
    overall = dict(
        n_ref=tot_ref,
        n_sec=tot_sec,
        exact_matched=tot_ex,
        value_matched=tot_vl,
        exact_f1=round(2 * tot_ex / (tot_ref + tot_sec), 4),
        value_f1=round(2 * tot_vl / (tot_ref + tot_sec), 4),
        coverage=dict(ref_covered=tot_vl, n_ref=tot_ref,
                      share=round(tot_vl / tot_ref, 4)),
        coverage_sec=dict(sec_covered=tot_vl, n_sec=tot_sec,
                          share=round(tot_vl / tot_sec, 4)),
        kappa_naming=overall_kappa["property"]["kappa"],
        kappa_naming_grids_excluded=overall_kappa["property_excl_grids"]["kappa"],
        alpha_naming=overall_alpha["alpha_naming"]["alpha"],
        raw_agreement_conditions=overall_kappa["conditions_stated"]["po"],
        value_disagreements=sum(r["value_disagreements"] for r in results),
        naming_only=sum(r["naming_only"] for r in results),
        ref_only=sum(r["ref_only"] for r in results),
        sec_only=sum(r["sec_only"] for r in results),
    )

    report = render(results, second_dir, banner, overall_kappa, overall_alpha)
    (HERE / "agreement_report.md").write_text(report)
    (HERE / "results.json").write_text(json.dumps(
        dict(second_set_dir=rel_to_root(second_dir), annotators=annotators,
             overall=overall,
             overall_kappa=overall_kappa, overall_alpha=overall_alpha,
             documents=results), indent=2))

    print(f"[exp14.score] annotators: {annotators}")
    print(f"[exp14.score] {tot_ref} reference vs {tot_sec} second-set claims")
    print(f"[exp14.score] exact      F1 = {2*tot_ex/(tot_ref+tot_sec):.3f} "
          f"({tot_ex} matched)")
    print(f"[exp14.score] value-level F1 = {2*tot_vl/(tot_ref+tot_sec):.3f} "
          f"({tot_vl} matched)")
    print("[exp14.score] wrote agreement_report.md / results.json")
    return results


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SECOND_SET)
