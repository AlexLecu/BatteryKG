"""Krippendorff's alpha for the two-annotator claim-extraction task.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
Krippendorff's alpha for UNITIZING (alpha_U) measures reliability when coders
carve their own units out of a *continuum* — it is defined over the lengths of
units and of the gaps between them, so it needs each unit's offsets in that
continuum. Our annotations carry no offsets: a claim is located by page,
property, value and a compressed condition quotation, and one of the five
documents is image-only, so no text continuum exists for it at all. Recovering
spans by string-matching the quotations back into the extracted text would make
the reliability estimate a measurement of the string matcher. So alpha_U proper
is NOT computable from these files, and nothing here claims to be it.

What IS computable is Krippendorff's alpha for nominal data over an aligned
item set, which is the standard treatment for extraction tasks once the two
annotations have been put in correspondence:

  items      the union of both annotations, aligned 1:1 by the value-level
             matcher (a matched pair = one item; an unmatched claim = one item)
  codings    each item carries exactly two codings, one per annotator; the
             annotator who did not record that claim codes it as the
             distinguished category NOT_ANNOTATED
  delta      nominal (0 if the two codings are identical, 1 otherwise)

Three questions, three alphas:

  alpha_property   category = property name, or NOT_ANNOTATED
                   -> agreement on what to extract AND what to call it
  alpha_presence   category = ANNOTATED / NOT_ANNOTATED
                   -> agreement on what to extract, naming ignored
  alpha_naming     matched items only, category = property name
                   -> agreement on naming, given both found the claim

THE CAVEAT THAT MUST TRAVEL WITH THESE NUMBERS: the item universe is generated
by the annotators themselves, so the NOT_ANNOTATED category can only appear
where at least one of them proposed a claim. There is no inventory of "numbers
in the document that both annotators correctly ignored". Chance disagreement is
therefore estimated on the union of proposals, which makes these alphas
conservative relative to a task with a fixed item inventory. They are
"reliability conditional on the union of proposals", and should be reported with
that phrase attached.

Run:  python -m experiments.exp14_annotator_agreement.alpha
"""
from __future__ import annotations

import sys
from pathlib import Path

NOT_ANNOTATED = "∅"
ANNOTATED = "•"


def alpha_nominal(codings: list[tuple]) -> dict:
    """Krippendorff's alpha, nominal metric, over (coder_a, coder_b) per item.

    Computed through the coincidence matrix, so the reported observed and
    expected disagreements are the quantities in Krippendorff's formula rather
    than a two-coder shortcut.
    """
    items = [c for c in codings if c[0] is not None and c[1] is not None]
    n_items = len(items)
    if n_items == 0:
        return dict(n_items=0, alpha=None, d_o=None, d_e=None, observed=None)

    n = 2 * n_items                              # total pairable codings
    disagreeing = sum(1 for a, b in items if a != b)
    d_o = disagreeing / n_items                  # nominal delta, m_u = 2 for all

    marg: dict[str, int] = {}
    for a, b in items:
        marg[a] = marg.get(a, 0) + 1
        marg[b] = marg.get(b, 0) + 1
    sum_sq = sum(v * v for v in marg.values())
    d_e = (n * n - sum_sq) / (n * (n - 1))

    a = 1 - d_o / d_e if d_e > 0 else None
    return dict(n_items=n_items, alpha=None if a is None else round(a, 3),
                d_o=round(d_o, 4), d_e=round(d_e, 4),
                observed=round(1 - d_o, 3), categories=len(marg))


def build_codings(pairs, ref_only, sec_only) -> dict:
    """Turn a matched/unmatched split into the three coding lists."""
    prop, presence = [], []
    for g, h in pairs:
        prop.append((g.property, h.property))
        presence.append((ANNOTATED, ANNOTATED))
    for g in ref_only:
        prop.append((g.property, NOT_ANNOTATED))
        presence.append((ANNOTATED, NOT_ANNOTATED))
    for h in sec_only:
        prop.append((NOT_ANNOTATED, h.property))
        presence.append((NOT_ANNOTATED, ANNOTATED))
    naming = [(g.property, h.property) for g, h in pairs]
    return dict(property=prop, presence=presence, naming=naming)


def alphas(pairs, ref_only, sec_only) -> dict:
    c = build_codings(pairs, ref_only, sec_only)
    return dict(
        alpha_property=alpha_nominal(c["property"]),
        alpha_presence=alpha_nominal(c["presence"]),
        alpha_naming=alpha_nominal(c["naming"]),
        n_items=len(c["property"]),
    )


# --- self-check -------------------------------------------------------------
def _self_check() -> None:
    """Hand-computable cases, asserted so the formula cannot drift silently."""
    # perfect agreement
    assert alpha_nominal([("a", "a"), ("b", "b")])["alpha"] == 1.0

    # 4 items, A = a a b b, B = a b b b
    #   d_o = 1/4 = 0.25
    #   marginals over 8 codings: a=3, b=5 -> sum_sq = 34
    #   d_e = (64 - 34) / (8*7) = 0.535714
    #   alpha = 1 - 0.25/0.535714 = 0.5333
    r = alpha_nominal([("a", "a"), ("a", "b"), ("b", "b"), ("b", "b")])
    assert abs(r["alpha"] - 0.533) < 0.001, r

    # systematic disagreement -> alpha at or below 0
    r = alpha_nominal([("a", "b"), ("a", "b"), ("a", "b")])
    assert r["alpha"] is not None and r["alpha"] <= 0.0, r
    print("[exp14.alpha] self-check passed")


if __name__ == "__main__":
    _self_check()
    from experiments.exp14_annotator_agreement.common import DEFAULT_SECOND_SET, document_set
    from experiments.exp14_annotator_agreement.score import classify, match_views

    second = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SECOND_SET
    pooled_pairs, pooled_ref, pooled_sec = [], [], []
    for d in document_set(second):
        views = match_views(d["ref_claims"], d["sec_claims"])
        dis = classify(d["ref_claims"], d["sec_claims"], views)
        pairs = views["value_level"]
        ref_only = dis["ref_only"] + [g for g, _ in dis["value_disagreements"]]
        sec_only = dis["sec_only"] + [h for _, h in dis["value_disagreements"]]
        a = alphas(pairs, ref_only, sec_only)
        print(f"{d['label']:52} items={a['n_items']:4}  "
              f"a_prop={a['alpha_property']['alpha']}  "
              f"a_pres={a['alpha_presence']['alpha']}  "
              f"a_name={a['alpha_naming']['alpha']}")
        pooled_pairs += pairs
        pooled_ref += ref_only
        pooled_sec += sec_only
    tot = alphas(pooled_pairs, pooled_ref, pooled_sec)
    print(f"\nPOOLED items={tot['n_items']}  "
          f"alpha_property={tot['alpha_property']['alpha']}  "
          f"alpha_presence={tot['alpha_presence']['alpha']}  "
          f"alpha_naming={tot['alpha_naming']['alpha']}")
