"""Unit tests for experiment 13's OCR stage (no PDF, no tesseract, no API)."""
import numpy as np
import pytest

from experiments.exp13_ocr_panasonic.ocr import RULE_MIN_RUN, strip_rules, to_ink
from experiments.exp13_ocr_panasonic.score import graph_claim_keys


# --- table-rule removal ------------------------------------------------------
def test_strip_rules_erases_long_runs_and_keeps_glyph_strokes():
    """A ruled table border is a long ink run; a glyph stroke is a short one."""
    # tall enough that a full-height vertical run exceeds RULE_MIN_RUN
    ink = np.zeros((100, 200), dtype=bool)
    ink[10, :] = True                       # horizontal rule across the image
    ink[:, 150] = True                      # vertical rule
    ink[20, 30:38] = True                   # short stroke — a glyph
    ink[25:31, 60] = True                   # short vertical stroke — a glyph

    out = strip_rules(ink, min_run=RULE_MIN_RUN)

    assert not out[10, :].any(), "horizontal rule survived"
    assert not out[:, 150].any(), "vertical rule survived"
    assert out[20, 30:38].all(), "horizontal glyph stroke was erased"
    assert out[25:31, 60].all(), "vertical glyph stroke was erased"


def test_strip_rules_threshold_is_the_boundary():
    ink = np.zeros((5, 100), dtype=bool)
    ink[1, 0:RULE_MIN_RUN - 1] = True       # one px under the threshold: kept
    ink[3, 0:RULE_MIN_RUN] = True           # at the threshold: erased
    out = strip_rules(ink, min_run=RULE_MIN_RUN)
    assert out[1].any()
    assert not out[3].any()


def test_strip_rules_does_not_mutate_its_input():
    ink = np.zeros((5, 100), dtype=bool)
    ink[2, :] = True
    before = ink.copy()
    strip_rules(ink, min_run=RULE_MIN_RUN)
    assert np.array_equal(ink, before)


def test_to_ink_marks_dark_pixels():
    from PIL import Image
    img = Image.fromarray(np.array([[0, 255], [200, 10]], dtype=np.uint8), mode="L")
    assert to_ink(img).tolist() == [[True, False], [False, True]]


# --- gold subset derivation --------------------------------------------------
def test_graph_claims_derived_from_gold_notes():
    """The graph subset must come from the YAML's notes, not a hardcoded list."""
    keys = graph_claim_keys()
    assert keys == {("discharge_cutoff_v", "2.5")}


def test_graph_subset_partitions_the_gold():
    from src.agents.evaluation import load_gold
    from experiments.exp13_ocr_panasonic.run_ocr_extraction import GOLD_DOC
    gold = [g for g in load_gold() if g.doc == GOLD_DOC]
    keys = graph_claim_keys()
    graph = [g for g in gold if (g.property, str(g.value)) in keys]
    spec = [g for g in gold if (g.property, str(g.value)) not in keys]
    assert len(gold) == 14
    assert len(graph) == 1 and len(spec) == 13
