"""Unit tests for experiment 12's deterministic table grounding (no API calls,
no PDF required — the page-boundary fixture reproduces the real §7.6 split)."""
import pytest

from experiments.exp12_table_grounding.table_grid import (
    TableRegion,
    classify,
    compose_property_name,
    is_free_text,
    is_quantity,
    quantity_row_indices,
    render_rows,
    render_spec_row,
    stitch_regions,
    units_from_regions,
)


# --- cell-level quantity parsing ---------------------------------------------
def test_quantity_vs_free_text():
    for cell in ("-20℃", "0.50A", "100%", "2,500mAh", "45.0g max".split()[0], "18mΩ"):
        assert is_quantity(cell), cell
    for cell in ("CCCV, 1.25A, 4.20 ± 0.05 V, 125mA cut-off", "Relative capacity",
                 "Discharge temperature", "Standard 1.25A", "1.5 year -30~25℃(1*)"):
        assert not is_quantity(cell), cell
        assert is_free_text(cell), cell
    assert not is_quantity("") and not is_free_text("")


def test_quantity_row_indices_finds_axis_and_value_rows():
    rows = [["", "Charge temperature", "", "", "", "", "Discharge temperature"],
            ["", "0℃", "5℃", "25℃", "45℃", "50℃", "25℃"],
            ["Relative capacity", "80%", "90%", "100%", "95%", "95%", ""]]
    assert quantity_row_indices(rows) == [1, 2]


# --- page-boundary stitch (the §7.6 split) -----------------------------------
AXIS_HALF = [["Discharge temperature", "", "", "", ""],
             ["-20℃", "-10℃", "0℃", "25℃", "60℃"]]
VALUE_HALF = [["60%", "75%", "80%", "100%", "100%"]]


def _region(rid, pages, rows):
    return TableRegion(region_id=rid, pages=pages, rows=rows,
                       n_cols=max(len(r) for r in rows))


def test_stitch_rejoins_grid_split_across_pages():
    a = _region("p4_t1", [4], AXIS_HALF)
    b = _region("p5_t1", [5], VALUE_HALF)
    regions, log = stitch_regions([a, b])
    assert len(regions) == 1 and len(log) == 1
    merged = regions[0]
    assert merged.stitched and merged.pages == [4, 5]
    assert merged.rows == AXIS_HALF + VALUE_HALF
    assert classify(merged) == "grid"          # 2 quantity rows once rejoined
    # neither half is a grid on its own — that is why flat text lost this table
    assert classify(a) != "grid" and classify(b) != "grid"


@pytest.mark.parametrize("a_rows,b_rows,b_pages", [
    (AXIS_HALF, [["60%", "75%", "80%", "100%"]], [5]),          # column count differs
    (AXIS_HALF, [["Relative", "75%", "80%", "100%", "100%"]], [5]),  # free text in B
    (AXIS_HALF + VALUE_HALF, VALUE_HALF, [5]),                  # A already a grid
    (AXIS_HALF, VALUE_HALF, [6]),                               # pages not adjacent
])
def test_stitch_is_conservative(a_rows, b_rows, b_pages):
    regions, log = stitch_regions([_region("a", [4], a_rows),
                                   _region("b", b_pages, b_rows)])
    assert len(regions) == 2 and log == []
    assert not any(r.stitched for r in regions)


# --- classification + units ---------------------------------------------------
def test_classify_grid_spec_table_and_other():
    grid = _region("g", [5], [["Current", "0.50A", "5A", "10A", "15A", "20A"],
                              ["Relative Capacity", "100%", "97%", "100%", "97%", "95%"]])
    spec = _region("s", [3], [["Item", "Specification"],
                              ["3.2 Nominal voltage", "3.6V"],
                              ["3.9 Cell weight", "45.0g max"],
                              ["3.8 Discharge cut-off voltage", "2.5V"]])
    other = _region("o", [17], [["Version", "Date", "Changes/Author", "Reason"],
                                ["1.0", "'14-02-10", "In-Young Jang", "First version"]])
    assert classify(grid) == "grid"
    assert classify(spec) == "spec_table"
    assert classify(other) == "other"


def test_units_one_per_grid_one_per_spec_row_header_skipped():
    grid = _region("g", [5], [["Current", "0.50A", "5A"],
                              ["Relative Capacity", "100%", "97%"]])
    grid.kind = "grid"
    spec = _region("s", [3], [["Item", "Specification"],          # skipped: no digits
                              ["3.2 Nominal voltage", "3.6V"],
                              ["3.9 Cell weight", "45.0g max"],
                              ["3.10 Cell dimension", "Height : 64.85 ± 0.15mm"]])
    spec.kind = "spec_table"
    units = units_from_regions([grid, spec])
    assert [u.kind for u in units] == ["grid", "spec_row", "spec_row", "spec_row"]
    assert units[0].rows == grid.rows                 # a grid stays whole
    assert all(len(u.rows) == 1 for u in units[1:])   # a spec row is isolated
    assert "Specification" not in units[0].render     # the header row is skipped
    assert "3.2 Nominal voltage" in units[1].render


def test_render_pads_ragged_rows():
    assert render_rows([["a", "b"], ["c"]]) == "| a | b |\n| c |  |"


def test_spec_row_render_keeps_one_source_line_per_line():
    """A cell holding three storage rows must not collapse into one line —
    exp11's model read section 3.12 as a single claim."""
    row = ["3.12 Storage temperature\n(Recovery 90% after storage)",
           "1.5 year -30~25℃(1*)\n3 months -30~45℃(1*)\n1 month -30~60℃(1*)"]
    render = render_spec_row(row)
    assert render.splitlines() == [
        "Item: 3.12 Storage temperature (Recovery 90% after storage)",
        "Specification:",
        "  1.5 year -30~25℃(1*)",
        "  3 months -30~45℃(1*)",
        "  1 month -30~60℃(1*)",
    ]


def test_quantity_checks_ignore_intra_cell_line_breaks():
    assert is_quantity("100\n%") or is_quantity("100%")   # flattened before matching
    assert is_free_text("Relative\nCapacity")


# --- deterministic namer ------------------------------------------------------
GOLD_GRID_NAMES = {
    ("rel_discharge_capacity", "temperature_c", -20): "rel_discharge_capacity_m20c_pct",
    ("rel_discharge_capacity", "temperature_c", -10): "rel_discharge_capacity_m10c_pct",
    ("rel_discharge_capacity", "temperature_c", 0): "rel_discharge_capacity_0c_pct",
    ("rel_discharge_capacity", "temperature_c", 25): "rel_discharge_capacity_25c_pct",
    ("rel_discharge_capacity", "temperature_c", 60): "rel_discharge_capacity_60c_pct",
    ("rel_charge_capacity", "temperature_c", 0): "rel_charge_capacity_0c_pct",
    ("rel_charge_capacity", "temperature_c", 5): "rel_charge_capacity_5c_pct",
    ("rel_charge_capacity", "temperature_c", 25): "rel_charge_capacity_25c_pct",
    ("rel_charge_capacity", "temperature_c", 45): "rel_charge_capacity_45c_pct",
    ("rel_charge_capacity", "temperature_c", 50): "rel_charge_capacity_50c_pct",
    ("rel_capacity", "charge_mode", "std"): "rel_capacity_std_charge_pct",
    ("rel_capacity", "charge_mode", "rapid"): "rel_capacity_rapid_charge_pct",
    ("rel_discharge_capacity", "current_a", 0.50): "rel_discharge_capacity_0_5a_pct",
    ("rel_discharge_capacity", "current_a", 5): "rel_discharge_capacity_5a_pct",
    ("rel_discharge_capacity", "current_a", 10): "rel_discharge_capacity_10a_pct",
    ("rel_discharge_capacity", "current_a", 15): "rel_discharge_capacity_15a_pct",
    ("rel_discharge_capacity", "current_a", 20): "rel_discharge_capacity_20a_pct",
}


def test_namer_composes_every_gold_grid_name():
    """All 17 gold table-cell names must be reachable — otherwise the pilot
    could not score on the paper metric even with perfect grid reading."""
    assert len(GOLD_GRID_NAMES) == 17
    for (family, kind, value), expected in GOLD_GRID_NAMES.items():
        assert compose_property_name(family, kind, value) == expected


def test_namer_is_insensitive_to_numeric_formatting():
    assert (compose_property_name("rel_discharge_capacity", "current_a", 0.5)
            == compose_property_name("rel_discharge_capacity", "current_a", 0.50)
            == "rel_discharge_capacity_0_5a_pct")
    assert (compose_property_name("rel_charge_capacity", "temperature_c", 25.0)
            == "rel_charge_capacity_25c_pct")
    assert (compose_property_name("rel_discharge_capacity", "temperature_c", -20.0)
            == "rel_discharge_capacity_m20c_pct")


def test_namer_rejects_values_outside_the_closed_sets():
    with pytest.raises(ValueError):
        compose_property_name("relative_capacity", "temperature_c", 25)   # bad family
    with pytest.raises(ValueError):
        compose_property_name("rel_capacity", "voltage_v", 4.2)           # bad axis kind
    with pytest.raises(ValueError):
        compose_property_name("rel_capacity", "charge_mode", "trickle")   # bad mode
