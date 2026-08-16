# Annotation guideline — quantitative claims in lithium-ion cell datasheets

**Second independent annotator package.** Version 1.0.

## 0. What this is, and why it is blind

We are building a hand-labelled reference set of the **quantitative claims** that
battery-cell manufacturers make in their product datasheets. A first annotator
has already labelled these five documents. You are the **second, independent
annotator**: your labels will be compared against the first set to compute
inter-annotator agreement, which we report in a journal revision.

For that number to mean anything, your annotation must be produced **without
seeing the existing labels**. Concretely, please do **not** look at:

- any file in the project repository (in particular `data/claims/` and
  `data/gold/`), or any repository history, notebook or paper draft;
- any automatic-extraction output for these documents;
- any earlier annotation of these documents, in any format.

Everything you need is in this folder. If something in this guideline is
ambiguous, **do not** resolve it by looking at the project files — write your
reading down in the `annotator_notes` field and move on. A documented
disagreement is a useful result; a contaminated one is not.

The property vocabulary in §4 is a shared **codebook** — it is the labelling
scheme, not the answers. It does not tell you which properties appear in which
document, with which values, conditions, or how many claims a document holds.

Please annotate the documents in the order listed in §1, and do not go back and
revise an earlier document after seeing a later one, except to fix an outright
mistake (note it in `annotator_notes` if you do).

## 1. What is in this package

```
ANNOTATION_GUIDELINE.md          this file
documents/                       the five source PDFs
  a123_apr18650m1a.pdf
  lg_inr18650hg2.pdf
  panasonic_ncr18650b.pdf                    (image-only — see §8.7)
  panasonic_ncr18650b_full_spec_sanyo.pdf
  samsung_inr18650_25r.pdf
templates/                       one blank YAML per document — fill these in
  a123_apr18650m1a_annotation.yaml
  lg_inr18650hg2_annotation.yaml
  panasonic_ncr18650b_annotation.yaml
  panasonic_ncr18650b_full_spec_annotation.yaml
  samsung_inr18650_25r_annotation.yaml
```

The five documents are manufacturer specification sheets, and at least one is
marked confidential/proprietary by its publisher. Please keep this folder to
yourself and do not redistribute the PDFs.

Deliverable: the five filled-in template files, returned as they are (same
filenames). Nothing else is needed.

## 2. What counts as a claim

**Annotate every quantitative assertion the document makes about the cell.**
A claim is one *(property, value)* statement — a number (or a numeric range)
that the manufacturer asserts about the product, together with whatever test
conditions the document states for it.

Annotate:

- rows of specification tables (nominal capacity, voltage, currents, mass, …);
- values stated in running text or in section headings;
- cells of characteristic grids/tables (temperature dependence, rate dependence)
  — **each cell that states a number is its own claim** (§8.2);
- values you can only read off a plotted curve (§7.8);
- pass/acceptance criteria stated numerically ("capacity ≥ 72 % of nominal after
  N cycles") — these are claims too, flagged as such in `notes` (§7.9).

Do **not** annotate:

- qualitative statements with no number ("excellent high-rate performance");
- mechanical, safety and abuse test criteria (crush, nail penetration, vibration,
  drop, overcharge protection, shipping/transport tests);
- warranty periods, commercial terms, packaging, ordering codes, part numbers;
- general laboratory conditions that qualify the *whole test setup* rather than
  one property (a blanket "unless otherwise noted, measurements are taken at
  ambient temperature and humidity" section). These are not claims — but if such
  a condition applies to a specific claim, quote it in that claim's
  `stated_conditions.text`;
- dimensions given only as tolerances on a mechanical drawing, *unless* the
  document also states a dimension as a specification value — then it is a claim
  like any other, and will usually need a name you coin per §4.2.

If you are unsure whether something is a claim, **include it** and say why you
hesitated in `notes`. Over-inclusion is recoverable in analysis; omission is not.

## 3. The record format

Each template is a YAML file. Fill in the header, then add one block per claim
under `claims:`. YAML is whitespace-sensitive: keep the indentation of the
examples exactly (2 spaces per level, `- property:` at 2 spaces).

```yaml
  - property: nominal_capacity_ah      # snake_case name, §4
    value: 2.87                        # number, or [lo, hi] for a range
    unit: Ah                           # after normalisation, §5
    page: 2                            # physical PDF page, §6
    stated_conditions:
      text: "verbatim condition wording from the document, or: unspecified"
      temperature_c: 25                # parsed fields — ONLY if stated, §7.3
      discharge_current_a: 0.57
    notes: "optional free text"         # mandatory in the cases listed in §7.7-7.9
```

Field by field:

| field | required | content |
|---|---|---|
| `property` | yes | snake_case identifier from §4, or a new one you coin (§4.2) |
| `value` | yes | the number as printed, in the unit given by `unit`. A range is `[lo, hi]`. Never a string, never a `±`, never a `<`/`>` sign |
| `unit` | yes | `Ah`, `V`, `A`, `cycles`, `pct`, `degC`, `g`, `mOhm`, `Wh/kg`, `Wh/L`, `min`, `h`, `mm` |
| `page` | yes | physical PDF page the claim is printed on (§6) |
| `stated_conditions.text` | yes | verbatim/near-verbatim quotation of the conditions the document states — or exactly `unspecified` (§7.3) |
| `stated_conditions.<parsed>` | no | machine-readable copies of conditions **the document explicitly states** (§7.4) |
| `notes` | no | your reading caveats; mandatory for graph-read values (§7.8), pass criteria (§7.9), bounds (§7.7), resolved cross-references (§8.4), repeated values (§8.5) |

## 4. Property names

### 4.1 The controlled vocabulary

Use these names where they fit. The name encodes the unit, so a name and its
unit must agree.

| property | meaning |
|---|---|
| `nominal_capacity_ah` | rated/nominal capacity |
| `rated_capacity_ah` | rated capacity where stated separately from nominal |
| `minimum_capacity_ah` | guaranteed minimum capacity |
| `nominal_voltage_v` | nominal cell voltage |
| `charge_voltage_v` | CV charge voltage |
| `max_charge_voltage_v` | maximum allowed charge voltage |
| `discharge_cutoff_v` | end-of-discharge voltage |
| `std_charge_current_a` | standard charge current |
| `std_charge_time_h` | standard charge duration (hours) |
| `max_charge_current_a` | maximum allowed charge current |
| `fast_charge_current_a` | fast/rapid charge current |
| `std_discharge_current_a` | standard discharge current |
| `fast_discharge_current_a` | fast discharge current |
| `max_cont_discharge_a` | maximum continuous discharge current |
| `cycle_life_cycles` | cycle life claim (conditions are essential) |
| `cycle_life_retention_pct` | the capacity retention the cycle-life claim is defined at |
| `capacity_retention_at_temp_pct` | capacity retained at a stated temperature |
| `storage_capacity_remaining_pct` | capacity still present after stated storage |
| `storage_capacity_recovery_pct` | capacity recoverable (re-measured) after stated storage |
| `mass_g` | cell mass |
| `charge_temp_range_c` | allowed charge temperature range `[lo, hi]` |
| `discharge_temp_range_c` | allowed discharge temperature range `[lo, hi]` |
| `operating_temp_range_c` | allowed operating temperature range `[lo, hi]` |
| `storage_temp_range_c` | allowed storage temperature range `[lo, hi]` |
| `internal_impedance_mohm` | AC impedance (typically 1 kHz) |
| `gravimetric_energy_density_wh_kg` | energy density by mass |
| `volumetric_energy_density_wh_l` | energy density by volume |

Two distinctions that are easy to blur:

- `storage_capacity_remaining_pct` = capacity *still there* at the end of the
  storage period. `storage_capacity_recovery_pct` = capacity measured *after a
  recharge* following storage. If the document says "recovery", use the second.
- `charge_voltage_v` (the CV set-point of the standard charge protocol) vs.
  `max_charge_voltage_v` (an absolute limit). If one number serves both roles in
  the document, record it **once**, under the name the document's own wording
  supports, and explain in `notes`.

### 4.2 When nothing fits

Coin a name following the pattern `<quantity>_<unit-suffix>`, all lower case,
snake_case, e.g. `shelf_life_months`, `peak_power_w`. Unit
suffixes: `_ah _v _a _pct _c` (temperature in °C) `_g _mm _mohm _min _h
_cycles _wh_kg _wh_l`. Ranges keep the `_range_` infix.

**Every name you coin must also be listed in the header's `new_properties:`
block**, with a one-line meaning. That block is how we tell "you saw a property
we didn't" apart from "we named the same property differently".

Prefer reusing a vocabulary name with the varying quantity in
`stated_conditions` over coining a new name per variant — see §8.2.

## 5. Values and units

- **Normalise the unit, never the number's meaning.** `2,870mAh` → `value: 2.87`,
  `unit: Ah`. `1,350mA` → `value: 1.35`, `unit: A`.
  Keep the printed precision: `4.13V` → `4.13`, `70.15mm` → `70.15`.
- **Do not convert time units.** If the document says 115 min, record
  `value: 115`, `unit: min` (and coin `std_charge_time_min` per §4.2). Converting
  to hours would put a number in the file that the document does not contain.
- **Do not convert between A and C-rate.** Record what is printed; if the
  document gives a C-rate as a *condition*, put it in `discharge_c_rate` /
  `charge_c_rate`, not in `discharge_current_a`.
- `pct` for percentages: `83.7 %` → `value: 83.7`, `unit: pct`.
- `degC` for temperatures. Negative values are fine: `-17`.
- **Record what the document SAYS, not what is plausible.** If a value looks
  wrong or contradicts another section, record it as printed and flag it in
  `notes`. Contradictions are exactly what this study is about.

## 6. Page numbers

`page` is the **physical PDF page**, as your PDF viewer counts it: the first
page of the file is `page: 1`, whether it is a cover sheet or not.

Many datasheets carry their own printed footer ("3 / 16"), and it often does
**not** match the PDF page. Always record the PDF page. Every claim needs one so
that it can be re-checked against the document.

If one claim is genuinely printed across a page break, use the page where the
value itself appears.

## 7. Conditions

Test conditions are what make a manufacturer claim comparable with a laboratory
measurement, so they matter as much as the value.

### 7.1 `text` is a quotation

Quote the document. Near-verbatim is fine — you may compress a long table row
into one line and add a section number ("2.6.2 Fast discharge: CC 13,500mA;
end voltage 2.65V"), but do not paraphrase numbers or add any that are not there.

### 7.2 Which conditions belong to a claim

Those the document attaches to *that* value: the section's own test protocol,
the row's column headers, a footnote marker on that row, and general conditions
that the document explicitly scopes to the section. If a condition sits in a
different section and is only reachable by following a reference, see §8.4.

### 7.3 The `unspecified` rule — the important one

If the document states **no** test conditions for a value, write exactly:

```yaml
    stated_conditions:
      text: unspecified
```

**Never fill in a typical, conventional, or industry-standard value for a
condition the document does not state.** Not 25 °C, not 4.2 V, not 100 % DOD,
not "room temperature". The absence of a stated condition is itself one of the
findings this dataset exists to measure — inventing a plausible value destroys
that measurement.

The same applies partially: if a section states the temperature but not the
discharge rate, record the temperature and simply leave the rate out. An absent
field means "the document does not say", and that is a valid, informative state.

### 7.4 Parsed condition fields

Optional, machine-readable duplicates of what is already in `text`. Add them
only when the document explicitly states the condition. Commonly used:

`temperature_c`, `charge_temperature_c`, `discharge_temperature_c`,
`charge_current_a`, `discharge_current_a`, `charge_c_rate`, `discharge_c_rate`,
`charge_voltage_v`, `discharge_cutoff_v`, `end_current_ma`, `charge_time_min`,
`dod_pct`, `cycles`, `duration_days`, `end_condition` (a short string, e.g.
`"capacity >= 72% of nominal at cycle 437"`).

If you need a condition with no field above, add one in the same style
(`<quantity>_<unit>`) and mention it in `annotator_notes`.

### 7.5 Tolerances `X ± Y`

`value` gets **only the central value X**. The `± Y` stays in
`stated_conditions.text`, quoted. So `4.13 ± 0.07 V` →
`value: 4.13`, `text: "Max. charge voltage: 4.13 ± 0.07V"`.
Never write `value: [4.06, 4.20]` for a tolerance — square brackets are for
ranges the document itself states as a range (§7.6).

### 7.6 Ranges

A quantity the document states as an interval → `value: [lo, hi]`, low value
first, e.g. `-17 ~ 53°C` → `value: [-17, 53]`, `unit: degC`.

Ranges of *allowed operating envelope* (temperature ranges, ex-factory voltage
window) are ranges. A ± tolerance on a single target value is not (§7.5).

### 7.7 Bounds: `Max.`, `Min.`, `≥`, `≤`, "over"

Put the **bare number** in `value`; keep the bounding word in the quoted `text`;
and record the direction in `notes` as `upper bound` or `lower bound`.

`Max. 61.4 g` → `value: 61.4`, `text: "2.8 Weight: Max. 61.4 g"`,
`notes: "upper bound"`.
`Over 615 cycles` → `value: 615`, `notes: "lower bound; document says 'Over 615 cycles'"`.

### 7.8 Values read off a graph

If the number is only plotted, not printed: read it conservatively, round
conservatively, and write `notes: "read from graph"`. Say in `text` which figure
and which curve you read.

### 7.9 Pass criteria vs. typical values

Many cycle-life and storage numbers are *acceptance criteria* ("the 437th cycle
capacity shall be ≥ 72 % of nominal"), not expected performance. Record them,
and write `notes: "pass criterion, not typical life"`. If the document instead
presents a value as typical/nominal performance, say nothing (that is the
default reading).

## 8. Edge cases

### 8.1 Specification tables

Each row that states a number is one claim. A row that states two quantities
("Capacity / voltage, nominal: 2.87 Ah / 3.62 V") is **two** claims, each with
its own property, both quoting the same row in `text`.

### 8.2 Characteristic grids (temperature / rate dependence)

A 2-D grid of percentages is not one claim — **every grid cell that states a
value is a separate claim**. The row/column that varies goes into the claim's
`stated_conditions` (both in `text` and as a parsed field), not into the property
name, and the reference the percentages are relative to (e.g. "of nominal
capacity") belongs in `text` too. See example E in §9.

The 100 % reference cell of a normalised grid is a claim like any other.

If, and only if, a grid cell cannot be told apart from its siblings by its
conditions, coin distinct property names per §4.2 and record them in
`new_properties`.

### 8.3 Footnotes

A footnote attached to a row is part of that row's conditions — fold it into
`text`. A footnote that itself states a new number (e.g. a note defining the
retention threshold) is a **separate claim** if it asserts a property value, and
in that case `page` is the page the footnote is printed on.

### 8.4 Cross-references between sections

Where a claim's conditions are defined elsewhere ("charged per 2.3",
"discharged at the maximum continuous discharge current"), you may resolve the
reference and fill in the parsed condition fields — but you **must** flag it:
`notes: "cross-reference resolved: 'charge per 2.3' = CCCV 1.35A, 4.13V"`.
If a reference is ambiguous, leave the field out and describe the ambiguity in
`notes`. Resolving is optional; flagging it is not.

### 8.5 Repeated values

The same value often appears in several places (a summary table and a detailed
section). Record it **once**, at the occurrence with the most complete
conditions, and list the other locations in `notes`
("repeated in the summary table on page 1, there without conditions"). Do not create duplicate claims for one assertion.

Conversely, one property may legitimately appear **several times with different
conditions** — three storage-temperature rows for three durations, two cycle-life
figures at two discharge rates. Those are separate claims that share a property
name; distinguish them purely through `stated_conditions` (example D in §9).

### 8.6 Cycle life specifically

Cycle life is the most consequential claim in these documents, so give it extra
care:

- A cycle-life statement at *n* different discharge (or charge) rates is *n*
  separate `cycle_life_cycles` claims, one per rate.
- If the statement also defines the end-of-life capacity threshold ("≥ 72 % of
  nominal"), record that threshold **twice**: as a string in the claim's
  `end_condition`, and as its own `cycle_life_retention_pct` claim (value 72,
  unit `pct`) quoting the same sentence. This is deliberate redundancy.
- If the charge regime, temperature or threshold is not stated — leave it out
  (§7.3). A cycle-life claim with unstated conditions is a finding.

### 8.7 The image-only document

`documents/panasonic_ncr18650b.pdf` is a scanned/graphical sheet: its text
cannot be selected or searched, and it contains plotted curves. **Annotate it by
eye, directly from the PDF**, zooming as needed. Expect to use §7.8 (read from
graph) for some values. If a character is genuinely illegible, do not guess:
leave the claim out and record it in `annotator_notes` as illegible.

### 8.8 Two documents for the same cell

Two of the five documents describe the same commercial cell (a short marketing
sheet and a full specification). Annotate them **independently, as separate
documents**, even where they repeat each other or contradict each other. Do not
carry a value from one file into the other, and do not reconcile them — the
disagreement between them is data.

## 9. Worked examples

The examples below use a **fictional datasheet**, invented for this guideline.
No such cell exists, and none of these numbers is taken from any of the five
documents you are annotating. They illustrate the format only.

### The fictional source excerpt

```
NORTHWIND CELL TECHNOLOGIES — Product Specification NW-PS-40X, Rev. A
Model NDR21700-40X                                  [FICTIONAL — not a real cell]

-- PDF page 2 --------------------------------------------------------------
 2.1  Nominal capacity ............ 2,870mAh   (0.57A discharge to 2.65V, 25°C)
 2.2  Minimum capacity ............ Min. 2,790mAh
 2.3  Standard charge ............. CCCV 1.35A, 4.13 ± 0.07V, 65mA cut-off
 2.6  Max. continuous discharge ... 17,500mA
 2.9  Operating temp. (discharge) . -17 ~ 53°C

-- PDF page 5 --------------------------------------------------------------
 4.2.3  Cycle life
   (a) charge per 2.3, discharge 7.0A CC to 2.65V, 25 ± 2°C:
       437th cycle discharge capacity shall be >= 72% of nominal capacity
   (b) as (a) but discharged at 13.5A CC:
       268th cycle discharge capacity shall be >= 72% of nominal capacity

-- PDF page 7 --------------------------------------------------------------
 4.4  Temperature dependency of discharge capacity (charged at 25°C, 1.35A;
      discharged at 0.57A; % of the capacity measured at 25°C)
         discharge temperature :  -15°C    0°C    25°C    45°C
         discharge capacity    :  83.7%   91.2%   100%    98.4%
```

### A — a simple value with stated conditions (§3, §5)

`2,870mAh` becomes `2.87 Ah`; the conditions are quoted and also parsed.

```yaml
  - property: nominal_capacity_ah
    value: 2.87
    unit: Ah
    page: 2
    stated_conditions:
      text: "2.1 Nominal capacity: 2,870mAh (0.57A discharge to 2.65V, 25°C)"
      temperature_c: 25
      discharge_current_a: 0.57
      discharge_cutoff_v: 2.65
```

### B — a bound and a tolerance (§7.5, §7.7)

`Min. 2,790mAh`: the number goes in `value`, the word `Min.` stays in the
quotation, the direction goes in `notes`.

```yaml
  - property: minimum_capacity_ah
    value: 2.79
    unit: Ah
    page: 2
    stated_conditions:
      text: "2.2 Minimum capacity: Min. 2,790mAh — no test conditions stated"
    notes: "lower bound"
```

`4.13 ± 0.07V`: **only the central value** 4.13 goes in `value`. The `± 0.07`
appears nowhere except in the quoted text.

```yaml
  - property: charge_voltage_v
    value: 4.13
    unit: V
    page: 2
    stated_conditions:
      text: "2.3 Standard charge: CCCV 1.35A, 4.13 ± 0.07V, 65mA cut-off"
      end_current_ma: 65
```

Note that section 2.3 also states a *current*, which is a second claim from the
same row (§8.1):

```yaml
  - property: std_charge_current_a
    value: 1.35
    unit: A
    page: 2
    stated_conditions:
      text: "2.3 Standard charge: CCCV 1.35A, 4.13 ± 0.07V, 65mA cut-off"
      charge_voltage_v: 4.13
      end_current_ma: 65
```

### C — a range (§7.6)

`-17 ~ 53°C` is an interval the document itself states as an interval.

```yaml
  - property: discharge_temp_range_c
    value: [-17, 53]
    unit: degC
    page: 2
    stated_conditions:
      text: "2.9 Operating temperature (discharge): -17 ~ 53°C"
```

### D — one section, two rates → two claims, plus the retention claim (§8.6)

Section 4.2.3 states cycle life at two discharge currents. That is **two**
`cycle_life_cycles` claims, told apart only by their conditions — never one
claim with two values.

```yaml
  - property: cycle_life_cycles
    value: 437
    unit: cycles
    page: 5
    stated_conditions:
      text: "4.2.3 (a) Cycle life: charge per 2.3 (CCCV 1.35A, 4.13V), discharge
             7.0A CC to 2.65V, 25 ± 2°C; 437th cycle discharge capacity
             >= 72% of nominal capacity"
      temperature_c: 25
      charge_current_a: 1.35
      discharge_current_a: 7.0
      discharge_cutoff_v: 2.65
      end_condition: "capacity >= 72% of nominal at cycle 437"
    notes: "pass criterion, not typical life; cross-reference resolved:
            'charge per 2.3' = CCCV 1.35A, 4.13V, 65mA cut-off"

  - property: cycle_life_cycles
    value: 268
    unit: cycles
    page: 5
    stated_conditions:
      text: "4.2.3 (b) Cycle life: as (a) but discharged at 13.5A CC;
             268th cycle discharge capacity >= 72% of nominal capacity"
      temperature_c: 25
      charge_current_a: 1.35
      discharge_current_a: 13.5
      discharge_cutoff_v: 2.65
      end_condition: "capacity >= 72% of nominal at cycle 268"
    notes: "pass criterion, not typical life"

  - property: cycle_life_retention_pct
    value: 72
    unit: pct
    page: 5
    stated_conditions:
      text: "4.2.3: end-of-life threshold for both rates — discharge capacity
             >= 72% of nominal capacity"
    notes: "the threshold the 4.2.3 cycle-life claims are defined at; stated
            once for both rates, recorded once"
```

### E — a characteristic-grid cell (§8.2)

The grid in 4.4 has four cells, so it yields **four** claims. Two of them:

```yaml
  - property: capacity_retention_at_temp_pct
    value: 83.7
    unit: pct
    page: 7
    stated_conditions:
      text: "4.4 Temperature dependency of discharge capacity: charged at 25°C
             / 1.35A, discharged at 0.57A at -15°C -> 83.7% of the capacity
             measured at 25°C"
      charge_temperature_c: 25
      discharge_temperature_c: -15
      discharge_current_a: 0.57

  - property: capacity_retention_at_temp_pct
    value: 98.4
    unit: pct
    page: 7
    stated_conditions:
      text: "4.4: charged at 25°C / 1.35A, discharged at 0.57A at 45°C -> 98.4%
             of the capacity measured at 25°C"
      charge_temperature_c: 25
      discharge_temperature_c: 45
      discharge_current_a: 0.57
```

The remaining two cells of the grid (0 °C and the 25 °C reference cell) are
recorded the same way.

The varying temperature lives in the conditions, **not** in the property name,
so all four cells share one property name and stay distinguishable.

## 10. Omissions — properties verified absent

Each template ends with an `omissions:` block. After you have read a document
**exhaustively**, list the properties you actively looked for and confirmed the
document does not state:

Continuing the fictional excerpt of §9 — which states a capacity, a voltage and
currents, but no impedance and no energy density — its omissions block would
read:

```yaml
omissions:
  - property: internal_impedance_mohm
    note: "no AC or DC internal resistance figure anywhere; searched the
           specification table (2.x), the characteristics section (4.x) and the
           footnotes"
  - property: gravimetric_energy_density_wh_kg
    note: "no Wh/kg or Wh/L figure in the specification table (2.x) or the
           characteristics section (4.x); the document gives capacity and
           voltage but never an energy figure"
```

Rules:

- An omission is a claim about your search, so only record one if you actually
  searched. Say **where** you searched in `note`.
- An empty `omissions: []` means "I checked and found nothing worth recording";
  if you did **not** do an exhaustive absence check for a document, delete the
  whole block and say so in `annotator_notes`. Those two states must not be
  confused.
- Record an omission when a property from §4.1 is one a reader of *this*
  document would reasonably expect it to state and it does not. Do not
  enumerate every vocabulary entry the document happens not to use, and do not
  record a property that would make no sense for the document's type.
- Judge each document on its own (§8.8). If two documents describe the same cell
  and only one states a property, that is an omission of the other — record it
  from what that document itself does or does not say, not by comparison.

## 11. Before you send the files back

- [ ] Every claim has `property`, `value`, `unit`, `page`.
- [ ] No `value` contains a `±`, a `<`/`>`/`≥` sign, a unit, or a word — those
      live in `text` and `notes`.
- [ ] Every `stated_conditions` has a `text`, and unstated conditions are
      `unspecified` rather than filled in with typical values (§7.3).
- [ ] No parsed condition field holds a number that is not printed in the
      document (except a cross-reference you flagged per §8.4).
- [ ] mAh → Ah and mA → A conversions done; minutes left as minutes.
- [ ] Grid cells are individual claims (§8.2).
- [ ] Cycle-life claims split by rate, with `cycle_life_retention_pct` recorded
      separately where a threshold is stated (§8.6).
- [ ] Names you coined are listed in `new_properties`.
- [ ] `omissions` filled in, or deleted with an explanation.
- [ ] Header filled in: `annotator`, `annotated_date`, `annotator_notes`.
- [ ] The file still parses as YAML. If you can run Python:
      `python -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]).read())" <file>`
      — no output means it is fine. If you cannot, any online YAML validator
      does the same, or just send it and we will check.

Thank you — and please do send along anything about the documents that surprised
you, whether or not it fits the schema. That commentary is often as useful as
the labels.
