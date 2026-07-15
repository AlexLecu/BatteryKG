"""First claim-vs-measured discrepancy computation — A123 APR18650M1A only.

Compares the datasheet cycle-life claim against the distribution of measured
`cycle_life_nominal` across the Severson instances of the same commercial cell.

The comparison is recorded HONESTLY: the Severson cells were fast-charged
(3.6C-8C two-step policies — the study's whole point), while a datasheet
cycle-life claim assumes the manufacturer's standard charge. Unless the claim's
stated conditions match the measurement conditions, `conditions_comparable` is
false and the relative gap is context, not a verdict.

Creates:  (Discrepancy)-[:CONTRASTS]->(Claim), (Discrepancy)-[:ABOUT]->(Cell)
with props: relative_gap, conditions_comparable, rationale, measured stats.

Run:  python -m src.kg.discrepancy
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import processed_parquet
from src.ingestion.qc import cycle_life_table
from src.kg.connection import get_driver

MODEL = "A123 APR18650M1A"
MEASUREMENT_SOURCE = "severson_mit_2019"

# Severson measurement conditions (from the study; see data/README.md)
MEASURED_CONDITIONS = ("30 degC chamber; fast charge 3.6C-8C two-step policies; "
                       "4C discharge to 2.0 V; 100% DOD; EOL = 80% of 1.1 Ah nominal")

_CLAIMS_QUERY = """
MATCH (cl:Claim {property: 'cycle_life_cycles'})-[:ABOUT]->(:Cell {model: $model})
RETURN cl.claim_id AS claim_id, cl.value AS value, cl.conditions_text AS conditions,
       cl.cond_temperature_c AS temp, cl.cond_discharge_c_rate AS dis_c,
       cl.cond_dod_pct AS dod, cl.cond_end_condition AS end_cond
"""

_DISCREPANCY_CYPHER = """
MATCH (cl:Claim {claim_id: $claim_id})
MATCH (c:Cell {model: $model})
MERGE (d:Discrepancy {discrepancy_id: $did})
  SET d.property = 'cycle_life_cycles',
      d.claim_value = $claim_value,
      d.measured_median = $med, d.measured_min = $mn, d.measured_max = $mx,
      d.measured_iqr_lo = $q1, d.measured_iqr_hi = $q3,
      d.n_measurements = $n,
      d.measurement_source = $meas_source,
      d.measured_conditions = $meas_conditions,
      d.claim_conditions = $claim_conditions,
      d.relative_gap = $gap,
      d.conditions_comparable = $comparable,
      d.rationale = $rationale
MERGE (d)-[:CONTRASTS]->(cl)
MERGE (d)-[:ABOUT]->(c)
"""


def measured_distribution() -> dict:
    """Distribution of measured cycle_life_nominal over all 124 instances."""
    df = pd.read_parquet(processed_parquet("severson_mit"))
    life = cycle_life_table(df)
    v = life["cycle_life_nominal"].dropna().to_numpy(float)
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "min": float(v.min()), "max": float(v.max()),
        "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
    }


def compute_discrepancies(driver=None) -> list[dict]:
    own = driver is None
    driver = driver or get_driver()
    try:
        meas = measured_distribution()
        results = []
        with driver.session() as s:
            claims = s.run(_CLAIMS_QUERY, model=MODEL).data()
            if not claims:
                raise RuntimeError(
                    f"no cycle_life_cycles claims for {MODEL} — run: python -m src.kg.claims")
            for cl in claims:
                claim_value = float(cl["value"])
                gap = (meas["median"] - claim_value) / claim_value
                # comparable only if the claim's charging regime matches the
                # measured one; Severson is fast-charge by design, datasheet
                # cycle life assumes standard charging -> not comparable.
                comparable = False
                rationale = (
                    f"Claim: {claim_value:.0f} cycles under '{cl['conditions']}'. "
                    f"Measured: median {meas['median']:.0f} cycles "
                    f"(IQR {meas['q1']:.0f}-{meas['q3']:.0f}, range {meas['min']:.0f}-"
                    f"{meas['max']:.0f}, n={meas['n']}) under '{MEASURED_CONDITIONS}'. "
                    "NOT directly comparable, despite partial overlap (100% DOD both; "
                    "4C measured vs 5C claimed discharge): (1) the measured cells were "
                    "deliberately fast-charged at 3.6C-8C to induce accelerated "
                    "degradation, while the claim's charge regime is unstated "
                    "(presumably the datasheet's 1.5A standard charge); (2) the claim "
                    "states no temperature and, critically, no end-of-life capacity "
                    "threshold, whereas the measurement uses 80% of nominal. The gap "
                    "quantifies the claim-vs-use-case spread under fast charging, not "
                    "a false claim."
                )
                did = f"{MODEL}:cycle_life:{cl['claim_id']}:{MEASUREMENT_SOURCE}"
                s.run(_DISCREPANCY_CYPHER, claim_id=cl["claim_id"], model=MODEL,
                      did=did, claim_value=claim_value,
                      med=meas["median"], mn=meas["min"], mx=meas["max"],
                      q1=meas["q1"], q3=meas["q3"], n=meas["n"],
                      meas_source=MEASUREMENT_SOURCE,
                      meas_conditions=MEASURED_CONDITIONS,
                      claim_conditions=cl["conditions"],
                      gap=gap, comparable=comparable, rationale=rationale)
                results.append({"claim_id": cl["claim_id"], "claim_value": claim_value,
                                "claim_conditions": cl["conditions"],
                                "measured": meas, "relative_gap": gap,
                                "conditions_comparable": comparable})
        return results
    finally:
        if own:
            driver.close()


# =========================================================================
# Act two: claimed capacity vs lygte-info.dk independent measurements
# =========================================================================
CURRENT_MATCH_TOL = 0.20        # discharge currents comparable within 20%

_CAPACITY_CLAIMS_QUERY = """
MATCH (cl:Claim)-[:ABOUT]->(c:Cell)
WHERE cl.property IN ['nominal_capacity_ah', 'minimum_capacity_ah', 'rated_capacity_ah']
RETURN c.model AS model, cl.claim_id AS claim_id, cl.property AS property,
       cl.value AS value, cl.conditions_text AS conditions,
       cl.cond_discharge_current_a AS current_a
ORDER BY model, property
"""

_LYGTE_CAPACITY_QUERY = """
MATCH (m:Measurement {metric: 'measured_capacity_ah'})-[:ABOUT]->(c:Cell)
MATCH (m)-[:MEASURED_BY]->(s:Source {type: 'independent_test'})
RETURN c.model AS model, m.measurement_id AS mid, m.value AS value,
       m.chart_only AS chart_only, m.cond_discharge_current_a AS current_a,
       m.source_fragment AS fragment, s.source_id AS source_id
"""

_CAPACITY_DISCREPANCY_CYPHER = """
MATCH (cl:Claim {claim_id: $claim_id})
MATCH (c:Cell {model: $model})
MERGE (d:Discrepancy {discrepancy_id: $did})
  SET d.property = $property,
      d.claim_value = $claim_value, d.measured_value = $measured_value,
      d.claim_current_a = $claim_current, d.measured_current_a = $measured_current,
      d.relative_gap = $gap,
      d.conditions_comparable = $comparable,
      d.rationale = $rationale,
      d.measurement_source = $source_id
MERGE (d)-[:CONTRASTS]->(cl)
MERGE (d)-[:ABOUT]->(c)
"""


def currents_comparable(a: float | None, b: float | None,
                        tol: float = CURRENT_MATCH_TOL) -> bool:
    """Discharge currents comparable when both stated and within `tol`
    (relative to the larger)."""
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(abs(a), abs(b))


def compute_capacity_discrepancies(driver=None) -> list[dict]:
    """Claimed nominal/minimum capacity vs lygte-measured capacity.

    Honest handling of the current data: on all three cached review pages the
    measured capacity exists ONLY as chart images (chart_only=true, value
    null), so no numeric comparison is computable — this function then creates
    NO Discrepancy nodes and reports the blockage explicitly. The comparison
    logic below is exercised by unit tests on synthetic data and will activate
    if valued capacity measurements ever enter the graph.
    """
    own = driver is None
    driver = driver or get_driver()
    results: list[dict] = []
    try:
        with driver.session() as s:
            claims = s.run(_CAPACITY_CLAIMS_QUERY).data()
            meas = s.run(_LYGTE_CAPACITY_QUERY).data()
            valued = [m for m in meas if m["value"] is not None]
            chart_only = [m for m in meas if m["value"] is None and m["chart_only"]]
            if not valued:
                print(f"[kg.discrepancy] capacity reconciliation: 0 computable — "
                      f"{len(chart_only)} lygte capacity records are chart-only "
                      "(no numeric values in page text; image extraction not "
                      "attempted by policy). No Discrepancy nodes created.")
                return []
            for cl in claims:
                cands = [m for m in valued if m["model"] == cl["model"]]
                if not cands:
                    continue
                # primary: nearest test current to the claim's stated current
                claim_i = cl["current_a"]
                if claim_i is not None:
                    cands_sorted = sorted(
                        cands, key=lambda m: abs((m["current_a"] or 1e9) - claim_i))
                else:
                    cands_sorted = sorted(cands, key=lambda m: m["current_a"] or 1e9)
                m0 = cands_sorted[0]
                comparable = currents_comparable(claim_i, m0["current_a"])
                gap = (m0["value"] - float(cl["value"])) / float(cl["value"])
                rationale = (
                    f"Claim {cl['property']}={cl['value']} Ah at "
                    f"{claim_i if claim_i is not None else 'unstated'} A vs lygte "
                    f"measured {m0['value']} Ah at {m0['current_a']} A "
                    f"({'currents match within 20%' if comparable else 'current mismatch'}). "
                    f"Source fragment: {m0['fragment']}")
                did = f"{cl['model']}:capacity:{cl['claim_id']}:{m0['source_id']}"
                s.run(_CAPACITY_DISCREPANCY_CYPHER, claim_id=cl["claim_id"],
                      model=cl["model"], did=did, property=cl["property"],
                      claim_value=float(cl["value"]), measured_value=m0["value"],
                      claim_current=claim_i, measured_current=m0["current_a"],
                      gap=gap, comparable=comparable, rationale=rationale,
                      source_id=m0["source_id"])
                results.append({"model": cl["model"], "property": cl["property"],
                                "gap": gap, "comparable": comparable})
                # secondary: highest-current measurement vs nominal claim (the
                # 'marketing number vs real load' gap) — never comparable
                if cl["property"] == "nominal_capacity_ah":
                    m_hi = max(cands, key=lambda m: m["current_a"] or 0)
                    if m_hi["current_a"] and m_hi is not m0:
                        gap_hi = (m_hi["value"] - float(cl["value"])) / float(cl["value"])
                        did_hi = (f"{cl['model']}:capacity_high_current:"
                                  f"{cl['claim_id']}:{m_hi['source_id']}")
                        rationale_hi = (
                            f"Nominal claim {cl['value']} Ah vs measured "
                            f"{m_hi['value']} Ah at {m_hi['current_a']} A — the "
                            "high-load gap; informative but NOT a like-for-like "
                            "comparison (nominal capacity is defined at low current).")
                        s.run(_CAPACITY_DISCREPANCY_CYPHER, claim_id=cl["claim_id"],
                              model=cl["model"], did=did_hi,
                              property="nominal_capacity_ah_at_high_load",
                              claim_value=float(cl["value"]), measured_value=m_hi["value"],
                              claim_current=claim_i, measured_current=m_hi["current_a"],
                              gap=gap_hi, comparable=False, rationale=rationale_hi,
                              source_id=m_hi["source_id"])
                        results.append({"model": cl["model"],
                                        "property": "nominal_capacity_ah_at_high_load",
                                        "gap": gap_hi, "comparable": False})
        return results
    finally:
        if own:
            driver.close()


if __name__ == "__main__":
    for r in compute_discrepancies():
        m = r["measured"]
        print(f"[kg.discrepancy] {MODEL}")
        print(f"  claim: {r['claim_value']:.0f} cycles  ({r['claim_conditions']})")
        print(f"  measured: median {m['median']:.0f} (IQR {m['q1']:.0f}-{m['q3']:.0f}, "
              f"range {m['min']:.0f}-{m['max']:.0f}, n={m['n']})")
        print(f"  relative gap (median vs claim): {r['relative_gap']:+.1%}")
        print(f"  conditions comparable: {r['conditions_comparable']}")
    compute_capacity_discrepancies()
