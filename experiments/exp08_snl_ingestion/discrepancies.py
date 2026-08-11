"""Step 5a — claim-vs-measured discrepancies unlocked by the SNL ingestion.

Both comparisons read their MEASURED side out of the knowledge graph, not out of
a parquet: these are queries the graph could not answer an hour ago and can now,
which is the point of the exercise.

Two records are created, matching the property shape of the existing
claim-vs-measured Discrepancy (src/kg/discrepancy.py) so the app and any
downstream query treat them uniformly:

  nominal_capacity_ah   1.1 Ah claimed vs 30 SNL cells' measured initial capacity
  cycle_life_cycles     1000 cycles claimed vs the SNL cells that reach EOL

Both carry conditions_comparable = False, and the rationale says why in the
datasheet's own words. For capacity the reason is not the 0.5C-vs-4C rate gap
that separates SNL from Severson — it is that the datasheet states NO
measurement conditions for its capacity number at all, so there is nothing to
compare conditions against. Under the project's own rule (currents_comparable,
20% tolerance) a null claim current cannot be declared comparable to anything.
The omission is itself the finding.

Run:  python -m experiments.exp08_snl_ingestion.discrepancies --confirm
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from experiments.exp08_snl_ingestion.load_snl import SOURCE_ID
from experiments.exp08_snl_ingestion.stage import OUT
from src.kg.connection import get_driver
from src.kg.discrepancy import _DISCREPANCY_CYPHER, currents_comparable

MODEL = "A123 APR18650M1A"
RPT_CURRENT_A = 0.55            # 0.5C of 1.1 Ah — the rate SNL's RPTs run at
MEASURED_CONDITIONS = (
    "25/15/35 degC chamber; 0.5C CC-CV charge; RPT capacity check at 0.5C over the "
    "full 3.6-2.0 V window; ageing at 0.5-3C discharge and 0-100/20-80/40-60% DoD; "
    "EOL = 80% of 1.1 Ah nominal")

_MEASURED_QUERY = """
MATCH (m:Measurement {metric: $metric})-[:ABOUT]->(ci:CellInstance)
MATCH (m)-[:MEASURED_BY]->(:Source {source_id: $source})
MATCH (ci)-[:INSTANCE_OF]->(:Cell {model: $model})
RETURN ci.study_cell_id AS cell, m.value AS value, m.censored AS censored,
       ci.temperature_c AS temp, ci.dod_window AS dod,
       ci.c_rate_discharge AS c_dis, ci.n_ageing_cycles AS n_ageing
ORDER BY cell
"""

_CLAIM_QUERY = """
MATCH (cl:Claim {property: $property})-[:ABOUT]->(:Cell {model: $model})
RETURN cl.claim_id AS claim_id, cl.value AS value,
       cl.conditions_text AS conditions,
       cl.cond_discharge_c_rate AS dis_c_rate,
       cl.cond_discharge_current_a AS dis_current_a,
       cl.cond_dod_pct AS dod, cl.cond_temperature_c AS temp
"""


def _stats(v: np.ndarray) -> dict:
    return {"n": int(v.size), "median": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max()),
            "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75))}


def fetch(session, metric: str) -> pd.DataFrame:
    return pd.DataFrame(session.run(_MEASURED_QUERY, metric=metric,
                                    source=SOURCE_ID, model=MODEL).data())


def capacity_discrepancy(session, write: bool) -> dict:
    cl = session.run(_CLAIM_QUERY, property="nominal_capacity_ah", model=MODEL).single()
    df = fetch(session, "initial_capacity_ah")
    v = df.loc[~df["censored"], "value"].to_numpy(float)
    st = _stats(v)
    claim = float(cl["value"])
    gap = (st["median"] - claim) / claim

    # the project's own comparability rule, applied honestly
    comparable = currents_comparable(cl["dis_current_a"], RPT_CURRENT_A)
    rationale = (
        f"Claim: nominal capacity {claim:.3f} Ah. The datasheet states no "
        f"measurement conditions for it — the entry reads \"{cl['conditions']}\". "
        f"Measured: median {st['median']:.4f} Ah (range {st['min']:.4f}-{st['max']:.4f}, "
        f"IQR {st['q1']:.4f}-{st['q3']:.4f}, n={st['n']}) as the initial RPT capacity "
        f"at {RPT_CURRENT_A} A (0.5C) over the full 3.6-2.0 V window, under "
        f"'{MEASURED_CONDITIONS}'. ALL {st['n']} cells sit below the claimed value, by "
        f"{100 * (1 - st['max'] / claim):.1f}% to {100 * (1 - st['min'] / claim):.1f}%. "
        "conditions_comparable = FALSE, and the reason is not the measurement: the "
        "claim carries no stated discharge current, temperature or cut-off, so there "
        "is no condition set to compare against and the project's currents_comparable "
        "rule cannot return true against a null. The omission is itself the finding — "
        "a capacity number with no conditions attached cannot be falsified, only "
        "contextualised. The consistent one-directional shortfall across 30 cells from "
        "an independent laboratory is that context.")
    did = f"{MODEL}:capacity:{cl['claim_id']}:{SOURCE_ID}"
    if write:
        session.run(_DISCREPANCY_CYPHER, claim_id=cl["claim_id"], model=MODEL, did=did,
                    claim_value=claim, med=st["median"], mn=st["min"], mx=st["max"],
                    q1=st["q1"], q3=st["q3"], n=st["n"], meas_source=SOURCE_ID,
                    meas_conditions=MEASURED_CONDITIONS, claim_conditions=cl["conditions"],
                    gap=gap, comparable=comparable, rationale=rationale)
        session.run("MATCH (d:Discrepancy {discrepancy_id: $did}) "
                    "SET d.property = 'nominal_capacity_ah'", did=did)
    return {"discrepancy_id": did, "property": "nominal_capacity_ah",
            "claim_value": claim, "claim_conditions": cl["conditions"],
            "measured": st, "relative_gap": gap, "conditions_comparable": comparable,
            "cells_below_claim": int((v < claim).sum()),
            "shortfall_pct_range": [round(100 * (1 - st["max"] / claim), 2),
                                    round(100 * (1 - st["min"] / claim), 2)],
            "rationale": rationale, "per_cell": df.to_dict("records")}


def cycle_life_discrepancy(session, write: bool) -> dict:
    cl = session.run(_CLAIM_QUERY, property="cycle_life_cycles", model=MODEL).single()
    df = fetch(session, "cycle_life_nominal")
    reached = df[~df["censored"]]
    v = reached["value"].to_numpy(float)
    if v.size == 0:
        return {"skipped": "no SNL cell reached EOL"}
    st = _stats(v)
    claim = float(cl["value"])
    gap = (st["median"] - claim) / claim
    conds = sorted({f"{int(r.temp)}degC/{r.c_dis}/{r.dod}" for r in reached.itertuples()})
    comparable = False
    rationale = (
        f"Claim: over {claim:.0f} cycles under '{cl['conditions']}'. Measured: median "
        f"{st['median']:.0f} cycles (range {st['min']:.0f}-{st['max']:.0f}, n={st['n']} "
        f"of {len(df)} SNL cells; the other {int(df['censored'].sum())} are right-censored, "
        "still above 80% of nominal when testing stopped) under "
        f"'{MEASURED_CONDITIONS}'. Conditions present in the measured set: "
        f"{', '.join(conds)}. conditions_comparable = FALSE: (1) the claim specifies 5C "
        f"discharge, the EOL-reaching SNL cells ran at {', '.join(sorted({r.c_dis for r in reached.itertuples()}))}; "
        "(2) the claim states no charge regime, no temperature and — critically — no "
        "end-of-life capacity threshold, while the measurement uses 80% of nominal; "
        "(3) the measured median is censored-biased, since only the fastest-degrading "
        "cells have reached EOL at all. Note the SIGN: these cells exceed the claim by "
        f"{100 * gap:+.0f}%, whereas the Severson instances of the same cell against the "
        "same claim fall SHORT by ~26%. Same cell, same datasheet number, opposite "
        "direction — Severson fast-charges at 3.6C-8C by design while SNL charges at "
        "0.5C. That contrast is the clearest available argument for why the gate reports "
        "conditions_comparable at all.")
    did = f"{MODEL}:cycle_life:{cl['claim_id']}:{SOURCE_ID}"
    if write:
        session.run(_DISCREPANCY_CYPHER, claim_id=cl["claim_id"], model=MODEL, did=did,
                    claim_value=claim, med=st["median"], mn=st["min"], mx=st["max"],
                    q1=st["q1"], q3=st["q3"], n=st["n"], meas_source=SOURCE_ID,
                    meas_conditions=MEASURED_CONDITIONS, claim_conditions=cl["conditions"],
                    gap=gap, comparable=comparable, rationale=rationale)
    return {"discrepancy_id": did, "property": "cycle_life_cycles",
            "claim_value": claim, "claim_conditions": cl["conditions"],
            "measured": st, "relative_gap": gap, "conditions_comparable": comparable,
            "n_censored": int(df["censored"].sum()), "conditions_present": conds,
            "rationale": rationale,
            "per_cell": reached[["cell", "value", "temp", "dod", "c_dis"]].to_dict("records")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm", action="store_true", help="write Discrepancy nodes")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    driver = get_driver()
    try:
        with driver.session() as s:
            before = s.run("MATCH (d:Discrepancy) RETURN count(*) AS c").single()["c"]
            cap = capacity_discrepancy(s, args.confirm)
            life = cycle_life_discrepancy(s, args.confirm)
            after = s.run("MATCH (d:Discrepancy) RETURN count(*) AS c").single()["c"]
    finally:
        driver.close()

    out = {"wrote_to_graph": args.confirm,
           "discrepancy_nodes_before": before, "discrepancy_nodes_after": after,
           "capacity": cap, "cycle_life": life}
    (OUT / "discrepancies.json").write_text(json.dumps(out, indent=1, default=str))
    pd.DataFrame(cap["per_cell"]).to_csv(OUT / "discrepancy_capacity_per_cell.csv", index=False)

    for r in (cap, life):
        print("=" * 78)
        print(f"{r['property']}   [{r['discrepancy_id']}]")
        print(f"  claim    : {r['claim_value']}")
        print(f"  measured : median {r['measured']['median']:.4f} "
              f"(n={r['measured']['n']}, {r['measured']['min']:.4f}-{r['measured']['max']:.4f})")
        print(f"  gap      : {100 * r['relative_gap']:+.1f}%")
        print(f"  comparable: {r['conditions_comparable']}")
        print(f"  rationale: {r['rationale'][:300]}...")
    print(f"\nDiscrepancy nodes: {before} -> {after} "
          f"({'written' if args.confirm else 'DRY RUN — nothing written'})")
    return out


if __name__ == "__main__":
    main()
