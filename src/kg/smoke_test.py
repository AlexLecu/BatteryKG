"""Post-load sanity report for the knowledge graph.

Prints and saves to outputs/kg_smoke_test.txt:
  - node counts per label, edge counts per type
  - mean / median cycle_life_nominal
  - the 10 lowest-coverage_xgroup instances (policy string + cycle lives)

Run (after load + coverage):  python -m src.kg.smoke_test
"""
from __future__ import annotations

from src.config import OUTPUTS
from src.kg.connection import get_driver
from src.kg.schema import NODE_LABELS, RELATIONSHIP_TYPES


def run_smoke_test(driver=None) -> str:
    own = driver is None
    driver = driver or get_driver()
    lines: list[str] = []

    def out(s: str = "") -> None:
        lines.append(s)
        print(s)

    try:
        with driver.session() as s:
            out("=== BatteryKG knowledge-graph smoke test ===")

            out("\nNode counts per label:")
            for label in NODE_LABELS:
                n = s.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()["c"]
                out(f"  {label:14s} {n}")

            out("\nEdge counts per type:")
            for rt in RELATIONSHIP_TYPES:
                n = s.run(f"MATCH ()-[r:`{rt}`]->() RETURN count(r) AS c").single()["c"]
                out(f"  {rt:14s} {n}")
            for view in ("condition", "behavior"):
                n = s.run("MATCH ()-[r:SIMILAR_TO {view: $v}]->() RETURN count(r) AS c",
                          v=view).single()["c"]
                out(f"    SIMILAR_TO[{view}] {n}")

            rec = s.run(
                "MATCH (m:Measurement {metric:'cycle_life_nominal'}) "
                "WHERE m.value IS NOT NULL "
                "RETURN count(m) AS n, avg(m.value) AS mean, "
                "percentileCont(m.value, 0.5) AS median, "
                "min(m.value) AS mn, max(m.value) AS mx"
            ).single()
            out("\ncycle_life_nominal (Measurement.value):")
            out(f"  n={rec['n']}  mean={rec['mean']:.1f}  median={rec['median']:.1f}  "
                f"min={rec['mn']:.0f}  max={rec['mx']:.0f}")

            out("\n10 lowest coverage_xgroup instances:")
            out(f"  {'instance':10s} {'policy_norm':22s} {'cov_x':>7s} {'cov_all':>7s} {'life_nom':>8s}")
            q = """
            MATCH (ci:CellInstance)
            OPTIONAL MATCH (ci)<-[:ABOUT]-(m:Measurement {metric:'cycle_life_nominal'})
            RETURN ci.study_cell_id AS id, ci.charge_policy_norm AS pol,
                   ci.coverage_xgroup AS covx, ci.coverage_all AS cova, m.value AS life
            ORDER BY ci.coverage_xgroup ASC, ci.study_cell_id ASC
            LIMIT 10
            """
            for r in s.run(q):
                covx = f"{r['covx']:.3f}" if r["covx"] is not None else "NA"
                cova = f"{r['cova']:.3f}" if r["cova"] is not None else "NA"
                life = f"{r['life']:.0f}" if r["life"] is not None else "NA"
                out(f"  {r['id']:10s} {str(r['pol']):22s} {covx:>7s} {cova:>7s} {life:>8s}")

            out("\nClaims per cell:")
            rows = s.run(
                "MATCH (cl:Claim)-[:ABOUT]->(c:Cell) "
                "RETURN c.model AS model, count(cl) AS n ORDER BY model").data()
            if rows:
                for r in rows:
                    out(f"  {r['model']:26s} {r['n']}")
            else:
                out("  (none loaded)")

            out("\nDiscrepancies (claim vs measured):")
            drows = s.run(
                "MATCH (d:Discrepancy)-[:ABOUT]->(c:Cell) "
                "WHERE d.claim_value IS NOT NULL AND d.measured_median IS NOT NULL "
                "RETURN c.model AS model, d.claim_value AS claim, "
                "d.measured_median AS med, d.relative_gap AS gap, "
                "d.conditions_comparable AS comparable").data()
            if drows:
                for r in drows:
                    out(f"  {r['model']}: claim {r['claim']:.0f} vs measured median "
                        f"{r['med']:.0f} cycles -> gap {r['gap']:+.1%} "
                        f"(conditions_comparable={r['comparable']})")
            else:
                out("  (none computed)")

            out("\nDiscrepancies (claim vs claim — spec consistency):")
            crows = s.run(
                "MATCH (d:Discrepancy {kind: 'claim_vs_claim'})-[:ABOUT]->(c:Cell) "
                "RETURN c.model AS model, d.verdict AS verdict, count(*) AS n "
                "ORDER BY model, verdict").data()
            if crows:
                for r in crows:
                    out(f"  {r['model']:26s} {r['verdict']:26s} {r['n']}")
            else:
                out("  (none computed)")

        OUTPUTS.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS / "kg_smoke_test.txt"
        path.write_text("\n".join(lines) + "\n")
        print(f"\nsaved {path}")
        return "\n".join(lines)
    finally:
        if own:
            driver.close()


if __name__ == "__main__":
    run_smoke_test()
