"""Graph-coverage metric.

coverage(instance) = sum of the top-k SIMILAR_TO edge weights, in two variants:
  - coverage_all     : over all neighbours
  - coverage_xgroup  : excluding same-policy-group neighbours (the honest value
                       for grouped cross-validation, where an instance may not
                       borrow strength from replicates of its own condition)

Both are written back as CellInstance properties.

Run:  python -m src.kg.coverage
"""
from __future__ import annotations

from src.kg.connection import get_driver
from src.kg.features import coverage_for_edges

DEFAULT_K = 5

# `$only IS NULL` -> every instance; otherwise restrict to the given ids. The
# scoping keeps tests from reading/writing the working graph (see tests).
# Edges are filtered to the requested similarity view.
_READ = """
MATCH (ci:CellInstance)
WHERE $only IS NULL OR ci.study_cell_id IN $only
OPTIONAL MATCH (ci)-[r:SIMILAR_TO {view: $view}]->()
RETURN ci.study_cell_id AS id,
       collect({w: r.weight, sg: r.same_policy_group}) AS edges
"""

# property names carry the view suffix for non-condition views so the original
# (condition-based) coverage_all / coverage_xgroup semantics are preserved.
_WRITE_TEMPLATE = """
UNWIND $rows AS row
MATCH (ci:CellInstance {{study_cell_id: row.id}})
SET ci.`{all_prop}` = row.cov_all, ci.`{xgroup_prop}` = row.cov_xgroup
"""


def _prop_names(view: str) -> tuple[str, str]:
    suffix = "" if view == "condition" else f"_{view}"
    return f"coverage_all{suffix}", f"coverage_xgroup{suffix}"


def compute_coverage(driver=None, k: int = DEFAULT_K,
                     only_ids: list[str] | None = None,
                     view: str = "condition") -> dict:
    """Compute coverage for all instances (or only `only_ids`) in one view."""
    own = driver is None
    driver = driver or get_driver()
    all_prop, xgroup_prop = _prop_names(view)
    try:
        with driver.session() as session:
            records = session.run(_READ, only=only_ids, view=view).data()
            rows = []
            for rec in records:
                edges = [(e["w"], bool(e["sg"])) for e in rec["edges"] if e["w"] is not None]
                cov_all, cov_xgroup = coverage_for_edges(edges, k)
                rows.append({"id": rec["id"], "cov_all": cov_all, "cov_xgroup": cov_xgroup})
            session.run(_WRITE_TEMPLATE.format(all_prop=all_prop, xgroup_prop=xgroup_prop),
                        rows=rows)
        print(f"[kg.coverage] wrote {all_prop} / {xgroup_prop} (k={k}) "
              f"for {len(rows)} instances")
        return {"instances": len(rows), "k": k, "view": view}
    finally:
        if own:
            driver.close()


if __name__ == "__main__":
    for v in ("condition", "behavior"):
        compute_coverage(view=v)
