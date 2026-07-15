"""Load HUST instances into the KG + extend behavior-space SIMILAR_TO edges
across studies.

New Source (hust_2022) and CellInstances under the EXISTING A123 APR18650M1A
Cell node (same commercial cell — phase-0 verified). After loading, the
behavior-view SIMILAR_TO edges are rebuilt over the UNION of Severson and HUST
instances (same feature set — all four behavior features are computable from
HUST), so neighborhoods genuinely span studies. Condition-view edges remain
per-study (HUST's charge policy is constant across its cells).

Uses the frozen loader machinery via import only; no frozen code is modified.

Run:  python -m src.kg.hust_load
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED
from src.ingestion import cell_metadata as cm
from src.ingestion.qc import cycle_life_table
from src.kg.connection import get_driver
from src.kg.features import parse_policy
from src.kg.load import (
    _EDGE_CLEAR_CYPHER,
    _EDGE_CYPHER,
    BEHAVIOR_FEATURE_COLS,
    FEATURE_COLS,
    SourceSpec,
    build_severson_instances,
    load_dataframe,
    similarity_edges,
)

HUST_SOURCE = SourceSpec(
    source_id="hust_2022",
    type="academic_cycling_study",
    citation="Ma et al. 2022, Energy & Environmental Science (DOI 10.1039/d2ee01676a); "
             "dataset DOI 10.17632/nsc7hnsg4s.2 (CC BY 4.0)")


def build_hust_instances() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "hust_cycles.parquet")
    life = cycle_life_table(df)
    meta = (df.groupby("cell_id")
              .agg(charge_policy_raw=("charge_policy_raw", "first"),
                   charge_policy_norm=("charge_policy_norm", "first"),
                   policy_group_id=("policy_group_id", "first"))
              .reset_index())
    inst = life.merge(meta, on="cell_id", how="left")
    feats = pd.read_parquet(PROCESSED / "hust_features.parquet")
    inst = inst.merge(feats, on="cell_id", how="left")
    inst = inst.rename(columns={"cell_id": "study_cell_id", "flags": "qc_flag"})
    pol = inst["charge_policy_norm"].map(parse_policy)
    inst["c_rate_1"] = [p.c_rate_1 if p else np.nan for p in pol]
    inst["c_rate_2"] = [p.c_rate_2 if p else np.nan for p in pol]
    inst["soc_transition_pct"] = [p.soc_transition_pct if p else np.nan for p in pol]
    inst["batch"] = pd.NA                       # Severson-specific covariate
    return inst


def load(driver=None) -> dict:
    own = driver is None
    driver = driver or get_driver()
    try:
        hust = build_hust_instances()
        result = load_dataframe(
            hust, cell=cm.SEVERSON_CELL, source=HUST_SOURCE,
            parquet_ref=str(PROCESSED / "hust_cycles.parquet"), driver=driver,
            edge_views=(("condition", FEATURE_COLS),
                        ("behavior", BEHAVIOR_FEATURE_COLS)))
        print(f"[kg.hust] {result['instances']} instances loaded "
              f"({result['edges_by_view']})")

        # cross-study behavior edges: rebuild the behavior view over the UNION
        sev = build_severson_instances()
        cols = ["study_cell_id", "policy_group_id"] + BEHAVIOR_FEATURE_COLS
        combined = pd.concat([sev[cols], hust[cols]], ignore_index=True)
        edges = similarity_edges(combined, k=10,
                                 feature_cols=BEHAVIOR_FEATURE_COLS,
                                 view="behavior")
        ids = combined["study_cell_id"].astype(str).tolist()
        with driver.session() as s:
            s.run(_EDGE_CLEAR_CYPHER, ids=ids, view="behavior")
            s.run(_EDGE_CYPHER, edges=edges)
        cross = sum(1 for e in edges
                    if e["src"].startswith("HUST_") != e["dst"].startswith("HUST_"))
        print(f"[kg.hust] behavior view rebuilt across studies: {len(edges)} edges, "
              f"{cross} cross-study")
        return {"instances": result["instances"], "behavior_edges": len(edges),
                "cross_study_edges": cross}
    finally:
        if own:
            driver.close()


if __name__ == "__main__":
    load()
