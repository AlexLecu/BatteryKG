"""Graph-augmented GBM: baseline features + per-fold KG neighbor features.

For each cell and each similarity view ('condition', 'behavior'), its stored
top-10 SIMILAR_TO neighbors are restricted to the CURRENT TRAINING FOLD, the
top-k (k=5) by weight are kept, and four scalars are derived:

  {view}_nbr_wmean_log_life  weighted mean of neighbors' log10 cycle life
  {view}_nbr_wstd_log_life   weighted std of the same
  {view}_nbr_mean_weight     mean edge weight of the kept neighbors
  {view}_coverage_train      sum of kept edge weights (graph coverage INTO the
                             training set — the abstention signal)

Leakage discipline: neighbor labels are only ever read from training cells, and
`graph_feature_row` asserts at runtime that no selected neighbor is a test cell.
Under leave-one-policy-group-out CV, same-group replicates sit in the test fold
together, so they can never lend their labels to each other.

Run:  python -m src.models.graph_model
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.common import lopgo_folds, make_model, metric_table
from src.models.dataset import BASE_FEATURES, TARGET, build_dataset

VIEWS = ("condition", "behavior")
K_GRAPH = 5

_EDGES_QUERY = """
MATCH (a:CellInstance)-[r:SIMILAR_TO {view: $view}]->(b:CellInstance)
RETURN a.study_cell_id AS src, b.study_cell_id AS dst,
       r.weight AS weight, r.same_policy_group AS same_group
ORDER BY src, weight DESC, dst
"""


def fetch_edges(driver=None, views=VIEWS) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """Read SIMILAR_TO edges from the KG: {view: {src: [(dst, weight), ...]}}.

    Lists are sorted by weight descending (dst ascending on ties).
    """
    from src.kg.connection import get_driver
    own = driver is None
    driver = driver or get_driver()
    try:
        out: dict[str, dict[str, list[tuple[str, float]]]] = {}
        with driver.session() as s:
            for view in views:
                nbrs: dict[str, list[tuple[str, float]]] = {}
                for rec in s.run(_EDGES_QUERY, view=view):
                    nbrs.setdefault(rec["src"], []).append((rec["dst"], float(rec["weight"])))
                out[view] = nbrs
        return out
    finally:
        if own:
            driver.close()


def select_train_neighbors(nbrs: list[tuple[str, float]], train_ids: set[str],
                           k: int = K_GRAPH) -> list[tuple[str, float]]:
    """Restrict a (weight-sorted) neighbor list to training cells, keep top-k."""
    return [(dst, w) for dst, w in nbrs if dst in train_ids][:k]


def graph_feature_row(selected: list[tuple[str, float]], train_ids: set[str],
                      test_ids: set[str], y_log_train: dict[str, float],
                      view: str) -> dict[str, float]:
    """Neighbor scalars from an already-selected neighbor list.

    Runtime leakage tripwire: every selected neighbor must be a training cell.
    """
    for dst, _w in selected:
        assert dst in train_ids and dst not in test_ids, (
            f"LEAKAGE: neighbor '{dst}' (view={view}) is not a training cell")

    if not selected:
        return {f"{view}_nbr_wmean_log_life": np.nan,
                f"{view}_nbr_wstd_log_life": np.nan,
                f"{view}_nbr_mean_weight": np.nan,
                f"{view}_coverage_train": 0.0}

    w = np.array([wt for _, wt in selected], float)
    y = np.array([y_log_train[dst] for dst, _ in selected], float)
    wmean = float(np.sum(w * y) / np.sum(w))
    wstd = float(np.sqrt(np.sum(w * (y - wmean) ** 2) / np.sum(w)))
    return {f"{view}_nbr_wmean_log_life": wmean,
            f"{view}_nbr_wstd_log_life": wstd,
            f"{view}_nbr_mean_weight": float(np.mean(w)),
            f"{view}_coverage_train": float(np.sum(w))}


def graph_feature_columns(views=VIEWS) -> list[str]:
    return [f"{v}_{stem}" for v in views
            for stem in ("nbr_wmean_log_life", "nbr_wstd_log_life",
                         "nbr_mean_weight", "coverage_train")]


def fold_graph_features(cell_ids, edges_by_view, train_ids: set[str],
                        test_ids: set[str], y_log_train: dict[str, float],
                        k: int = K_GRAPH) -> pd.DataFrame:
    """Graph features for `cell_ids`, using train-fold neighbors only."""
    rows = []
    for cid in cell_ids:
        row: dict = {"cell_id": cid}
        for view, nbrs in edges_by_view.items():
            own_train = train_ids - {cid}   # a train cell never counts itself
            sel = select_train_neighbors(nbrs.get(cid, []), own_train, k)
            row.update(graph_feature_row(sel, own_train, test_ids, y_log_train, view))
        rows.append(row)
    return pd.DataFrame(rows)


def run_graph_model(df: pd.DataFrame, edges_by_view,
                    base_features: list[str] = BASE_FEATURES,
                    k: int = K_GRAPH) -> pd.DataFrame:
    """Grouped-CV predictions with per-fold graph features appended."""
    gcols = graph_feature_columns(tuple(edges_by_view))
    rows = []
    for gid, train, test in lopgo_folds(df):
        train_ids = set(df.iloc[train]["cell_id"])
        test_ids = set(df.iloc[test]["cell_id"])
        y_log_train = dict(zip(df.iloc[train]["cell_id"], df.iloc[train][TARGET]))

        gf = fold_graph_features(df["cell_id"].tolist(), edges_by_view,
                                 train_ids, test_ids, y_log_train, k)
        fold_df = df.merge(gf, on="cell_id", how="left")
        feats = base_features + gcols

        tr = fold_df[fold_df["cell_id"].isin(train_ids)]
        te = fold_df[fold_df["cell_id"].isin(test_ids)]
        model = make_model()
        model.fit(tr[feats], tr[TARGET])
        preds = model.predict(te[feats])
        for (_, r), p in zip(te.iterrows(), preds):
            rows.append({
                "cell_id": r["cell_id"],
                "fold": gid,
                "batch": int(r["batch"]),
                "y_true_log": float(r[TARGET]),
                "y_pred_log": float(p),
                **{c: float(r[c]) if pd.notna(r[c]) else np.nan for c in gcols},
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = build_dataset()
    data = data[~data["is_anomalous"]].reset_index(drop=True)
    edges = fetch_edges()
    preds = run_graph_model(data, edges)
    print(f"[graph_model] {len(preds)} cells, {preds['fold'].nunique()} folds")
    m = metric_table(preds, "graph")
    print(f"[graph_model] rmse_log={m['rmse_log']:.4f}  "
          f"rmse_cycles={m['rmse_cycles']:.1f}  mape={m['mape_pct']:.2f}%")
