"""Graph-free GBM baseline (Severson-style).

XGBoost on [early-cycle features + parsed policy features + batch] under
leave-one-policy-group-out CV. No graph information — this is the model the
graph-augmented variant has to beat.

Run:  python -m src.models.baseline
"""
from __future__ import annotations

import pandas as pd

from src.models.common import lopgo_folds, make_model, metric_table
from src.models.dataset import BASE_FEATURES, TARGET, build_dataset


def run_baseline(df: pd.DataFrame, features: list[str] = BASE_FEATURES) -> pd.DataFrame:
    """Grouped-CV predictions: one row per cell with y_true_log / y_pred_log."""
    rows = []
    for gid, train, test in lopgo_folds(df):
        model = make_model()
        model.fit(df.iloc[train][features], df.iloc[train][TARGET])
        preds = model.predict(df.iloc[test][features])
        for i, p in zip(test, preds):
            rows.append({
                "cell_id": df.iloc[i]["cell_id"],
                "fold": gid,
                "batch": int(df.iloc[i]["batch"]),
                "y_true_log": float(df.iloc[i][TARGET]),
                "y_pred_log": float(p),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = build_dataset()
    data = data[~data["is_anomalous"]].reset_index(drop=True)
    preds = run_baseline(data)
    print(f"[baseline] {len(preds)} cells, {preds['fold'].nunique()} folds")
    m = metric_table(preds, "baseline")
    print(f"[baseline] rmse_log={m['rmse_log']:.4f}  "
          f"rmse_cycles={m['rmse_cycles']:.1f}  mape={m['mape_pct']:.2f}%")
