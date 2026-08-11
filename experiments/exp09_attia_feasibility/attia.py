"""Feasibility check: Attia et al. 2020 (Nature 578) as a third external dataset.

Batch 4 of the MIT/Stanford/Toyota series (2019-01-24) — the closed-loop
optimization study. Same A123 APR18650M1A cells, same lab, same equipment,
30 degC, 4C discharge; what changes is the charge protocol family (multi-step
constant-current, optimized by the CLO loop) rather than the measurement basis.

That makes it a genuinely different proposition from HUST or SNL: those changed
the discharge rate and therefore the feature basis, so their cells could not be
placed in the bank's behaviour space on the paper's own definition. Attia keeps
the 4C discharge, so dQ(V) between cycles 10 and 100 means the SAME thing here
as in the bank. If any external cell is ever going to clear the gate, it is one
of these.

Report only. Nothing is written to the graph, the bank, or any model.

Run:  python -m experiments.exp09_attia_feasibility.attia
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise ImportError("needs h5py: pip install h5py") from e

from app.common import _zscore_with, load_artifacts
from src.config import OUTPUTS, RAW_SEVERSON
from src.ingestion import cell_metadata as cm
from src.ingestion.download import SEVERSON_BATCH4
from src.ingestion.features import _read_cell, compute_early_features
from src.ingestion.qc import cycle_life_table
from src.ingestion.schema import conform
from src.ingestion.severson import STUDY as SEVERSON_STUDY
from src.ingestion.severson import _decode_char_array, _load_batch
from src.kg.features import parse_policy

OUT = OUTPUTS / "experiment_09_attia_feasibility"
BATCH4 = RAW_SEVERSON / "2019-01-24_batchdata_updated_struct_errorcorrect.mat"
BATCH_NUM = 4
STUDY = "attia_clo"
Q_STAR = 41.0                      # deployed percentile, fixed in-study (exp06)
ABSOLUTE_THRESHOLD = 4.080029      # the superseded absolute gate, for reference


def inspect_file() -> dict:
    """What the batch-4 file actually contains — no assumptions."""
    with h5py.File(BATCH4, "r") as f:
        batch = f["batch"]
        n = int(batch["summary"].shape[0])
        summ_keys = sorted(f[batch["summary"][0, 0]].keys())
        cyc_keys = sorted(f[batch["cycles"][0, 0]].keys())
        top = sorted(batch.keys())
        policies = []
        for i in range(min(n, 5)):
            try:
                policies.append(_decode_char_array(f, batch["policy_readable"][i, 0]))
            except Exception:
                policies.append("(undecodable)")
    return {"n_cells_in_file": n, "batch_keys": top, "summary_keys": summ_keys,
            "cycles_keys": cyc_keys, "policy_examples": policies}


# Batch-4 data quirk, found by inspection and handled explicitly rather than
# silently: exactly one cell (b4c31) stores 834 entries in every measurement
# array (QDischarge/QCharge/chargetime/IR) but only 833 cycle numbers. The
# trailing measurement cannot be assigned a cycle index, so all arrays are
# truncated to the common length. This touches no cycle <= 100, so early-cycle
# features are unaffected; the only value at risk is the very last capacity
# point of that one cell.
LENGTH_MISMATCH_CELLS: dict[str, int] = {}


def _truncate_common(*arrays):
    n = min(len(a) for a in arrays)
    return [np.asarray(a)[:n] for a in arrays], n


def build_cycles() -> pd.DataFrame:
    """Tidy per-cycle frame, same construction as the Severson loader."""
    cells = _load_batch(BATCH4, BATCH_NUM)
    cell = cm.SEVERSON_CELL
    frames = []
    for cell_id, d in cells.items():
        cyc, qd = d["cycle"], d["qd"]
        if len(cyc) != len(qd):
            LENGTH_MISMATCH_CELLS[cell_id] = len(qd) - len(cyc)
            (cyc, qd), _ = _truncate_common(cyc, qd)
        ok = np.isfinite(cyc) & np.isfinite(qd) & (qd > 0)
        cyc, qd = cyc[ok], qd[ok]
        if cyc.size == 0:
            continue
        frames.append(pd.DataFrame({
            "cell_id": cell_id, "study": STUDY,
            "cycle_index": np.round(cyc).astype("int64"),
            "discharge_capacity_ah": qd,
            "chemistry": cell.chemistry,
            "nominal_capacity_ah": cell.nominal_capacity_ah,
            "temperature_c": cm.SEVERSON_TEMPERATURE_C,
            "dod_window": cm.SEVERSON_DOD_WINDOW,
            "c_rate_charge": d["policy"] or pd.NA,
            "c_rate_discharge": cm.SEVERSON_C_RATE_DISCHARGE,
        }))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["cell_id", "cycle_index"], keep="first")
    return conform(df)


def _read_cell_tolerant(f, batch, i: int) -> dict | None:
    """`src.ingestion.features._read_cell` with the length quirk absorbed.

    Byte-for-byte the same logic, plus a truncation to the common array length.
    Only used when the shipped reader raises on the mismatched cell; the two are
    asserted to agree on unaffected cells before it is trusted.
    """
    from src.ingestion.features import CYCLE_MAX, DQ_EARLY, DQ_LATE
    summ = f[batch["summary"][i, 0]]
    cyc = np.array(summ["cycle"]).squeeze().astype(float)
    qd = np.array(summ["QDischarge"]).squeeze().astype(float)
    ct = (np.array(summ["chargetime"]).squeeze().astype(float)
          if "chargetime" in summ else None)
    arrs = [cyc, qd] + ([ct] if ct is not None else [])
    trunc, _ = _truncate_common(*arrs)
    cyc, qd = trunc[0], trunc[1]
    ct = trunc[2] if ct is not None else None

    cycles = f[batch["cycles"][i, 0]]

    def qdlin(n: int):
        idx = np.where(cyc == n)[0]
        if not len(idx):
            return None
        j = int(idx[0])
        assert cyc[j] <= CYCLE_MAX, "leakage: Qdlin read beyond cycle 100"
        return np.array(f[cycles["Qdlin"][j, 0]]).squeeze().astype(float)

    q_early, q_late = qdlin(DQ_EARLY), qdlin(DQ_LATE)
    if q_early is None or q_late is None or q_early.size < 2 or q_late.size < 2:
        return None
    keep = cyc <= CYCLE_MAX
    return {"cyc": cyc[keep], "qd": qd[keep],
            "ct": ct[keep] if ct is not None else None,
            "qdlin_early": q_early, "qdlin_late": q_late}


def build_features() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Early-cycle features via the paper's own extractor (<=100 boundary enforced)."""
    rows, skipped, tolerant = [], [], []
    with h5py.File(BATCH4, "r") as f:
        batch = f["batch"]
        n = int(batch["summary"].shape[0])
        # trust check: the tolerant reader must reproduce the shipped one exactly
        # on cells where no truncation happens
        for probe in range(min(3, n)):
            a, b = _read_cell(f, batch, probe), _read_cell_tolerant(f, batch, probe)
            if a is not None and b is not None:
                assert all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]))
                           for k in ("cyc", "qd", "qdlin_early", "qdlin_late")), \
                    "tolerant reader diverges from src.ingestion.features._read_cell"
        for i in range(n):
            cell_id = f"b{BATCH_NUM}c{i}"
            try:
                rec = _read_cell(f, batch, i)
            except (ValueError, IndexError):
                rec = _read_cell_tolerant(f, batch, i)
                tolerant.append(cell_id)
            if rec is None:
                skipped.append(cell_id)
                continue
            feats = compute_early_features(**rec)      # asserts the <=100 boundary
            feats["cell_id"] = cell_id
            rows.append(feats)
    return (pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True),
            skipped, tolerant)


def behavior_coverage(meta: dict, bank: pd.DataFrame, q: pd.DataFrame,
                      id_col: str, exclude_own_group: bool = False) -> np.ndarray:
    """Served-path behaviour coverage against `bank` (self-excluded by cell id)."""
    sc = meta["scaler"]["behavior"]
    cols, k = sc["columns"], meta["k_neighbors"]
    B = _zscore_with(bank[cols].to_numpy(float), sc["mean"], sc["std"])
    Q = _zscore_with(q[cols].to_numpy(float), sc["mean"], sc["std"])
    W = 1.0 / (1.0 + np.linalg.norm(Q[:, None, :] - B[None, :, :], axis=2))
    W = np.where(q[id_col].to_numpy()[:, None] == bank["cell_id"].to_numpy()[None, :],
                 -np.inf, W)
    if exclude_own_group:
        W = np.where(q["policy_group_id"].to_numpy()[:, None]
                     == bank["policy_group_id"].to_numpy()[None, :], -np.inf, W)
    return np.sort(W, axis=1)[:, -k:].sum(axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not BATCH4.exists() or BATCH4.stat().st_size != SEVERSON_BATCH4.expected_bytes:
        have = BATCH4.stat().st_size if BATCH4.exists() else 0
        raise SystemExit(
            f"batch-4 file incomplete ({have:,} of "
            f"{SEVERSON_BATCH4.expected_bytes:,} bytes).\n"
            "  python -m src.ingestion.download attia")

    info = inspect_file()
    print(f"[file] {info['n_cells_in_file']} cells; summary keys: {info['summary_keys']}")
    print(f"[file] policy examples: {info['policy_examples']}")

    cycles = build_cycles()
    life = cycle_life_table(cycles)
    feats, skipped, tolerant = build_features()
    print(f"[load] {cycles['cell_id'].nunique()} cells, {len(cycles)} cycle rows; "
          f"features for {len(feats)}, skipped {len(skipped)}")

    # --- coverage against the frozen Severson-only bank ------------------------
    art, err = load_artifacts()
    if art is None:
        raise SystemExit(f"artifacts unavailable: {err}")
    meta = art["meta"]
    bank = art["bank"].copy()
    cov = behavior_coverage(meta, bank, feats, "cell_id")
    feats = feats.assign(behavior_coverage=cov)

    # --- deployed gate: study-stratified percentile, whole-bank fallback -------
    # No Attia cell is in the bank, so the stratified rule has < 5 same-study
    # reference cells and falls back to the whole-bank percentile, scored with
    # same-policy edges excluded (the in-study scoring rule).
    bank_loo = behavior_coverage(meta, bank, bank.assign(cell_id=bank["cell_id"]),
                                 "cell_id", exclude_own_group=True)
    thr_q = float(np.percentile(bank_loo, Q_STAR))
    retained = cov >= thr_q
    retained_abs = cov >= ABSOLUTE_THRESHOLD

    merged = life.merge(feats, on="cell_id", how="outer")
    merged.to_csv(OUT / "attia_cells.csv", index=False)

    reached = merged["reached_eol_nominal"].fillna(False)
    summary = {
        "source": {"study": "Attia et al. 2020, Nature 578 (closed-loop optimization)",
                   "file": BATCH4.name, "bytes": BATCH4.stat().st_size,
                   "route": "src.ingestion.download attia (file id from BatteryML)"},
        "cells": {"in_file": info["n_cells_in_file"],
                  "loaded": int(cycles["cell_id"].nunique()),
                  "with_features": int(len(feats)),
                  "feature_skipped": skipped},
        "data_quality": {
            "length_mismatch_cells": LENGTH_MISMATCH_CELLS,
            "note": ("summary arrays longer than the cycle array; truncated to the "
                     "common length. Affects no cycle <= 100, so early-cycle "
                     "features are untouched."),
            "read_via_tolerant_reader": tolerant},
        "protocol": {"temperature_c": cm.SEVERSON_TEMPERATURE_C,
                     "dod_window": cm.SEVERSON_DOD_WINDOW,
                     "discharge": cm.SEVERSON_C_RATE_DISCHARGE,
                     "matches_bank_discharge": True,
                     "charge_policy_examples": info["policy_examples"]},
        "cycle_life": {
            "n_total": int(len(life)),
            "reached_eol_nominal": int(reached.sum()),
            "right_censored": int((~reached).sum()),
            "completeness_pct": round(100 * float(reached.mean()), 1),
            "min": float(life["cycle_life_nominal"].min()),
            "median": float(life["cycle_life_nominal"].median()),
            "max": float(life["cycle_life_nominal"].max()),
            "qc_anomalous": int(life["is_anomalous"].sum()),
            "qc_flags": life["flags"].value_counts().to_dict(),
        },
        "coverage": {
            "min": float(cov.min()), "q25": float(np.percentile(cov, 25)),
            "median": float(np.median(cov)), "q75": float(np.percentile(cov, 75)),
            "max": float(cov.max()), "mean": float(cov.mean()),
        },
        "bank_reference_coverage": {
            "min": float(bank_loo.min()), "median": float(np.median(bank_loo)),
            "max": float(bank_loo.max())},
        "gate": {
            "rule": "study-stratified percentile, fallback: whole-bank percentile "
                    "(no Attia cell in the bank -> < 5 same-study reference cells)",
            "q_star": Q_STAR, "threshold": thr_q,
            "retained": int(retained.sum()), "n": int(len(cov)),
            "retention_pct": round(100 * float(retained.mean()), 1),
            "absolute_gate_threshold": ABSOLUTE_THRESHOLD,
            "absolute_gate_retained": int(retained_abs.sum()),
        },
        "bank_untouched": {"cells": int(len(bank)), "attia_added": 0,
                           "models_retrained": 0, "predictions_made": 0},
    }
    if retained.any():
        summary["retained_cells"] = (
            feats.loc[retained, ["cell_id", "behavior_coverage"]]
            .sort_values("behavior_coverage", ascending=False)
            .round(4).to_dict("records"))

    (OUT / "attia_feasibility.json").write_text(json.dumps(summary, indent=1, default=str))

    print(f"\n[cycle life] {summary['cycle_life']['reached_eol_nominal']}/"
          f"{summary['cycle_life']['n_total']} reach EOL "
          f"({summary['cycle_life']['completeness_pct']}%); "
          f"range {summary['cycle_life']['min']:.0f}-{summary['cycle_life']['max']:.0f}, "
          f"median {summary['cycle_life']['median']:.0f}")
    c = summary["coverage"]
    print(f"[coverage]  min {c['min']:.2f}  median {c['median']:.2f}  max {c['max']:.2f}")
    print(f"[bank ref]  min {bank_loo.min():.2f}  median {np.median(bank_loo):.2f}  "
          f"max {bank_loo.max():.2f}")
    g = summary["gate"]
    print(f"[gate]      q={Q_STAR:g} -> threshold {g['threshold']:.4f}; "
          f"retained {g['retained']}/{g['n']} ({g['retention_pct']}%)")
    print(f"            absolute 4.08 would retain {g['absolute_gate_retained']}/{g['n']}")
    if retained.any():
        print("\n[retained cells]")
        for r in summary["retained_cells"]:
            print(f"   {r['cell_id']:8s} coverage {r['behavior_coverage']:.4f}")
    print(f"\n-> {OUT / 'attia_feasibility.json'}")
    return summary


if __name__ == "__main__":
    main()
