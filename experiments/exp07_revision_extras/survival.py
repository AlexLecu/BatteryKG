"""Task 3 (appendix addition) — Kaplan-Meier survival under the initial-capacity EOL.

Under the alternative EOL definition (80% of each cell's own measured initial
capacity rather than 80% of nameplate) only 46 of the 124 Severson cells reach
threshold within their recorded life; the other 78 are right-censored at their
last observed cycle. Dropping them, as a plain regression on complete cases
must, discards the longest-lived cells and biases any lifetime estimate
downwards. Kaplan-Meier uses them.

Implemented directly (product-limit estimator, Greenwood variance, k-sample
log-rank) rather than pulling in a survival package: both are short, exact, and
deterministic, and the repo does not otherwise depend on one. The two-group
log-rank is cross-checked against the scalar formula.

Run:  python -m experiments.exp07_revision_extras.survival
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src.config import OUTPUTS, ROOT, processed_parquet
from src.ingestion.qc import cycle_life_table

OUT = OUTPUTS / "experiment_07_revision_extras"


# --- estimator ----------------------------------------------------------------
def kaplan_meier(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Product-limit estimator with Greenwood 95% CI (log-log transformed).

    time  observed time (event time, or censoring time)
    event 1 = EOL reached, 0 = right-censored
    """
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    order = np.argsort(time, kind="stable")
    time, event = time[order], event[order]
    n = len(time)

    rows, S, cum_var = [], 1.0, 0.0
    rows.append({"t": 0.0, "n_risk": n, "n_event": 0, "survival": 1.0,
                 "ci_lo": 1.0, "ci_hi": 1.0})
    for t in np.unique(time[event == 1]):
        at_risk = int((time >= t).sum())
        d = int(((time == t) & (event == 1)).sum())
        S *= (1.0 - d / at_risk)
        if at_risk > d:
            cum_var += d / (at_risk * (at_risk - d))     # Greenwood, on log S
        lo = hi = S
        if 0.0 < S < 1.0 and cum_var > 0:
            se_loglog = np.sqrt(cum_var) / abs(np.log(S))
            z = stats.norm.ppf(0.975)
            lo = S ** np.exp(z * se_loglog)
            hi = S ** np.exp(-z * se_loglog)
        rows.append({"t": float(t), "n_risk": at_risk, "n_event": d,
                     "survival": float(S), "ci_lo": float(min(lo, hi)),
                     "ci_hi": float(max(lo, hi))})
    return pd.DataFrame(rows)


def median_survival(km: pd.DataFrame) -> float:
    below = km[km["survival"] <= 0.5]
    return float(below["t"].iloc[0]) if len(below) else float("nan")


def logrank(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> dict:
    """k-sample log-rank test. Returns chi2, df, p, and per-group O/E."""
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    group = np.asarray(group)
    groups = np.unique(group)
    k = len(groups)
    O = np.zeros(k)
    E = np.zeros(k)
    V = np.zeros((k, k))

    for t in np.unique(time[event == 1]):
        at_risk = time >= t
        n_j = int(at_risk.sum())
        d_j = int(((time == t) & (event == 1)).sum())
        if n_j <= 1:
            continue
        n_ij = np.array([int((at_risk & (group == g)).sum()) for g in groups], float)
        d_ij = np.array([int(((time == t) & (event == 1) & (group == g)).sum())
                         for g in groups], float)
        O += d_ij
        E += n_ij * d_j / n_j
        factor = d_j * (n_j - d_j) / (n_j - 1) / (n_j ** 2)
        V += factor * (np.diag(n_ij * n_j) - np.outer(n_ij, n_ij))

    z = (O - E)[:-1]                       # drop one group: the vector sums to zero
    Vr = V[:-1, :-1]
    chi2 = float(z @ np.linalg.pinv(Vr) @ z)
    df = k - 1
    return {"chi2": chi2, "df": df, "p_value": float(stats.chi2.sf(chi2, df)),
            "observed": dict(zip(map(str, groups), O.round(2))),
            "expected": dict(zip(map(str, groups), E.round(2)))}


# --- data ---------------------------------------------------------------------
def survival_table() -> pd.DataFrame:
    """Per-cell (time, event, batch) under the initial-capacity EOL definition."""
    cycles = pd.read_parquet(processed_parquet("severson_mit"))
    life = cycle_life_table(cycles)
    last = (cycles.groupby("cell_id")["cycle_index"].max()
            .rename("last_cycle").reset_index())
    batch = cycles.groupby("cell_id")["batch"].first().rename("batch").reset_index()
    d = life[["cell_id", "cycle_life_initial", "reached_eol_initial",
              "initial_capacity_ah", "final_capacity_ah",
              "nominal_capacity_ah"]].merge(last, on="cell_id").merge(batch, on="cell_id")
    d["event"] = d["reached_eol_initial"].astype(int)
    d["time"] = np.where(d["event"] == 1, d["cycle_life_initial"], d["last_cycle"])
    d["thr_initial"] = 0.8 * d["initial_capacity_ah"]
    d["thr_nominal"] = 0.8 * d["nominal_capacity_ah"]
    return d.sort_values("cell_id").reset_index(drop=True)


def censoring_diagnostic(d: pd.DataFrame) -> dict:
    """Is the censoring uninformative, as Kaplan-Meier assumes?

    It is not. The initial-capacity threshold (0.8 x measured initial) sits
    BELOW the nominal one (0.8 x 1.1 Ah) because these cells start above
    nameplate, so a cell has to be cycled past the study's own stopping point to
    register an event. Whether that happened is a property of the test schedule,
    and the test schedule differs by batch — so censoring is batch-dependent and
    the stratified comparison below measures stopping policy, not degradation.
    """
    g = d.groupby("batch")
    return {
        "note": ("censoring is administrative (test stopped) and batch-dependent; "
                 "the KM independent-censoring assumption is violated across batches"),
        "by_batch": {int(b): {
            "n": int(len(s)), "events": int(s["event"].sum()),
            "median_final_capacity_ah": round(float(s["final_capacity_ah"].median()), 4),
            "median_threshold_initial_ah": round(float(s["thr_initial"].median()), 4),
            "median_threshold_nominal_ah": round(float(s["thr_nominal"].median()), 4),
            "median_observed_cycles": round(float(s["time"].median()), 1),
        } for b, s in g},
    }


def figure(d: pd.DataFrame, km_all: pd.DataFrame, km_by: dict, lr: dict) -> None:
    """Paper figure -> paper/figures/km_survival.{pdf,png} (+ a copy in outputs/).

    Font sizes are left at the paper_style defaults throughout; nothing is
    shrunk below them, so the figure stays legible at \\textwidth.

    The log-rank statistic is deliberately NOT shown. Censoring here is
    administrative and batch-dependent (only batch 2 was cycled past the study's
    stopping point), so the test measures the test schedule rather than
    degradation; putting p on the panel invites exactly the reading the text
    argues against. The number stays in survival_summary.json.
    """
    import matplotlib.pyplot as plt
    from src.viz.paper_style import (BLUE, GREEN, INK_MUT, ORANGE, W_FULL,
                                     apply_style, save_fig)
    apply_style()
    # sized so the tight-trimmed output matches the ~5.8 in width of the other
    # paper figures: placed at \textwidth they then scale identically, and the
    # 7.5/8.5/9 pt paper_style sizes render at the same size on the page
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 3.05))

    # --- left: whole population, Greenwood band, censoring ticks --------------
    ax1.step(km_all["t"], km_all["survival"], where="post", lw=1.8, color=BLUE)
    ax1.fill_between(km_all["t"], km_all["ci_lo"], km_all["ci_hi"], step="post",
                     color=BLUE, alpha=0.16, lw=0, label="95% CI (Greenwood)")
    cens = d[d["event"] == 0]
    ax1.plot(cens["time"], np.interp(cens["time"], km_all["t"], km_all["survival"]),
             "|", ms=5, color=INK_MUT, mew=0.9, ls="none", label="right-censored")
    ax1.axhline(0.5, ls=":", lw=1.0, color=INK_MUT)
    ax1.set_xlabel("cycles")
    ax1.set_ylabel("survival probability")
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_title(f"All {len(d)} cells ({int(d['event'].sum())} events, "
                  f"{int((1 - d['event']).sum())} censored)", pad=5)
    ax1.legend(loc="lower left", handlelength=1.4)

    # --- right: by batch -------------------------------------------------------
    for (b, km), color in zip(sorted(km_by.items()), (BLUE, ORANGE, GREEN)):
        sub = d[d["batch"] == b]
        ax2.step(km["t"], km["survival"], where="post", lw=1.7, color=color,
                 label=f"batch {b} ($n={len(sub)}$, {int(sub['event'].sum())} events)")
        c = sub[sub["event"] == 0]
        ax2.plot(c["time"], np.interp(c["time"], km["t"], km["survival"]), "|",
                 ms=4.5, color=color, mew=0.9, ls="none")
    ax2.axhline(0.5, ls=":", lw=1.0, color=INK_MUT)
    ax2.set_xlabel("cycles")
    ax2.set_ylabel("survival probability")
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_title("By batch", pad=5)
    ax2.legend(loc="lower right", handlelength=1.4)
    # kept short and inside the axes: the earlier four-line version overran the
    # right edge and lost its last line
    ax2.annotate("censoring is batch-dependent:\nonly batch 2 was cycled past\n"
                 "the study's stopping point",
                 xy=(0.70, 0.74), xycoords="axes fraction", ha="center", va="top",
                 color=INK_MUT)

    fig.suptitle("Kaplan\u2013Meier survival, initial-capacity EOL definition "
                 "(ticks mark right-censored cells)")
    fig.tight_layout()
    save_fig(fig, "km_survival")
    for ext in ("pdf", "png"):
        (OUT / f"km_survival.{ext}").write_bytes(
            (ROOT / "paper" / "figures" / f"km_survival.{ext}").read_bytes())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = survival_table()
    km_all = kaplan_meier(d["time"], d["event"])
    km_by = {int(b): kaplan_meier(g["time"], g["event"]) for b, g in d.groupby("batch")}
    lr = logrank(d["time"], d["event"], d["batch"])

    # cross-check the k-sample machinery against the scalar 2-group formula
    two = d[d["batch"].isin([1, 2])]
    lr2 = logrank(two["time"], two["event"], two["batch"])
    O = np.array(list(lr2["observed"].values())); E = np.array(list(lr2["expected"].values()))
    assert abs(O.sum() - E.sum()) < 1e-6, "log-rank O/E bookkeeping broken"

    summary = {
        "n_cells": int(len(d)), "n_events": int(d["event"].sum()),
        "n_censored": int((1 - d["event"]).sum()),
        "median_survival_cycles": median_survival(km_all),
        "naive_median_complete_cases_only": float(
            d.loc[d["event"] == 1, "time"].median()),
        "logrank_all_batches": lr,
        "logrank_batch1_vs_2": {k: lr2[k] for k in ("chi2", "df", "p_value")},
        "by_batch": {int(b): {"n": int(len(g)), "events": int(g["event"].sum()),
                              "median_survival": median_survival(km_by[int(b)])}
                     for b, g in d.groupby("batch")},
        "censoring_diagnostic": censoring_diagnostic(d),
    }
    d.to_csv(OUT / "survival_data.csv", index=False)
    km_all.to_csv(OUT / "survival_km_all.csv", index=False)
    for b, km in km_by.items():
        km.to_csv(OUT / f"survival_km_batch{b}.csv", index=False)
    (OUT / "survival_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    figure(d, km_all, km_by, lr)

    print(json.dumps(summary, indent=1, default=str))
    return d, km_all, km_by, lr, summary


if __name__ == "__main__":
    main()
