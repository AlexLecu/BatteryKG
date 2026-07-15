"""Regenerate every paper figure into paper/figures/ (PDF + PNG, 300 dpi;
600 dpi PNG copies land in paper/figures/png/ for Overleaf).

Reproducible from processed data + cached extraction artifacts; the model
figures re-run the (deterministic, seeded) CV experiments against the pinned
Severson-only similarity edges (see publication_edges — the live KG's
behavior view gains cross-study edges whenever a new study is ingested,
which would silently shift the in-study numbers).

MANUSCRIPT_FIGURES below is the explicit manifest of what the paper includes;
SUPPLEMENTARY_FIGURES are still generated but no longer part of the paper.

Run:  python -m src.viz.make_paper_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import processed_parquet
from src.ingestion.qc import EOL_FRACTION, cycle_life_table
from src.viz.paper_style import (
    BLUE, GOLD, GREEN, INK_MUT, ORANGE, PINK, W_FULL, W_SINGLE,
    apply_style, save_fig,
)

apply_style()

# --- manuscript figure manifest ---------------------------------------------------
# The figures the paper includes: paper/figures/<name>.pdf + png/<name>.png.
# kg_schema / system_architecture are draw.io exports (sources in
# paper/figures/src/); risk_coverage_cross_study comes from src.models.experiment_04.
MANUSCRIPT_FIGURES = [
    "kg_schema",
    "system_architecture",
    "fade_severson",
    "risk_coverage_main",
    "risk_coverage_sensitivity",
    "extraction_stages",
    "risk_coverage_cross_study",
]
# Generated but retired from the manuscript; kept as repo/report artifacts.
SUPPLEMENTARY_FIGURES = [
    "cycle_life_hist_by_batch",
    "pred_vs_actual",
    "risk_coverage_k_sensitivity",     # panel (a) of risk_coverage_sensitivity
    "risk_coverage_view_sensitivity",  # panel (b) of risk_coverage_sensitivity
]


# --- Fig 1: Severson capacity fade ------------------------------------------------
# Excluded from the PLOT only: QC-flagged cells whose vertical measurement-artifact
# spikes clutter the figure. They remain in every analysis and dataset.
FADE_PLOT_EXCLUDE = ("b1c0", "b1c18", "b2c12", "b2c44")


def fig_fade():
    df = pd.read_parquet(processed_parquet("severson_mit"))
    df = df[~df["cell_id"].isin(FADE_PLOT_EXCLUDE)]
    life = cycle_life_table(df).set_index("cell_id")
    fig, ax = plt.subplots(figsize=(W_FULL, 3.1))
    finite = life["cycle_life_nominal"].dropna()
    norm = mpl.colors.Normalize(vmin=finite.min(), vmax=finite.max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    for cell_id, g in df.groupby("cell_id"):
        cl = life.loc[cell_id, "cycle_life_nominal"]
        ax.plot(g["cycle_index"], g["discharge_capacity_ah"], lw=0.5, alpha=0.6,
                color=sm.to_rgba(cl) if pd.notna(cl) else "lightgrey")
    thr = EOL_FRACTION * 1.1
    ax.axhline(thr, ls="--", lw=0.8, color="crimson")
    ax.annotate("EOL = 80% of nominal (0.88 Ah)", xy=(1400, thr), xytext=(1400, 0.845),
                fontsize=7, color="crimson")
    lo = df["discharge_capacity_ah"].quantile(0.005)
    hi = df["discharge_capacity_ah"].quantile(0.995)
    ax.set_ylim(lo * 0.97, hi * 1.02)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("cycle life to 80% of nominal")
    ax.set_xlabel("cycle")
    ax.set_ylabel("discharge capacity [Ah]")
    save_fig(fig, "fade_severson")


# --- Fig 2: cycle life by batch --------------------------------------------------------
def fig_hist_by_batch():
    df = pd.read_parquet(processed_parquet("severson_mit"))
    life = cycle_life_table(df)
    life["batch"] = life["cell_id"].str.extract(r"^b(\d)").astype(int)
    d = life[life["reached_eol_nominal"]]
    fig, ax = plt.subplots(figsize=(W_FULL, 2.7))
    colors = {1: BLUE, 2: ORANGE, 3: GREEN}
    bins = np.linspace(d["cycle_life_nominal"].min(), d["cycle_life_nominal"].max(), 25)
    for b in (1, 2, 3):
        g = d[d["batch"] == b]
        ax.hist(g["cycle_life_nominal"], bins=bins, alpha=0.6, color=colors[b],
                label=f"batch {b} (n={len(g)}, median {g['cycle_life_nominal'].median():.0f})")
    ax.set_xlabel("cycle life to 80% of nominal capacity")
    ax.set_ylabel("cells")
    ax.legend()
    save_fig(fig, "cycle_life_hist_by_batch")


# --- model-dependent figures (re-run the deterministic experiments) ----------------------
EDGES_SNAPSHOT = (Path(__file__).resolve().parents[2]
                  / "data" / "kg_snapshots" / "severson_edges_publication.json")


def publication_edges():
    """Severson-only SIMILAR_TO edges from the committed publication snapshot.

    The live KG's behavior view is rebuilt over the union of studies whenever
    a new study is ingested (src/kg/hust_load.py) — after the HUST load, 31
    Severson cells carry HUST cells in their stored top-10 behavior neighbors,
    displacing Severson neighbors and shifting the in-study CV numbers. The
    manuscript experiments are defined on the Severson-only graph, so the
    paper figures read the pinned edge set (verified to reproduce the
    published headline exactly: graph model RMSE 135.15, MAPE 9.69) instead
    of querying Neo4j. Provenance is documented in the snapshot's _meta.
    """
    snap = json.loads(EDGES_SNAPSHOT.read_text())
    # view order MUST stay ("condition", "behavior"): downstream code derives
    # the model's feature-column order from this dict's key order, and XGBoost
    # results depend on column order (the snapshot file itself is key-sorted).
    from src.models.graph_model import VIEWS
    return {view: {src: [(d, float(w)) for d, w in nbrs]
                   for src, nbrs in snap["views"][view].items()}
            for view in VIEWS}


def _model_results():
    from src.models.baseline import run_baseline
    from src.models.dataset import build_dataset
    from src.models.graph_model import run_graph_model
    data = build_dataset()
    df = data[~data["is_anomalous"]].reset_index(drop=True)
    edges = publication_edges()
    return df, edges, run_baseline(df), run_graph_model(df, edges)


def fig_pred_vs_actual(base, graph):
    from src.models.common import mape_cycles, rmse_cycles
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.9), sharex=True, sharey=True)
    for ax, pred, color, title in [(axes[0], base, BLUE, "Graph-free GBM (baseline)"),
                                   (axes[1], graph, ORANGE, "Graph-augmented GBM")]:
        t, p = 10 ** pred["y_true_log"], 10 ** pred["y_pred_log"]
        lims = [120, 2600]
        ax.plot(lims, lims, ls="--", lw=0.8, color=INK_MUT)
        ax.scatter(t, p, s=9, alpha=0.75, color=color, edgecolors="white", linewidths=0.3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ticks = [200, 500, 1000, 2000]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(mpl.ticker.FixedLocator(ticks))
            axis.set_major_formatter(mpl.ticker.FixedFormatter([str(x) for x in ticks]))
            axis.set_minor_formatter(mpl.ticker.NullFormatter())
        r = rmse_cycles(pred["y_true_log"], pred["y_pred_log"])
        m = mape_cycles(pred["y_true_log"], pred["y_pred_log"])
        ax.set_title(f"{title}\nRMSE {r:.0f} cycles, MAPE {m:.1f}%")
        ax.set_xlabel("actual cycle life")
    axes[0].set_ylabel("predicted cycle life")
    save_fig(fig, "pred_vs_actual")


def fig_risk_coverage_main(graph):
    from src.models.abstention import risk_coverage_sweep
    sweep = risk_coverage_sweep(graph["behavior_coverage_train"].to_numpy(),
                                graph["y_true_log"].to_numpy(),
                                graph["y_pred_log"].to_numpy())
    d = sweep.sort_values("frac_retained")
    fig, ax = plt.subplots(figsize=(W_FULL, 2.8))
    ax.plot(d["frac_retained"], d["rmse_random"], lw=1.4, ls="--", color=INK_MUT,
            label="random abstention (mean of 500 draws)")
    ax.plot(d["frac_retained"], d["rmse_retained"], lw=1.6, color=ORANGE,
            label="coverage-gated (behavior view, $k{=}5$)")
    ax.set_xlabel("fraction of cells retained")
    ax.set_ylabel("RMSE on retained cells [cycles]")
    ax.set_xlim(0, 1.02)
    ax.legend(loc="lower right")
    save_fig(fig, "risk_coverage_main")


def _plot_k_panel(ax, rand, sweep, cov):
    ax.plot(rand["frac_retained"], rand["rmse_random"], lw=1.3, ls="--",
            color=INK_MUT, label="random abstention")
    for k, color in [(3, BLUE), (5, ORANGE), (10, GREEN)]:
        d = sweep(cov[f"behavior_k{k}"].to_numpy())
        ax.plot(d["frac_retained"], d["rmse_retained"], lw=1.3, color=color,
                label=f"behavior $k{{=}}{k}$")
    ax.set_xlabel("fraction of cells retained")
    ax.set_xlim(0, 1.02)


def _plot_view_panel(ax, rand, sweep, cov, qs):
    ax.plot(rand["frac_retained"], rand["rmse_random"], lw=1.3, ls="--",
            color=INK_MUT, label="random abstention")
    for label, sig, color in [("behavior $k{=}5$", cov["behavior_k5"].to_numpy(), ORANGE),
                              ("condition $k{=}5$", cov["condition_k5"].to_numpy(), BLUE),
                              ("mean(views) $k{=}5$", cov["mean_k5"].to_numpy(), GREEN),
                              ("GBM quantile spread", -qs, GOLD)]:
        d = sweep(sig)
        ax.plot(d["frac_retained"], d["rmse_retained"], lw=1.3, color=color, label=label)
    ax.set_xlabel("fraction of cells retained")
    ax.set_xlim(0, 1.02)


def fig_sensitivity(df, edges, graph):
    """Single-panel figures (kept for the experiment reports) plus the merged
    two-panel manuscript figure risk_coverage_sensitivity (a: k, b: view)."""
    from src.models.abstention import risk_coverage_sweep
    from src.models.sensitivity import coverage_signals, quantile_spread_signal
    yt, yp = graph["y_true_log"].to_numpy(), graph["y_pred_log"].to_numpy()
    cov = coverage_signals(df, edges).set_index("cell_id").loc[graph["cell_id"]]
    qs = (quantile_spread_signal(df, edges).set_index("cell_id")
          .loc[graph["cell_id"], "quantile_spread"].to_numpy())

    def sweep(sig):
        return risk_coverage_sweep(sig, yt, yp).sort_values("frac_retained")

    rand = sweep(cov["behavior_k5"].to_numpy())[["frac_retained", "rmse_random"]]

    # k sensitivity (single panel, report artifact)
    fig, ax = plt.subplots(figsize=(W_FULL, 2.8))
    _plot_k_panel(ax, rand, sweep, cov)
    ax.set_ylabel("RMSE on retained cells [cycles]")
    ax.legend(loc="upper left")
    save_fig(fig, "risk_coverage_k_sensitivity")

    # view sensitivity + uncertainty gate (single panel, report artifact)
    fig, ax = plt.subplots(figsize=(W_FULL, 2.8))
    _plot_view_panel(ax, rand, sweep, cov, qs)
    ax.set_ylabel("RMSE on retained cells [cycles]")
    ax.legend(loc="upper left", ncols=2)
    save_fig(fig, "risk_coverage_view_sensitivity")

    # merged two-panel manuscript figure (shared y-axis)
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.8), sharey=True)
    _plot_k_panel(axes[0], rand, sweep, cov)
    _plot_view_panel(axes[1], rand, sweep, cov, qs)
    axes[0].set_ylabel("RMSE on retained cells [cycles]")
    axes[0].set_title("(a)", loc="left", fontsize=9, fontweight="bold")
    axes[1].set_title("(b)", loc="left", fontsize=9, fontweight="bold")
    # half-width panels: upper-left would sit on the random-abstention curve;
    # lower right is empty in both panels (same choice as risk_coverage_main)
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower right")
    save_fig(fig, "risk_coverage_sensitivity")


# --- Fig: extraction pipeline stages (new bar chart) -----------------------------------------
def fig_extraction_stages():
    """F1 and hallucinations across raw -> consensus -> validated, recomputed
    from the cached extraction artifacts (no API calls)."""
    from src.agents.evaluation import combine, evaluate_document, load_gold
    from src.agents.extractor import RAW_DIR
    from src.agents.pdf_text import TEXT_DIR
    from src.agents.validator import consensus, validate_claims

    gold = load_gold()
    docs = ["a123_apr18650m1a", "lg_inr18650hg2", "panasonic_ncr18650b_full_spec_sanyo"]
    texts = {d: (TEXT_DIR / f"{d}.txt").read_text() for d in docs}
    runs = {d: [json.loads((RAW_DIR / f"{d}_run{r}_parsed.json").read_text())
                for r in (1, 2, 3)] for d in docs}

    def total(preds_by_doc):
        return combine([evaluate_document(d, gold, preds_by_doc[d], texts[d])
                        for d in docs])

    stages = {
        "raw\n(single run)": total({d: runs[d][0] for d in docs}),
        "consensus\n($\\geq$2/3)": total({d: consensus(runs[d]) for d in docs}),
        "consensus\n+ Validator": total({d: consensus(
            [validate_claims(c, texts[d], doc=d).accepted for c in runs[d]])
            for d in docs}),
    }
    names = list(stages)
    f1 = [s.f1 for s in stages.values()]
    hall = [s.hallucinations for s in stages.values()]

    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.5))
    x = np.arange(3)
    b0 = axes[0].bar(x, f1, width=0.55, color=[INK_MUT, BLUE, ORANGE])
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("F1 vs gold standard")
    axes[0].set_ylim(0, 0.9)
    axes[0].bar_label(b0, fmt="%.3f", fontsize=7.5)
    b1 = axes[1].bar(x, hall, width=0.55, color=[INK_MUT, BLUE, ORANGE])
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("hallucinated claims")
    axes[1].set_ylim(0, 3.6)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].bar_label(b1, fontsize=7.5)
    save_fig(fig, "extraction_stages")


def main() -> None:
    fig_fade()
    fig_hist_by_batch()
    df, edges, base, graph = _model_results()
    fig_pred_vs_actual(base, graph)
    fig_risk_coverage_main(graph)
    fig_sensitivity(df, edges, graph)
    fig_extraction_stages()
    print("[make_paper_figures] all figures regenerated")
    print(f"[make_paper_figures] manuscript set: {', '.join(MANUSCRIPT_FIGURES)}")
    print(f"[make_paper_figures] supplementary:  {', '.join(SUPPLEMENTARY_FIGURES)}")


if __name__ == "__main__":
    main()
