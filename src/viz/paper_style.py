"""Publication figure style for the paper (single source of truth).

Serif-compatible fonts (matches MDPI's Palatino body), colorblind-safe palette
(validated with the six-checks palette validator: all pairs pass CVD
separation), 300 dpi, consistent physical sizes. Every figure is written as
both PDF (vector, for LaTeX) and PNG (for quick viewing) into paper/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper" / "figures"

# CVD-validated categorical palette (scripts/validate_palette.js: ALL PASS)
BLUE, ORANGE, GREEN, GOLD, PINK = "#2563eb", "#ea580c", "#059669", "#b45309", "#db2777"
INK, INK_MUT = "#1f2937", "#6b7280"

# physical widths [inches] — MDPI single-column layout
W_SINGLE = 3.4          # half-width figure (side-by-side pairs)
W_FULL = 6.6            # full text-width figure


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Palatino", "TeX Gyre Pagella", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK_MUT,
        "ytick.color": INK_MUT,
        "legend.frameon": False,
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    """Write paper/figures/<name>.pdf and .png (300 dpi), plus a 600 dpi
    PNG into paper/figures/png/ (raster copies for Overleaf upload)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "png").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    fig.savefig(FIG_DIR / "png" / f"{name}.png", dpi=600,
                bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print(f"[paper_style] saved figures/{name}.pdf+.png (+png/ 600dpi)")
