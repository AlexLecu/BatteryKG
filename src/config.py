"""Project-wide paths and constants.

Single source of truth for where things live so every script/notebook resolves
the same directories regardless of the working directory it is launched from.
"""
from __future__ import annotations

from pathlib import Path

# Repo root = two levels up from this file (src/config.py -> src -> root)
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"                 # gitignored; download target
PROCESSED = DATA / "processed"     # tidy Parquet artifacts
OUTPUTS = ROOT / "outputs"         # figures, tables

# Per-study raw subdirectories
RAW_SEVERSON = RAW / "severson_mit"
RAW_SANDIA = RAW / "sandia"

for _d in (RAW, PROCESSED, OUTPUTS, RAW_SEVERSON, RAW_SANDIA):
    _d.mkdir(parents=True, exist_ok=True)


def processed_parquet(study: str) -> Path:
    """Canonical output path for a study's tidy per-cycle Parquet."""
    return PROCESSED / f"{study}_cycles.parquet"
