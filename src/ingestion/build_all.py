"""Build every study's tidy Parquet from raw downloads.

Reproducible entry point: assumes raw files are already downloaded
(`python -m src.ingestion.download all`). Each loader is independent, so a
missing dataset is reported and skipped rather than aborting the whole build.

Run:  python -m src.ingestion.build_all
"""
from __future__ import annotations

from src.ingestion import sandia, severson

LOADERS = [severson, sandia]


def main() -> None:
    for mod in LOADERS:
        name = mod.STUDY
        try:
            mod.build()
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")
        except Exception as e:  # keep going; report loudly
            print(f"[error] {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
