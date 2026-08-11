"""Reproducible dataset downloads.

Every raw artifact must be obtainable from this script (no manual steps).
Downloads are idempotent: a file already present with the expected size is
skipped; a partial file is resumed via an HTTP Range request.

Usage:
    python -m src.ingestion.download severson   # MIT/Severson batches 1-3 (~8.3 GB)
    python -m src.ingestion.download sandia     # Sandia 24-cell cycle data (~298 MB)
    python -m src.ingestion.download all
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import RAW_SANDIA, RAW_SEVERSON

_MATR = "https://data.matr.io/1/api/v1/file/{file_id}/download"


@dataclass(frozen=True)
class RemoteFile:
    url: str
    dest: Path
    expected_bytes: int          # for integrity check / skip decision
    note: str = ""


# MIT/Stanford/Toyota — Severson 2019 (Nature Energy): batches 1-3 = 124 LFP cells.
# Batch 4 (2019-01-24, Attia 2020 CLO) intentionally excluded here; it belongs to
# a later project phase. File IDs verified against microsoft/BatteryML.
SEVERSON_FILES = [
    RemoteFile(_MATR.format(file_id="5c86c0b5fa2ede00015ddf66"),
               RAW_SEVERSON / "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
               3_025_320_241, "batch1 (2017-05-12)"),
    RemoteFile(_MATR.format(file_id="5c86bf13fa2ede00015ddd82"),
               RAW_SEVERSON / "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
               2_007_331_155, "batch2 (2017-06-30)"),
    RemoteFile(_MATR.format(file_id="5c86bd64fa2ede00015ddbb2"),
               RAW_SEVERSON / "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
               3_236_690_412, "batch3 (2018-04-12)"),
]

# Optional 4th batch (Attia 2020 closed-loop optimization) — not part of the
# Severson 2019 study; kept here for later phases.
SEVERSON_BATCH4 = RemoteFile(
    _MATR.format(file_id="5dcef152110002c7215b2c90"),
    RAW_SEVERSON / "2019-01-24_batchdata_updated_struct_errorcorrect.mat",
    2_601_295_745, "batch4 (2019-01-24, Attia CLO — optional)")

SANDIA_FILES = [
    RemoteFile("https://www.sandia.gov/app/uploads/sites/163/Sandia_Cell_Cycle_Testing_Data.zip",
               RAW_SANDIA / "Sandia_Cell_Cycle_Testing_Data.zip",
               312_035_656, "24-cell cycle testing (LCO/LFP/NCA/NMC)"),
]

CHUNK = 1 << 20  # 1 MiB
MAX_RETRIES = 8


def _stream_once(rf: RemoteFile) -> None:
    """One resume-aware streaming attempt (raises on network error)."""
    have = rf.dest.stat().st_size if rf.dest.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    print(f"[{'resume' if have else 'get'}] {rf.dest.name} from byte {have:,}  ({rf.note})")

    with requests.get(rf.url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(rf.dest, mode) as fh, tqdm(
            total=rf.expected_bytes, initial=have, unit="B", unit_scale=True,
            desc=rf.dest.name[:28],
        ) as bar:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))


def download_file(rf: RemoteFile) -> None:
    rf.dest.parent.mkdir(parents=True, exist_ok=True)

    if rf.dest.exists() and rf.dest.stat().st_size == rf.expected_bytes:
        print(f"[skip] {rf.dest.name} already complete ({rf.expected_bytes:,} bytes)")
        return
    if rf.dest.exists() and rf.dest.stat().st_size > rf.expected_bytes:
        print(f"[warn] {rf.dest.name} larger than expected; re-downloading")
        rf.dest.unlink()

    # Resume-on-failure: flaky large downloads (matr.io read-timeouts) recover
    # by re-requesting from the current file size.
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _stream_once(rf)
            break
        except (requests.exceptions.RequestException, OSError) as e:
            have = rf.dest.stat().st_size if rf.dest.exists() else 0
            if have >= rf.expected_bytes:
                break
            wait = min(30, 2 ** attempt)
            print(f"[retry {attempt}/{MAX_RETRIES}] {rf.dest.name} at {have:,} bytes "
                  f"after {type(e).__name__}; sleeping {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"{rf.dest.name}: exhausted retries")

    final = rf.dest.stat().st_size
    status = "ok" if final == rf.expected_bytes else "warn"
    print(f"[{status}] {rf.dest.name} ({final:,} bytes, expected {rf.expected_bytes:,})")


def download(which: str) -> None:
    groups = {
        "severson": SEVERSON_FILES,
        "sandia": SANDIA_FILES,
        # batch 4 is the Attia 2020 closed-loop-optimization study: same cell,
        # lab and equipment as Severson, new multi-step charge protocols. It is
        # NOT part of the Severson 124-cell modelling set and is downloaded on
        # request only, so `severson` and `all` stay reproducible as published.
        "attia": [SEVERSON_BATCH4],
        "all": SEVERSON_FILES + SANDIA_FILES,
    }
    if which not in groups:
        raise SystemExit(f"unknown target '{which}'; choose from {list(groups)}")
    for rf in groups[which]:
        download_file(rf)


if __name__ == "__main__":
    download(sys.argv[1] if len(sys.argv) > 1 else "all")
