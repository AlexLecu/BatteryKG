"""lygte-info.dk (HKJ) independent battery tests — fetch, cache, parse.

Respectful fetching: each review page is fetched exactly ONCE with an
academic-research User-Agent and cached under data/raw/lygte/ (retrieval date
alongside); every parse works from the cache. No crawling beyond the three
specific review pages. URLs + dates are documented in data/README.md.

Conservative parsing: only numbers explicitly present in the page TEXT are
extracted, each with a `source_fragment` (the surrounding sentence). The
measured discharge-capacity data on these pages exists ONLY as chart images
(<cell>-Capacity.png etc.); per policy we do NOT attempt image extraction —
those measurement types are recorded with `chart_only: true` and a null value.

Output: data/processed/lygte_measurements.json

Run:  python -m src.ingestion.lygte
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from src.config import PROCESSED, RAW

RAW_LYGTE = RAW / "lygte"
OUT_JSON = PROCESSED / "lygte_measurements.json"
USER_AGENT = ("BatteryKG-research/1.0 (academic research, MDPI Batteries "
              "submission; contact: lecu.alex@yahoo.com)")

# The three review pages (located via web search; the A123 review exists).
PAGES = [
    {
        "cell_model": "A123 APR18650M1A",
        "source_id": "lygte_a123_apr18650m1a",
        "cache": "a123_18650_1100mah.html",
        "url": "https://lygte-info.dk/review/batteries2012/A123%2018650%201100mAh%20(Yellow)%20UK.html",
        "title": "Test of A123 18650 1100mAh (Yellow)",
    },
    {
        "cell_model": "Panasonic NCR18650B",
        "source_id": "lygte_panasonic_ncr18650b",
        "cache": "panasonic_ncr18650b.html",
        "url": "https://lygte-info.dk/review/batteries2012/Panasonic%20NCR18650B%203400mAh%20(Green)%20UK.html",
        "title": "Test of Panasonic NCR18650B 3400mAh (Green)",
    },
    {
        "cell_model": "LG Chem 18650HG2",
        "source_id": "lygte_lg_18650hg2",
        "cache": "lg_18650_hg2.html",
        "url": "https://lygte-info.dk/review/batteries2012/LG%2018650%20HG2%203000mAh%20(Brown)%20UK.html",
        "title": "Test of LG 18650 HG2 3000mAh (Brown)",
    },
]


def fetch_all(force: bool = False) -> str:
    """Fetch any uncached page once (polite pacing); returns retrieval date."""
    RAW_LYGTE.mkdir(parents=True, exist_ok=True)
    date_file = RAW_LYGTE / "RETRIEVED.txt"
    fetched = False
    for page in PAGES:
        dest = RAW_LYGTE / page["cache"]
        if dest.exists() and not force:
            continue
        r = requests.get(page["url"], headers={"User-Agent": USER_AGENT}, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        fetched = True
        print(f"[lygte] fetched {page['cache']} ({len(r.content):,} bytes)")
        time.sleep(2)
    if fetched or not date_file.exists():
        from datetime import datetime, timezone
        date_file.write_text(datetime.now(timezone.utc).strftime("%Y-%m-%d") + "\n")
    return date_file.read_text().strip()


# --- parsing ------------------------------------------------------------------
def _text_of(html: str) -> str:
    text = re.sub(r"<img[^>]*>", " ", html)
    text = re.sub(r"<[^>]+>", "\n", text)
    return re.sub(r"[ \t]+", " ", text)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]


# text-statement patterns -> (property, unit, value_group, condition builder)
_STATEMENT_PATTERNS = [
    # "At 20A the cell reaches 81°C"
    (re.compile(r"At (\d+(?:\.\d+)?)A the cell reaches (\d+)\s?°?C", re.I),
     lambda m: {"property": "measured_cell_temp_c", "value": float(m.group(2)),
                "unit": "degC",
                "conditions": {"discharge_current_a": float(m.group(1))}}),
    # "the cell do reach 90°C after termination" (30A context handled below)
    (re.compile(r"At (\d+(?:\.\d+)?)A I terminated due to temperature.*?reach (\d+)\s?°?C",
                re.I | re.S),
     lambda m: {"property": "measured_cell_temp_c", "value": float(m.group(2)),
                "unit": "degC",
                "conditions": {"discharge_current_a": float(m.group(1)),
                               "note": "discharge terminated early due to temperature"}}),
    # "able to deliver 30A, but only for two minutes"
    (re.compile(r"deliver (\d+(?:\.\d+)?)A, but only for (\w+) minutes?", re.I),
     lambda m: {"property": "measured_sustained_current_a", "value": float(m.group(1)),
                "unit": "A",
                "conditions": {"duration_min": {"two": 2.0, "one": 1.0,
                                                "three": 3.0}.get(m.group(2).lower())}}),
    # "in my test I only discharges to 2.8 volt"
    (re.compile(r"I only discharges? to (\d+(?:\.\d+)?) volt", re.I),
     lambda m: {"property": "test_discharge_cutoff_v", "value": float(m.group(1)),
                "unit": "V",
                "conditions": {"note": "review's test cutoff; full capacity not measured"}}),
]

# chart-image measurement types (data exists only as images -> chart_only)
_CHART_TYPES = [
    ("measured_capacity_ah", r"-Capacity\.png"),
    ("measured_energy_wh", r"-Energy\.png"),
]


def parse_review(html: str) -> dict:
    """Parse one cached review page -> {spec_echo, measurements}."""
    text = _text_of(html)

    # 1) 'Official specifications' echo (datasheet restatement, NOT measurements
    #    — kept because the echoes sometimes disagree with the datasheets).
    #    Parsed from the HTML <ul> (the site uses unclosed <li> tags).
    spec_echo = []
    m = re.search(r"Official specifications:\s*<ul>(.*?)</ul>", html, re.S | re.I)
    if m:
        for item in re.split(r"<li>", m.group(1))[1:]:
            line = re.sub(r"<[^>]+>", " ", item)
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                spec_echo.append(line)

    # 2) explicit numeric statements, each with its sentence as fragment
    measurements = []
    for sent in _sentences(text):
        for rx, build in _STATEMENT_PATTERNS:
            mm = rx.search(sent)
            if mm:
                rec = build(mm)
                rec["chart_only"] = False
                rec["source_fragment"] = sent[:240]
                measurements.append(rec)

    # 3) chart-only measurement types (no image extraction attempted)
    for prop, img_rx in _CHART_TYPES:
        imgs = re.findall(rf'<img src="[^"]*/([^"/]*{img_rx})"', html)
        if imgs:
            measurements.append({
                "property": prop, "value": None, "unit": None, "conditions": {},
                "chart_only": True,
                "source_fragment": f"data presented only as chart image: "
                                   f"{imgs[0].replace('%20', ' ')}",
            })
    return {"spec_echo": spec_echo, "measurements": measurements}


def build() -> dict:
    retrieved = fetch_all()
    out = {"retrieved": retrieved, "user_agent": USER_AGENT, "reviews": []}
    for page in PAGES:
        cache = RAW_LYGTE / page["cache"]
        if not cache.exists():
            print(f"[lygte] MISSING cache {page['cache']} — skipping")
            continue
        html = cache.read_text(encoding="iso-8859-1")
        parsed = parse_review(html)
        n_meas = sum(1 for m in parsed["measurements"] if not m["chart_only"])
        n_chart = sum(1 for m in parsed["measurements"] if m["chart_only"])
        out["reviews"].append({**{k: page[k] for k in
                                  ("cell_model", "source_id", "url", "title")},
                               "retrieved": retrieved,
                               "n_specimens": None,     # not stated on these pages
                               **parsed})
        print(f"[lygte] {page['cell_model']}: {n_meas} text measurements, "
              f"{n_chart} chart-only types, {len(parsed['spec_echo'])} spec-echo lines")
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"[lygte] -> {OUT_JSON}")
    return out


if __name__ == "__main__":
    build()
