"""Pipeline status — read-only dashboard over the KG and report files.

Placeholder for later Literature Monitor activity.
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.common import OUTPUTS, ROOT, make_driver, run_query  # noqa: E402

st.set_page_config(page_title="Status — BatteryKG", page_icon="📊", layout="wide")
st.title("Pipeline status")

# --- KG counts -----------------------------------------------------------------
st.subheader("Knowledge graph")
driver, err = make_driver()
if driver is None:
    st.error(f"Knowledge graph unavailable — {err}")
else:
    labels = ["Cell", "CellInstance", "Chemistry", "Source", "Claim",
              "Measurement", "Discrepancy"]
    counts = {}
    for lb in labels:
        counts[lb] = run_query(driver, f"MATCH (n:`{lb}`) RETURN count(n) AS c")[0]["c"]
    cols = st.columns(len(labels))
    for col, (lb, n) in zip(cols, counts.items()):
        col.metric(lb, n)

    edge_counts = run_query(driver, """
        MATCH ()-[r:SIMILAR_TO]->() RETURN r.view AS view, count(r) AS c""")
    st.caption("SIMILAR_TO edges: " + ", ".join(
        f"{r['view']}: {r['c']}" for r in edge_counts))

    st.markdown("**Claims per cell**")
    per_cell = run_query(driver, """
        MATCH (cl:Claim)-[:ABOUT]->(c:Cell)
        RETURN c.model AS cell, count(cl) AS claims ORDER BY cell""")
    if per_cell:
        st.dataframe(pd.DataFrame(per_cell), use_container_width=False)
    else:
        st.info("No claims loaded.")
    driver.close()

# --- extraction pipeline metrics (from the experiment report) ---------------------
st.subheader("Claim-extraction pipeline (experiment 02/02b)")
report = OUTPUTS / "experiment_02_extraction.md"
if not report.exists():
    st.warning("Report outputs/experiment_02_extraction.md not found — run "
               "`python -m src.agents.experiment_02` and `experiment_02b`. "
               "No metrics are shown because none are available.")
else:
    text = report.read_text()
    # parse the 02b pipeline table rows: | name | n | TP | FP | FN | P | R | F1 | cond | hall |
    rows = re.findall(
        r"\|\s*([A-Za-z][^|]*?)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*"
        r"\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|",
        text.split("## 02b")[1].split("###")[0] if "## 02b" in text else "")
    if rows:
        df = pd.DataFrame(rows, columns=["pipeline stage", "pred", "TP", "FP", "FN",
                                         "precision", "recall", "F1",
                                         "cond. acc.", "hallucinations"])
        c1, c2, c3 = st.columns(3)
        c1.metric("F1 (validated pipeline)", df["F1"].iloc[-1])
        c2.metric("Hallucinations (validated)", df["hallucinations"].iloc[-1])
        c3.metric("Stages", len(df))
        st.dataframe(df, use_container_width=True)
        st.caption(f"Parsed from {report.relative_to(ROOT)} — the canonical "
                   "experiment report.")
    else:
        st.warning("Could not parse the 02b pipeline table from the report — "
                   "showing nothing rather than a guess.")

# --- artifact freshness --------------------------------------------------------------
st.subheader("Artifact freshness")
artifacts = {
    "processed Severson parquet": ROOT / "data" / "processed" / "severson_mit_cycles.parquet",
    "early-cycle features": ROOT / "data" / "processed" / "severson_features.parquet",
    "serving model (meta.json)": ROOT / "app" / "artifacts" / "meta.json",
    "experiment 01 report": OUTPUTS / "experiment_01_prediction.md",
    "experiment 02 report": report,
}
rows = []
for name, p in artifacts.items():
    if p.exists():
        rows.append({"artifact": name,
                     "last updated": datetime.fromtimestamp(p.stat().st_mtime)
                     .strftime("%Y-%m-%d %H:%M"),
                     "status": "present"})
    else:
        rows.append({"artifact": name, "last updated": "—", "status": "MISSING"})
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.divider()
st.subheader("🛰️ Literature Monitor")
import json  # noqa: E402

run_log = ROOT / "data" / "monitor" / "run_log.jsonl"
cand_file = ROOT / "data" / "monitor" / "candidates.jsonl"
if not run_log.exists():
    st.info("No monitor runs yet. Start one with "
            "`python -m src.agents.literature_monitor run` "
            "(schedulable via cron; see README — deliberately not auto-scheduled).")
else:
    runs = [json.loads(x) for x in run_log.read_text().splitlines() if x.strip()]
    last = runs[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last run (UTC)", last["finished"][:16].replace("T", " "))
    c2.metric("Queries issued", len(last["queries"]))
    c3.metric("Hits", last["total_hits"])
    c4.metric("New candidates", last["new_candidates"])

    if cand_file.exists():
        cands = [json.loads(x) for x in cand_file.read_text().splitlines() if x.strip()]
        cdf = pd.DataFrame(cands)
        st.markdown("**Candidates by status:** " + ", ".join(
            f"{k}: {v}" for k, v in cdf["status"].value_counts().items()))
        st.markdown("**Triage table** (LLM verdicts are advisory only — "
                    "promotion is a human action):")
        show = cdf[["candidate_id", "triage_verdict", "title", "venue", "year",
                    "matched_target", "triage_rationale", "status"]].copy()
        order = {"yes": 0, "maybe": 1, "unparsed": 2, "no": 3}
        show = show.sort_values(by="triage_verdict",
                                key=lambda s: s.map(order).fillna(9))
        st.dataframe(show, use_container_width=True, height=320)
        pending = cdf[cdf["status"] == "pending_review"]
        if len(pending):
            st.code(f"python -m src.agents.promote <candidate_id> --confirm",
                    language="bash")
            st.caption(f"{len(pending)} candidate(s) awaiting human review. "
                       "Nothing is promoted automatically, regardless of verdict.")
    else:
        st.warning("Run log exists but candidates.jsonl is missing.")
