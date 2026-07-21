"""BatteryKG demo — landing page."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.common import is_snapshot, make_driver, neo4j_settings  # noqa: E402

st.set_page_config(page_title="BatteryKG", page_icon="🔋", layout="wide")

st.title("🔋 BatteryKG")
st.markdown("### We check what battery makers promise against what independent "
            "tests measure — and our predictor refuses to guess when it "
            "shouldn't.")

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    with st.container(border=True):
        st.markdown("#### 🔋 Will it last?")
        st.caption("Design a fast-charging recipe and get a lifetime "
                   "prediction — or an honest refusal.")
        st.page_link("pages/1_🔋_Will_It_Last.py", label="**Play →**")
with c2:
    with st.container(border=True):
        st.markdown("#### ⚖️ Promise vs reality")
        st.caption("The datasheet's claim drawn over how long the cells "
                   "actually lasted.")
        st.page_link("pages/2_⚖️_Promise_vs_Reality.py", label="**Compare →**")
with c3:
    with st.container(border=True):
        st.markdown("#### 🔍 Who is lying?")
        st.caption("Official spec sheets for the same battery that disagree "
                   "with each other.")
        st.page_link("pages/3_🔍_Who_Is_Lying.py", label="**Investigate →**")

st.divider()
driver, err = make_driver()
if driver is None:
    footer = f"Knowledge graph offline — {err}"
elif is_snapshot(driver):
    footer = (f"Read-only demo: knowledge-graph answers served from a static "
              f"snapshot recorded {driver.generated_at} — same data, no live "
              "database.")
    driver.close()
else:
    footer = f"Live knowledge graph connected at {neo4j_settings()['uri']}."
    driver.close()
st.caption(f"{footer} · Every number comes from the knowledge graph, the "
           "trained model artifacts, or an experiment report — nothing is "
           "mocked. · [GitHub](https://github.com/AlexLecu/BatteryKG)")
