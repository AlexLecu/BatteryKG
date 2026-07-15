"""BatteryKG demo — landing page."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.common import load_artifacts, make_driver, neo4j_settings  # noqa: E402

st.set_page_config(page_title="BatteryKG", page_icon="🔋", layout="wide")

st.title("🔋 BatteryKG")
st.caption("Reconciling manufacturer claims with measured degradation — "
           "knowledge graph, gated prediction, and validated claim extraction.")

st.markdown("""
**Pages**

1. **Cell Explorer** — datasheet claims per commercial cell, with stated test
   conditions and their gaps; measured cycle-life distribution and the
   claim-vs-measured discrepancy verdict.
2. **Prediction with Abstention** — cycle-life prediction that *refuses* when
   the cell's knowledge-graph neighborhood is too sparse.
3. **Graph Neighborhood** — interactive view of a cell's similarity
   neighborhood (behavior vs. condition space).
4. **Pipeline Status** — live knowledge-graph counts and extraction-pipeline
   quality metrics.

Every displayed number is read from the knowledge graph, a trained model
artifact, or an experiment report — nothing is mocked.
""")

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Knowledge graph")
    driver, err = make_driver()
    if driver is None:
        st.error(f"Not connected — {err}")
        st.code("docker compose up -d neo4j", language="bash")
    else:
        st.success(f"Connected to {neo4j_settings()['uri']}")
        driver.close()
with c2:
    st.subheader("Model artifacts")
    art, err = load_artifacts()
    if art is None:
        st.error(f"Not available — {err}")
    else:
        m = art["meta"]
        st.success(f"Trained {m['trained_at']} on {m['n_training_cells']} cells "
                   f"(CV RMSE {m['cv_metrics']['rmse_cycles']:.0f} cycles, "
                   f"MAPE {m['cv_metrics']['mape_pct']:.1f}%)")
