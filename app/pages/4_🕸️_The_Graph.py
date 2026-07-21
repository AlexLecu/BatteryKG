"""Graph neighborhood — interactive SIMILAR_TO network around one instance."""
import sys
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.common import make_driver, run_query  # noqa: E402

st.set_page_config(page_title="The graph — BatteryKG", page_icon="🕸️", layout="wide")
st.title("🕸️ The graph")

driver, err = make_driver()
if driver is None:
    st.error(f"Knowledge graph unavailable — {err}")
    st.code("docker compose up -d neo4j", language="bash")
    st.stop()

instances = run_query(driver, """
    MATCH (ci:CellInstance) RETURN ci.study_cell_id AS id ORDER BY id""")
if not instances:
    st.warning("No CellInstance nodes — run `python -m src.kg.load`.")
    st.stop()

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    center = st.selectbox("Cell instance", [i["id"] for i in instances])
with c2:
    view = st.radio("Similarity view", ["behavior", "condition"], horizontal=True)
with c3:
    k = st.slider("neighbors", 3, 10, 5)

rows = run_query(driver, """
    MATCH (a:CellInstance {study_cell_id: $center})-[r:SIMILAR_TO {view: $view}]->(b)
    OPTIONAL MATCH (b)<-[:ABOUT]-(m:Measurement {metric: 'cycle_life_nominal'})
    RETURN b.study_cell_id AS nbr, r.weight AS weight,
           r.same_policy_group AS same_group, m.value AS life,
           b.charge_policy_norm AS policy
    ORDER BY r.weight DESC LIMIT $k
""", center=center, view=view, k=k)
center_life = run_query(driver, """
    MATCH (:CellInstance {study_cell_id: $center})<-[:ABOUT]-
          (m:Measurement {metric: 'cycle_life_nominal'})
    RETURN m.value AS life""", center=center)

if not rows:
    st.warning(f"No SIMILAR_TO edges in the '{view}' view for {center} — "
               "run `python -m src.kg.load`.")
    st.stop()

# node colors: viridis over cycle life (same encoding as the paper's fade figure)
lives = [r["life"] for r in rows if r["life"] is not None]
if center_life and center_life[0]["life"] is not None:
    lives.append(center_life[0]["life"])
norm = mpl.colors.Normalize(vmin=min(lives), vmax=max(lives))
cmap = plt.cm.viridis

def life_color(life):
    if life is None:
        return "#9ca3af"
    return mpl.colors.to_hex(cmap(norm(life)))

try:
    from pyvis.network import Network
    net = Network(height="560px", width="100%", bgcolor="#ffffff",
                  font_color="#1f2937", directed=False, cdn_resources="in_line")
    c_life = center_life[0]["life"] if center_life else None
    net.add_node(center, label=center, size=28, color=life_color(c_life),
                 borderWidth=3,
                 title=f"{center} — cycle life {c_life:.0f}" if c_life else center)
    for r in rows:
        title = (f"{r['nbr']} — cycle life {r['life']:.0f}, policy {r['policy']}"
                 if r["life"] is not None else r["nbr"])
        net.add_node(r["nbr"], label=r["nbr"], size=16, color=life_color(r["life"]),
                     title=title)
        net.add_edge(center, r["nbr"], value=float(r["weight"]),
                     width=1 + 8 * float(r["weight"]),
                     dashes=bool(r["same_group"]),
                     title=f"weight {r['weight']:.3f}"
                           + (" (same policy group)" if r["same_group"] else ""))
    with tempfile.NamedTemporaryFile("r+", suffix=".html", delete=False) as f:
        net.save_graph(f.name)
        html = Path(f.name).read_text()
    components.html(html, height=580)
except Exception as e:                            # noqa: BLE001
    st.error(f"Network rendering failed ({type(e).__name__}: {e}); table fallback:")

st.caption("Node color = measured cycle life (viridis, matching the paper's fade "
           "figure); edge width = similarity weight; dashed edge = same charge-policy "
           "group (excluded from coverage under grouped evaluation).")

import pandas as pd  # noqa: E402
st.dataframe(pd.DataFrame(rows).style.format({"weight": "{:.3f}", "life": "{:.0f}"}),
             use_container_width=True)
driver.close()
