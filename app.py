"""
app.py

Minimal Streamlit dashboard for Cohort. Loads the sample dataset, runs the
same coordination-detection pipeline as src/pipeline.py, and shows the
flagged clusters + graph interactively.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import load_data, build_coordination_graph, score_clusters, visualize

st.set_page_config(page_title="Cohort", page_icon="🕸️", layout="wide")

st.title("🕸️ Cohort")
st.caption("Detecting bot-driven misinformation campaigns by finding coordinated account clusters, not just flagging individual posts.")

with st.spinner("Loading sample data and running the coordination pipeline..."):
    df = load_data()
    G = build_coordination_graph(df)
    results = score_clusters(G, df)
    visualize(G, results)

col1, col2, col3 = st.columns(3)
col1.metric("Posts analyzed", len(df))
col2.metric("Accounts", df["account_id"].nunique())
col3.metric("Coordinated clusters flagged", len(results))

st.divider()

st.subheader("Flagged Clusters")
if results:
    table = pd.DataFrame([{
        "Cluster": r["cluster_id"],
        "Accounts": r["size"],
        "Density": r["density"],
        "Avg. Account Age (days)": r["avg_account_age_days"],
        "Avg. Followers": r["avg_followers"],
        "Coordination Score": r["coordination_score"],
    } for r in results])
    st.dataframe(table, use_container_width=True)

    for r in results:
        with st.expander(f"Cluster {r['cluster_id']} — {r['size']} accounts, score {r['coordination_score']}"):
            st.write(", ".join(r["accounts"]))
else:
    st.info("No coordinated clusters found in this sample.")

st.divider()
st.subheader("Coordination Graph")
st.image("outputs/cluster_graph.png", caption="Red = flagged coordinated cluster · Gray = organic activity", use_container_width=True)

st.caption("Sample dataset for this demo. Live platform API ingestion is a planned next step.")
