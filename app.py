"""
app.py

Cohort's interactive dashboard. Loads a dataset (the bundled sample, or a
CSV the user uploads), runs the detection pipeline live with user-adjustable
thresholds, and shows flagged clusters plus the coordination graph.

Run locally:   streamlit run app.py
Deployed via:  Streamlit Community Cloud, pointed at this repo (see README)
"""

import io
import sys
import os

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import (
    load_data,
    run_pipeline,
    clusters_to_dataframe,
    build_figure,
    TEXT_SIMILARITY_THRESHOLD,
    TIME_WINDOW_MINUTES,
    MIN_CLUSTER_SIZE,
)

st.set_page_config(page_title="Cohort", page_icon="\U0001F578\uFE0F", layout="wide")

# ---- light styling: keep it simple, no custom CSS frameworks, just spacing/color tweaks ----
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.7rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar: data source + tunable parameters ----------------
with st.sidebar:
    st.header("\U0001F578\uFE0F Cohort")
    st.caption("Detecting bot-driven misinformation campaigns by finding coordinated account clusters.")

    st.divider()
    st.subheader("Data")
    uploaded = st.file_uploader(
        "Upload your own posts CSV",
        type=["csv"],
        help="Expected columns: post_id, account_id, account_age_days, "
             "follower_count, following_count, timestamp, text",
    )
    use_sample = st.checkbox("Use bundled sample dataset", value=(uploaded is None))

    st.divider()
    st.subheader("Detection settings")
    similarity_threshold = st.slider(
        "Text similarity threshold", min_value=0.2, max_value=0.9,
        value=TEXT_SIMILARITY_THRESHOLD, step=0.05,
        help="How similar two posts' wording must be (cosine similarity) to count as coordinated.",
    )
    time_window = st.slider(
        "Time window (minutes)", min_value=1, max_value=60,
        value=TIME_WINDOW_MINUTES, step=1,
        help="How close together in time two posts must be to count as coordinated.",
    )
    min_cluster_size = st.slider(
        "Minimum cluster size to flag", min_value=2, max_value=10,
        value=MIN_CLUSTER_SIZE, step=1,
    )

    st.divider()
    st.caption("Sample dataset for demo purposes. Live platform API ingestion is a planned next step.")

# ---------------- Load data ----------------
if uploaded is not None and not use_sample:
    df = pd.read_csv(uploaded, parse_dates=["timestamp"])
    data_label = f"your upload ({uploaded.name})"
else:
    df = load_data()
    data_label = "the bundled sample dataset"

# ---------------- Header ----------------
st.title("Cohort")
st.caption(
    f"Analyzing **{data_label}** \u2014 flagging clusters of accounts that post near-identical "
    "content in tight time windows, rather than judging any single post."
)

# ---------------- Run pipeline ----------------
with st.spinner("Running the coordination detection pipeline..."):
    G, results = run_pipeline(
        df,
        similarity_threshold=similarity_threshold,
        time_window_minutes=time_window,
        min_cluster_size=min_cluster_size,
    )

# ---------------- Top metrics ----------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Posts analyzed", len(df))
m2.metric("Accounts", df["account_id"].nunique())
m3.metric("Clusters flagged", len(results))
top_score = results[0]["coordination_score"] if results else 0
m4.metric("Top coordination score", f"{top_score:.1f}" if results else "\u2014")

st.divider()

# ---------------- Two-column layout: table + graph ----------------
left, right = st.columns([1, 1.3])

with left:
    st.subheader("Flagged Clusters")
    if results:
        table = clusters_to_dataframe(results).rename(columns={
            "cluster_id": "Cluster", "size": "Accounts", "density": "Density",
            "avg_account_age_days": "Avg. Age (days)", "avg_followers": "Avg. Followers",
            "coordination_score": "Score", "accounts": "Account IDs",
        })
        st.dataframe(
            table.drop(columns=["Account IDs"]),
            use_container_width=True, hide_index=True,
        )
        csv_buffer = io.StringIO()
        table.to_csv(csv_buffer, index=False)
        st.download_button(
            "\u2b07\ufe0f Download results as CSV", csv_buffer.getvalue(),
            file_name="cohort_flagged_clusters.csv", mime="text/csv",
        )

        for r in results:
            with st.expander(f"Cluster {r['cluster_id']} details \u2014 {r['size']} accounts, score {r['coordination_score']}"):
                st.write(", ".join(r["accounts"]))
    else:
        st.info("No coordinated clusters found with the current settings. Try lowering the similarity threshold or widening the time window.")

with right:
    st.subheader("Coordination Graph")
    fig = build_figure(G, results)
    st.pyplot(fig, use_container_width=True)
    st.caption("Red = flagged coordinated cluster \u00b7 Gray = organic activity")

st.divider()
with st.expander("How this works"):
    st.markdown(
        """
        **Cohort doesn't ask "is this post true or false" \u2014 it asks "are these accounts moving together."**

        1. **Ingest** \u2014 load posts (sample data here; a live platform API is the planned next step)
        2. **Extract signals** \u2014 account age, follower ratio, and a TF-IDF text-similarity fingerprint per post
        3. **Cluster** \u2014 connect two accounts in a graph if their posts are highly similar in wording *and*
           posted within the configured time window
        4. **Score** \u2014 rank each connected cluster 0\u2013100 based on density, account age, follower count, and size

        Flagged clusters are meant for **human review** \u2014 this tool does not auto-remove any content.
        """
    )
