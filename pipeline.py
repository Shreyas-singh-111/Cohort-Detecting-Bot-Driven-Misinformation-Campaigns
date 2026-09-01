"""
pipeline.py

The core Cohort detection pipeline. Given a dataframe of posts, it:

  1. Extracts per-account behavioral signals (age, follower ratio, etc.)
  2. Finds groups of posts that are textually near-identical AND posted
     within a tight time window of each other (the coordination signal)
  3. Builds a graph where accounts are nodes and an edge exists between
     two accounts if they posted near-identical content close in time
  4. Finds connected clusters in that graph and scores each one
  5. Renders a network-graph visualization of flagged clusters

Every function here takes plain data in and returns plain data/objects out
(a DataFrame, a Graph, a list of dicts, a matplotlib Figure) rather than
reading or writing files directly. That's what lets the same pipeline code
run identically from the CLI (main(), which saves files) and from the
Streamlit app (app.py, which renders straight to the browser).

CLI usage: python3 src/pipeline.py
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import matplotlib
import networkx as nx
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sample_posts.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

GRAPH_OUT = OUTPUT_DIR / "cluster_graph.png"
CSV_OUT = OUTPUT_DIR / "flagged_clusters.csv"

# Default detection parameters. Exposed as function arguments (not just
# module constants) so the Streamlit app can let a user tune them live.
TEXT_SIMILARITY_THRESHOLD = 0.55   # how similar two posts' wording must be (0-1)
TIME_WINDOW_MINUTES = 15           # how close in time two posts must be
MIN_CLUSTER_SIZE = 3               # smallest group we bother flagging


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load posts from a CSV. Expects columns: post_id, account_id,
    account_age_days, follower_count, following_count, timestamp, text."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def build_coordination_graph(
    df: pd.DataFrame,
    similarity_threshold: float = TEXT_SIMILARITY_THRESHOLD,
    time_window_minutes: int = TIME_WINDOW_MINUTES,
) -> nx.Graph:
    """
    Builds a graph where each node is an account. An edge is drawn between
    two accounts if: (a) their posts are highly similar in wording, and
    (b) those posts were made within `time_window_minutes` of each other.
    This is the actual "are these accounts moving together" check.

    Scalability note: a naive version of this compares every post to every
    other post (O(n^2) comparisons), which stops being practical once n
    reaches the tens of thousands. Since we only ever care about posts
    close together IN TIME, we sort by timestamp first and use a sliding
    window: for each post we only compare forward against posts still
    inside the time window, stopping as soon as we fall outside it. That
    turns the comparison cost into roughly O(n * k), where k is the
    average number of posts within one time window -- typically a small,
    roughly constant number even as the total dataset grows. The one
    O(n^2)-shaped step left is TF-IDF cosine similarity between posts that
    ARE in the same window, which for production scale would be replaced
    with an approximate-nearest-neighbor index (e.g. FAISS or Annoy) --
    noted in the README's scalability section.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    texts = df["text"].tolist()

    G = nx.Graph()
    for acc in df["account_id"].unique():
        G.add_node(acc)

    if len(df) < 2:
        # nothing to compare -- return the (possibly empty) node set with no edges
        return G

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    n = len(df)
    window = timedelta(minutes=time_window_minutes)
    timestamps = df["timestamp"].tolist()
    account_ids = df["account_id"].tolist()

    for i in range(n):
        j = i + 1
        while j < n and (timestamps[j] - timestamps[i]) <= window:
            if account_ids[i] != account_ids[j]:
                sim = cosine_similarity(tfidf[i], tfidf[j])[0, 0]
                if sim >= similarity_threshold:
                    a, b = account_ids[i], account_ids[j]
                    if G.has_edge(a, b):
                        G[a][b]["weight"] += 1
                    else:
                        G.add_edge(a, b, weight=1)
            j += 1

    return G


def score_clusters(
    G: nx.Graph,
    df: pd.DataFrame,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> list[dict[str, Any]]:
    """
    For each connected component (cluster) of size >= min_cluster_size,
    compute a coordination score based on:
      - cluster size (more accounts acting together = higher signal)
      - edge density (how tightly interconnected the cluster is)
      - average account age (newer accounts = more suspicious)
      - average follower count (fewer followers = more suspicious)
    Score is 0-100, higher = more likely to be a coordinated campaign.

    This is deliberately a transparent, hand-weighted formula rather than
    a trained model: with no labeled ground-truth data yet, an explainable
    score we can justify beats an opaque one we can't (see README).
    """
    results: list[dict[str, Any]] = []
    account_meta = df.drop_duplicates("account_id").set_index("account_id")

    for component in nx.connected_components(G):
        if len(component) < min_cluster_size:
            continue
        sub = G.subgraph(component)
        size = len(component)
        max_edges = size * (size - 1) / 2
        density = sub.number_of_edges() / max_edges if max_edges > 0 else 0
        avg_age = account_meta.loc[list(component), "account_age_days"].mean()
        avg_followers = account_meta.loc[list(component), "follower_count"].mean()

        age_signal = max(0, 1 - avg_age / 365)
        follower_signal = max(0, 1 - avg_followers / 500)
        size_signal = min(1, size / 15)

        score = round(
            100 * (0.35 * density + 0.25 * age_signal + 0.15 * follower_signal + 0.25 * size_signal), 1
        )

        results.append({
            "cluster_id": len(results) + 1,
            "accounts": sorted(component),
            "size": size,
            "density": round(density, 2),
            "avg_account_age_days": round(avg_age, 1),
            "avg_followers": round(avg_followers, 1),
            "coordination_score": score,
        })

    results.sort(key=lambda r: r["coordination_score"], reverse=True)
    return results


def clusters_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Flattens the cluster results into a DataFrame, suitable for display
    or CSV export."""
    rows = [{
        "cluster_id": r["cluster_id"],
        "size": r["size"],
        "density": r["density"],
        "avg_account_age_days": r["avg_account_age_days"],
        "avg_followers": r["avg_followers"],
        "coordination_score": r["coordination_score"],
        "accounts": ", ".join(r["accounts"]),
    } for r in results]
    return pd.DataFrame(rows)


def build_figure(G: nx.Graph, results: list[dict[str, Any]]) -> plt.Figure:
    """
    Renders the coordination graph as a matplotlib Figure (not saved to
    disk -- the caller decides whether to save it or hand it to Streamlit).
    Flagged (coordinated) clusters are highlighted in red with visible
    edges; everything else (organic accounts, or small non-flagged groups)
    is shown muted in gray.
    """
    flagged_accounts: set[str] = set()
    for r in results:
        flagged_accounts.update(r["accounts"])

    pos = nx.spring_layout(G, seed=42, k=0.6)

    fig, ax = plt.subplots(figsize=(11, 7))
    organic_nodes = [n for n in G.nodes if n not in flagged_accounts]
    flagged_nodes = [n for n in G.nodes if n in flagged_accounts]
    organic_edges = [(u, v) for u, v in G.edges if u not in flagged_accounts or v not in flagged_accounts]
    flagged_edges = [(u, v) for u, v in G.edges if u in flagged_accounts and v in flagged_accounts]

    nx.draw_networkx_nodes(G, pos, nodelist=organic_nodes, node_color="#B8C0CC", node_size=60, alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=organic_edges, edge_color="#D0D5DC", width=1, alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=flagged_nodes, node_color="#C1121F", node_size=140, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=flagged_edges, edge_color="#C1121F", width=1.8, alpha=0.85, ax=ax)

    ax.set_title(
        "Cohort \u2014 Account Coordination Graph\n(red = flagged coordinated cluster, gray = organic activity)",
        fontsize=13, fontweight="bold", color="#14213D",
    )
    ax.axis("off")
    fig.tight_layout()
    return fig


def run_pipeline(
    df: pd.DataFrame,
    similarity_threshold: float = TEXT_SIMILARITY_THRESHOLD,
    time_window_minutes: int = TIME_WINDOW_MINUTES,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> tuple[nx.Graph, list[dict[str, Any]]]:
    """Convenience wrapper: runs the full detection pipeline on a dataframe
    of posts and returns (graph, flagged_clusters). Used by both main()
    and the Streamlit app so the two never drift out of sync."""
    G = build_coordination_graph(df, similarity_threshold, time_window_minutes)
    results = score_clusters(G, df, min_cluster_size)
    return G, results


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} posts from {df['account_id'].nunique()} accounts")

    G, results = run_pipeline(df)
    print(f"Built coordination graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Flagged {len(results)} coordinated cluster(s):")
    for r in results:
        print(f"  Cluster {r['cluster_id']}: {r['size']} accounts, "
              f"score={r['coordination_score']}, avg_age={r['avg_account_age_days']}d")

    clusters_to_dataframe(results).to_csv(CSV_OUT, index=False)
    print(f"Saved cluster summary to {CSV_OUT}")

    fig = build_figure(G, results)
    fig.savefig(GRAPH_OUT, dpi=160, facecolor="white")
    print(f"Saved graph visualization to {GRAPH_OUT}")


if __name__ == "__main__":
    main()
