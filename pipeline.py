"""
pipeline.py

The core Cohort pipeline. Reads posts from data/sample_posts.csv and:

  1. Extracts per-account behavioral signals (age, follower ratio, etc.)
  2. Finds groups of posts that are textually near-identical AND posted
     within a tight time window of each other (the coordination signal)
  3. Builds a graph where accounts are nodes and an edge exists between
     two accounts if they posted near-identical content close in time
  4. Finds connected clusters in that graph and scores each one
  5. Saves a network-graph visualization (outputs/cluster_graph.png)
     and a CSV summary of flagged clusters (outputs/flagged_clusters.csv)

Run: python3 src/pipeline.py
"""

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import timedelta

DATA_PATH = "data/sample_posts.csv"
GRAPH_OUT = "outputs/cluster_graph.png"
CSV_OUT = "outputs/flagged_clusters.csv"

TEXT_SIMILARITY_THRESHOLD = 0.55   # how similar two posts' wording must be
TIME_WINDOW_MINUTES = 15           # how close in time two posts must be
MIN_CLUSTER_SIZE = 3               # smallest group we bother flagging


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


def build_coordination_graph(df):
    """
    Builds a graph where each node is an account. An edge is drawn between
    two accounts if: (a) their posts are highly similar in wording, and
    (b) those posts were made within TIME_WINDOW_MINUTES of each other.
    This is the actual "are these accounts moving together" check.
    """
    texts = df["text"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf)

    G = nx.Graph()
    for acc in df["account_id"].unique():
        G.add_node(acc)

    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] < TEXT_SIMILARITY_THRESHOLD:
                continue
            time_diff = abs((df.iloc[i]["timestamp"] - df.iloc[j]["timestamp"]).total_seconds())
            if time_diff > TIME_WINDOW_MINUTES * 60:
                continue
            acc_i, acc_j = df.iloc[i]["account_id"], df.iloc[j]["account_id"]
            if acc_i == acc_j:
                continue
            if G.has_edge(acc_i, acc_j):
                G[acc_i][acc_j]["weight"] += 1
            else:
                G.add_edge(acc_i, acc_j, weight=1)
    return G


def score_clusters(G, df):
    """
    For each connected component (cluster) of size >= MIN_CLUSTER_SIZE,
    compute a coordination score based on:
      - cluster size (more accounts acting together = higher signal)
      - edge density (how tightly interconnected the cluster is)
      - average account age (newer accounts = more suspicious)
    Score is 0-100, higher = more likely to be a coordinated campaign.
    """
    results = []
    account_meta = df.drop_duplicates("account_id").set_index("account_id")

    for component in nx.connected_components(G):
        if len(component) < MIN_CLUSTER_SIZE:
            continue
        sub = G.subgraph(component)
        size = len(component)
        max_edges = size * (size - 1) / 2
        density = sub.number_of_edges() / max_edges if max_edges > 0 else 0
        avg_age = account_meta.loc[list(component), "account_age_days"].mean()
        avg_followers = account_meta.loc[list(component), "follower_count"].mean()

        # newer accounts + low followers + high density + larger size -> higher score
        age_signal = max(0, 1 - avg_age / 365)          # newer = closer to 1
        follower_signal = max(0, 1 - avg_followers / 500)
        size_signal = min(1, size / 15)

        score = round(100 * (0.35 * density + 0.25 * age_signal + 0.15 * follower_signal + 0.25 * size_signal), 1)

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


def save_cluster_csv(results):
    rows = []
    for r in results:
        rows.append({
            "cluster_id": r["cluster_id"],
            "size": r["size"],
            "density": r["density"],
            "avg_account_age_days": r["avg_account_age_days"],
            "avg_followers": r["avg_followers"],
            "coordination_score": r["coordination_score"],
            "accounts": ", ".join(r["accounts"]),
        })
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"Saved cluster summary to {CSV_OUT}")


def visualize(G, results):
    """
    Draws the full graph. Flagged (coordinated) clusters are highlighted
    in red with visible edges; everything else (isolated/organic accounts,
    or small non-flagged groups) is shown muted in gray.
    """
    flagged_accounts = set()
    for r in results:
        flagged_accounts.update(r["accounts"])

    pos = nx.spring_layout(G, seed=42, k=0.6)

    plt.figure(figsize=(11, 7))
    organic_nodes = [n for n in G.nodes if n not in flagged_accounts]
    flagged_nodes = [n for n in G.nodes if n in flagged_accounts]

    organic_edges = [(u, v) for u, v in G.edges if u not in flagged_accounts or v not in flagged_accounts]
    flagged_edges = [(u, v) for u, v in G.edges if u in flagged_accounts and v in flagged_accounts]

    nx.draw_networkx_nodes(G, pos, nodelist=organic_nodes, node_color="#B8C0CC", node_size=60, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=organic_edges, edge_color="#D0D5DC", width=1, alpha=0.6)

    nx.draw_networkx_nodes(G, pos, nodelist=flagged_nodes, node_color="#C1121F", node_size=140)
    nx.draw_networkx_edges(G, pos, edgelist=flagged_edges, edge_color="#C1121F", width=1.8, alpha=0.85)

    plt.title("Cohort \u2014 Account Coordination Graph\n(red = flagged coordinated cluster, gray = organic activity)",
              fontsize=13, fontweight="bold", color="#14213D")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPH_OUT, dpi=160, facecolor="white")
    plt.close()
    print(f"Saved graph visualization to {GRAPH_OUT}")


def main():
    df = load_data()
    print(f"Loaded {len(df)} posts from {df['account_id'].nunique()} accounts")

    G = build_coordination_graph(df)
    print(f"Built coordination graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    results = score_clusters(G, df)
    print(f"Flagged {len(results)} coordinated cluster(s):")
    for r in results:
        print(f"  Cluster {r['cluster_id']}: {r['size']} accounts, "
              f"score={r['coordination_score']}, avg_age={r['avg_account_age_days']}d")

    save_cluster_csv(results)
    visualize(G, results)


if __name__ == "__main__":
    main()
