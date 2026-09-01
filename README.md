# Cohort

**Detecting bot-driven misinformation campaigns by finding the coordinated networks behind them — not just flagging individual posts.**

Built for **Omnikon National Hackathon 2026** — Problem Statement `Omni_CyberTech_12`

Team: **Shreyas Singh** (CSE Core) & **Utkarsh Singh** (ECE)

**Live demo:** _add your Streamlit Community Cloud URL here after deploying (see [Deployment](#deployment) below)_

---

## The Problem

Most misinformation-detection tools ask *"is this one post true or false?"* — evaluated in isolation, usually too slow to matter during a fast-moving event. What that approach misses is the actual signature of an organized campaign: **hundreds of accounts posting near-identical content within a tight time window**, working together to make a narrative look bigger and more "real" than it is.

Real-world examples that motivated this project:

- **Doppelganger campaign** (Europe) — a Russian-linked network cloned real news outlets and pushed coordinated posts across thousands of accounts, documented by EU DisinfoLab and Meta's takedown reports.
- **Recent India–Pakistan tensions** — state-linked and anonymous accounts on both sides pushed competing, unverified claims in real time during recent conflict, flagged by independent fact-checkers as coordinated amplification rather than isolated posts.
- **Hurricane conspiracy theories** (USA) — a non-military example showing the same account-coordination pattern outside any conflict setting, proving this is a general phenomenon, not just a wartime tool.

## Our Approach

Instead of asking whether a single post is true, **Cohort** asks whether a *cluster* of accounts is moving together. That coordination — not the content of any one post — is what's hard to fake and hard to hide.

### Pipeline

```
Ingest → Extract Signals → Cluster → Score & Visualize
```

1. **Ingest** — load posts (bundled sample dataset, or a CSV you upload in the app)
2. **Extract signals** — account age, follower count, posting timings and a TF-IDF text-similarity fingerprint per post
3. **Cluster** — build a graph where an edge connects two accounts if their posts are highly similar in wording (cosine similarity ≥ threshold) **and** posted within a configurable time window of each other; find connected components
4. **Score & visualize** — rank each cluster with a coordination score (0–100, based on density, account age, follower count, and cluster size) and render the full account graph, with flagged clusters highlighted in red against organic activity in gray

Cohort flags clusters for human review — it does not auto-remove anyone's content.

## Try It Yourself

- **Live app:** see the link at the top of this README once deployed
- **Locally:**

```bash
git clone https://github.com/<your-username>/cohort-omnikon.git
cd cohort-omnikon
pip install -r requirements.txt

python3 generate_data.py   # regenerate the sample dataset (optional, already included)
python3 pipeline.py        # run the pipeline from the CLI
streamlit run app.py           # or launch the interactive dashboard
```

The dashboard lets you upload your own CSV (same columns as `sample_posts.csv`) and adjust the similarity threshold, time window, and minimum cluster size live, rather than only running on the bundled sample.

## Deployment

We deployed the dashboard using **Streamlit Community Cloud** (free, no server management):

1. Push this repo to GitHub (already done)
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
3. Click "New app", select this repo, branch `main`, and set the main file to `app.py`
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically and gives a public URL within a couple of minutes

## Scalability

The version submitted for Phase 2 compared every post against every other post (O(n²)), which would not hold up past a few thousand posts. For Round 3 we fixed the actual bottleneck rather than just noting it:

- **Time-windowed comparison**: posts are sorted by timestamp, and each post is only compared against others still inside the configurable time window (a sliding window), instead of the full dataset. This turns the comparison cost into roughly **O(n × k)**, where *k* is the average number of posts within one time window — a small, roughly constant number even as total data grows, instead of scaling with the square of the dataset size.
- **What's still O(n²)-shaped**: computing TF-IDF cosine similarity between the posts that *are* in the same window. At real platform scale (millions of posts/day), the next step would be swapping exact pairwise cosine similarity for an **approximate-nearest-neighbor index** (e.g. FAISS or Annoy) over post embeddings, so similarity lookups stop scaling with the number of posts in a window at all.
- **Beyond the algorithm**: at production scale we'd also move off a single in-memory NetworkX graph and into a proper graph database (e.g. Neo4j) or a streaming architecture (ingest → feature extraction → clustering as separate, horizontally-scalable stages) rather than one monolithic script.

None of this is implemented yet — it's an honest account of the next bottleneck and the concrete plan to remove it, not a claim that it's already solved.

## Testing

Basic tests live in `test_pipeline.py` and check the core claims the project rests on: that a planted coordinated cluster gets correctly flagged, that organic/unrelated activity does *not* get falsely flagged, that similarity without matching timing is correctly ignored, and that the pipeline doesn't crash on empty input.

```bash
python3 test_pipeline.py
# or, if pytest is installed:
python3 -m pytest test_pipeline.py -v
```

## What's Built

- ✅ Detection pipeline (TF-IDF similarity + time-windowed graph clustering + coordination scoring), with the O(n²) → O(n·k) scalability fix described above
- ✅ Synthetic sample dataset generator (`src/generate_data.py`)
- ✅ Interactive Streamlit dashboard — CSV upload, live-adjustable detection parameters, downloadable results
- ✅ Basic automated tests (`tests/test_pipeline.py`)
- ✅ Public deployment (Streamlit Community Cloud)
- ⏳ Live platform API ingestion — not yet built, planned next
- ⏳ Embedding-based similarity (currently TF-IDF / near-exact-match style) — planned next
- ⏳ Human-review workflow UI before any flag is acted on — planned next

On our test dataset, the pipeline correctly identifies the single planted coordinated cluster (14 accounts, coordination score 92.9) and does not flag any organic accounts as false positives.

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Language | Python | Core logic and data processing |
| Data | pandas | Wrangling post/account data |
| Similarity | scikit-learn (TF-IDF + cosine similarity) | Detecting near-identical wording between posts |
| Graph | NetworkX | Building the coordination graph, finding connected clusters |
| Visualization | matplotlib | Rendering the network graph |
| Dashboard | Streamlit | Interactive demo UI, deployed via Streamlit Community Cloud |
| Testing | Plain Python asserts (pytest-compatible) | Verifying detection correctness |

We're first-year students and are learning parts of this stack as we build — the clustering approach here (TF-IDF + time-window graphing) is intentionally simple and explainable rather than a black-box ML model, which also makes it easier to justify why a given cluster got flagged.

## Challenges We Faced

- Tuning the similarity threshold and time window took a few passes — too loose and organic accounts who happened to post around the same trending topic got falsely grouped together; too strict and the planted cluster split into fragments instead of one connected group.
- Deciding what should count toward a "coordination score" was harder than expected — we settled on density + account age + follower count + size because each maps to a real-world red flag, but there's no ground-truth dataset to formally validate the weights against yet.
- The original Phase 2 pipeline compared every post to every other post — it worked on our small sample but wouldn't scale. Rewriting it to only compare posts within a sliding time window (instead of noting scalability as a future concern) was the main technical push for Round 3.

## Who This Helps

- **Fact-checkers & journalists** — a ranked list of suspicious clusters instead of an endless stream of individual posts
- **Platform trust & safety teams** — an earlier signal for coordinated activity, before a narrative fully takes hold
- **Everyday people, during a crisis** — less exposure to manufactured "consensus" during wars, disasters, or elections

---

*Omnikon National Hackathon 2026 · Round 3 (Final) Submission*
