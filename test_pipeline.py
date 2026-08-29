"""
test_pipeline.py

Basic tests for the Cohort detection pipeline. These aren't exhaustive --
they check the core claim the whole project rests on: that the pipeline
correctly separates a planted coordinated cluster from organic activity,
and that it behaves sanely on edge cases (empty input, no coordination).

Run: python3 -m pytest tests/ -v
(or, without pytest installed: python3 tests/test_pipeline.py)
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import build_coordination_graph, score_clusters, run_pipeline


def _make_post(post_id, account_id, text, timestamp, age=500, followers=200, following=200):
    return {
        "post_id": post_id, "account_id": account_id, "account_age_days": age,
        "follower_count": followers, "following_count": following,
        "timestamp": timestamp, "text": text,
    }


def test_detects_planted_coordinated_cluster():
    """Five accounts posting near-identical text within a 2-minute window
    should be flagged as a single coordinated cluster."""
    base = datetime(2026, 1, 1, 12, 0, 0)
    rows = []
    for i in range(5):
        rows.append(_make_post(
            i, f"bot_{i}", "BREAKING: this claim is completely fabricated, share now",
            base + timedelta(seconds=i * 20), age=5, followers=3,
        ))
    df = pd.DataFrame(rows)

    G, results = run_pipeline(df, similarity_threshold=0.5, time_window_minutes=5, min_cluster_size=3)

    assert len(results) == 1, "expected exactly one flagged cluster"
    assert results[0]["size"] == 5, "expected all 5 planted accounts in the cluster"
    assert results[0]["coordination_score"] > 50, "expected a high coordination score"


def test_no_false_positive_on_organic_activity():
    """Accounts posting unrelated content at random, spread-out times
    should never be flagged."""
    base = datetime(2026, 1, 1, 12, 0, 0)
    topics = [
        "just finished my morning run", "anyone know a good place to eat",
        "watching the match tonight", "studying for finals is rough",
        "beautiful sunset today",
    ]
    rows = [
        _make_post(i, f"user_{i}", topics[i], base + timedelta(hours=i * 3))
        for i in range(len(topics))
    ]
    df = pd.DataFrame(rows)

    G, results = run_pipeline(df, similarity_threshold=0.5, time_window_minutes=5, min_cluster_size=3)

    assert len(results) == 0, "organic, unrelated posts should not be flagged"


def test_similar_text_outside_time_window_not_flagged():
    """Even identical text shouldn't be flagged if posted far apart in
    time -- coordination requires BOTH similarity and timing."""
    base = datetime(2026, 1, 1, 12, 0, 0)
    rows = [
        _make_post(0, "user_a", "check out this deal today", base),
        _make_post(1, "user_b", "check out this deal today", base + timedelta(hours=6)),
        _make_post(2, "user_c", "check out this deal today", base + timedelta(hours=12)),
    ]
    df = pd.DataFrame(rows)

    G, results = run_pipeline(df, similarity_threshold=0.5, time_window_minutes=15, min_cluster_size=3)

    assert len(results) == 0, "identical text posted hours apart should not count as coordinated"


def test_empty_dataframe_does_not_crash():
    """The pipeline should handle an empty dataset gracefully rather than
    raising an exception."""
    df = pd.DataFrame(columns=[
        "post_id", "account_id", "account_age_days",
        "follower_count", "following_count", "timestamp", "text",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    G, results = run_pipeline(df)
    assert G.number_of_nodes() == 0
    assert results == []


if __name__ == "__main__":
    tests = [
        test_detects_planted_coordinated_cluster,
        test_no_false_positive_on_organic_activity,
        test_similar_text_outside_time_window_not_flagged,
        test_empty_dataframe_does_not_crash,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -- {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
