"""
generate_data.py

Generates a synthetic sample dataset of social-media-style posts for the
Cohort demo. This is NOT scraped from any real platform -- it's a stand-in
dataset that mimics the shape of real data (organic posts + a coordinated
bot cluster) so the pipeline can be demoed without needing live API access
during Phase 2.

Output: data/sample_posts.csv
Columns: post_id, account_id, account_age_days, follower_count,
         following_count, timestamp, text
"""

import csv
import random
from datetime import datetime, timedelta

random.seed(42)

OUT_PATH = "data/sample_posts.csv"

# --- organic, unrelated post templates (varied wording, varied topics) ---
ORGANIC_TEMPLATES = [
    "just finished my morning run, feeling great today",
    "anyone know a good place to eat near campus?",
    "watching the match tonight, hope our team wins",
    "studying for finals is rough this semester",
    "beautiful sunset today, had to take a picture",
    "can't believe how expensive groceries have gotten",
    "finally fixed my laptop after weeks of trying",
    "new episode of my favorite show dropped, so good",
    "traffic was insane on the highway this morning",
    "trying out a new recipe for dinner tonight",
    "excited for the weekend, finally some rest",
    "my internet has been so slow all week",
    "started reading a new book, really enjoying it so far",
    "went for a long walk, needed to clear my head",
    "coffee shop downtown has the best pastries",
]

# --- the coordinated narrative that the bot cluster pushes, near-identical wording ---
COORDINATED_TEMPLATES = [
    "BREAKING: sources confirm the reports are completely fabricated, share this now",
    "BREAKING: sources confirm the reports are totally fabricated, share now",
    "BREAKING: sources say the reports are completely fabricated, please share",
    "URGENT: sources confirm these reports are fabricated, spread the word",
    "BREAKING: multiple sources confirm reports are fabricated, RT this now",
]


def random_time(base, spread_minutes):
    return base + timedelta(minutes=random.uniform(-spread_minutes, spread_minutes))


def make_organic_accounts(n):
    accounts = []
    for i in range(n):
        accounts.append({
            "account_id": f"user_{i:04d}",
            "account_age_days": random.randint(150, 2200),
            "follower_count": random.randint(20, 3000),
            "following_count": random.randint(50, 1200),
        })
    return accounts


def make_bot_accounts(n):
    accounts = []
    for i in range(n):
        accounts.append({
            "account_id": f"bot_{i:04d}",
            "account_age_days": random.randint(1, 25),       # very new accounts
            "follower_count": random.randint(0, 15),          # almost no followers
            "following_count": random.randint(300, 900),      # follows a lot, follows nobody back
        })
    return accounts


def main():
    rows = []
    post_id = 0

    # --- organic activity spread across a week ---
    organic_accounts = make_organic_accounts(60)
    window_start = datetime(2026, 8, 10, 0, 0, 0)
    for acc in organic_accounts:
        num_posts = random.randint(1, 4)
        for _ in range(num_posts):
            ts = window_start + timedelta(
                days=random.uniform(0, 7),
                hours=random.uniform(0, 23),
            )
            rows.append({
                "post_id": post_id,
                "account_id": acc["account_id"],
                "account_age_days": acc["account_age_days"],
                "follower_count": acc["follower_count"],
                "following_count": acc["following_count"],
                "timestamp": ts.isoformat(),
                "text": random.choice(ORGANIC_TEMPLATES),
            })
            post_id += 1

    # --- coordinated cluster: two separate "bursts" of near-identical posts ---
    bot_accounts = make_bot_accounts(14)
    burst_times = [
        datetime(2026, 8, 12, 14, 5, 0),
        datetime(2026, 8, 13, 9, 40, 0),
    ]
    for burst_center in burst_times:
        for acc in bot_accounts:
            ts = random_time(burst_center, spread_minutes=4)  # all within ~8 min window
            rows.append({
                "post_id": post_id,
                "account_id": acc["account_id"],
                "account_age_days": acc["account_age_days"],
                "follower_count": acc["follower_count"],
                "following_count": acc["following_count"],
                "timestamp": ts.isoformat(),
                "text": random.choice(COORDINATED_TEMPLATES),
            })
            post_id += 1

    rows.sort(key=lambda r: r["timestamp"])

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "post_id", "account_id", "account_age_days",
            "follower_count", "following_count", "timestamp", "text"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} posts from {len(organic_accounts)} organic accounts "
          f"and {len(bot_accounts)} coordinated accounts to {OUT_PATH}")


if __name__ == "__main__":
    main()
