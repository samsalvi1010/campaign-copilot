"""Generate a realistic fake marketing dataset: data/sample_campaigns.csv

One row = one campaign, in one country, on one day.
Run with:  python scripts/generate_data.py

Two "stories" are hidden in the data for the analysis to find:
  1. TikTok has very cheap clicks, but almost nobody buys (low conversion rate).
  2. Meta performance drops during week 3 of September (Mon 14 - Sun 20 Sep 2026).
"""
from pathlib import Path

import numpy as np
import pandas as pd

# A fixed seed means we get the exact same "random" numbers on every run,
# so anyone who runs this script gets an identical CSV.
rng = np.random.default_rng(seed=42)

# --- Settings for each channel -------------------------------------------
# base_impressions: typical daily impressions for one campaign in one country
# ctr: click-through rate (clicks / impressions)
# cpc: cost per click in euros
# cvr: conversion rate (conversions / clicks)
# order_value: average revenue per conversion in euros
CHANNELS = {
    "Google Ads": {"base_impressions": 20000, "ctr": 0.035, "cpc": 0.90, "cvr": 0.040, "order_value": 70},
    "Meta":       {"base_impressions": 40000, "ctr": 0.012, "cpc": 0.60, "cvr": 0.025, "order_value": 60},
    # Story 1: TikTok has cheap clicks (low cpc) but a very low conversion rate.
    "TikTok":     {"base_impressions": 60000, "ctr": 0.015, "cpc": 0.25, "cvr": 0.004, "order_value": 45},
    "LinkedIn":   {"base_impressions": 8000,  "ctr": 0.006, "cpc": 4.50, "cvr": 0.030, "order_value": 150},
    # Email cpc includes the email tool and design costs, spread over each click
    "Email":      {"base_impressions": 10000, "ctr": 0.040, "cpc": 0.60, "cvr": 0.050, "order_value": 65},
}

# Campaign names inside each channel
CAMPAIGNS = {
    "Google Ads": ["Search - Brand", "Search - Generic"],
    "Meta":       ["Prospecting", "Retargeting"],
    "TikTok":     ["Awareness Video", "Spark Ads"],
    "LinkedIn":   ["B2B Lead Gen"],
    "Email":      ["Newsletter", "Abandoned Cart"],
}

# Bigger markets get more impressions (1.0 = France as the reference)
COUNTRY_SIZE = {"FR": 1.0, "DE": 1.2, "ES": 0.7, "IT": 0.8}

WEEKEND_FACTOR = 0.7  # weekends get 30% less traffic

# Story 2: dates when Meta performance drops (week 3 of September)
META_DROP_START = pd.Timestamp("2026-09-14")
META_DROP_END = pd.Timestamp("2026-09-20")

# --- Build the rows -------------------------------------------------------
rows = []
for date in pd.date_range("2026-07-01", "2026-10-31", freq="D"):
    is_weekend = date.day_of_week >= 5  # 5 = Saturday, 6 = Sunday

    for channel, settings in CHANNELS.items():
        ctr = settings["ctr"]
        cpc = settings["cpc"]
        cvr = settings["cvr"]

        # Story 2: during the drop, Meta ads get fewer clicks, each click costs
        # more, and fewer of those clicks turn into sales.
        if channel == "Meta" and META_DROP_START <= date <= META_DROP_END:
            ctr = ctr * 0.6
            cpc = cpc * 1.8
            cvr = cvr * 0.5

        for campaign in CAMPAIGNS[channel]:
            for country, size in COUNTRY_SIZE.items():
                # Impressions: base volume x market size x weekend effect,
                # plus +/- ~15% random day-to-day noise
                impressions = settings["base_impressions"] * size
                if is_weekend:
                    impressions = impressions * WEEKEND_FACTOR
                impressions = int(impressions * rng.normal(1.0, 0.15))
                impressions = max(impressions, 0)  # never negative

                # binomial = "each impression has a ctr chance of becoming a click"
                clicks = rng.binomial(impressions, ctr)
                conversions = rng.binomial(clicks, cvr)

                # Spend and revenue get a little noise too (+/- ~10%)
                spend = clicks * cpc * rng.normal(1.0, 0.10)
                revenue = conversions * settings["order_value"] * rng.normal(1.0, 0.10)

                rows.append({
                    "date": date.date(),
                    "channel": channel,
                    "campaign": campaign,
                    "country": country,
                    "impressions": impressions,
                    "clicks": clicks,
                    "spend_eur": round(max(spend, 0), 2),
                    "conversions": conversions,
                    "revenue_eur": round(max(revenue, 0), 2),
                })

df = pd.DataFrame(rows)

# Save next to the project's data/ folder, no matter where the script is run from
output_path = Path(__file__).resolve().parent.parent / "data" / "sample_campaigns.csv"
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")
