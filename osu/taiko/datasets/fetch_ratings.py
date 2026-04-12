"""Fetch user ratings for all beatmaps via osu! API v1 and update manifest.json.

The 'rating' field is a 1-10 average of human votes submitted after passing a map.

Usage:
    cd osu/taiko/datasets
    python fetch_ratings.py taiko_v2
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

API_V1 = "https://osu.ppy.sh/api"
API_KEY = "9f9e5ed188a7df5c6c9ddb4cc9dfc58cb3637d7b"


def fetch_beatmapset(beatmapset_id):
    """Fetch all beatmaps in a set via API v1."""
    resp = requests.get(f"{API_V1}/get_beatmaps", params={
        "k": API_KEY,
        "s": beatmapset_id,
    })
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset directory name (e.g. taiko_v2)")
    args = parser.parse_args()

    ds_dir = os.path.join(os.path.dirname(__file__), args.dataset)
    manifest_path = os.path.join(ds_dir, "manifest.json")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    charts = manifest["charts"]
    print(f"Charts: {len(charts)}")

    # Collect unique beatmapset IDs
    bset_ids = sorted(set(c["beatmapset_id"] for c in charts))
    print(f"Unique beatmapset IDs: {len(bset_ids)}")

    # Check how many already have rating
    already = sum(1 for c in charts if "rating" in c)
    if already > 0:
        print(f"Already have rating: {already}/{len(charts)}")

    # Fetch ratings per beatmapset
    # API v1 returns rating per beatmap_id within the set
    bid_to_rating = {}
    errors = 0

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(fetch_beatmapset, bset_id): bset_id for bset_id in bset_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching ratings"):
            try:
                data = future.result()
            except Exception:
                errors += 1
                continue
            for bm in data:
                bid = str(bm["beatmap_id"])
                bid_to_rating[bid] = float(bm.get("rating", 0))

    print(f"\nFetched ratings for {len(bid_to_rating)} beatmaps ({errors} errors)")

    # Update manifest
    n_updated = 0
    n_missing = 0
    for c in charts:
        bid = str(c.get("beatmap_id", ""))
        if bid in bid_to_rating:
            c["rating"] = bid_to_rating[bid]
            n_updated += 1
        else:
            n_missing += 1

    print(f"Updated {n_updated}/{len(charts)} charts ({n_missing} missing)")

    # Stats
    ratings = [c["rating"] for c in charts if "rating" in c]
    if ratings:
        ratings.sort()
        print(f"\nRating stats:")
        print(f"  min:    {min(ratings):.3f}")
        print(f"  max:    {max(ratings):.3f}")
        print(f"  mean:   {sum(ratings)/len(ratings):.3f}")
        print(f"  median: {ratings[len(ratings)//2]:.3f}")

    # Save
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved to {manifest_path}")


if __name__ == "__main__":
    main()
