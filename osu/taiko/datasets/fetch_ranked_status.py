"""Fetch ranked status for all beatmaps and update manifest.json.

Uses osu! API v2 (OAuth client credentials).

Usage:
    cd osu/taiko/datasets
    python fetch_ranked_status.py taiko_v2
"""

import argparse
import json
import os
import time

import requests

# osu! API v2 OAuth
CLIENT_ID = "51590"
CLIENT_SECRET = "5Nk6wGHIwugkLNpbDEhMv1UVSfZsCyrZQkB29X9r"
TOKEN_URL = "https://osu.ppy.sh/oauth/token"
API_BASE = "https://osu.ppy.sh/api/v2"


def get_token():
    resp = requests.post(TOKEN_URL, json={
        "client_id": int(CLIENT_ID),
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": "public",
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_beatmaps_batch(token, beatmap_ids):
    """Fetch up to 50 beatmaps in one request."""
    headers = {"Authorization": f"Bearer {token}"}
    params = [("ids[]", str(bid)) for bid in beatmap_ids]
    resp = requests.get(f"{API_BASE}/beatmaps", headers=headers, params=params)
    if resp.status_code == 429:
        print("  Rate limited, waiting 60s...")
        time.sleep(60)
        resp = requests.get(f"{API_BASE}/beatmaps", headers=headers, params=params)
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

    # Collect unique beatmap IDs
    all_ids = set()
    for c in charts:
        bid = c.get("beatmap_id")
        if bid:
            all_ids.add(str(bid))
    print(f"Unique beatmap IDs: {len(all_ids)}")

    # Check how many already have ranked_status
    already = sum(1 for c in charts if "ranked_status" in c)
    if already > 0:
        print(f"Already have ranked_status: {already}/{len(charts)}")

    # Get token
    print("Authenticating...")
    token = get_token()
    print("  OK")

    # Batch fetch (50 per request)
    id_list = sorted(all_ids)
    batch_size = 50
    id_to_status = {}
    id_to_info = {}

    n_batches = (len(id_list) + batch_size - 1) // batch_size
    print(f"Fetching {len(id_list)} beatmaps in {n_batches} batches...")

    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        batch_num = i // batch_size + 1

        try:
            data = fetch_beatmaps_batch(token, batch)
        except Exception as e:
            print(f"  Batch {batch_num}: ERROR {e}")
            continue

        # API returns {"beatmaps": [...]} or just [...]
        beatmaps = data if isinstance(data, list) else data.get("beatmaps", [])

        for bm in beatmaps:
            bid = str(bm["id"])
            status = bm.get("status", "unknown")
            ranked = bm.get("ranked", -3)
            id_to_status[bid] = status
            id_to_info[bid] = {
                "ranked_status": status,
                "ranked_int": ranked,
                "star_rating": bm.get("difficulty_rating", 0),
                "playcount": bm.get("playcount", 0),
                "passcount": bm.get("passcount", 0),
                "favourite_count": bm.get("beatmapset", {}).get("favourite_count", 0),
                "play_count_set": bm.get("beatmapset", {}).get("play_count", 0),
            }

        found = len(beatmaps)
        missing = len(batch) - found
        if missing > 0:
            # Mark missing IDs
            found_ids = {str(bm["id"]) for bm in beatmaps}
            for bid in batch:
                if bid not in found_ids:
                    id_to_status[bid] = "not_found"
                    id_to_info[bid] = {"ranked_status": "not_found", "ranked_int": -99}

        print(f"  Batch {batch_num}/{n_batches}: {found} found, {missing} missing")

        # Gentle rate limiting
        if batch_num % 10 == 0:
            time.sleep(1)

    # Update manifest
    n_updated = 0
    for c in charts:
        bid = str(c.get("beatmap_id", ""))
        if bid in id_to_info:
            for k, v in id_to_info[bid].items():
                c[k] = v
            n_updated += 1

    print(f"\nUpdated {n_updated}/{len(charts)} charts")

    # Summary
    status_counts = {}
    for c in charts:
        s = c.get("ranked_status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
    print("\nRanked status distribution:")
    for s, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {count} ({count/len(charts)*100:.1f}%)")

    # Save
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved to {manifest_path}")


if __name__ == "__main__":
    main()
