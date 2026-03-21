"""Tally votes from evaluators and reveal results.

Reads votes.json and the mapping .txt files to compute which model won.

Usage:
    python experiments/experiment_42ar/results/tally_votes.py
"""
import json
import os
import re
import unicodedata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPILED_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "compiled")
VOTES_PATH = os.path.join(SCRIPT_DIR, "votes.json")


def _normalize(s):
    """Normalize a song name for fuzzy matching: lowercase, ascii-only, alphanum+spaces."""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode()  # strip non-ascii
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_mapping(song, mappings):
    """Find the mapping for a song name, with fuzzy matching."""
    # exact match
    if song in mappings:
        return song, mappings[song]

    # substring match
    for s, m in mappings.items():
        if song.lower() in s.lower() or s.lower() in song.lower():
            return s, m

    # normalized fuzzy match: check if normalized names share enough words
    norm_song = _normalize(song)
    best_match = None
    best_score = 0
    for s, m in mappings.items():
        norm_s = _normalize(s)
        # count shared words
        words_song = set(norm_song.split())
        words_s = set(norm_s.split())
        shared = words_song & words_s
        # score = shared words / min word count (Jaccard-like)
        if len(words_song) > 0 and len(words_s) > 0:
            score = len(shared) / min(len(words_song), len(words_s))
            if score > best_score:
                best_score = score
                best_match = s

    if best_score >= 0.5:
        return best_match, mappings[best_match]

    return None, None


def load_mappings():
    """Load all song→label→model mappings."""
    mappings = {}
    for f in os.listdir(COMPILED_DIR):
        if not f.endswith("_mapping.txt"):
            continue
        song = f.replace("_mapping.txt", "")
        mapping = {}
        with open(os.path.join(COMPILED_DIR, f), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if " = " in line:
                    label, model = line.split(" = ", 1)
                    mapping[label.strip()] = model.strip()
        mappings[song] = mapping
    return mappings


def main():
    with open(VOTES_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # collect all votes: self (all songs) + evaluators (1 song each)
    all_votes = []

    for sr in data.get("self_rankings", []):
        if sr.get("rank_1"):
            all_votes.append({**sr, "name": "self"})

    for ev in data.get("evaluators", []):
        if ev.get("rank_1") and ev.get("song"):
            all_votes.append(ev)

    if not all_votes:
        print("No votes recorded yet. Edit votes.json to fill in rankings.")
        print("Use Alpha/Beta/Gamma for rank_1, rank_2, rank_3.")
        return

    mappings = load_mappings()

    if not mappings:
        print("No mapping files found in compiled/. Run compile_videos.py first.")
        return

    # tally: points per model (1st=3pts, 2nd=2pts, 3rd=1pt)
    model_points = {"exp14": 0, "exp35c": 0, "exp42": 0}
    model_firsts = {"exp14": 0, "exp35c": 0, "exp42": 0}
    model_lasts = {"exp14": 0, "exp35c": 0, "exp42": 0}
    n_votes = 0

    n_self = sum(1 for v in all_votes if v.get("name") == "self")
    n_other = len(all_votes) - n_self

    print(f"{'='*70}")
    print(f"  VOTE TALLY (self: {n_self} songs, evaluators: {n_other})")
    print(f"{'='*70}")
    print()

    for ev in all_votes:
        song = ev["song"]
        matched_song, mapping = _find_mapping(song, mappings)
        if matched_song:
            song = matched_song

        if not mapping:
            print(f"  WARNING: no mapping for song '{song}' (evaluator: {ev['name']})")
            continue

        r1_model = mapping.get(ev["rank_1"])
        r2_model = mapping.get(ev["rank_2"])
        r3_model = mapping.get(ev["rank_3"])

        if not r1_model:
            print(f"  WARNING: invalid rank_1 '{ev['rank_1']}' for {ev['name']}")
            continue

        display_song = song.encode("ascii", "replace").decode()[:35]
        print(f"  {ev['name']:15s} | {display_song:35s} | 1st={ev['rank_1']}({r1_model})  "
              f"2nd={ev['rank_2']}({r2_model})  3rd={ev['rank_3']}({r3_model})")

        if r1_model:
            model_points[r1_model] += 3
            model_firsts[r1_model] += 1
        if r2_model:
            model_points[r2_model] += 2
        if r3_model:
            model_points[r3_model] += 1
            model_lasts[r3_model] += 1
        n_votes += 1

    print()
    print(f"{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  {'Model':12s}  {'Points':>8s}  {'1st':>5s}  {'3rd':>5s}  {'Avg':>6s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*6}")

    for model in sorted(model_points, key=lambda m: -model_points[m]):
        pts = model_points[model]
        firsts = model_firsts[model]
        lasts = model_lasts[model]
        avg = pts / max(n_votes, 1)
        print(f"  {model:12s}  {pts:8d}  {firsts:5d}  {lasts:5d}  {avg:6.2f}")

    winner = max(model_points, key=lambda m: model_points[m])
    print(f"\n  Winner: {winner} ({model_points[winner]} points, {model_firsts[winner]} first-place votes)")

    # save results back
    data["mappings_revealed"] = True
    data["final_results"] = {
        "n_votes": n_votes,
        "points": model_points,
        "firsts": model_firsts,
        "lasts": model_lasts,
        "winner": winner,
    }
    with open(VOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {VOTES_PATH}")


if __name__ == "__main__":
    main()
