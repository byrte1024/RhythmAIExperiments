"""Tally votes from evaluators and reveal results.

Reads votes.json and the mapping .txt files to compute which model won.
Updated for 4 models (Alpha/Beta/Gamma/Delta) with 4/3/2/1 scoring.
"""
import json
import os
import re
import unicodedata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPILED_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "compiled")
VOTES_PATH = os.path.join(SCRIPT_DIR, "votes.json")


def _normalize(s):
    """Normalize a song name for fuzzy matching."""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_mapping(song, mappings):
    """Find the mapping for a song name, with fuzzy matching."""
    if song in mappings:
        return song, mappings[song]
    for s, m in mappings.items():
        if song.lower() in s.lower() or s.lower() in song.lower():
            return s, m
    norm_song = _normalize(song)
    best_match = None
    best_score = 0
    for s, m in mappings.items():
        norm_s = _normalize(s)
        words_song = set(norm_song.split())
        words_s = set(norm_s.split())
        shared = words_song & words_s
        if len(words_song) > 0 and len(words_s) > 0:
            score = len(shared) / min(len(words_song), len(words_s))
            if score > best_score:
                best_score = score
                best_match = s
    if best_score >= 0.5:
        return best_match, mappings[best_match]
    return None, None


def load_mappings():
    """Load all song->label->model mappings."""
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

    all_votes = []
    for sr in data.get("self_rankings", []):
        if sr.get("rank_1"):
            all_votes.append({**sr, "name": "self"})
    for ev in data.get("evaluators", []):
        if ev.get("rank_1") and ev.get("song"):
            all_votes.append(ev)

    if not all_votes:
        print("No votes recorded yet. Edit votes.json to fill in rankings.")
        return

    mappings = load_mappings()
    if not mappings:
        print("No mapping files found in compiled/. Run compile_videos.py first.")
        return

    # tally: 4pts for 1st, 3pts for 2nd, 2pts for 3rd, 1pt for 4th
    model_points = {"exp14": 0, "exp44": 0, "exp45": 0, "exp53": 0}
    model_firsts = {"exp14": 0, "exp44": 0, "exp45": 0, "exp53": 0}
    model_lasts = {"exp14": 0, "exp44": 0, "exp45": 0, "exp53": 0}
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
            display_song = song.encode("ascii", "replace").decode()[:35]
            print(f"  WARNING: no mapping for song '{display_song}' (evaluator: {ev['name']})")
            continue

        ranks = [ev.get("rank_1"), ev.get("rank_2"), ev.get("rank_3"), ev.get("rank_4")]
        points = [4, 3, 2, 1]

        r1_model = mapping.get(ev["rank_1"])
        if not r1_model:
            print(f"  WARNING: invalid rank_1 '{ev['rank_1']}' for {ev['name']}")
            continue

        display_song = song.encode("ascii", "replace").decode()[:35]
        rank_str = "  ".join(f"{i+1}st={ranks[i]}({mapping.get(ranks[i], '?')})" if ranks[i] else "" for i in range(4))
        print(f"  {ev['name']:15s} | {display_song:35s} | {rank_str}")

        for i, rank_label in enumerate(ranks):
            if rank_label and rank_label in mapping:
                model = mapping[rank_label]
                if model in model_points:
                    model_points[model] += points[i]
                    if i == 0:
                        model_firsts[model] += 1
                    if i == 3:
                        model_lasts[model] += 1
        n_votes += 1

    print()
    print(f"{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  {'Model':12s}  {'Points':>8s}  {'1st':>5s}  {'4th':>5s}  {'Avg':>6s}")
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
