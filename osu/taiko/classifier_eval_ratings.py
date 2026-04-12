"""Evaluate how well the quality evaluator correlates with osu! user ratings.

Scores all charts in the dataset, then computes:
- Spearman correlation between model score and osu! rating
- Per-star-rating-tier correlation (controls for difficulty)
- Top/bottom ranked charts
- Pairwise accuracy on rating pairs (same criteria as training)

Usage:
    python classifier_eval_ratings.py --checkpoint runs/eval_experiment_66_1_p2/checkpoints/best.pt --dataset taiko_v2
"""
import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from collections import defaultdict

from classifier_model import ChartQualityEvaluator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_FRAMES = 2000
MAX_EVENTS = 256


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = ChartQualityEvaluator(
        d_model=ckpt_args.get("d_model", 256),
        n_layers=ckpt_args.get("n_layers", 6),
        n_heads=ckpt_args.get("n_heads", 8),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    eval_step = ckpt.get("eval_step", "?")
    print(f"Loaded evaluator from eval {eval_step}")
    return model


def score_chart_windows(model, mel, events, star_rating, device, n_windows=8):
    total_frames = mel.shape[1]
    if total_frames <= WINDOW_FRAMES:
        starts = [0]
    else:
        starts = np.linspace(0, total_frames - WINDOW_FRAMES, n_windows, dtype=int)
        starts = sorted(set(starts))

    scores = []
    with torch.no_grad():
        for start in starts:
            end = start + WINDOW_FRAMES
            mel_w = mel[:, start:min(total_frames, end)].astype(np.float32)
            if mel_w.shape[1] < WINDOW_FRAMES:
                mel_w = np.pad(mel_w, ((0, 0), (0, WINDOW_FRAMES - mel_w.shape[1])))

            mask = (events >= start) & (events < end)
            evt_w = events[mask].astype(np.int64) - start
            n_evt = min(len(evt_w), MAX_EVENTS)
            evt_arr = np.zeros(MAX_EVENTS, dtype=np.int64)
            evt_mask = np.ones(MAX_EVENTS, dtype=bool)
            if n_evt > 0:
                evt_arr[:n_evt] = evt_w[:n_evt]
                evt_mask[:n_evt] = False

            mel_t = torch.from_numpy(mel_w).unsqueeze(0).to(device)
            evt_t = torch.from_numpy(evt_arr).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(evt_mask).unsqueeze(0).to(device)
            star_t = torch.tensor([star_rating], dtype=torch.float32, device=device)

            score = model(mel_t, evt_t, mask_t, star_t).item()
            scores.append(score)

    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="taiko_v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    ds_dir = os.path.join(SCRIPT_DIR, "datasets", args.dataset)
    with open(os.path.join(ds_dir, "manifest.json")) as f:
        manifest = json.load(f)

    model = load_model(args.checkpoint, args.device)
    mel_dir = os.path.join(ds_dir, "mels")
    evt_dir = os.path.join(ds_dir, "events")
    mel_cache = {}

    # score all charts
    results = []
    for chart in tqdm(manifest["charts"], desc="Scoring charts"):
        mel_file = chart["mel_file"]
        if mel_file not in mel_cache:
            mel_cache[mel_file] = np.load(os.path.join(mel_dir, mel_file), mmap_mode="r")
        mel = mel_cache[mel_file]
        events = np.load(os.path.join(evt_dir, chart["event_file"]))
        star = chart.get("star_rating", 4.0)

        score = score_chart_windows(model, mel, events, star, args.device, args.n_windows)
        results.append({
            "chart_id": chart.get("chart_id", ""),
            "beatmapset_id": chart.get("beatmapset_id", ""),
            "star_rating": star,
            "rating": chart.get("rating", None),
            "quality_score": score,
        })

    # ── overall correlation ──
    rated = [(r["quality_score"], r["rating"], r["star_rating"], r["chart_id"], r["beatmapset_id"])
             for r in results if r["rating"] is not None]
    print(f"\n{'='*60}")
    print(f"Scored {len(results)} charts, {len(rated)} with ratings")

    qs = np.array([r[0] for r in rated])
    rs = np.array([r[1] for r in rated])
    rho, p = spearmanr(qs, rs)
    r_p, p_p = pearsonr(qs, rs)
    print(f"\nOverall correlation (all charts):")
    print(f"  Spearman: rho={rho:.4f}, p={p:.2e}")
    print(f"  Pearson:  r={r_p:.4f}, p={p_p:.2e}")

    # ── per-beatmapset (one score per set, avoids duplicate ratings) ──
    bset_scores = defaultdict(list)
    bset_ratings = {}
    for q, r, sr, cid, bset in rated:
        bset_scores[bset].append(q)
        bset_ratings[bset] = r
    bset_mean_q = {bset: np.mean(scores) for bset, scores in bset_scores.items()}
    bset_list = [(bset_mean_q[b], bset_ratings[b]) for b in bset_mean_q]
    bq = np.array([x[0] for x in bset_list])
    br = np.array([x[1] for x in bset_list])
    rho_b, p_b = spearmanr(bq, br)
    print(f"\nPer-beatmapset correlation ({len(bset_list)} sets):")
    print(f"  Spearman: rho={rho_b:.4f}, p={p_b:.2e}")

    # ── per star-rating tier ──
    print(f"\nPer star-rating tier:")
    tiers = [(0, 2, "<2*"), (2, 3, "2-3*"), (3, 4, "3-4*"), (4, 5, "4-5*"), (5, 6, "5-6*"), (6, 99, "6+*")]
    for lo, hi, name in tiers:
        tier_data = [(q, r) for q, r, sr, _, _ in rated if lo <= sr < hi]
        if len(tier_data) < 10:
            print(f"  {name:8s}: n={len(tier_data):4d} (too few)")
            continue
        tq = np.array([x[0] for x in tier_data])
        tr = np.array([x[1] for x in tier_data])
        rho_t, p_t = spearmanr(tq, tr)
        print(f"  {name:8s}: n={len(tier_data):4d}, spearman={rho_t:+.4f}, p={p_t:.2e}")

    # ── pairwise accuracy on rating pairs (same criteria as training) ──
    print(f"\nPairwise accuracy on cross-set rating pairs:")
    print(f"  (star_rating within 0.5, rating gap >= 1.0)")
    rng = np.random.default_rng(42)
    n_test_pairs = 5000
    correct = 0
    total = 0
    # index by beatmapset
    bset_charts = defaultdict(list)
    for r in results:
        if r["rating"] is not None and r["star_rating"] is not None:
            bset_charts[r["beatmapset_id"]].append(r)

    bset_list_for_pairs = list(bset_charts.keys())
    for _ in range(n_test_pairs):
        i, j = rng.choice(len(bset_list_for_pairs), size=2, replace=False)
        bset_a, bset_b = bset_list_for_pairs[i], bset_list_for_pairs[j]
        ca = rng.choice(bset_charts[bset_a])
        cb = rng.choice(bset_charts[bset_b])

        if abs(ca["star_rating"] - cb["star_rating"]) > 0.5:
            continue
        gap = abs(ca["rating"] - cb["rating"])
        if gap < 1.0:
            continue

        if ca["rating"] > cb["rating"]:
            better, worse = ca, cb
        else:
            better, worse = cb, ca

        if better["quality_score"] > worse["quality_score"]:
            correct += 1
        total += 1

    if total > 0:
        print(f"  {correct}/{total} = {correct/total:.1%} (baseline: 50%)")
    else:
        print(f"  No valid pairs found")

    # ── top/bottom 10 ──
    rated_sorted = sorted(rated, key=lambda x: x[0], reverse=True)
    print(f"\nTop 10 (highest quality score):")
    for q, r, sr, cid, bset in rated_sorted[:10]:
        print(f"  {q:+7.2f} | rating={r:.1f} | *{sr:.1f} | {cid[:70]}")
    print(f"\nBottom 10 (lowest quality score):")
    for q, r, sr, cid, bset in rated_sorted[-10:]:
        print(f"  {q:+7.2f} | rating={r:.1f} | *{sr:.1f} | {cid[:70]}")

    # ── save ──
    output = args.output or f"quality_scores_{args.dataset}.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} scores to {output}")


if __name__ == "__main__":
    main()
