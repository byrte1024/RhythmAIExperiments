"""Inference script for chart quality evaluator (exp 66-1).

Score a chart's quality by evaluating multiple windows and averaging.
Can also compare two charts pairwise.

Usage:
    # Score a single chart
    python classifier_inference.py --checkpoint runs/eval_exp66/checkpoints/best.pt \
        --mel path/to/mel.npy --events path/to/events.npy --star-rating 4.5

    # Score all charts in a dataset
    python classifier_inference.py --checkpoint runs/eval_exp66/checkpoints/best.pt \
        --dataset taiko_v2 --output scores.json

    # Compare generated vs ground truth
    python classifier_inference.py --checkpoint runs/eval_exp66/checkpoints/best.pt \
        --compare-csv generated.csv --compare-gt events.npy --mel path/to/mel.npy --star-rating 4.5
"""
import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from classifier_model import ChartQualityEvaluator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_FRAMES = 2000
MAX_EVENTS = 256
BIN_MS = 5.0


def load_model(checkpoint_path, device):
    """Load evaluator from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    model = ChartQualityEvaluator(
        d_model=ckpt_args.get("d_model", 256),
        n_layers=ckpt_args.get("n_layers", 6),
        n_heads=ckpt_args.get("n_heads", 8),
        dropout=0.0,  # no dropout at inference
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    eval_step = ckpt.get("eval_step", "?")
    val_acc = ckpt.get("val_metrics", {}).get("pair_accuracy", "?")
    print(f"Loaded evaluator from eval {eval_step}, pair_accuracy={val_acc}")
    return model


def extract_windows(mel, events, n_windows=16):
    """Extract uniformly spaced windows from a full mel + events.

    Returns list of (mel_window, events_in_window, event_mask) tuples.
    """
    total_frames = mel.shape[1]
    windows = []

    if total_frames <= WINDOW_FRAMES:
        starts = [0]
    else:
        # uniformly spaced, avoiding going past end
        starts = np.linspace(0, total_frames - WINDOW_FRAMES, n_windows, dtype=int)
        starts = sorted(set(starts))  # deduplicate if song is short

    for start in starts:
        end = start + WINDOW_FRAMES
        mel_w = mel[:, start:min(total_frames, end)].astype(np.float32)
        if mel_w.shape[1] < WINDOW_FRAMES:
            pad = WINDOW_FRAMES - mel_w.shape[1]
            mel_w = np.pad(mel_w, ((0, 0), (0, pad)), mode="constant")

        # events in window
        mask = (events >= start) & (events < end)
        evt_w = events[mask].astype(np.int64) - start
        n_evt = min(len(evt_w), MAX_EVENTS)
        evt_arr = np.zeros(MAX_EVENTS, dtype=np.int64)
        evt_mask = np.ones(MAX_EVENTS, dtype=bool)
        if n_evt > 0:
            evt_arr[:n_evt] = evt_w[:n_evt]
            evt_mask[:n_evt] = False

        windows.append((mel_w, evt_arr, evt_mask))

    return windows


def score_chart(model, mel, events, star_rating, device, n_windows=16):
    """Score a single chart. Returns (mean_score, per_window_scores)."""
    windows = extract_windows(mel, events, n_windows)
    scores = []

    with torch.no_grad():
        for mel_w, evt_arr, evt_mask in windows:
            mel_t = torch.from_numpy(mel_w).unsqueeze(0).to(device)
            evt_t = torch.from_numpy(evt_arr).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(evt_mask).unsqueeze(0).to(device)
            star_t = torch.tensor([star_rating], dtype=torch.float32, device=device)

            score = model(mel_t, evt_t, mask_t, star_t).item()
            scores.append(score)

    return float(np.mean(scores)), scores


def score_dataset(model, dataset_name, device, n_windows=16, output_path=None):
    """Score all charts in a dataset."""
    ds_dir = os.path.join(SCRIPT_DIR, "datasets", dataset_name)
    manifest_path = os.path.join(ds_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    mel_dir = os.path.join(ds_dir, "mels")
    evt_dir = os.path.join(ds_dir, "events")

    results = []
    mel_cache = {}

    for chart in tqdm(manifest["charts"], desc="Scoring charts"):
        mel_file = chart["mel_file"]
        if mel_file not in mel_cache:
            mel_cache[mel_file] = np.load(os.path.join(mel_dir, mel_file), mmap_mode="r")
        mel = mel_cache[mel_file]
        events = np.load(os.path.join(evt_dir, chart["event_file"]))
        star = chart.get("star_rating", 4.0)

        mean_score, window_scores = score_chart(model, mel, events, star, device, n_windows)

        results.append({
            "chart_id": chart.get("chart_id", ""),
            "beatmapset_id": chart.get("beatmapset_id", ""),
            "star_rating": star,
            "rating": chart.get("rating", None),
            "quality_score": mean_score,
            "n_windows": len(window_scores),
        })

    # sort by score
    results.sort(key=lambda x: x["quality_score"], reverse=True)

    # summary stats
    scores = [r["quality_score"] for r in results]
    print(f"\nScored {len(results)} charts")
    print(f"  mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
    print(f"  min={np.min(scores):.3f}, max={np.max(scores):.3f}")

    # correlation with rating (if available)
    ratings = [(r["quality_score"], r["rating"]) for r in results if r["rating"] is not None]
    if len(ratings) > 10:
        from scipy.stats import spearmanr
        qs, rs = zip(*ratings)
        rho, p = spearmanr(qs, rs)
        print(f"  Spearman(quality_score, rating): rho={rho:.3f}, p={p:.2e}")

    # top/bottom 5
    print("\nTop 5:")
    for r in results[:5]:
        print(f"  {r['quality_score']:+.3f} | ★{r['star_rating']:.1f} | "
              f"rating={r.get('rating', '?')} | {r['chart_id'][:80]}")
    print("\nBottom 5:")
    for r in results[-5:]:
        print(f"  {r['quality_score']:+.3f} | ★{r['star_rating']:.1f} | "
              f"rating={r.get('rating', '?')} | {r['chart_id'][:80]}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved scores to {output_path}")

    return results


def compare_charts(model, mel, events_a, events_b, star_rating, device, n_windows=16):
    """Compare two charts on the same audio. Returns (score_a, score_b, a_wins)."""
    score_a, _ = score_chart(model, mel, events_a, star_rating, device, n_windows)
    score_b, _ = score_chart(model, mel, events_b, star_rating, device, n_windows)
    return score_a, score_b, score_a > score_b


def load_events_from_csv(csv_path):
    """Load events from a prediction CSV (time_ms column → bin indices)."""
    import csv
    events = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_ms = float(row.get("time_ms", row.get("time", 0)))
            events.append(int(round(time_ms / BIN_MS)))
    return np.array(events, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Chart quality evaluator inference")
    parser.add_argument("--checkpoint", required=True, help="Path to evaluator checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-windows", type=int, default=16, help="Windows per chart")

    # single chart scoring
    parser.add_argument("--mel", help="Path to mel spectrogram .npy")
    parser.add_argument("--events", help="Path to events .npy")
    parser.add_argument("--star-rating", type=float, default=4.0)

    # dataset scoring
    parser.add_argument("--dataset", help="Dataset name (e.g. taiko_v2)")
    parser.add_argument("--output", help="Output JSON path for dataset scoring")

    # comparison mode
    parser.add_argument("--compare-csv", help="Generated chart CSV to compare")
    parser.add_argument("--compare-gt", help="Ground truth events .npy to compare against")

    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)

    if args.dataset:
        # score full dataset
        output = args.output or f"{args.dataset}_quality_scores.json"
        score_dataset(model, args.dataset, args.device, args.n_windows, output)

    elif args.compare_csv and args.compare_gt and args.mel:
        # compare generated vs ground truth
        mel = np.load(args.mel)
        events_gen = load_events_from_csv(args.compare_csv)
        events_gt = np.load(args.compare_gt)

        score_gen, score_gt, gt_wins = compare_charts(
            model, mel, events_gen, events_gt, args.star_rating, args.device, args.n_windows)

        print(f"Generated: {score_gen:+.3f}")
        print(f"Ground truth: {score_gt:+.3f}")
        print(f"Winner: {'ground truth' if gt_wins else 'generated'} "
              f"(delta={abs(score_gt - score_gen):.3f})")

    elif args.mel and args.events:
        # score single chart
        mel = np.load(args.mel)
        events = np.load(args.events)

        mean_score, window_scores = score_chart(
            model, mel, events, args.star_rating, args.device, args.n_windows)

        print(f"Quality score: {mean_score:+.3f}")
        print(f"Per-window: {['%.3f' % s for s in window_scores]}")

    else:
        parser.print_help()
        print("\nProvide --dataset, --mel+--events, or --compare-csv+--compare-gt+--mel")


if __name__ == "__main__":
    main()
