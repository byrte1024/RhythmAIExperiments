"""Overlap analysis: S1 (audio proposer) vs S2 (context predictor).

For each val sample, compare S1 and S2 predictions. Classify into 4 quadrants:
  - Both correct
  - S1 only correct
  - S2 only correct
  - Neither correct

S1 prediction modes:
  - MAX: highest confidence token in B_PRED range
  - FIRST_THRESH: first token above threshold
  - ORACLE_THRESH: best token above threshold (closest to target)

Usage:
    cd osu/taiko
    python experiments/experiment_65_s2/overlap_analysis.py \
        --s1-checkpoint runs/detect_experiment_58/checkpoints/best.pt \
        --s2-checkpoint runs/s2_experiment_65/checkpoints/best.pt
"""

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, TAIKO_DIR)

from detection_s2_model import ContextPredictor
from detection_s2_train import S2Dataset, N_CLASSES, B_PRED, C_EVENTS, MIN_CURSOR_BIN
from detection_train import OnsetDataset

# S1 needs the full model to get proposal logits
A_BINS = 500
B_BINS = 500


def load_s1_model(checkpoint_path, device):
    """Load S1 proposer from a ProposeSelectDetector checkpoint."""
    from detection_model import ProposeSelectDetector, EventEmbeddingDetector

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    state_keys = set(ckpt["model"].keys())

    has_propose_select = "proposer_head.0.weight" in state_keys
    if not has_propose_select:
        print("WARNING: checkpoint is not ProposeSelectDetector, using full model logits instead")
        return None, None

    # Detect architecture
    event_proj_key = next((k for k in state_keys if "event_proj.0.weight" in k), None)
    has_gap_ratios = False
    if event_proj_key:
        w = ckpt["model"][event_proj_key]
        if w.shape[1] > ckpt_args.get("d_model", 384) * 3:
            has_gap_ratios = True

    n_proposer = sum(1 for k in state_keys if k.startswith("proposer_layers.") and ".self_attn.in_proj_weight" in k)
    n_selector = sum(1 for k in state_keys if k.startswith("selector_layers.") and ".self_attn.in_proj_weight" in k)

    a_bins = ckpt_args.get("a_bins", 500)
    b_bins = ckpt_args.get("b_bins", 500)
    b_pred = ckpt_args.get("b_pred", 0)
    if b_pred <= 0:
        b_pred = b_bins
    n_classes = b_pred + 1

    model = ProposeSelectDetector(
        n_mels=80, d_model=ckpt_args.get("d_model", 384),
        n_proposer_layers=n_proposer, n_selector_layers=n_selector,
        n_heads=ckpt_args.get("n_heads", 8), n_classes=n_classes,
        max_events=C_EVENTS, gap_ratios=has_gap_ratios,
        a_bins=a_bins, b_bins=b_bins,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()

    return model, {"a_bins": a_bins, "b_bins": b_bins, "b_pred": b_pred, "n_classes": n_classes}


def load_s2_model(checkpoint_path, device):
    """Load S2 ContextPredictor."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]

    model = ContextPredictor(
        d_model=ckpt_args.get("d_model", 256),
        n_gru_layers=ckpt_args.get("n_gru_layers", 4),
        n_classes=N_CLASSES,
        max_events=C_EVENTS,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model


def is_hit(pred, target, stop=250):
    """Check if prediction is a HIT (<=3% or ±1 frame)."""
    if target == stop:
        return pred == stop
    if pred == stop:
        return False
    pct_err = abs(pred / (target + 1) - 1)
    frame_err = abs(pred - target)
    return pct_err <= 0.03 or frame_err <= 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-checkpoint", required=True)
    parser.add_argument("--s2-checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    ds_dir = os.path.join(TAIKO_DIR, "datasets", "taiko_v2")
    with open(os.path.join(ds_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Same val split as training
    charts = manifest["charts"]
    song_to_charts = {}
    for i, c in enumerate(charts):
        sid = c.get("beatmapset_id", str(i))
        song_to_charts.setdefault(sid, []).append(i)
    songs = list(song_to_charts.keys())
    random.seed(42)
    random.shuffle(songs)
    n_val = max(1, int(len(songs) * 0.1))
    val_songs = set(songs[:n_val])
    val_idx = [i for i, c in enumerate(charts) if c.get("beatmapset_id", str(i)) in val_songs]

    print(f"Val charts: {len(val_idx)}")

    # ── Load S2 ──
    print(f"Loading S2: {args.s2_checkpoint}")
    s2_model = load_s2_model(args.s2_checkpoint, args.device)
    s2_ds = S2Dataset(manifest, ds_dir, val_idx, augment=False)
    s2_loader = DataLoader(s2_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    print(f"S2 val samples: {len(s2_ds)}")

    # Collect S2 predictions
    print("Running S2...")
    s2_preds = []
    s2_probs_list = []
    s2_targets = []
    with torch.no_grad():
        for gaps, ratios, mask, cond, target in tqdm(s2_loader, desc="S2"):
            gaps, ratios, mask, cond = (
                gaps.to(args.device), ratios.to(args.device),
                mask.to(args.device), cond.to(args.device),
            )
            logits = s2_model(gaps, ratios, mask, cond)
            probs = F.softmax(logits, dim=-1)
            s2_preds.append(logits.argmax(dim=-1).cpu().numpy())
            s2_probs_list.append(probs.cpu().numpy())
            s2_targets.append(target.numpy())

    s2_preds = np.concatenate(s2_preds)
    s2_probs = np.concatenate(s2_probs_list)
    s2_targets = np.concatenate(s2_targets)
    print(f"  S2 HIT: {np.mean([is_hit(p, t) for p, t in zip(s2_preds, s2_targets)]):.1%}")

    # ── Load S1 ──
    print(f"\nLoading S1: {args.s1_checkpoint}")
    s1_model, s1_cfg = load_s1_model(args.s1_checkpoint, args.device)

    if s1_model is None:
        print("ERROR: S1 checkpoint is not ProposeSelectDetector")
        return

    # S1 needs mel data — use the main OnsetDataset
    import detection_train as dt
    dt.A_BINS = s1_cfg["a_bins"]
    dt.B_BINS = s1_cfg["b_bins"]
    dt.N_CLASSES = s1_cfg["n_classes"]
    dt.WINDOW = dt.A_BINS + dt.B_BINS

    s1_ds = OnsetDataset(manifest, ds_dir, val_idx, augment=False)
    s1_loader = DataLoader(s1_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    assert len(s1_ds) == len(s2_ds), f"Dataset size mismatch: S1={len(s1_ds)} S2={len(s2_ds)}"

    # Collect S1 proposal confidences
    print("Running S1...")
    s1_proposal_confs = []  # (N, n_tokens) per-token sigmoid
    s1_targets = []
    b_pred = s1_cfg["b_pred"]
    cursor_token = s1_cfg["a_bins"] // 4

    with torch.no_grad():
        for batch in tqdm(s1_loader, desc="S1"):
            mel, evt, mask, cond = batch[0].to(args.device), batch[1].to(args.device), \
                                    batch[2].to(args.device), batch[3].to(args.device)
            target = batch[4]

            output = s1_model(mel, evt, mask, cond)
            # ProposeSelectDetector returns (logits, proposal_logits)
            proposal_logits = output[1]  # (B, n_tokens)
            proposal_conf = torch.sigmoid(proposal_logits).cpu().numpy()

            s1_proposal_confs.append(proposal_conf)
            if target.dim() == 1:
                s1_targets.append(target.numpy())
            else:
                s1_targets.append(target[:, 0].numpy())  # multi-onset: use o1

    s1_confs = np.concatenate(s1_proposal_confs)  # (N, n_tokens)
    s1_targets = np.concatenate(s1_targets)

    # Verify targets match
    assert np.array_equal(s1_targets, s2_targets), "Target mismatch between S1 and S2!"
    targets = s1_targets
    N = len(targets)
    stop = N_CLASSES - 1

    print(f"\nSamples: {N}")
    print(f"Non-STOP: {(targets != stop).sum()} ({(targets != stop).mean():.1%})")

    # ── S1 prediction modes ──
    # Map tokens back to bins: token i covers bins [i*4 - a_bins, i*4 - a_bins + 4)
    # For B_PRED range: bins 0..b_pred-1 correspond to tokens cursor_token..cursor_token+b_pred//4
    n_tokens = s1_confs.shape[1]
    pred_start_token = cursor_token  # token for bin 0
    pred_end_token = min(cursor_token + (b_pred + 3) // 4, n_tokens)
    pred_tokens = pred_end_token - pred_start_token

    # Extract prediction-range confidences and map to bins
    s1_pred_confs = s1_confs[:, pred_start_token:pred_end_token]  # (N, pred_tokens)

    def token_to_bin(token_idx):
        """Convert token index (within pred range) to bin."""
        return token_idx * 4

    # MODE 1: MAX — highest confidence token
    s1_max_tokens = s1_pred_confs.argmax(axis=1)
    s1_max_preds = np.array([token_to_bin(t) for t in s1_max_tokens])

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = {}
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    def compute_quadrants(s1_p, s2_p, targets, label):
        """Compute overlap quadrants."""
        both = 0; s1_only = 0; s2_only = 0; neither = 0
        n_ns = 0
        for i in range(len(targets)):
            t = targets[i]
            h1 = is_hit(int(s1_p[i]), int(t))
            h2 = is_hit(int(s2_p[i]), int(t))
            if t != stop:
                n_ns += 1
                if h1 and h2: both += 1
                elif h1: s1_only += 1
                elif h2: s2_only += 1
                else: neither += 1

        r = {
            "label": label,
            "n_samples": n_ns,
            "both_correct": both,
            "s1_only": s1_only,
            "s2_only": s2_only,
            "neither": neither,
            "both_pct": both / max(n_ns, 1),
            "s1_only_pct": s1_only / max(n_ns, 1),
            "s2_only_pct": s2_only / max(n_ns, 1),
            "neither_pct": neither / max(n_ns, 1),
            "s1_hit": (both + s1_only) / max(n_ns, 1),
            "s2_hit": (both + s2_only) / max(n_ns, 1),
            "union_hit": (both + s1_only + s2_only) / max(n_ns, 1),
            "theoretical_ceiling": (both + s1_only + s2_only) / max(n_ns, 1),
        }
        return r

    # ── MAX mode ──
    r = compute_quadrants(s1_max_preds, s2_preds, targets, "S1_MAX")
    results["S1_MAX"] = r
    print(f"\n{'='*60}")
    print(f"S1 MAX (highest confidence token)")
    print(f"  S1 HIT: {r['s1_hit']:.1%}  S2 HIT: {r['s2_hit']:.1%}")
    print(f"  Both:   {r['both_pct']:.1%}  ({r['both_correct']})")
    print(f"  S1only: {r['s1_only_pct']:.1%}  ({r['s1_only']})")
    print(f"  S2only: {r['s2_only_pct']:.1%}  ({r['s2_only']})")
    print(f"  Neither:{r['neither_pct']:.1%}  ({r['neither']})")
    print(f"  Union (theoretical ceiling): {r['theoretical_ceiling']:.1%}")

    # ── THRESHOLD modes ──
    for thresh in thresholds:
        # FIRST_THRESH: first token above threshold in B_PRED range
        s1_first = np.full(N, stop, dtype=np.int64)
        # ORACLE_THRESH: best (closest to target) token above threshold
        s1_oracle = np.full(N, stop, dtype=np.int64)

        for i in range(N):
            above = np.where(s1_pred_confs[i] >= thresh)[0]
            if len(above) > 0:
                # First
                s1_first[i] = token_to_bin(above[0])
                # Oracle: closest to target
                t = targets[i]
                if t != stop:
                    bins_above = np.array([token_to_bin(a) for a in above])
                    closest_idx = np.argmin(np.abs(bins_above - t))
                    s1_oracle[i] = bins_above[closest_idx]
                else:
                    s1_oracle[i] = stop

        # FIRST
        r_first = compute_quadrants(s1_first, s2_preds, targets, f"S1_FIRST_{thresh}")
        results[f"S1_FIRST_{thresh}"] = r_first

        # ORACLE
        r_oracle = compute_quadrants(s1_oracle, s2_preds, targets, f"S1_ORACLE_{thresh}")
        results[f"S1_ORACLE_{thresh}"] = r_oracle

        print(f"\n--- Threshold {thresh} ---")
        print(f"  FIRST:  S1={r_first['s1_hit']:.1%}  Union={r_first['theoretical_ceiling']:.1%}  S2only={r_first['s2_only_pct']:.1%}")
        print(f"  ORACLE: S1={r_oracle['s1_hit']:.1%}  Union={r_oracle['theoretical_ceiling']:.1%}  S2only={r_oracle['s2_only_pct']:.1%}")

    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<20} {'S1 HIT':>8} {'S2 HIT':>8} {'Both':>8} {'S1only':>8} {'S2only':>8} {'Neither':>8} {'Union':>8}")
    print("-" * 80)
    for key in sorted(results.keys()):
        r = results[key]
        print(f"{r['label']:<20} {r['s1_hit']:>7.1%} {r['s2_hit']:>7.1%} {r['both_pct']:>7.1%} {r['s1_only_pct']:>7.1%} {r['s2_only_pct']:>7.1%} {r['neither_pct']:>7.1%} {r['theoretical_ceiling']:>7.1%}")

    # Save
    with open(os.path.join(output_dir, "overlap_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/overlap_results.json")

    # ── Graphs ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Quadrant pie chart for MAX mode
    r = results["S1_MAX"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie
    ax = axes[0]
    sizes = [r["both_correct"], r["s1_only"], r["s2_only"], r["neither"]]
    labels = [f"Both\n{r['both_pct']:.1%}", f"S1 only\n{r['s1_only_pct']:.1%}",
              f"S2 only\n{r['s2_only_pct']:.1%}", f"Neither\n{r['neither_pct']:.1%}"]
    colors = ["#6bc46d", "#4a90d9", "#e6a817", "#eb4528"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="", startangle=90)
    ax.set_title(f"S1 (MAX) vs S2 Overlap\nS1={r['s1_hit']:.1%}  S2={r['s2_hit']:.1%}  Union={r['theoretical_ceiling']:.1%}")

    # Bar chart: union ceiling by mode
    ax = axes[1]
    modes = []
    ceilings = []
    s2_onlys = []
    for key in ["S1_MAX"] + [f"S1_FIRST_{t}" for t in thresholds] + [f"S1_ORACLE_{t}" for t in thresholds]:
        if key in results:
            r2 = results[key]
            modes.append(r2["label"].replace("S1_", ""))
            ceilings.append(r2["theoretical_ceiling"] * 100)
            s2_onlys.append(r2["s2_only_pct"] * 100)

    x = np.arange(len(modes))
    ax.bar(x, ceilings, color="#6bc46d", alpha=0.8, label="Union ceiling")
    ax.bar(x, s2_onlys, color="#e6a817", alpha=0.8, label="S2-only contribution")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of samples")
    ax.set_title("Theoretical ceiling (S1+S2 union) by S1 mode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "overlap_analysis.png"), dpi=150)
    plt.close(fig)
    print(f"Saved graph to {output_dir}/overlap_analysis.png")


if __name__ == "__main__":
    main()
