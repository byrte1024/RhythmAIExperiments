"""Compare evaluator scores against human rankings from exp 42-AR and 53-AR.

Scores the same songs x models that humans ranked, then checks
whether the evaluator's ranking matches human preference.
Reports total, self-only, and evaluator-only agreement.

Usage:
    python classifier_eval_human.py \
        --checkpoint runs/eval_experiment_66_2/checkpoints/best.pt \
        --checkpoint2 runs/eval_experiment_66_1/checkpoints/best.pt
"""
import os
import json
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau

from classifier_model import ChartQualityEvaluator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_MS = 5.0
WINDOW_FRAMES = 2000
MAX_EVENTS = 256


# ──────────────────────────────────────────────
#  Model loading and scoring
# ──────────────────────────────────────────────

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
    return model


def load_csv_events(csv_path):
    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("time_ms"):
                continue
            parts = line.strip().split(",")
            if parts:
                try:
                    events.append(int(round(float(parts[0]) / BIN_MS)))
                except ValueError:
                    continue
    return np.array(events, dtype=np.int64)


def score_chart(model, mel, events, star_rating, device, n_windows=16):
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

            s = model(mel_t, evt_t, mask_t, star_t).item()
            scores.append(s)

    return float(np.mean(scores))


# ──────────────────────────────────────────────
#  Exp 53-AR data loading
# ──────────────────────────────────────────────

EXP53AR_DIR = os.path.join(SCRIPT_DIR, "experiments", "experiment_53ar")

SONGS_53 = [
    ("01", "arashi_five", "Arashi - Five"),
    ("02", "sakurazaka46_growing_up_train", "Sakurazaka46 - The growing up train"),
    ("03", "camellia_denkoh_sekka", "Camellia - Denkoh Sekka"),
    ("04", "redalice_tpazolite_xterfusion", "REDALiCE x t+pazolite - Xterfusion"),
    ("05", "courtney_barnett_stay_in_your_lane", "Courtney Barnett - Stay in Your Lane"),
    ("06", "mon_rovia_heavy_foot", "Mon Rovia - Heavy Foot"),
    ("07", "roccow_when_the_leaves_leaf", "RoccoW - When the Leaves Leaf"),
    ("08", "supernovayuli_one_more_time", "supernovayuli - One More Time"),
    ("09", "conan_gray_the_best", "Conan Gray - The Best"),
    ("10", "miley_cyrus_younger_you", "Miley Cyrus - Younger You"),
]
MODELS_53 = ["exp14", "exp44", "exp45", "exp53"]


def load_53ar_mappings():
    mappings = {}
    compiled_dir = os.path.join(EXP53AR_DIR, "compiled")
    for song_num, _, _ in SONGS_53:
        for f in os.listdir(compiled_dir):
            if f.startswith(f"{song_num}_") and f.endswith("_mapping.txt"):
                m = {}
                with open(os.path.join(compiled_dir, f)) as fh:
                    for line in fh:
                        if " = " in line:
                            label, model = line.strip().split(" = ")
                            m[label.strip()] = model.strip()
                mappings[song_num] = m
    return mappings


def load_53ar_votes():
    with open(os.path.join(EXP53AR_DIR, "results", "votes.json")) as f:
        return json.load(f)


def score_53ar(model, device, n_windows=16):
    charts_dir = os.path.join(EXP53AR_DIR, "charts")
    scores = {}
    for song_num, song_stem, song_name in SONGS_53:
        scores[song_num] = {}
        # find mel (same audio for all models, use first available)
        mel_file = None
        for f in sorted(os.listdir(charts_dir)):
            if f.startswith(f"{song_num}_") and f.endswith(f"_{MODELS_53[0]}_mel.npy"):
                mel_file = f
                break
        if mel_file is None:
            continue
        mel = np.load(os.path.join(charts_dir, mel_file))

        for m in MODELS_53:
            csv_file = None
            for f in sorted(os.listdir(charts_dir)):
                if f.startswith(f"{song_num}_") and f.endswith(f"_{m}.csv"):
                    csv_file = f
                    break
            if csv_file is None:
                continue
            events = load_csv_events(os.path.join(charts_dir, csv_file))
            if len(events) == 0:
                continue
            scores[song_num][m] = score_chart(model, mel, events, 4.0, device, n_windows)

        print(f"  53-AR Song {song_num}: {' '.join(f'{m}={scores[song_num].get(m, 0):+.3f}' for m in MODELS_53)}")
    return scores


# ──────────────────────────────────────────────
#  Exp 42-AR data loading
# ──────────────────────────────────────────────

EXP42AR_DIR = os.path.join(SCRIPT_DIR, "experiments", "experiment_42ar")
MODELS_42 = ["exp14", "exp35c", "exp42"]


def load_42ar_mappings():
    mappings = {}
    compiled_dir = os.path.join(EXP42AR_DIR, "compiled")
    for f in sorted(os.listdir(compiled_dir)):
        if f.endswith("_mapping.txt"):
            song_name = f.replace("_mapping.txt", "")
            m = {}
            with open(os.path.join(compiled_dir, f), encoding="utf-8") as fh:
                for line in fh:
                    if " = " in line:
                        label, model = line.strip().split(" = ")
                        m[label.strip()] = model.strip()
            mappings[song_name] = m
    return mappings


def load_42ar_votes():
    with open(os.path.join(EXP42AR_DIR, "results", "votes.json")) as f:
        return json.load(f)


def score_42ar(model, device, n_windows=16):
    charts_dir = os.path.join(EXP42AR_DIR, "charts")
    scores = {}
    # songs are stored as charts/{model_name}/{song_name}.csv
    # find all song names from exp14 directory
    exp14_dir = os.path.join(charts_dir, "exp14")
    song_names = sorted(set(
        f.replace(".csv", "").replace("_mel.npy", "").replace("_stats.json", "").replace("_wave.npy", "")
        for f in os.listdir(exp14_dir) if f.endswith(".csv")
    ))

    for song_name in song_names:
        scores[song_name] = {}
        mel_path = os.path.join(exp14_dir, f"{song_name}_mel.npy")
        if not os.path.exists(mel_path):
            continue
        mel = np.load(mel_path)

        for m in MODELS_42:
            csv_path = os.path.join(charts_dir, m, f"{song_name}.csv")
            if not os.path.exists(csv_path):
                continue
            events = load_csv_events(csv_path)
            if len(events) == 0:
                continue
            scores[song_name][m] = score_chart(model, mel, events, 4.0, device, n_windows)

        short = song_name[:40]
        print(f"  42-AR {short}: {' '.join(f'{m}={scores[song_name].get(m, 0):+.3f}' for m in MODELS_42)}")
    return scores


# ──────────────────────────────────────────────
#  Human vote processing
# ──────────────────────────────────────────────

POINTS = {1: 4, 2: 3, 3: 2, 4: 1}  # for 4 models
POINTS_3 = {1: 3, 2: 2, 3: 1}       # for 3 models


def process_votes(votes_data, mappings, song_key_fn, models, n_ranks):
    """Process votes into per-song per-model points, split by self/evaluator.

    Returns: (self_points, eval_points, total_points)
    Each is dict[song_key] → {model: points}
    """
    pts = POINTS if n_ranks == 4 else POINTS_3

    def _process_vote_list(vote_list, label):
        song_points = {}
        for vote in vote_list:
            # skip empty votes
            if vote.get("rank_1") is None:
                continue
            song_key = song_key_fn(vote["song"])
            if song_key is None:
                continue
            mapping = mappings.get(song_key, {})
            if not mapping:
                continue

            if song_key not in song_points:
                song_points[song_key] = {m: 0 for m in models}

            for rank_idx in range(1, n_ranks + 1):
                rank_key = f"rank_{rank_idx}"
                blind_label = vote.get(rank_key)
                if blind_label and blind_label in mapping:
                    model = mapping[blind_label]
                    song_points[song_key][model] += pts[rank_idx]

        return song_points

    self_pts = _process_vote_list(votes_data.get("self_rankings", []), "self")
    eval_pts = _process_vote_list(votes_data.get("evaluators", []), "eval")

    # total: merge both
    total_pts = {}
    for src in [self_pts, eval_pts]:
        for song_key, model_pts in src.items():
            if song_key not in total_pts:
                total_pts[song_key] = {m: 0 for m in models}
            for m, p in model_pts.items():
                total_pts[song_key][m] += p

    return self_pts, eval_pts, total_pts


# ──────────────────────────────────────────────
#  Comparison
# ──────────────────────────────────────────────

def compare(eval_scores, human_points, models, label):
    """Compare evaluator vs human for a set of songs. Returns metrics dict."""
    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")

    n_first_match = 0
    n_songs = 0
    pairwise_correct = 0
    pairwise_total = 0

    print(f"  {'Song':<40s} {'Human #1':>10s} {'Eval #1':>10s} {'Match':>6s}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*6}")

    for song_key in sorted(eval_scores.keys()):
        if song_key not in human_points:
            continue
        if not eval_scores[song_key]:
            continue

        hp = human_points[song_key]
        # skip songs with no votes
        if all(v == 0 for v in hp.values()):
            continue

        human_ranked = sorted(models, key=lambda m: hp.get(m, 0), reverse=True)
        eval_ranked = sorted(models, key=lambda m: eval_scores[song_key].get(m, -999), reverse=True)

        match = "YES" if human_ranked[0] == eval_ranked[0] else ""
        if human_ranked[0] == eval_ranked[0]:
            n_first_match += 1
        n_songs += 1

        short_key = str(song_key)[:40]
        print(f"  {short_key:<40s} {human_ranked[0]:>10s} {eval_ranked[0]:>10s} {match:>6s}")

        # pairwise
        for i, m_a in enumerate(models):
            for m_b in models[i+1:]:
                if m_a not in eval_scores[song_key] or m_b not in eval_scores[song_key]:
                    continue
                h_a, h_b = hp.get(m_a, 0), hp.get(m_b, 0)
                e_a, e_b = eval_scores[song_key][m_a], eval_scores[song_key][m_b]
                if h_a == h_b:
                    continue
                if (h_a > h_b) == (e_a > e_b):
                    pairwise_correct += 1
                pairwise_total += 1

    random_baseline = 1.0 / len(models)
    pairwise_baseline = 0.5

    print(f"\n  #1 match: {n_first_match}/{n_songs} ({n_first_match/max(n_songs,1):.0%}) "
          f"— random: {random_baseline:.0%}")

    if pairwise_total > 0:
        print(f"  Pairwise: {pairwise_correct}/{pairwise_total} = "
              f"{pairwise_correct/pairwise_total:.1%} — random: 50%")

    # global ranking
    global_eval = {m: [] for m in models}
    global_human = {m: 0 for m in models}
    for song_key in eval_scores:
        for m, s in eval_scores[song_key].items():
            global_eval[m].append(s)
        if song_key in human_points:
            for m, p in human_points[song_key].items():
                global_human[m] += p

    global_eval_mean = {m: np.mean(v) if v else -999 for m, v in global_eval.items()}
    eval_ranked = sorted(models, key=lambda m: global_eval_mean[m], reverse=True)
    human_ranked = sorted(models, key=lambda m: global_human[m], reverse=True)

    print(f"\n  Global ranking:")
    print(f"    Human:     {' > '.join(f'{m}({global_human[m]})' for m in human_ranked)}")
    print(f"    Evaluator: {' > '.join(f'{m}({global_eval_mean[m]:+.3f})' for m in eval_ranked)}")

    # rank correlation on global
    h_ranks = [human_ranked.index(m) + 1 for m in models]
    e_ranks = [eval_ranked.index(m) + 1 for m in models]
    if len(models) >= 3:
        rho, p_rho = spearmanr(h_ranks, e_ranks)
        tau, p_tau = kendalltau(h_ranks, e_ranks)
        print(f"    Spearman: rho={rho:+.3f} (p={p_rho:.3f})")
        print(f"    Kendall:  tau={tau:+.3f} (p={p_tau:.3f})")
    else:
        rho, tau = 0, 0

    return {
        "first_match": n_first_match,
        "n_songs": n_songs,
        "first_match_pct": n_first_match / max(n_songs, 1),
        "pairwise_correct": pairwise_correct,
        "pairwise_total": pairwise_total,
        "pairwise_accuracy": pairwise_correct / max(pairwise_total, 1),
        "global_eval_ranking": eval_ranked,
        "global_human_ranking": human_ranked,
        "spearman": rho,
    }


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def run_experiment(model, device, n_windows, ckpt_label):
    """Run full human comparison for one model checkpoint."""
    all_results = {}

    # ── Exp 53-AR ──
    print(f"\n{'='*70}")
    print(f"  EXP 53-AR — {ckpt_label}")
    print(f"{'='*70}")

    mappings_53 = load_53ar_mappings()
    votes_53 = load_53ar_votes()
    scores_53 = score_53ar(model, device, n_windows)

    def song_key_53(song_name):
        # match by song index (votes are in order)
        for i, (sn, _, sname) in enumerate(SONGS_53):
            # try name matching
            if song_name.split(" - ")[-1][:10].lower() in sname.lower():
                return sn
        return None

    self_pts, eval_pts, total_pts = process_votes(
        votes_53, mappings_53, song_key_53, MODELS_53, n_ranks=4)

    # also use index-based matching for self votes
    for i, vote in enumerate(votes_53.get("self_rankings", [])):
        if i < len(SONGS_53):
            sn = SONGS_53[i][0]
            mapping = mappings_53.get(sn, {})
            if mapping and sn not in self_pts:
                self_pts[sn] = {m: 0 for m in MODELS_53}
            if mapping and sn in self_pts:
                for rank_idx in range(1, 5):
                    label = vote.get(f"rank_{rank_idx}")
                    model_name = mapping.get(label)
                    if model_name:
                        self_pts[sn][model_name] = max(self_pts[sn].get(model_name, 0), POINTS[rank_idx])
                        if sn not in total_pts:
                            total_pts[sn] = {m: 0 for m in MODELS_53}
                        total_pts[sn][model_name] = max(total_pts[sn].get(model_name, 0), POINTS[rank_idx])

    r_self = compare(scores_53, self_pts, MODELS_53, "53-AR Self votes only")
    r_eval = compare(scores_53, eval_pts, MODELS_53, "53-AR External evaluators only")
    r_total = compare(scores_53, total_pts, MODELS_53, "53-AR All votes")

    all_results["53ar"] = {
        "self": r_self, "eval": r_eval, "total": r_total,
        "scores": {k: v for k, v in scores_53.items()},
    }

    # ── Exp 42-AR ──
    print(f"\n{'='*70}")
    print(f"  EXP 42-AR — {ckpt_label}")
    print(f"{'='*70}")

    mappings_42 = load_42ar_mappings()
    votes_42 = load_42ar_votes()
    scores_42 = score_42ar(model, device, n_windows)

    def song_key_42(song_name):
        # direct match — mapping keys are full song names
        if song_name in mappings_42:
            return song_name
        # partial match
        for key in mappings_42:
            if song_name[:20] in key or key[:20] in song_name:
                return key
        return None

    self_pts_42, eval_pts_42, total_pts_42 = process_votes(
        votes_42, mappings_42, song_key_42, MODELS_42, n_ranks=3)

    # index-based fallback for self votes
    song_names_42 = sorted(scores_42.keys())
    for i, vote in enumerate(votes_42.get("self_rankings", [])):
        song_name = vote["song"]
        song_key = song_key_42(song_name)
        if song_key is None:
            # try matching by order in the scores dict
            for sk in song_names_42:
                if song_name[:15] in sk or sk[:15] in song_name:
                    song_key = sk
                    break
        if song_key and song_key not in self_pts_42:
            mapping = mappings_42.get(song_key, {})
            if not mapping:
                # try matching mapping key
                for mk in mappings_42:
                    if song_name[:15] in mk or mk[:15] in song_name:
                        mapping = mappings_42[mk]
                        break
            if mapping:
                self_pts_42[song_key] = {m: 0 for m in MODELS_42}
                total_pts_42.setdefault(song_key, {m: 0 for m in MODELS_42})
                for rank_idx in range(1, 4):
                    label = vote.get(f"rank_{rank_idx}")
                    model_name = mapping.get(label)
                    if model_name:
                        self_pts_42[song_key][model_name] += POINTS_3[rank_idx]
                        total_pts_42[song_key][model_name] += POINTS_3[rank_idx]

    r_self_42 = compare(scores_42, self_pts_42, MODELS_42, "42-AR Self votes only")
    r_eval_42 = compare(scores_42, eval_pts_42, MODELS_42, "42-AR External evaluators only")
    r_total_42 = compare(scores_42, total_pts_42, MODELS_42, "42-AR All votes")

    all_results["42ar"] = {
        "self": r_self_42, "eval": r_eval_42, "total": r_total_42,
        "scores": {k: v for k, v in scores_42.items()},
    }

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {ckpt_label}")
    print(f"{'='*70}")
    print(f"\n  {'Dataset':<15s} {'Split':<10s} {'#1 match':>10s} {'Pairwise':>10s} {'Spearman':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for ds, ds_label in [("53ar", "53-AR"), ("42ar", "42-AR")]:
        for split in ["self", "eval", "total"]:
            r = all_results[ds][split]
            print(f"  {ds_label:<15s} {split:<10s} "
                  f"{r['first_match']}/{r['n_songs']} ({r['first_match_pct']:.0%}):>10s "
                  f"{r['pairwise_accuracy']:.1%}:>10s "
                  f"{r.get('spearman', 0):+.3f}:>10s")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare evaluator vs human preference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint2", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-windows", type=int, default=16)
    parser.add_argument("--output", default="human_eval_comparison.json")
    args = parser.parse_args()

    output = {}

    print(f"\n*** Checkpoint 1: {args.checkpoint} ***")
    model1 = load_model(args.checkpoint, args.device)
    label1 = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))
    output["ckpt1"] = run_experiment(model1, args.device, args.n_windows, label1)

    if args.checkpoint2:
        print(f"\n\n*** Checkpoint 2: {args.checkpoint2} ***")
        model2 = load_model(args.checkpoint2, args.device)
        label2 = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint2)))
        output["ckpt2"] = run_experiment(model2, args.device, args.n_windows, label2)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
