"""Run our models + TaikoNation on TaikoNation's exact evaluation charts.

Computes GT matching + TaikoNation metrics on all 10 charts for each model.
Compares directly to TaikoNation's published numbers.

Usage:
    cd osu/taiko
    python experiments/experiment_63b/run_comparison.py
"""

import json
import os
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CHART_DIR = os.path.join(SCRIPT_DIR, "charts")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

BIN_MS = 4.9887
TN_STEP_MS = 23

# Our models to test
MODELS = {
    "exp14": os.path.join(TAIKO_DIR, "runs", "detect_experiment_14", "checkpoints", "best.pt"),
    "exp44": os.path.join(TAIKO_DIR, "runs", "detect_experiment_44", "checkpoints", "best.pt"),
    "exp45": os.path.join(TAIKO_DIR, "runs", "detect_experiment_45", "checkpoints", "best.pt"),
    "exp58": os.path.join(TAIKO_DIR, "runs", "detect_experiment_58", "checkpoints", "best.pt"),
}

# TaikoNation published results (on these exact charts)
PUBLISHED = {
    "TaikoNation": {"over_pspace": 21.328, "hi_pspace": 94.117, "dc_human": 74.987, "dc_rand": 50.405},
    "DDC": {"over_pspace": 15.938, "hi_pspace": 83.160, "dc_human": 77.900, "dc_rand": 49.938},
    "Human GT": {"over_pspace": 14.453, "dc_rand": 50.170},
}


# ═══════════════════════════════════════════════════════════════
#  TaikoNation Metrics
# ═══════════════════════════════════════════════════════════════

def events_ms_to_binary(events_ms, step_ms=TN_STEP_MS):
    if len(events_ms) == 0:
        return np.array([], dtype=np.int32)
    max_time = int(max(events_ms)) + step_ms
    n_steps = max_time // step_ms + 1
    binary = np.zeros(n_steps, dtype=np.int32)
    for t in events_ms:
        idx = int(t) // step_ms
        if 0 <= idx < n_steps:
            binary[idx] = 1
    return binary


def tn_dc_rand(chart, rng):
    noise = rng.integers(low=0, high=2, size=len(chart))
    if len(chart) == 0:
        return 0.0
    return float((chart == noise).sum() / len(chart) * 100)


def tn_dc_human(ai_chart, human_chart):
    limit = min(len(ai_chart), len(human_chart))
    if limit == 0:
        return 0.0
    start = 0
    for i in range(limit):
        if human_chart[i] == 1:
            start = i
            break
    total = limit - start
    if total <= 0:
        return 0.0
    return float((ai_chart[start:limit] == human_chart[start:limit]).sum() / total * 100)


def tn_oc_human(ai_chart, human_chart, buffer=1):
    limit = min(len(ai_chart), len(human_chart))
    if limit == 0:
        return 0.0
    start = 0
    for i in range(limit):
        if human_chart[i] == 1:
            start = i
            break
    total = limit - start
    if total <= 0:
        return 0.0
    similarity = 0
    for i in range(start, limit):
        if ai_chart[i] == 1:
            matched = False
            for b in range(-buffer, buffer + 1):
                j = i + b
                if 0 <= j < limit and human_chart[j] == 1:
                    matched = True
                    break
            if matched:
                similarity += 1
        elif human_chart[i] == 0:
            similarity += 1
    return float(similarity / total * 100)


def tn_over_pspace(chart, scale=8):
    patterns = set()
    last_ind = len(chart) - scale + 1
    if last_ind <= 0:
        return 0.0
    for i in range(last_ind):
        patterns.add(tuple(chart[i:i + scale]))
    return float(len(patterns) / 2**scale * 100)


def tn_hi_pspace(ai_chart, human_chart, scale=8):
    ai_p = set()
    hu_p = set()
    for i in range(len(ai_chart) - scale + 1):
        ai_p.add(tuple(ai_chart[i:i + scale]))
    for i in range(len(human_chart) - scale + 1):
        hu_p.add(tuple(human_chart[i:i + scale]))
    if len(hu_p) == 0:
        return 0.0
    return float(len(ai_p & hu_p) / len(hu_p) * 100)


def compute_tn_metrics(pred_ms, gt_ms, rng):
    pb = events_ms_to_binary(pred_ms)
    gb = events_ms_to_binary(gt_ms)
    if len(pb) < 16 or len(gb) < 16:
        return None
    ml = max(len(pb), len(gb))
    pp = np.zeros(ml, dtype=np.int32)
    gp = np.zeros(ml, dtype=np.int32)
    pp[:len(pb)] = pb
    gp[:len(gb)] = gb
    return {
        "over_pspace": tn_over_pspace(pp),
        "hi_pspace": tn_hi_pspace(pp, gp),
        "dc_human": tn_dc_human(pp, gp),
        "oc_human": tn_oc_human(pp, gp),
        "dc_rand": tn_dc_rand(pp, rng),
    }


# ═══════════════════════════════════════════════════════════════
#  GT Matching
# ═══════════════════════════════════════════════════════════════

def compute_gt_metrics(pred_ms, gt_ms):
    if len(pred_ms) == 0 or len(gt_ms) == 0:
        return None
    ps = np.sort(pred_ms)
    gs = np.sort(gt_ms)

    def _c(arr, v):
        i = np.searchsorted(arr, v)
        best = float("inf")
        for j in [i - 1, i, i + 1]:
            if 0 <= j < len(arr):
                best = min(best, abs(arr[j] - v))
        return best

    ge = np.array([_c(ps, g) for g in gs])
    pe = np.array([_c(gs, p) for p in ps])

    n_pred, n_gt = len(ps), len(gs)
    pd = n_pred / max((ps[-1] - ps[0]) / 1000, 0.1) if n_pred > 1 else 0
    gd = n_gt / max((gs[-1] - gs[0]) / 1000, 0.1) if n_gt > 1 else 0

    return {
        "n_pred": n_pred, "n_gt": n_gt,
        "matched_rate": float((ge <= 25).mean()),
        "close_rate": float((ge <= 50).mean()),
        "far_rate": float((ge > 100).mean()),
        "hallucination_rate": float((pe > 100).mean()),
        "gt_error_mean": float(ge.mean()),
        "gt_error_median": float(np.median(ge)),
        "density_ratio": pd / max(gd, 0.01),
    }


# ═══════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════

def load_csv_events_ms(csv_path):
    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith("time_ms") or not line:
                continue
            parts = line.split(",")
            if parts:
                events.append(int(parts[0]))
    return np.array(events, dtype=np.float64)


def run_inference(checkpoint, audio_path, output_csv, density_mean, density_peak, density_std, hop_ms=75):
    cmd = [
        sys.executable, os.path.join(TAIKO_DIR, "detection_inference.py"),
        "--checkpoint", checkpoint,
        "--audio", audio_path,
        "--output", output_csv,
        "--density-mean", str(density_mean),
        "--density-peak", str(density_peak),
        "--density-std", str(density_std),
        "--hop-ms", str(hop_ms),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"      ERROR: {result.stderr[-300:]}")
        return False
    return True


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    manifest_path = os.path.join(CHART_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("ERROR: Run download_taikonation_charts.py first!")
        sys.exit(1)

    with open(manifest_path) as f:
        charts = json.load(f)

    print(f"Charts: {len(charts)}")

    # Check model checkpoints
    available_models = {}
    for name, path in MODELS.items():
        if os.path.exists(path):
            available_models[name] = path
            print(f"  {name}: OK")
        else:
            print(f"  {name}: NOT FOUND ({path})")

    tn_rng = np.random.default_rng(2009000042)  # TaikoNation's seed

    # Per-model results
    model_results = {m: [] for m in available_models}
    gt_self_results = []  # Human GT self-metrics

    csv_dir = os.path.join(RESULTS_DIR, "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    for ci, chart in enumerate(charts):
        events_path = chart.get("events_path")
        audio_path = chart.get("audio_path")

        if not events_path or not os.path.exists(events_path):
            print(f"  [{ci+1}] SKIP: no events")
            continue
        if not audio_path or not os.path.exists(audio_path):
            print(f"  [{ci+1}] SKIP: no audio")
            continue

        with open(events_path) as f:
            event_data = json.load(f)

        gt_ms = np.array(event_data["hit_times_ms"], dtype=np.float64)
        artist = chart.get("artist", "?")
        title = chart.get("title", "?")
        diff = chart.get("difficulty", "?")
        bsid = chart.get("beatmapset_id", "?")

        print(f"\n  [{ci+1}/{len(charts)}] {artist} - {title} [{diff}] ({len(gt_ms)} objects)")

        # GT self-metrics
        gt_binary = events_ms_to_binary(gt_ms)
        if len(gt_binary) >= 16:
            gt_self_results.append({
                "over_pspace": tn_over_pspace(gt_binary),
                "dc_rand": tn_dc_rand(gt_binary, tn_rng),
            })

        # Estimate density from GT
        if len(gt_ms) >= 2:
            duration_s = (gt_ms[-1] - gt_ms[0]) / 1000.0
            density = len(gt_ms) / max(duration_s, 1.0)
        else:
            density = 5.0
        # Rough density params
        d_mean = density
        d_peak = density * 2
        d_std = density * 0.3

        # Run each model
        for model_name, ckpt_path in available_models.items():
            safe = f"{bsid}_{model_name}"
            output_csv = os.path.join(csv_dir, f"{safe}_predicted.csv")

            if not os.path.exists(output_csv):
                print(f"    {model_name}: running inference...")
                ok = run_inference(ckpt_path, audio_path, output_csv, d_mean, d_peak, d_std)
                if not ok:
                    continue
            else:
                print(f"    {model_name}: (cached)")

            pred_ms = load_csv_events_ms(output_csv)
            if len(pred_ms) < 2:
                print(f"    {model_name}: too few predictions ({len(pred_ms)})")
                continue

            gt_result = compute_gt_metrics(pred_ms, gt_ms)
            tn_result = compute_tn_metrics(pred_ms, gt_ms, tn_rng)

            if gt_result and tn_result:
                combined = {**gt_result, **tn_result, "chart": f"{artist} - {title}"}
                model_results[model_name].append(combined)
                print(f"    {model_name}: close={gt_result['close_rate']:.1%} "
                      f"hall={gt_result['hallucination_rate']:.1%} "
                      f"DCHum={tn_result['dc_human']:.1f}% "
                      f"P-Space={tn_result['over_pspace']:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    #  Results
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"RESULTS ({len(charts)} TaikoNation evaluation charts)")
    print(f"{'='*100}")

    print(f"\n{'Model':<15} {'Close%':>7} {'Match%':>7} {'Far%':>6} {'Hall%':>6} {'d_ratio':>8} {'err_med':>8} | {'OverPS':>7} {'HI-PS':>6} {'DCHum':>6} {'OCHum':>6} {'DCRand':>7}")
    print("-" * 110)

    # Our models
    for model_name in available_models:
        results = model_results[model_name]
        if not results:
            print(f"  {model_name:<13} (no data)")
            continue
        avg = lambda k: float(np.mean([r[k] for r in results]))
        print(f"  {model_name:<13} {avg('close_rate'):>6.1%} {avg('matched_rate'):>6.1%} "
              f"{avg('far_rate'):>5.1%} {avg('hallucination_rate'):>5.1%} "
              f"{avg('density_ratio'):>7.2f} {avg('gt_error_median'):>7.0f}ms | "
              f"{avg('over_pspace'):>6.1f}% {avg('hi_pspace'):>5.1f}% "
              f"{avg('dc_human'):>5.1f}% {avg('oc_human'):>5.1f}% {avg('dc_rand'):>6.1f}%")

    # Human GT self-metrics
    if gt_self_results:
        avg_ps = np.mean([r["over_pspace"] for r in gt_self_results])
        avg_dr = np.mean([r["dc_rand"] for r in gt_self_results])
        print(f"  {'Human GT':<13} {'---':>7} {'---':>7} {'---':>6} {'---':>6} {'---':>8} {'---':>8} | "
              f"{avg_ps:>6.1f}% {'---':>5} {'---':>5} {'---':>5} {avg_dr:>6.1f}%")

    # Published baselines
    print(f"\n  --- Published (TaikoNation paper, SAME charts) ---")
    for name, r in PUBLISHED.items():
        print(f"  {name:<13} {'---':>7} {'---':>7} {'---':>6} {'---':>6} {'---':>8} {'---':>8} | "
              f"{r.get('over_pspace', 0):>6.1f}% {r.get('hi_pspace', 0):>5.1f}% "
              f"{r.get('dc_human', 0):>5.1f}% {r.get('oc_human', 0):>5.1f}% {r.get('dc_rand', 0):>6.1f}%")

    # Save
    save_data = {
        "n_charts": len(charts),
        "models": {},
        "human_gt": {},
        "published": PUBLISHED,
    }
    for model_name in available_models:
        results = model_results[model_name]
        if not results:
            continue
        avg = lambda k: float(np.mean([r[k] for r in results]))
        save_data["models"][model_name] = {k: avg(k) for k in results[0] if k != "chart"}

    if gt_self_results:
        save_data["human_gt"] = {
            "over_pspace": float(np.mean([r["over_pspace"] for r in gt_self_results])),
            "dc_rand": float(np.mean([r["dc_rand"] for r in gt_self_results])),
        }

    with open(os.path.join(RESULTS_DIR, "comparison_results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/comparison_results.json")


if __name__ == "__main__":
    main()
