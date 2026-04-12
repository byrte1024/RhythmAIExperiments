#!/usr/bin/env bash
# Experiment 66-1b: Run AR inference for exp 14, 45, 58, 62 and evaluate
# with the corruption-only (P1) and rating-finetuned (P2) quality evaluators.
#
# Usage: bash run_eval_66_1b.sh
set -e
cd "$(dirname "$0")"

EVAL_CKPT_P1="runs/eval_experiment_66_1/checkpoints/best.pt"
EVAL_CKPT_P2="runs/eval_experiment_66_1_p2/checkpoints/best.pt"
EXP_DIR="experiments/experiment_66_1b"
OUTPUT_DIR="$EXP_DIR/results"
mkdir -p "$OUTPUT_DIR"

# ═══════════════════════════════════════════════════════════════
#  Step 1: Run AR inference for each detector model
# ═══════════════════════════════════════════════════════════════

DETECTORS=(
    "14:runs/detect_experiment_14/checkpoints/best.pt"
    "45:runs/detect_experiment_45/checkpoints/best.pt"
    "58:runs/detect_experiment_58/checkpoints/best.pt"
    "62:runs/detect_experiment_62/checkpoints/best.pt"
)

echo "========================================"
echo "Step 1: AR inference (run_ar.py)"
echo "========================================"

for entry in "${DETECTORS[@]}"; do
    IFS=':' read -r exp_num ckpt_path <<< "$entry"
    label="exp${exp_num}_best"
    ar_dir="$EXP_DIR/ar_eval/$label"

    if [ -f "$ar_dir/songs.json" ]; then
        n_csvs=$(ls "$ar_dir/csvs/song_density/"*_predicted.csv 2>/dev/null | wc -l)
        if [ "$n_csvs" -ge 28 ]; then
            echo "  exp $exp_num: already have $n_csvs CSVs, skipping"
            continue
        fi
    fi

    echo "  exp $exp_num: running AR inference..."
    python run_ar.py experiment_66_1b "$ckpt_path" --label "$label"
done

echo ""

# ═══════════════════════════════════════════════════════════════
#  Step 2: Score with quality evaluator (P1 and P2)
# ═══════════════════════════════════════════════════════════════

echo "========================================"
echo "Step 2: Quality evaluation (classifier_eval_ar.py)"
echo "========================================"

for entry in "${DETECTORS[@]}"; do
    IFS=':' read -r exp_num ckpt_path <<< "$entry"
    label="exp${exp_num}_best"
    ar_dir="$EXP_DIR/ar_eval/$label"
    out_json="$OUTPUT_DIR/eval_exp${exp_num}.json"

    if [ ! -f "$ar_dir/songs.json" ]; then
        echo "  exp $exp_num: no AR data, skipping"
        continue
    fi

    echo ""
    echo "  ── exp $exp_num ──"
    python classifier_eval_ar.py \
        --checkpoint "$EVAL_CKPT_P1" \
        --checkpoint2 "$EVAL_CKPT_P2" \
        --ar-dir "$ar_dir" \
        --regime song_density \
        --output "$out_json"
done

echo ""

# ═══════════════════════════════════════════════════════════════
#  Step 3: Cross-model comparison summary
# ═══════════════════════════════════════════════════════════════

echo "========================================"
echo "Step 3: Cross-model summary"
echo "========================================"

python -u -c "
import json, numpy as np, os, sys
sys.stdout.reconfigure(encoding='utf-8')

output_dir = '$OUTPUT_DIR'
exps = [14, 45, 58, 62]

print(f\"{'Exp':>5s} {'GT win%':>8s} {'GT mean':>8s} {'Gen mean':>9s} {'Diff':>8s} {'Close%':>8s} {'Hall%':>8s} {'Metro':>8s}\")
print(f\"{'---':>5s} {'---':>8s} {'---':>8s} {'---':>9s} {'---':>8s} {'---':>8s} {'---':>8s} {'---':>8s}\")

for exp_num in exps:
    path = os.path.join(output_dir, f'eval_exp{exp_num}.json')
    if not os.path.exists(path):
        print(f'{exp_num:5d} (no data)')
        continue
    data = json.load(open(path))
    r1 = data.get('results_1', [])
    if not r1:
        continue

    gt_scores = [r['gt_score'] for r in r1]
    gen_scores = [r['gen_score'] for r in r1]
    diffs = [r['diff'] for r in r1]
    gt_wins = sum(1 for r in r1 if r['gt_wins'])
    n = len(r1)
    close = np.mean([r.get('gt_close_rate', 0) for r in r1])
    hall = np.mean([r.get('gt_hallucination_rate', 0) for r in r1])
    metro = np.mean([r.get('pat_max_metro_streak', 0) for r in r1])

    print(f'{exp_num:5d} {gt_wins/n:7.1%} {np.mean(gt_scores):+8.2f} {np.mean(gen_scores):+9.2f} '
          f'{np.mean(diffs):+8.2f} {close:7.1%} {hall:7.1%} {metro:8.1f}')
"

echo ""
echo "Done. Results saved to $OUTPUT_DIR/"
