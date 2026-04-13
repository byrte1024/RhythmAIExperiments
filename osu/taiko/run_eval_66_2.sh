#!/usr/bin/env bash
# Experiment 66-2: Evaluate bidirectional corruption evaluator on AR-generated charts.
# Reuses AR inference outputs from 66-1b.
#
# Usage: bash run_eval_66_2.sh
set -e
cd "$(dirname "$0")"

CKPT="runs/eval_experiment_66_2/checkpoints/best.pt"
AR_BASE="experiments/experiment_66_1b/ar_eval"
OUT="experiments/experiment_66_2/results"
mkdir -p "$OUT"

for exp in 14 45 58 62; do
    echo "── exp $exp ──"
    python classifier_eval_ar.py \
        --checkpoint "$CKPT" \
        --ar-dir "$AR_BASE/exp${exp}_best" \
        --regime song_density \
        --output "$OUT/eval_exp${exp}.json"
    echo ""
done

# cross-model summary
echo "========================================"
echo "Cross-model summary"
echo "========================================"

python -u -c "
import json, numpy as np, os, sys
sys.stdout.reconfigure(encoding='utf-8')

out = '$OUT'
exps = [14, 45, 58, 62]

print(f\"{'Exp':>5s} {'GT win%':>8s} {'GT mean':>8s} {'Gen mean':>9s} {'Diff':>8s} {'close%':>8s} {'hall%':>8s} {'metro':>8s} {'gap_cv':>8s}\")
print('-'*80)

for e in exps:
    path = os.path.join(out, f'eval_exp{e}.json')
    if not os.path.exists(path): continue
    data = json.load(open(path))
    r = data.get('results_1', [])
    if not r: continue

    gt_w = sum(1 for x in r if x['gt_wins'])
    n = len(r)
    def avg(k):
        v = [x.get(k) for x in r if x.get(k) is not None]
        return np.mean(v) if v else 0

    print(f'{e:5d} {gt_w/n:7.1%} {np.mean([x[\"gt_score\"] for x in r]):+8.2f} '
          f'{np.mean([x[\"gen_score\"] for x in r]):+9.2f} '
          f'{np.mean([x[\"diff\"] for x in r]):+8.2f} '
          f'{avg(\"gt_close_rate\"):7.1%} {avg(\"gt_hallucination_rate\"):7.1%} '
          f'{avg(\"pat_max_metro_streak\"):8.1f} {avg(\"pat_gap_cv\"):8.3f}')
"

echo ""
echo "Done. Results in $OUT/"
