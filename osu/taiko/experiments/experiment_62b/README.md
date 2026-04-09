# Experiment 62b - Multi-Onset Truncation Ablation

## Purpose

Exp 62 trained a 4-onset model and showed improved pattern diversity (P-Space 12.0% vs exp58's 10.1%) with only -0.9pp close rate loss. But per-sample metrics showed large gaps between onsets (o1=74.9%, o2=58.3%, o3=49.7%, o4=43.0%) and strict_increasing was only 70%.

**Question**: Is the diversity gain coming from using all 4 onsets, or did multi-onset *training* improve even the first onset's pattern quality?

## Method

Run AR inference with exp62's best checkpoint, but truncate how many onsets get placed:

| Run | Max onsets | What it tests |
|---|---|---|
| mo1 | 1 (o1 only) | Does multi-onset training improve single-onset AR quality? |
| mo2 | 2 (o1+o2) | Is 2 onsets the sweet spot? |
| mo3 | 3 (o1+o2+o3) | Diminishing returns from o4? |
| all | 4 (baseline) | Same as exp62 AR results |

Compare against exp58 (single-onset trained, single-onset inference) as baseline.

No new training — purely inference-time ablation on existing exp62 checkpoint.

## Launch

```bash
cd osu/taiko
python run_ar.py experiment_62b runs/detect_experiment_62/checkpoints/best.pt --max-onsets 1
python run_ar.py experiment_62b runs/detect_experiment_62/checkpoints/best.pt --max-onsets 2
python run_ar.py experiment_62b runs/detect_experiment_62/checkpoints/best.pt --max-onsets 3

python analyze_ar.py experiment_62b detect_experiment_62_best_mo1
python analyze_ar.py experiment_62b detect_experiment_62_best_mo2
python analyze_ar.py experiment_62b detect_experiment_62_best_mo3
```

## Result

*Pending*

## Lesson

*Pending*
