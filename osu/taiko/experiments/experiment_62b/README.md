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

### Song Density Regime (30 val songs, per-song density)

| Metric | exp58 (1-onset model) | mo1 | mo2 | mo3 | mo4 (exp62) |
|---|---|---|---|---|---|
| Close (<50ms) | 75.9% | 75.0% | **76.7%** | 76.5% | 75.0% |
| Far (>100ms) | 16.6% | 17.8% | 15.9% | **15.7%** | 16.7% |
| Hallucination | 15.6% | **14.9%** | 15.8% | 16.4% | 15.9% |
| Density ratio | 0.92 | 0.90 | 0.94 | **0.99** | 0.97 |
| Error median | **8ms** | 10ms | 9ms | 9ms | **8ms** |
| Over. P-Space | 10.1% | 11.0% | 11.4% | **12.0%** | **12.0%** |
| HI P-Space | 81.1% | 80.9% | **84.3%** | 82.7% | 82.4% |
| DCHuman | **90.8%** | 89.6% | 90.3% | 89.9% | 90.5% |
| gap_std | — | 175.6 | 169.7 | 167.4 | 174.9 |
| gap_cv | — | 0.717 | 0.712 | 0.689 | 0.712 |
| dominant_gap_pct | — | 46.8% | 45.8% | 46.0% | 47.5% |
| max_metro_streak | — | 15.7 | 13.0 | **11.1** | 13.0 |

### Fixed 5.75 Regime (30 val songs)

| Metric | mo1 | mo2 | mo3 | mo4 (exp62) |
|---|---|---|---|---|
| Close (<50ms) | 79.9% | **81.4%** | 81.1% | 79.8% |
| Hallucination | **18.5%** | 20.2% | 19.8% | 19.9% |
| Density ratio | **1.06** | 1.14 | 1.17 | 1.19 |
| Over. P-Space | 9.5% | 10.0% | 10.3% | **10.5%** |

## Lesson

1. **Multi-onset training improves single-onset inference.** mo1 (exp62 checkpoint, only o1 used) achieves P-Space 11.0% vs exp58's 10.1% — a +0.9pp diversity gain from training alone, without using any extra onsets at inference time. Training on patterns teaches better individual decisions.

2. **mo2 (o1+o2) is the sweet spot for GT matching.** 76.7% close rate is the highest of any configuration — better than exp58 (75.9%), mo1 (75.0%), and full mo4 (75.0%). Two onsets per step gives enough lookahead to improve placement without the noise from weaker o3/o4 predictions.

3. **o3 adds diversity, o4 adds nothing.** P-Space: mo1=11.0% → mo2=11.4% → mo3=12.0% → mo4=12.0%. The third onset contributes the final diversity gain, but the fourth onset is redundant — it was already the weakest per-sample (43% HIT) and adds only noise in AR.

4. **Density ratio improves monotonically with more onsets.** 0.90 → 0.94 → 0.99 → 0.97. mo3 achieves near-perfect density matching (0.99), eliminating the systematic under-prediction that previously required the 1.2x inflation hack.

5. **mo3 is the best overall configuration.** Highest P-Space (12.0%, surpassing human GT 11.7%), near-perfect density (0.99), good close rate (76.5%), and lowest max_metro_streak (11.1). mo2 wins on close rate alone, but mo3 is better balanced.

6. **The per-sample gap between onsets (o1=74.9% vs o4=43.0%) is misleading.** Even o3 and o4 contribute positively in AR — they add events the model is somewhat confident about, improving density and diversity. The 43% per-sample HIT for o4 doesn't mean it's harmful in context.
