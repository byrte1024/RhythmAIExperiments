# Experiment 59-H - Extended Model Comparison

## Hypothesis

[Exp 59-G](../experiment_59g/README.md) validated the synthetic evaluator on the 53-AR models. Now test models that weren't in 53-AR to see where they rank. Includes the propose-select ATH (exp58) and notable experiments from the 50-series.

### Models

| Model | HIT% | Key trait | Role |
|---|---|---|---|
| exp44 | 73.7% | Gentle augmentation | Reference (53-AR volunteer winner) |
| exp53 | 72.1% | B_AUDIO/B_PRED split | Reference (53-AR 3rd place) |
| exp50b | ~73.2% | Anti-entropy loss (w=0.5) | New — bimodal entropy, untested AR |
| exp51 | 67.5% | Streak-ratio loss weighting | New — failed per-sample but untested AR |
| exp55 | 73.6% | Auxiliary ratio head | New — best val loss ever |
| exp58 | 74.6% | Propose-select two-stage | New — per-sample ATH |

exp44 and exp53 included as reference frame from 53-AR human evaluation.

### Density Regimes

Both regimes tested (from [59-G](../experiment_59g/README.md) findings):
- **song_density**: per-song actual density from manifest (more reliable for synthetic ranking)
- **fixed_53ar**: d_mean=5.75, d_peak=11.1, d_std=1.5 (how AR is typically done in practice — fixed density across songs)

360 total inference runs (6 models × 30 songs × 2 regimes).

### Launch

```bash
cd osu/taiko
python experiments/experiment_59h/extended_comparison.py
```

## Result

30 songs × 6 models × 2 density regimes = 360 inference runs.

### Rankings (song_density regime):

| Rank | Expert | Volunteer | General |
|---|---|---|---|
| 1st | **exp51** (+99.2) | **exp51** (+56.8) | **exp51** (+90.9) |
| 2nd | exp50b (-5.0) | exp44 (+4.9) | exp58 (+11.8) |
| 3rd | exp44 (-13.9) | exp58 (+2.2) | exp44 (+2.1) |
| 4th | exp55 (-22.9) | exp55 (-9.3) | exp55 (-17.6) |
| 5th | exp58 (-25.7) | exp50b (-15.0) | exp50b (-25.8) |
| 6th | exp53 (-31.8) | exp53 (-39.5) | exp53 (-61.5) |

fixed_53ar regime produces the same #1 (exp51) and same last (exp53) with minor shuffling in the middle.

### Key findings:

1. **exp51 dominates by ~100+ points** across all evaluators and both density regimes. This is the model that *failed* per-sample metrics (67.5% HIT, worst in the group).

2. **Reference frame holds**: exp44 always above exp53, matching 53-AR human rankings.

3. **exp58** (74.6% HIT ATH) lands 2nd-3rd — solid but not dominant. Close to exp44 despite having 1pp higher HIT.

4. **exp53 consistently last** — the B_PRED split model that placed 3rd in 53-AR human eval continues to score lowest on pattern variety.

### Critical caveat on exp51:

exp51's massive synthetic score could reflect **chaotic randomness, not musical creativity**. At 67.5% HIT, it predicts many wrong onsets. A model that predicts semi-randomly also has high gap_std and gap_cv. The synthetic evaluator measures variety but cannot distinguish "interesting patterns" from "confused predictions." Only human evaluation can resolve this.

## Lesson

1. **Per-sample accuracy and synthetic AR quality are inversely correlated** in this test. The worst per-sample model (exp51, 67.5%) scores highest on pattern variety. The best (exp58, 74.6%) scores mid-pack. This echoes the 42-AR finding where exp14 (lowest HIT) won human evaluation.

2. **The synthetic evaluator has a fundamental blind spot**: it rewards variety regardless of whether that variety is intentional (musical intelligence) or accidental (prediction errors). exp51 needs human validation before conclusions can be drawn.

3. **exp51 should be included in the next human AR evaluation** to test whether its synthetic dominance translates to human preference or is exposed as noise.

4. **exp58 is a safe bet** — competitive with exp44 (the known human-preferred model) across evaluators, with the highest per-sample accuracy. If exp51 turns out to be noise, exp58 is the likely best overall model.
