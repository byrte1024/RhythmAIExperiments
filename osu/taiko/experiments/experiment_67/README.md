# Experiment 67 — Ratio-Based Onset Prediction

## Hypothesis

All 70+ experiments predict onset positions as bin offsets (0-250). This mashes together three separate questions: "what's the rhythm?", "where am I in it?", and "what comes next?" The model must rediscover the rhythmic grid every prediction.

Ratio-based prediction decomposes this into three sequential heads:
1. **Divisor** — the dominant rhythmic gap (the "beat")
2. **Offset** — cursor distance from the last event
3. **Ratio** — what multiple of the divisor is the next onset?

The final position: `onset = divisor × ratio − offset`

### Data analysis (from experiment_67/analyze_ratios.py)

- 91.6% of targets hit clean musical ratios (0.5x, 1.0x, 2.0x, etc.)
- Three ratios cover 82%: 1.0x (47%), 0.5x (17%), 2.0x (18%)
- 97.4% coverage with top-3 dominant gaps
- Dominant gap median: 30 bins (150ms)

The ratio space is highly structured — the model classifies among ~255 ratio bins instead of 251 arbitrary frame offsets.

## Architecture

ProposeSelectDetector backbone (same as exp58) + RatioHeads (3 output heads).

### Backbone (identical to exp58)
```
mel → Conv stem → 250 audio tokens
  → Stage 1 Proposer (4 layers, pure audio) → proposal logits
  → Proposal embedding added to tokens
  → Stage 2 Selector (8 layers, events + FiLM + proposals)
  → cursor_token (384-dim)
```

### Head 1: Divisor (auxiliary)
```
cursor_token → LayerNorm → Linear(384, 192) → GELU → Linear(192, 250)
Target: dominant gap from last 128 events (most frequent gap cluster)
Loss: CE, weight α = 0.1 (stop gradient from primary loss)
```

### Head 2: Offset (auxiliary)
```
cursor_token → LayerNorm → Linear(384, 192) → GELU → Linear(192, 100)
Target: cursor distance from last event (0 normally, >0 after STOP hops)
Loss: CE, weight α = 0.1 (stop gradient from primary loss)
```

### Head 3: Ratio (primary, sees Head 1+2)
```
divisor_value = softmax_expectation(Head1)
offset_value = softmax_expectation(Head2)
divisor_emb = MLP(divisor_value) → (384,)
offset_emb = MLP(offset_value) → (384,)
ratio_input = cursor_token + divisor_emb + offset_emb
→ LayerNorm → Linear(384, 384) → GELU → Linear(384, 256)
→ 255 ratio bins (log-spaced 0.125x-8.0x) + STOP
```

### Ratio Bins
255 bins log-spaced from 0.125 to 8.0, center at 1.0x:
- 42 bins per octave, ~1.65% resolution
- 6 octaves: 0.125-0.25, 0.25-0.5, 0.5-1.0, 1.0-2.0, 2.0-4.0, 4.0-8.0
- All musical ratios within 0.7% of a bin

### Loss
```
Loss A (Heads 1+2): divisor_CE + offset_CE [stop gradient from Loss B]
Loss B (Head 3): OnsetLoss on derived position + ratio hill loss
  - Hill loss: expected distance in log-ratio space from correct bin
  - Dynamic ratio target: (target + offset) / divisor_pred

Total = loss_B + 0.1 * loss_A + 0.5 * s1_loss
```

### Warmup
Eval 1: Only s1_loss + loss_A. Divisor/offset heads train. Ratio head frozen.
Eval 2+: All losses active. Ratio head unfreezes. Configurable via `--ratio-freeze-evals`.

## Configuration

| Param | Value |
|---|---|
| Backbone | ProposeSelectDetector (same as exp58) |
| RatioHeads | 761K extra params |
| Total params | ~24.3M |
| A_BINS / B_BINS | 500 / 500 |
| B_PRED | 250 |
| Ratio bins | 255 (log 0.125-8.0) + STOP = 256 |
| Divisor bins | 250 |
| Offset bins | 100 |
| Warm-start | exp58 eval 2 (S1 only) |
| Proposer freeze | 0 evals (S1 warm-started) |
| Ratio freeze | 1 eval (Head 3 frozen while Head 1+2 warm up) |
| ramp_alpha | 2.5 |
| Cursor offset aug | 30% (shift cursor between events) |
| Gap ratios | ON |
| Density jitter | ±10% at 30% |

## Metrics

Standard HIT/MISS/accuracy on derived bin position, plus:
- **Divisor accuracy**: % matching dominant gap (exact and ±3 bins)
- **Offset accuracy**: % matching cursor-to-event distance
- **Ratio HIT/MISS**: in ratio space
- **Ratio stop F1**
- **Graphs**: ratio scatter, ratio distributions, divisor/offset distributions

## Launch

```bash
cd osu/taiko
python detection_train.py taiko_v2 --run-name detect_experiment_67 \
    --model-type ratio_propose_select \
    --a-bins 500 --b-pred 250 --gap-ratios \
    --density-jitter-rate 0.30 --density-jitter-pct 0.10 \
    --proposer-freeze-evals 0 \
    --ratio-freeze-evals 1 \
    --warm-start runs/detect_experiment_58/checkpoints/eval_002.pt \
    --ramp-alpha 2.5 \
    --epochs 50 --batch-size 48 --evals-per-epoch 4 --workers 3
```

- `--proposer-freeze-evals 0`: S1 warm-started from exp58 eval 2, no freeze needed
- `--ratio-freeze-evals 1`: Head 3 (ratio) frozen for 1 eval while Heads 1+2 (divisor/offset) warm up

## Result

Stopped after eval 2 (epoch 1.50). Early but revealing.

### Per-Sample Metrics

| Eval | Derived HIT | Derived MISS | Div Acc | Div ±3 | Ratio HIT | Ratio MISS | StpF1 | Val Loss |
|---|---|---|---|---|---|---|---|---|
| 1 (warmup) | 1.4% | 90.7% | 25.0% | 43.5% | 1.8% | 86.6% | 0.000 | 9.206 |
| 2 | **27.7%** | **43.4%** | **65.8%** | **73.9%** | **11.8%** | **44.9%** | **0.531** | **3.288** |

Massive improvement from eval 1 → 2 as ratio head unfreezes. Divisor already at 65.8% accuracy, offset 99.3%.

### Ratio Analysis Graphs (eval 2)

**Ratio scatter (target vs predicted):**
- Clear horizontal banding — the model has a few preferred ratio outputs and snaps to them
- Almost never predicts below 1.0x (avoids downward ratios like 0.5x)
- Concentrates predictions around 1.0x, 2.0x, 3.0x bands

**Divisor distribution:**
- Nearly perfect match between predicted and actual dominant gap distributions
- Scatter shows clear banding at harmonics (predicting 250 on a 500 is fine, 300 on 500 is bad)
- This is expected and acceptable — divisor just needs a consistent reference frame

**Offset distribution:**
- All predictions at 0, all targets at 0
- The 30% cursor offset augmentation may not be working correctly — needs investigation

**Framewise heatmap (NEW failure mode):**
- The derived bin heatmap shows a **blurry, diffuse diagonal** instead of the sharp bright line that direct bin prediction produces
- This is because the final position is computed from two noisy predictions (divisor × ratio), and the multiplication amplifies errors
- Small errors in divisor (±3 bins) multiply with ratio to create ±3×ratio frame-level errors
- This "lossy ratio" effect is a fundamental tradeoff: the ratio decomposition trades frame precision for structural understanding

### Inference Test (manual)

Despite only 27.7% derived HIT, AR inference shows **extremely promising qualitative results**:
- The model clearly attempts to follow rhythmic patterns
- Onset placement follows the music's beat structure
- Heavily out of sync due to compounding divisor×ratio errors
- But the CHARACTER of the output is musical — not metronomic garbage

## Lesson

1. **The ratio head collapses to a few discrete values.** The ratio scatter shows clear horizontal bands — the model outputs only a handful of unique ratio values instead of using the full 255-bin space. This is the primary failure mode. 256 classes should give smooth ratio predictions, but the model snaps to ~5-10 favorite outputs. This may be because the loss landscape around 1.0x is very steep (47% of targets), making other values hard to learn, or the softmax is too sharp. Potential fixes: temperature on ratio logits, label smoothing, or fewer ratio bins (the 256-bin space may be too fine for the model to explore).

2. **Divisor is surprisingly accurate.** 73.9% within ±3 bins at eval 2. The model quickly learns "what's the beat?" from audio + context. Harmonic banding (predicting 2x or 0.5x of true gap) is acceptable since ratio compensates.

3. **Ratio has an upward bias.** The model almost never predicts ratios below 1.0x (0.5x double-time etc.). It defaults to 1.0x-4.0x range. This may be because the training data is dominated by ratio=1.0 (47%) and the model hasn't yet learned the rarer downward ratios.

4. **Lossy ratio multiplication creates a blurry heatmap.** The framewise heatmap shows a diffuse diagonal instead of the sharp bright line that direct bin prediction produces. Two noise sources multiply: divisor error × ratio error = amplified frame error. This is a fundamental tradeoff — decomposed prediction gains structural understanding but loses frame precision.

5. **Ratio bin collapse + lossy multiplication compound.** The few horizontal bands in ratio space become vertical bands in frame space (since the same ratio × different divisors = different frames). Combined with divisor's harmonic banding, the frame-level heatmap becomes a grid of fuzzy blobs instead of a clean diagonal. This is worse than direct bin prediction for frame precision, even though the structural understanding is better.

6. **Offset augmentation needs fixing.** All offsets are 0 in eval — the 30% cursor shift augmentation isn't being applied. This means the model hasn't learned to handle AR STOP hops. Needs debugging.

7. **AR quality is promising despite low per-sample HIT.** The model produces musically-structured output even at 27.7% HIT. This suggests that ratio-based reasoning captures something that raw bin prediction doesn't — the model "understands" rhythmic structure even when imprecise. With more training and offset fix, this could surpass direct bin prediction on AR quality.
