# Experiment 41 - Deep Entropy Analysis (Diagnostic)

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

The model's entropy rises sharply with target distance. Exp [40](../experiment_40/README.md) proved this isn't from undertraining — more exposure to distant bins didn't help. The question: **why is entropy higher?**

Two competing explanations:
- **A) More valid onsets in the window** — at target=200, there might be 5 onsets between cursor and target. The model correctly hedges across all of them. Entropy reflects genuine ambiguity.
- **B) The model can't read distant audio** — the cursor bottleneck means distant targets require multi-hop attention, degrading information quality. Entropy reflects model weakness.

If entropy correlates more strongly with `n_onsets_between_cursor_and_target` than with `target_distance` itself, it's explanation A (fundamental ambiguity). If entropy correlates with distance independent of onset count, it's explanation B (model limitation).

### Method

Measure correlations between entropy and:
- Target distance (bin offset)
- Number of future onsets in window
- Number of onsets SKIPPED (between cursor and the PREDICTED position, for overpredictions)
- Context length (past events)
- Density conditioning
- Audio features at target (mel energy, spectral flux)
- Prediction correctness (HIT/MISS)
- Top-1 confidence

Break down entropy by distance bins, by onset count, and by obstacle count.

## Result

**The skip count is the primary predictor of failure, not target distance or model weakness.**

### Correlations with entropy

| Feature | r | Interpretation |
|---|---|---|
| top1_confidence | **-0.846** | Tautological — low confidence = high entropy |
| target_distance | **+0.569** | Further targets = more uncertainty |
| n_future_onsets | **-0.543** | More onsets = lower entropy (denser = easier) |
| density_mean | **-0.544** | Nearly identical to n_future_onsets — density is the same signal |
| is_hit | -0.368 | Wrong predictions have higher entropy |
| n_skipped_onsets | +0.271 | Skipping onsets increases entropy |
| context_length | **-0.210** | More context = lower entropy (context helps!) |
| target_mel_energy | -0.206 | Louder targets are easier |
| target_spectral_flux | +0.073 | Negligible |

### The smoking gun: entropy and HIT by skip count

Skip = how many real onsets ahead of the target did the model predict. Skip=1 means the model predicted the 2nd onset instead of the 1st.

| Skip | Samples | % | Entropy | Conf | HIT | Avg target dist |
|---|---|---|---|---|---|---|
| under | 9,534 | 19.3% | 2.473 | 0.324 | **46.5%** | 54.5 |
| 0 (hit/near) | 33,021 | 66.9% | 2.271 | 0.380 | **93.7%** | 36.0 |
| 1 | 5,533 | 11.2% | 2.780 | 0.287 | **0.0%** | 30.2 |
| 2 | 942 | 1.9% | 3.103 | 0.251 | **0.0%** | 32.1 |
| 3 | 214 | 0.4% | 3.211 | 0.255 | **0.0%** | 28.4 |

**Target distance is nearly identical across all skip counts (~28-36 bins).** The model doesn't fail because the target is far — it fails because there's a more salient onset further ahead.

- **Skip 0 (67%)**: 93.7% HIT — when the model doesn't overshoot past a real onset, it's almost always correct
- **Skip 1 (11%)**: 0% HIT — the model jumped to the 2nd onset. Always wrong by definition.
- **Skip 2+ (2%)**: 0% HIT — jumped even further
- **Underprediction (19%)**: 46.5% HIT — predicted before the target, some within tolerance (46.5% HIT means many are close misses)

### Key correlations explained

**n_future_onsets negative correlation (-0.543)**: More future onsets = denser section = shorter gaps = lower target distance = easier predictions. Dense sections are easy; sparse sections are hard. Density_mean shows the same signal (-0.544).

**context_length negative correlation (-0.210)**: More past context = lower entropy. **The model genuinely uses context** — more past events help it be more confident. This validates the mel ramp approach and suggests more context could help further.

**target_distance positive correlation (+0.569)**: Confounded with density — further targets exist in sparser sections with fewer onsets. But the skip analysis shows distance alone isn't the cause: skip=0 has 85.4% HIT at avg distance 40, while skip=1 has 43.5% HIT at avg distance 35. The distance is LOWER for the failures.

### Entropy by target distance

| Range | N | Entropy | Conf | HIT | FutOnsets | Skipped |
|---|---|---|---|---|---|---|
| 0-15 | 7,270 | 1.814 | 0.422 | 70.3% | 27.0 | 0.5 |
| 15-30 | 27,539 | 2.026 | 0.404 | 76.4% | 19.8 | 0.4 |
| 30-60 | 25,798 | 2.579 | 0.321 | 70.1% | 13.8 | 0.3 |
| 60-100 | 9,799 | 2.964 | 0.298 | 70.3% | 9.2 | 0.3 |
| 100-200 | 3,028 | 3.346 | 0.260 | 54.4% | 5.5 | 0.3 |
| 200-500 | 641 | 3.615 | 0.242 | 38.4% | 3.3 | 0.3 |

Entropy rises with distance, but HIT only drops significantly at 100+ bins. The 15-100 range maintains ~70-76% HIT despite rising entropy — the model is uncertain but still usually correct.

## Lesson

- **The failure modes are precisely quantified:**
  - **Skip 1+ (13.5%)**: model jumps to a further real onset. Always 0% HIT. 5,533 samples skip exactly 1 — fixing these pushes HIT from 71.6% to ~83%.
  - **Underprediction (19.3%)**: model predicts before the target. 46.5% HIT means many are close misses, 53.5% are genuine errors.
  - **Skip 0 (66.9%)**: 93.7% HIT — the model is excellent when it doesn't overshoot past a real onset.
- **Skip count, not target distance, is the failure mode.** Target distance is ~30-36 bins across all skip counts. The competing onset causes the failure, not the distance.
- **Context helps: r=-0.213.** More past events = lower entropy. The mel ramps are working. This is the strongest evidence that context has untapped potential.
- **Density/future onsets are the same signal as distance (r=-0.543/-0.544).** Dense sections have short gaps = easy. Sparse sections have long gaps = hard. Not because of distance but because sparse sections have more competing onsets per window.
- **Two problems to solve:**
  1. **Entropy increasing with target distance** — partly fundamental (sparse sections are harder), partly the cursor bottleneck. Context correlation (-0.213) suggests room to improve.
  2. **Skip-1 overpredictions (11.2%)** — model predicts the 2nd onset instead of the 1st. The sharper transient at the further onset (exp [39-E](../experiment_39e/README.md): 78% have stronger flux) pulls the model past the nearest one.
  3. **Underpredictions (19.3%)** — model predicts too early. Some are near-misses (46.5% HIT), others are genuine hallucinations or rhythm interpolation errors.
