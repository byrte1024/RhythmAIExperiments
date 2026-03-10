# Experiment 15 - Context Aux Loss + Density Benchmarks

## Hypothesis

Experiment 14 proved the data alignment fix was transformative (50.5% acc, 69% HIT, 30% miss at E8), but revealed a new bottleneck: the context path is dormant. no_events accuracy (50.2%) matches full accuracy (50.4%), meaning the context path contributes almost nothing to the final prediction. The model's ~50% accuracy ceiling is the audio-only ceiling.

**Why the context path is dormant:** The context path found a local minimum — just amplify whatever audio already ranks highest. Since audio is right ~69% of the time, agreeing with audio is a safe strategy. The context path has no incentive to develop independent opinions or override audio's ranking. This manifests as:
- no_events ≈ full accuracy (context adds nothing)
- Top-1 to top-3 gap of ~20% (correct answer often ranked 2nd/3rd by audio, context never overrides)
- Ray patterns in scatter (harmonic confusion that event spacing would directly resolve, but context doesn't intervene)
- Inference ignoring density conditioning

**Why this approach failed in exp 12 but should work now:** Experiment 12 tried `main + 0.1 audio_aux + 0.1 context_aux` and the audio path collapsed into mode collapse. But that was on misaligned data where audio couldn't learn effectively — stealing gradient from it was fatal. Now on clean data, audio is strong and self-sufficient (49-50% accuracy with no events at all). Adding context aux gradient on top rather than redistributing from audio should be safe.

### Changes

**Loss:** `main + 0.2 * audio_aux + 0.1 * context_aux` (was `main + 0.2 * audio_aux`)
- Audio aux stays at 0.2 (unchanged, not starved)
- Context aux added at 0.1 (new direct training signal)
- Total aux weight 0.3 (was 0.2)
- The context path now gets its own gradient pushing it to independently predict the correct answer, not just rubber-stamp audio

**New ablation benchmarks:**
- **Zero density**: conditioning vector set to [0, 0, 0] — tests if density affects predictions at all
- **Random density**: conditioning randomized — tests if the model uses density information or ignores it

Everything else identical to exp 14: same architecture (~21M params), same dataset (taiko_v2 with correct BIN_MS), same AR augmentations.

### Expected outcomes

- Context path begins contributing: no_events accuracy should drop below full accuracy (gap of 5-10% would indicate meaningful context contribution)
- Top-1 to top-3 gap narrows as context learns to select from audio's candidates
- Accuracy breaks past 50% audio-only ceiling
- Ray patterns (harmonic confusion) reduce as context disambiguates 2x/0.5x intervals
- Density benchmarks establish baseline for whether FiLM conditioning is working

### Risk

Adding 0.1 context aux increases total gradient. If the model destabilizes, the context aux may need to be reduced to 0.05. Watch for:
- val_loss increasing or oscillating compared to exp 14
- no_events accuracy *increasing* (would mean audio is degrading)
- Pred distribution collapsing (spikiness, fewer unique values)

## Result

*Pending.*

## Lesson

*Pending.*
