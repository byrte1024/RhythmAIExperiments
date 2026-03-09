# Experiment 12 — Stronger Context Path + AR Augmentation

## Hypothesis

Experiment 11 showed the two-path architecture working well — audio is now the primary signal (no_events=36.8% >> no_audio=15.5%) and the audio path proposes excellent candidates (top-3 = 84%, top-10 = 95%). But two problems remain:

**1. The context path is the selection bottleneck.**

Top-K accuracy analysis showed all bars (top-1 through top-10) improving at the same linear rate across epochs. This means the audio path is getting better at proposing candidates, but the context path isn't getting any better at selecting from them. The gap between top-1 (65%) and top-3 (86%) stayed roughly constant — roughly 20% of samples have the correct answer ranked 2nd or 3rd, and the context path can't disambiguate.

To address this, three changes were made:
- **Event encoder widened**: d_event increased from 128 to 192 (with 6 attention heads instead of 4), giving the context path richer event representations to work with. Event encoder depth increased from 2 to 3 layers.
- **Context path deepened**: context_path_layers increased from 3 to 4, giving it more capacity for the selection task.
- **Dedicated context auxiliary loss**: The model now returns all three logit tensors (combined, audio, context). Loss changed from `main + 0.2 * audio_aux` to `main + 0.1 * audio_aux + 0.1 * context_aux`. Both paths now receive equal direct training signal. Previously, the audio path got 1.2x the gradient (main + aux) while the context path only got 1x (main only), which may have contributed to the context path underperforming.

Total parameter increase: ~21M → ~24.5M (+17%), all invested in the selector side (event_encoder 0.5M→1.5M, context_path 7.5M→9.9M).

**2. Autoregressive drift during inference.**

The model trains on ground truth event history but infers on its own predictions. Each prediction error shifts the event context for subsequent predictions, and these errors compound over the duration of a song. Inference results showed good local accuracy but increasing drift from ground truth over time.

To address this, event augmentation was redesigned to simulate the kinds of errors the model produces during autoregressive inference:
- **Recency-scaled jitter**: Instead of uniform ±4 bin jitter, noise scales from 1x (oldest events) to 3x (most recent events). This mimics real AR behavior where recent predictions are less reliable because they've had less opportunity to be corrected by subsequent context.
- **Global shift**: A uniform ±3 bin shift applied to ALL events simultaneously, simulating systematic timing drift where the model is consistently a bit early or late.
- **Random deletion (8%)**: Drop 1 to N/6 individual events, simulating missed beats during inference.
- **Random insertion (8%)**: Add 1 to N/6 spurious events at random positions within the existing event range, simulating false positive predictions.
- Existing augmentations (5% full dropout, 10% truncation) remain unchanged.

The augmentation rates are deliberately light — the goal is to expose the model to slightly messy event histories that resemble real inference output, not to corrupt the data so heavily that events become useless (the lesson from experiment 07).

## Result

*Training in progress.*

## Lesson

*Pending.*
