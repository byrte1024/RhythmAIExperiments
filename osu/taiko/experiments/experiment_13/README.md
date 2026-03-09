# Experiment 13 - AR Augmentation Only (Exp 11 Architecture)

## Hypothesis

Experiment 12 showed that increasing the context path capacity while reducing the audio aux loss was catastrophic — the audio proposer collapsed into mode collapse (226 unique preds, horizontal banding in scatter, top-10 only 65%). The lesson: the audio aux loss at 0.2 is load-bearing and the audio path must be strong before the context path can be useful.

Experiment 11 had excellent results (E5: 47.1% acc, 64.8% HIT, top-3 ~86%, top-10 ~95%) but suffered from autoregressive drift during inference — the model trains on ground truth event history but infers on its own noisy predictions, and errors compound over a song's duration.

This experiment keeps exp 11's architecture and loss exactly as-is, and adds only the AR-simulating augmentations from exp 12:

**Architecture (unchanged from exp 11):**
- d_model=384, d_event=128, enc_layers=4, enc_event_layers=2
- audio_path_layers=2, context_path_layers=3, n_heads=8
- Loss: `main + 0.2 * audio_aux` (no context aux)
- ~21M params

**New augmentations (kept from exp 12):**
- **Recency-scaled jitter**: Per-event noise scales from 1x (oldest) to 3x (most recent), simulating how AR errors are larger for recent predictions. Plus a global ±3 bin shift for systematic drift.
- **Random deletion (8%)**: Drop 1 to N/6 individual events to simulate missed beats.
- **Random insertion (8%)**: Add 1 to N/6 spurious events to simulate false positives.
- All existing augmentations unchanged (5% full dropout, 10% truncation, mel augmentations).

The hypothesis is that the AR augmentations will:
1. Reduce autoregressive drift during inference by making the model robust to noisy event histories
2. Maintain or improve the strong per-sample accuracy from exp 11
3. Show up as better metronome/time_shifted/random_events benchmark scores (model more robust to bad context)

This is a controlled test: the only variable vs exp 11 is the event augmentation.

## Result

*Training in progress.*

## Lesson

*Pending.*
