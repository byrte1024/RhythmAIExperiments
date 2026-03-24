# Experiment 49 - Virtual Tokens for Out-of-Window Context

## Hypothesis

Exp 48 showed 14.2% of failures are universal across all architectures, with 2x/0.5x metric confusion as the dominant error. The model can't distinguish beat from sub-beat because it only sees 2.5s of audio + 2.5s of in-window events. A full musical phrase at 120 BPM is 8 seconds — the model can't see one.

The model HAS 128 context events spanning ~25 seconds of history, but events outside the 2.5s audio window are discarded because there's no audio token to place them on.

**Virtual tokens** solve this by adding V extra tokens before the audio sequence. Out-of-window events get placed on these tokens via the same scatter-add mechanism. The transformer attends to virtual tokens alongside audio tokens — one unified sequence, no cross-attention.

### Architecture

```
[V virtual tokens] + [250 audio tokens] = V+250 total tokens

Virtual tokens:  learned watermark embedding + position embedding + event embeddings (scatter-add)
Audio tokens:    conv stem output + position embedding + event embeddings (scatter-add, unchanged)
Cursor:          token V+125 (was 125)
```

Virtual tokens are NOT silence — they carry a learned watermark that signals "this is context, not audio." The transformer can attend to them for rhythm history without confusing them with quiet audio sections.

### Event mapping

- In-window events (offset >= -500): mapped to audio tokens 0-124 (unchanged)
- Out-of-window events (offset < -500): mapped to virtual tokens 0 to V-1
  - Linear mapping: earlier events → lower token indices
  - More recent out-of-window events → closer to the audio boundary (token V-1)

### Key parameters

- `--n-virtual-tokens V` (default 32): number of virtual tokens
- Default V=32 covers ~25s of history at ~0.8s per token resolution

### Risks

- Virtual tokens add V*d_model parameters and increase attention cost by (V/250)^2
- The watermark must be distinctive enough that the model doesn't confuse virtual tokens with silence
- Out-of-window events have no audio association — the model must learn to use rhythm-only information

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_49 --model-type event_embed --n-virtual-tokens 32 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
