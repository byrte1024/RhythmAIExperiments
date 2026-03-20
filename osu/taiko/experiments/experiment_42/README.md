# Experiment 42 - Event Embedding Detector

## Hypothesis

40+ experiments of context fusion have established:
- Separate gap tokens get drowned in self-attention (exp 25-30, delta collapses to ~1.5%)
- Cross-attention creates magnitude imbalance and banding (exp 31-33)
- FiLM can't encode sequences (exp 34, 64-dim bottleneck)
- Mel ramps work (exp 35-C, 5% delta, 71.6% HIT) but are synthetic signals the model processes differently from audio
- Context helps confidence (r=-0.213, exp 41) — more context = lower entropy
- Skip-1 errors are structural (~11%, exp 41-B) — model picks sharper transient at further onset

**Event embedding detector** takes the core insight from mel ramps (context in the audio pathway) but implements it properly: instead of synthetic ramps in the mel spectrogram, add **learned embeddings directly to audio tokens** at event positions.

After the conv stem produces 250 audio tokens, each token corresponding to a past event position gets an enriched embedding:
- **Event presence** — learned vector marking "there's a mapped onset here"
- **Gap before** — sinusoidal encoding of the gap from the previous event
- **Gap after** — sinusoidal encoding of the gap to the next event (except the most recent event, where this would reveal the target)

These three are projected through an MLP and added to the audio token. The self-attention then processes 250 tokens — some are pure audio, some are audio + event context. No separate tokens, no cross-attention, no synthetic mel signals.

### Why this should work

- **Same pathway as audio** — event info enriches existing tokens, processed by the same attention layers
- **Not drowned** — no competing token set. Event embeddings modify existing tokens, not compete with them
- **Sequential** — events are at their actual temporal positions (unlike FiLM's single vector)
- **Learned** — the model discovers what "event here" means in its own representation space
- **Gap encoding** — directly tells the model "the rhythm has been 75, 75, 150" through gap_before/gap_after
- **Not bypassable** — embeddings are added to token features, no skip connection to route around

### Architecture

```
mel (80, 1000) → Conv stem → 250 tokens (d_model=384)
                                    ↓
              Add event embeddings at past event token positions:
                event_proj(presence + gap_before_emb + gap_after_emb)
                                    ↓
              Self-attention (8 layers, FiLM conditioning)
                                    ↓
              Cursor at token 125 → output head → 501 logits
```

16.1M params. No GapEncoder, no mel ramps, no separate context tokens.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_42 --model-type event_embed --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
