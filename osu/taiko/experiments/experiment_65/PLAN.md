# Experiment 65 — 3-Stage Decomposed Onset Detector (PLANNING)

## Philosophy

Instead of one model trying to predict the next onset, decompose into 3 independent questions:
- **S1:** "Where does the audio suggest onsets?" (signal detection)
- **S2:** "What gap makes rhythmic sense?" (pattern continuation)
- **S3:** "Given both signals, what's the final answer?" (fusion)

S2 is the key innovation — a model that MUST use context because it has nothing else.

## S1: Audio Proposer

**Goal:** Per-bin onset confidence from audio alone.

### Architecture: Conformer (CNN-in-Transformer)

Literature says Conformer outperforms pure transformer for audio. Each block:
```
x → FFN(half) → MHSA → Conv1d → FFN(half) → x + residual
```

| Param | Value | Rationale |
|---|---|---|
| Input | mel (80 bands, hop=110, SR=22050) | Same as current |
| Conv stem | 2x stride-2 Conv1d → 4x downsample | Same as current |
| Layers | **8** (up from 4) | SOTA uses 6-12 |
| d_model | 384 | Keep same |
| n_heads | 8 | Keep same |
| Conv kernel in Conformer | 31 | Standard for audio conformer |
| Dropout | 0.1 | Standard |
| Output | per-token sigmoid (n_audio_tokens,) | Same as current |

**No conditioning, no events.** Pure audio → onset confidence.

**Training:** Focal BCE against binary onset targets (same as current S1).

**Estimated params:** ~12M (up from ~4M current proposer)

### Improvements over current S1
- Conformer blocks add local conv inductive bias (transient detection)
- 8 layers vs 4 gives deeper feature hierarchy
- Still bidirectional (sees full window)

---

## S2: Context Predictor (NEW)

**Goal:** Per-bin onset probability from rhythm pattern alone. No audio.

### Architecture: GRU + Transformer hybrid (TPP-inspired)

Based on DTPP (NeurIPS 2024) and TPP benchmark findings:

### Input Encoding (per event, 128 past events)

Each event i encoded as:
```
event_i = [
    log_gap_emb(log(gap_i)),           # sinusoidal encoding of log(gap)
    ratio_emb(log(gap_i / gap_{i-1})), # sinusoidal encoding of log-ratio
    abs_gap_emb(gap_i),                # sinusoidal encoding of raw gap (for scale)
]
→ Linear(3 * d_model, d_model) → event_token (d_model,)
```

### History Encoder

Two options to evaluate:

**Option A: GRU (simpler, TPP benchmark winner)**
```
event_tokens (128, d_model) → Bidirectional GRU(d_model, 4 layers) → context (d_model)
```

**Option B: Causal Transformer (more expressive)**
```
event_tokens (128, d_model) → 6 causal transformer layers → last token = context (d_model)
```

### Candidate Scoring (per-bin output)

For each of 250 possible next bins, encode what choosing that bin would mean:
```
candidate_b = [
    log_gap_emb(log(b)),                    # what absolute gap this creates
    ratio_emb(log(b / last_gap)),           # what ratio this creates
]
→ Linear(2 * d_model, d_model) → candidate_token (d_model,)
```

Score each candidate against the context:
```
score(b) = MLP(context ⊕ candidate_b)  # or dot product
→ 250 scores → softmax → probability distribution
```

### Alternative: Direct 250-class head
```
context (d_model) → Linear(d_model, 250) → softmax
```
Simpler but doesn't encode per-candidate features.

| Param | Value |
|---|---|
| d_model | 256 (smaller than S1 — less data to process) |
| History length | 128 events |
| Encoder | GRU 4-layer bidirectional OR Transformer 6-layer |
| Output | 250 bins + STOP = 251 classes |
| No audio input | By design |
| No S1 input | By design |

**Training:** Standard CE or LogNormMix NLL against ground truth next gap.

**Estimated params:** ~3-5M

### Key design insight from DTPP
The gap-ratio (which "bucket" — 1x, 0.5x, 2x) can be predicted by the Transformer,
while the exact timing within that bucket can be refined by a simple parametric model.
We may not need this decomposition if direct 251-class prediction works.

---

## S3: Fusion Selector

**Goal:** Final onset prediction. Sees audio + S1 confidence + S2 confidence + context.

### Architecture: Transformer with dual proposal embeddings

```
Input:
  - Audio tokens from conv stem (shared with S1 or separate)
  - S1 confidence → proposal_embed_s1 (same as current)
  - S2 confidence → proposal_embed_s2 (NEW, same mechanism)
  - Event embeddings (scatter-added, same as current)
  - Density FiLM conditioning

audio_tokens = conv_stem(mel)  # (B, n_tokens, d_model)
audio_tokens += proposal_embed_s1(s1_conf)  # add S1 signal
audio_tokens += proposal_embed_s2(s2_conf)  # add S2 signal
scatter_add(audio_tokens, event_embeddings)
FiLM(audio_tokens, density_cond)

for 8 transformer layers:
    audio_tokens = TransformerBlock(audio_tokens)
    audio_tokens = FiLM(audio_tokens, density_cond)

logits = output_head(audio_tokens[cursor_token])
```

| Param | Value |
|---|---|
| Conv stem | Same as S1 (shared or separate) |
| d_model | 384 |
| Layers | 8 |
| S1 embed | Linear(1, 384) → GELU → Linear(384, 384) |
| S2 embed | Linear(1, 384) → GELU → Linear(384, 384) |
| Event embeddings | Same as current (gap ratios + presence + gaps) |
| FiLM | Same as current |
| Output | 251 classes (or multi-onset) |

**Training:** OnsetLoss (hard + soft CE + ramp) against ground truth.

**Estimated params:** ~16M (similar to current)

---

## Total System

| Stage | Params | Input | Output |
|---|---|---|---|
| S1 | ~12M | audio only | per-bin confidence |
| S2 | ~3-5M | gap history only | per-bin confidence |
| S3 | ~16M | audio + S1 + S2 + events + density | final logits |
| **Total** | **~31-33M** | | |

vs current ProposeSelectDetector: ~23.5M

---

## Training Strategy

### Phase 1: Train S1 alone (freeze S2, S3)
- Same as current S1 freeze phase
- 2-4 evals, focal BCE

### Phase 2: Train S2 alone (separate, no gradients to S1/S3)
- Pure context prediction task
- Can be trained on the ENTIRE dataset independently
- CE loss against ground truth next gap
- This is the critical experiment — how good is pure context prediction?

### Phase 3: Train S3 with frozen S1 + S2
- S1 and S2 provide fixed confidence maps
- S3 learns to fuse them
- OnsetLoss

### Phase 4: End-to-end fine-tuning (optional)
- Unfreeze all, low LR
- Risk of S1/S2 collapsing — may need gradient scaling

---

## Inference Sampling Matrix

### Combination methods:
| Method | Formula |
|---|---|
| ADD | α·S1 + β·S2 + γ·S3 |
| MULTIPLY | S1^α · S2^β · S3^γ (normalized) |
| S1_ONLY | S1 |
| S2_ONLY | S2 |
| S3_ONLY | S3 |

### Sampling methods:
| Method | Description |
|---|---|
| MAX | argmax of combined distribution |
| FIRST_THRESHOLD | first bin above threshold |
| ALL_THRESHOLD | all bins above threshold (framewise-style) |
| TEMPERATURE | sample from distribution with temperature |

### Key experiments to run at inference:
1. S2_ONLY — how good is pure context? This answers THE CONTEXT ISSUE
2. S1_ONLY vs S3_ONLY — does S3 add value over raw audio?
3. ADD vs MULTIPLY — which fusion works better?
4. S3 with S2 zeroed vs S3 with S2 — does S2 signal help S3?

---

## Open Questions

1. **Should S1 and S3 share the conv stem?** Sharing saves compute but couples gradients. Separate is cleaner but more params.

2. **S2 candidate scoring vs direct head?** Per-candidate scoring is more principled but 250x more computation. Direct 251-class head is simpler and may work just as well given enough capacity.

3. **S2 history window?** TPP literature says 16-32 events may suffice. We have 128 — overkill or necessary for music?

4. **Should S2 see density conditioning?** It's "context only" but density IS context. Could help S2 predict appropriate gap distributions for the song's difficulty.

5. **Multi-onset?** Start with single-onset, add delta-encoding later if S2 works.

6. **Ramp loss in S2?** Same gradient-everywhere benefit applies to context prediction.
