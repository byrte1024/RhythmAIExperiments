# Experiment 66-1 — Full Architecture Specification

## Task

Score chart quality. Given a 10-second window of audio + chart events, output a scalar quality score. Trained via pairwise comparison (Bradley-Terry loss) on corruption pairs and cross-set human rating pairs.

## Input

| Input | Shape | Description |
|---|---|---|
| mel | (B, 80, 2000) | Mel spectrogram, 80 bands, 2000 frames (10s at 5ms/frame) |
| events | (B, N_max) | Event positions within window (bin indices, int64, zero-padded) |
| event_mask | (B, N_max) | Bool mask, True = padding (no event) |
| star_rating | (B,) | osu! star rating for difficulty conditioning |

N_max set to 256 (covers p99 of events-per-10s-window).

## Output

| Output | Shape | Description |
|---|---|---|
| score | (B,) | Scalar quality score (unbounded, higher = better) |

## Model: ChartQualityEvaluator

### Mel Compression

```
mel (B, 80, 2000)
  → Linear(80, 16)  # learned compression, kills timbral detail
  → (B, 16, 2000)
```

16 bins preserve onset/energy/coarse spectral shape. Destroy production quality, mixing, bitrate information.

### Audio Augmentation (rating pairs only, training only)

```
mel_compressed (B, 16, 2000):
  → + uniform(-0.3, 0.3) per-sample gain        # ±6dB in log-mel space
  → zero out 2-4 random freq bins                # random EQ destruction
  → zero out 2-4 random time segments (50-200 frames each)  # temporal masking
  → + gaussian noise, std ~ uniform(0, 0.3)      # SNR degradation
```

Not applied to corruption pairs — audio identical there, augmentation would add noise to a clean signal.

### Conv Stem

```
mel_compressed (B, 16, 2000)
  → Conv1d(16, 128, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 128)
  → Conv1d(128, 256, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 500, 256)
  → LayerNorm(256)
  → + SinusoidalPosEmb(positions 0..499)
  → x: (B, 500, 256) audio tokens
```

4x downsample: 2000 frames → 500 tokens. Each token covers ~20ms (4 mel frames).

### Star Rating Conditioning

```
star_bucket = clamp(floor(star_rating / 0.5), 0, 19)  # 20 buckets: 0-0.5*, 0.5-1*, ..., 9.5-10*
star_emb = Embedding(20, 256)(star_bucket)             # (B, 256)
x = x + star_emb.unsqueeze(1)                         # broadcast to all tokens
```

Model evaluates quality relative to the intended difficulty.

### Event Embeddings

For each event i in the window, compute 3 features:

**Ratio before:**
```
ratio_before[i] = gap[i] / gap[i-1], clamped [0.1, 10.0]
ratio_before[0] = 1.0  # first event: neutral
→ * 50 → int → SinusoidalPosEmb(256)  # range 5-500 for sinusoidal resolution
```

**Ratio after:**
```
ratio_after[i] = gap[i+1] / gap[i], clamped [0.1, 10.0]
ratio_after[-1] = 1.0  # last event: neutral
→ * 50 → int → SinusoidalPosEmb(256)
```

**Absolute gap (ms):**
```
gap_ms[i] = gap[i] * 5  # convert bins to ms
→ int → SinusoidalPosEmb(256)  # raw ms value, no scaling
```

Where `gap[i] = events[i] - events[i-1]` (inter-onset interval in bins).

**Projection:**
```
[ratio_before (256) | ratio_after (256) | gap_ms (256)]
→ Linear(768, 256) → GELU → Linear(256, 256)
→ event_emb (B, N_valid, 256)
```

**Scatter-add into audio tokens:**
```
token_pos[i] = events[i] // 4   # map event bin to audio token index
x[b].scatter_add_(0, token_pos.unsqueeze(-1).expand(-1, 256), event_embs)
```

Multiple events on the same token are summed (scatter_add_). Tokens with no events receive no event embedding — the model distinguishes "audio with event" from "audio without event" by the presence/absence of the additive signal.

### Transformer

```
for 6 layers:
    x = TransformerEncoderLayer(
        d_model=256, nhead=8, dim_feedforward=1024,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
```

No FiLM conditioning (star rating already added before transformer). Pre-norm for stability.

### Attention Pooling

```
query = Parameter(256,)  # single learnable query vector
attn_weights = softmax(x @ query / sqrt(256))  # (B, 500)
pooled = (attn_weights.unsqueeze(-1) * x).sum(dim=1)  # (B, 256)
```

The query learns to attend to quality-relevant positions (e.g., sections where events misalign with audio energy).

### Output Head

```
score = Linear(256, 1)(pooled).squeeze(-1)  # (B,)
```

Unbounded scalar. Higher = better quality.

## Loss

### Bradley-Terry with Adaptive Margin

For a pair (a, b) where a is known to be better:

```
diff = score_a - score_b
margin = alpha * level_gap
loss = -log(sigmoid(diff - margin) + 1e-8)
```

**Corruption pairs:** level_gap ∈ {1, 2, 3, 4} based on corruption level distance.

| Pair | level_gap |
|---|---|
| CLEAN vs LIGHT | 1 |
| CLEAN vs MED | 2 |
| CLEAN vs HIGH | 3 |
| CLEAN vs GARBAGE | 4 |
| LIGHT vs MED | 1 |
| LIGHT vs HIGH | 2 |
| LIGHT vs GARBAGE | 3 |
| MED vs HIGH | 1 |
| MED vs GARBAGE | 2 |
| HIGH vs GARBAGE | 1 |

**Rating pairs:** level_gap = |rating_a - rating_b| (continuous, typically 1.0-3.0).

**alpha = 0.1** (tunable). Controls how much larger score gap is demanded for wider quality differences.

## Corruption Recipes

Order of operations: delete → insert → per-event jitter → all-event jitter → sort → merge → clamp.

### CLEAN
No modification.

### LIGHT
```
per_event_jitter:  uniform(-2, +2) bins per event      # ±10ms
all_event_jitter:  uniform(-2, +2) bins, one roll       # ±10ms shift to all
insert_center:     for each gap, 1% chance to add event at midpoint
delete:            for each event, 1% chance to remove
insert_offset:     for each event, 1% chance to add event at (pos + X),
                   X sampled from global gap distribution
```

### MED
```
per_event_jitter:  uniform(-6, +6) bins per event      # ±30ms
all_event_jitter:  uniform(-6, +6) bins, one roll       # ±30ms
insert_center:     5% per gap
delete:            5% per event
insert_offset:     5% per event, X from global gap distribution
```

### HIGH
```
per_event_jitter:  uniform(-20, +20) bins per event    # ±100ms
all_event_jitter:  uniform(-50, +50) bins, one roll     # ±250ms
insert_center:     25% per gap
delete:            15% per event
insert_offset:     10% per event, X from global gap distribution
```

### GARBAGE
```
Keep same number of events as original chart.
Generate fully random gap sequence:
  - Sample each gap independently from global gap distribution
  - global_gap_dist: empirical distribution over 6.9M gaps from taiko_v2
  - Place events cumulatively from position 0
```

### Post-corruption cleanup
```
1. Sort events ascending
2. Merge events within 2 bins (10ms): keep first, delete duplicates
3. Remove any event at position < 0
4. Clamp all positions to [0, mel_frames - 1]
```

### Global gap distribution

Precomputed from all taiko_v2 charts (6.9M gaps). Stored as (gap_value, count) pairs. Used for:
- LIGHT/MED/HIGH insert_offset sampling
- GARBAGE gap generation

Key stats: 77.5% of gaps ≤ 250ms (50 bins). Peaks at 80ms (16 bins) and 160ms (32 bins). Range: 3-10596 bins.

## Data Sampling

### Pair construction per batch

**Corruption pairs (60% of batch):**
1. Sample a chart from taiko_v2
2. Sample two corruption levels (e.g., CLEAN and HIGH)
3. Apply each corruption to the same chart
4. Sample the same random 10s window from both
5. Label: less corrupted is better, level_gap = level distance

**Rating pairs (40% of batch):**
1. Sample two beatmapsets with rating gap ≥ 1.0
2. Filter: star_rating within ±0.5 of each other (pick one chart per set from matching difficulty)
3. Sample a random 10s window from each (different audio)
4. Apply audio augmentation to both
5. Label: higher-rated beatmapset is better, level_gap = |rating_a - rating_b|

### Window sampling

For a given chart with mel (80, T) and events array:
```
max_start = T - 2000
start = randint(0, max_start)
end = start + 2000
mel_window = mel[:, start:end]                # (80, 2000)
events_in_window = events[(events >= start) & (events < end)] - start  # shift to window-relative
```

Padding if chart is shorter than 10s (rare — min duration in dataset is ~30s).

### Validation split

Hold out 250 beatmapsets (~10%, by song, seed 42). Same split as detector training. Validation metrics:
- **Corruption accuracy**: pairwise accuracy per margin tier (1, 2, 3, 4)
- **Rating accuracy**: pairwise accuracy on held-out rating pairs
- **Mean score by level**: average model score for CLEAN, LIGHT, MED, HIGH, GARBAGE (should be monotonically decreasing)

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW |
| Batch size | 256 pairs (512 forward passes) |
| Phase 1 LR | 3e-4 |
| Phase 1 epochs | 20 |
| Phase 1 data | 100% corruption pairs |
| Phase 2 LR | 3e-5 |
| Phase 2 epochs | 10-15 |
| Phase 2 data | 60% corruption + 40% rating |
| Weight decay | 0.01 |
| Scheduler | CosineAnnealingLR |
| Dropout | 0.1 |
| Early stopping | Val rating accuracy (phase 2) |

## Inference Modes

### 1. Absolute score

```
Sample 8-16 windows uniformly across the song
Score each window independently
Return mean score
```

### 2. Pairwise comparison

```
Score chart A and chart B (same number of windows)
Higher mean score wins
```

### 3. Quality curve

```
Score windows at regular intervals (e.g., every 5s with 10s window, 50% overlap)
Plot score over time
Identify low-quality sections
```

## Estimated Parameters

| Component | Params |
|---|---|
| Mel compression | 80 × 16 = 1.3K |
| Conv stem | 16×128×7 + 128×256×7 = 244K |
| Star rating embedding | 20 × 256 = 5K |
| Event projection | 768×256 + 256×256 = 262K |
| Sinusoidal embeddings | 0 (no learned params) |
| Transformer (6 layers) | 6 × (4 × 256² + 2 × 256 × 1024) = 4.7M |
| Attention pool query | 256 |
| Output head | 256 + 1 = 257 |
| **Total** | **~5.2M** |

## Dataset: taiko_v2

10,048 charts, 2,489 beatmapsets. Mel: 80 bands, hop=110, SR=22050, n_fft=2048, 20-8000 Hz, ~5ms/frame. Events: int32 bin indices. Rating: 1-10 float (per-beatmapset), fetched via osu! API v1. 90/10 val split by song (seed 42).

## Environment

Running on CachyOS machine (same as exp 44-RE).

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 4060 (8 GB) |
| OS | CachyOS (Linux) |
| torch.compile | Available (Linux/Triton) |
