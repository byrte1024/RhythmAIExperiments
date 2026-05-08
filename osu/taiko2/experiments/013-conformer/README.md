# Experiment 013 — Conformer trunk

## Status

`Planned`

## Context

Across taiko1 (124 experiments) and taiko2 (#001–#012, 19
experiments), every successful intervention to date — exp 14's
data-alignment fix, event embeddings, time-stretch, sub-band
spectral-flux input channels — has come from the **input or data
side**. Loss-side, output-head, and reranking experiments
(roughly 30 across both codebases) have produced essentially zero
movement on per-step val miss.

What has stayed **constant** across all 143 experiments is the
core trunk: an 8-layer post-norm Transformer encoder (d_model=384,
n_heads=8, dim_ff=1536, GELU, dropout=0.1), fed by a 2-layer
Conv1d stem with sinusoidal position embeddings and FiLM
conditioning. Pure self-attention, no convolution inside the
trunk, no recurrence, no state-space layers.

[#012](../012-onset-channels/) reached val/single/onset/miss
**0.2331** [exp_012_onset_channels, step 349,690,
val/single/onset/miss] — the lowest in the codebase but only
0.75 pp below #007's 0.2406 [exp_007_time_stretch, step 372,132,
val/single/onset/miss]. Combined with [#011b](../011b-onset-disagreement/)'s
diagnostic that ratio-banding ridges and ~16 % structural-floor
failures recur across every architecture and intervention, the
remaining test surface narrows to **the trunk itself**.

This experiment swaps the 8-layer post-norm Transformer for an
8-layer Conformer (Gulati et al., INTERSPEECH 2020). Each
Conformer block adds a depthwise-convolution module + macaron-
style FFN sandwich on top of the standard self-attention layer.
The convolution module captures local temporal patterns that
pure attention has to learn from positional cues alone — exactly
the kind of structure relevant to onset detection (attack
envelopes, drum decay, beat-scale rhythms).

## Citations

- Direct baseline: [#007 — TimeStretch](../007-time-stretch/).
  Best val miss 0.2406 [exp_007_time_stretch, step 372,132,
  val/single/onset/miss], hit 0.7512 [same step,
  val/single/onset/hit]. Same dataset, same audio sampler, same
  loss, same augmentations — only the trunk changes.
- Best per-step ceiling so far: [#012](../012-onset-channels/).
  Best val miss 0.2331 [exp_012_onset_channels, step 349,690,
  val/single/onset/miss]. Used the input-feature axis (sub-band
  SF channels); #013 attacks the trunk axis instead. The two
  could stack later (`taiko2_v1_onset` + Conformer trunk) but
  for variable isolation #013 uses #007's `taiko2_v1` dataset
  (80 mel rows).
- Cross-experiment record motivating "trunk axis is the next
  thing to try": [`PERFORMANCE.md`](../../PERFORMANCE.md) +
  [`experiments/README.md`](../README.md). Trunk has been
  invariant across all 143 experiments (taiko1 + taiko2).
- External:
  - [Conformer: Convolution-augmented Transformer for Speech Recognition (Gulati et al., INTERSPEECH 2020)](https://arxiv.org/abs/2005.08100) —
    the canonical paper. Macaron FFN + MHSA + conv module + macaron FFN.
  - [Beat This! Accurate beat tracking without DBN postprocessing (Foscarin et al., ISMIR 2024)](https://arxiv.org/abs/2407.21658) —
    closest published reference for our task. Uses a transformer
    backbone with rotary embeddings and alternating frequency /
    time attention; not pure Conformer, but their hyperparameter
    scale informs ours.
  - [Multi-Convformer: Extending Conformer with Multiple Convolution Kernels (Li et al., 2024)](https://arxiv.org/abs/2407.03718) —
    confirms kernel size 17–31 is the productive band across
    audio tasks; we use 31 (canonical).
  - [Squeezeformer (Kim et al., 2022)](https://arxiv.org/abs/2206.00888) —
    Conformer variant with simplified macaron; we don't use it
    but the ablations there support keeping the macaron design.
  - [`torchaudio.models.Conformer` documentation](https://docs.pytorch.org/audio/stable/generated/torchaudio.models.Conformer.html) —
    reference implementation. Our `ConformerBlock` mirrors its
    internal layer with one change (exposed as a single block
    with a `nn.TransformerEncoderLayer`-compatible forward
    signature, so the parent detector's per-block FiLM hook
    keeps working).

---
<!--
Everything above this divider may be written freely.
Everything between the two dividers is PRE-RUN and must be filled
BEFORE the run. Do not edit it afterwards — use the amendment rule.
-->
─────────────────────────────────────────────────────────────────────

## Hypothesis

### Claim

If we replace the 8-layer post-norm Transformer trunk in
[#007](../007-time-stretch/)'s `EventEmbeddingDetector` with an
8-layer Conformer trunk (everything else identical — same conv
stem, same event-token mixer, same per-block FiLM placement,
same head, same training recipe), val/single/onset/miss will
reach a best value at least 1.0 pp BELOW #007's 0.2406 (i.e.
≤ 0.231) **and** at least matching #012's 0.2331 (i.e. ≤ 0.233),
because the depthwise-convolution module inside each Conformer
block captures local temporal patterns (attack envelopes, drum
decay, beat-scale rhythm) that pure attention has to derive from
positional cues alone — directly addressing the failure modes
that ratio-banding and the structural-floor analyses
([#011b](../011b-onset-disagreement/), taiko1 exp 65-S2v2b)
identified as universal across every architecture tested so far.

### Mechanism

Three effects, stacked:

1. **Local temporal pattern capture.** Pure attention over 250
   audio tokens (post stride-4 conv stem) can in principle attend
   anywhere, but it has to learn the "this attack envelope spans
   N adjacent tokens" pattern from scratch via positional cues.
   A depthwise-conv module with kernel=31 covers ~620 ms of local
   context (k=31 × 20 ms/token) — about half a beat at 60 BPM, two
   beats at 200 BPM. The literature consensus from speech ASR is
   that this conv-augmentation gives 2-4 pp improvement on
   downstream audio tasks at otherwise-matched setups.

2. **Macaron FFN sandwich.** Two half-residual FFNs (one before
   attention, one after conv) double the per-block FFN capacity.
   Gulati 2020's ablation showed ~0.4 pp WER regression when the
   macaron design is replaced by a single FFN, so the structure
   matters beyond just having more parameters. The model gets to
   transform features more thoroughly per block.

3. **Architectural diversity attacks the structural floor.** The
   ~16 % "both audio and context paths miss the same onsets" floor
   from taiko1 [exp 65-S2v2b](../../../taiko/experiments/experiment_65_s2v2b/)
   and the ratio-banding ridges visible in every direct-bin run
   suggest a representational limit in how the audio path
   processes mel features. Convolution-augmented attention is a
   genuinely different inductive bias from pure self-attention —
   if the floor is a representational issue rather than a data
   one, this is the experiment that should test it.

### Predicted numbers

Reference: [#007](../007-time-stretch/) E18 (best val miss),
[#012](../012-onset-channels/) E17 (best val miss).

| Metric | #007 best | #012 best | Predicted (#013, mature eval) | Notes |
|---|---:|---:|---:|---|
| val/single/onset/miss | 0.2406 | 0.2331 | **≤ 0.231** | must-have, ≥ 1.0 pp below #007 |
| val/single/onset/hit | 0.7512 | 0.7542 | ≥ 0.755 | paired with miss |
| val/single/onset/exact | 0.5748 | 0.5665 | ≥ 0.575 | conv module should help strict-frame too |
| val/single/onset/fhit (±2 fr) | 0.7508 | 0.7539 | ≥ 0.755 | local conv → tighter timing |
| val/single/onset/fgood (±7 fr) | 0.7637 | 0.7667 | ≥ 0.770 | mid-tolerance lift |
| val/single/onset/frame_err_p90 | 30 | 31 | ≤ 30 | tail should hold or improve |
| val/single/onset/stop_f1 | 0.5850 | 0.5556 | ≥ 0.580 | conv may help STOP boundary detection |
| AR `matched_rate` (gt_cond, median) | 0.7061 | 0.7080 | ≥ 0.710 | small lift; new SOTA candidate |
| AR `error_median_ms` | 8 | 8 | 8 | likely tied; conv helps placement, not max precision |
| train_noaug → val gap @ best eval | −3.50 pp | −2.62 pp | −2.0 to −3.5 pp | watch — bigger model = more overfit risk |

Observational (not gated):

- `ratio_error.png` should show the same `±log(2)` ridges as
  every other direct-bin run (#005, #007, #008). The conv module
  doesn't encode tempo octaves; it doesn't have a structural
  reason to attack that failure mode. This is a sanity check: if
  ridges DO compress, that would be a surprise and inform the
  next experiment; if they don't, the ceiling is somewhere else.
- Loss curve: should descend smoothly. Conformer is not known
  for instability — the per-block LayerNorms + half-residual
  FFNs make it robust. Watch for divergence in the first 20k
  steps; if it appears, the GroupNorm-vs-BatchNorm choice or LR
  is the suspect.

### Param-count flag (open question for follow-up)

Conformer-8 with these hyperparameters has **29.47 M params**
[computed by instantiating `ConformerDetector(config)` with
`config/model.json` and summing `p.numel() for p in model.parameters()`],
vs #007's **16.35 M** [#007 instantiated from
`007-time-stretch/config/model.json`] — **+80.2 % parameters**.

The intervention here is "swap each transformer layer with a
conformer block at the same depth." That intentionally adds
capacity (the macaron FFNs double the per-block FFN compute,
plus the conv module is new). If #013 wins, we will not know
from this run alone whether the win is from:

  (a) the Conformer block structure (conv + macaron), or
  (b) just the extra parameters.

A follow-up experiment is planned to disambiguate: train a
**Transformer-8 with d_model widened to ~512 and ffn=2048**
(matched param count to Conformer-8), and compare. If
matched-param Transformer matches #013, the win is capacity.
If it doesn't, the win is the Conformer architecture.

For #013 itself we accept the param-count confound and report
results both as "absolute miss number" and as "relative-to-#007
delta" so the comparison is honest.

## Success criteria

- **Must have:** val/single/onset/miss ≤ 0.231 by E18 (≥ 1.0 pp
  below #007's 0.2406; ≥ 0.2 pp below #012's 0.2331).
- **Must have:** training stable, no NaN, no Inf, runs to E20+.
- **Must have:** train_noaug → val gap not materially worse than
  #007's −3.50 pp at the best eval (−4 pp is the fails-if line).
- **Nice-to-have:** miss ≤ 0.225 — clear architectural win.
- **Nice-to-have:** AR `matched_rate` ≥ 0.710 (new taiko2 best).
- **Nice-to-have:** `±log(2)` ratio-banding ridges compress
  vs #007's heatmap (capability finding).
- **Fails if:** miss > 0.245 at every eval after E10 — Conformer
  hurt vs #007 baseline.
- **Fails if:** train_noaug → val gap > 4 pp at any post-warmup
  eval — bigger model overfits with same data.
- **Fails if:** loss diverges or stops decreasing in the first
  10 evals — architectural / numerical bug.

## Changes from baseline

Baseline: [#007 — TimeStretch](../007-time-stretch/).

- `models/conformer_block.py` (new) — `ConformerBlock(d_model,
  n_heads, ffn_dim, depthwise_kernel_size, dropout,
  use_group_norm)`. Mirrors the internal layer of
  `torchaudio.models.Conformer`, exposed as a single block with a
  `(B, T, d) -> (B, T, d)` forward (no mask arg) so it's a
  drop-in replacement for `nn.TransformerEncoderLayer`. Macaron
  FFN-1 (half-residual) + MHSA + conv module + macaron FFN-2
  (half-residual) + final LayerNorm. Conv module: LN ->
  PointwiseConv(d -> 2d) -> GLU -> DepthwiseConv1d(d, k, groups=d,
  padding=k//2) -> GroupNorm(num_groups=1) -> Swish ->
  PointwiseConv(d -> d) -> Dropout. Even kernel rejected at
  construction time.
- `models/conformer_detector.py` (new) — `ConformerDetector(EventEmbeddingDetector)`,
  thin subclass that overrides `__init__` to replace the parent's
  `self.layers` (8 × `nn.TransformerEncoderLayer`) with 8 ×
  `ConformerBlock`. Everything else inherited unchanged: conv
  stem, event-embedding mixer, FiLM, head. The per-block FiLM
  loop in `get_cursor_token` keeps working because
  `ConformerBlock.forward(x)` matches `nn.TransformerEncoderLayer.forward(x)`.
- `models/conformer_detector.py:ConformerDetectorConfig` (new)
  extends `EventEmbeddingConfig` with three Conformer-specific
  fields:
  - `ffn_dim: int = 1536` (macaron FFN hidden, = 4 × d_model).
  - `depthwise_conv_kernel_size: int = 31` (canonical, must be odd).
  - `use_group_norm: bool = True` (BeatThis convention; small-batch robust).
- `models/__init__.py` — exports `ConformerBlock`,
  `ConformerDetector`, `ConformerDetectorConfig`.
- `config/model.json` — `__class__` set to
  `osu.taiko2.models.conformer_detector:ConformerDetectorConfig`,
  same backbone fields as #007 plus the three Conformer fields
  (`ffn_dim: 1536`, `depthwise_conv_kernel_size: 31`,
  `use_group_norm: true`).
- `config/{adapter,data,loss,trainer,infer}.json` — **identical
  to #007's**. Loss is `OnsetLoss` (mixed hard + trapezoid soft
  CE, soft margin 3% / 20% / frame_tolerance 2, stop_weight 1.5).
  Trainer is AdamW LR 3e-4 cosine, batch 64, 50 epochs, evals
  per epoch 4. Augmentations identical to #007 (TimeStretch p=0.3
  max_scale=1.4, MelGainJitter, MelGaussianNoise, MelFreqJitter,
  SpecAugFreq, SpecAugTime, EventJitter, EventDropout,
  EventInsertion, PartialMetronome, PartialAdvMetronome,
  LargeTimeShift, ContextTruncation, ConditioningJitter).

## Run config

- Run name: `exp_013_conformer`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1` (80-row mel; same as #007).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_013_conformer \
      --config-dir osu/taiko2/experiments/013-conformer/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --benchmarks all --benchmark-fraction 0.05 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/013-conformer/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

─────────────────────────────────────────────────────────────────────
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
─────────────────────────────────────────────────────────────────────

## Results summary

_(To fill post-run.)_

## Visualizations

_(Post-run.)_

## Vs prediction

_(Post-run.)_

## Takeaways

_(Post-run.)_

## Followup questions

_(Post-run; the matched-param-count Transformer ablation is
already pre-flagged in the Param-count flag section above and
will land here as a concrete next-experiment proposal.)_
