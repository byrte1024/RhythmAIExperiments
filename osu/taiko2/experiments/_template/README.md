# Experiment NNN — {short title}

## Status

`Planned` (→ `Running` → `Complete` | `Abandoned` | `Superseded`)

## Context

{One paragraph. What problem motivates this experiment? Which prior
result prompts it? Don't editorialize; state the setup.}

## Citations

- Baseline: [taiko2 #NNN](../NNN-slug/) or [taiko1 exp N](../../../taiko/experiments/experiment_N/)
- Related prior: [#NNN](../NNN-slug/), [taiko1 exp N](../../../taiko/experiments/experiment_N/)
- External: [Paper / URL](https://...)

---
<!--
Everything above this divider may be written freely.
Everything between the two dividers is PRE-RUN and must be filled
BEFORE the run. Do not edit it afterwards — use the amendment rule.
-->
─────────────────────────────────────────────────────────────────────

## Hypothesis

### Claim

{One sentence, if-then-because form. Example: "If we double
`n_layers` from 8 to 16, HIT will rise 0.5-1.5 pp because the
bottleneck is representational capacity rather than data."}

### Mechanism

{2-4 sentences. Why you expect this. What evidence from prior
experiments supports it. What's the proposed causal chain.}

### Predicted numbers

| Metric | Current | Predicted | Notes |
|---|---:|---:|---|
| val/single/hit_e1 | 71.9% | +0.5 to +1.5pp | watched eval metric |
| val/single/miss | 27.5% | ≤ +0.5pp | should stay stable |
| train/overall/loss | 2.41 | ≤ 2.35 | capacity test |

## Success criteria

- **Must have:** {}
- **Nice-to-have:** {}
- **Fails if:** {}

## Changes from baseline

Baseline: [#NNN](../NNN-slug/) (or [taiko1 exp N](../../../taiko/experiments/experiment_N/))

- {Code diff: `models/event_embedding.py:L210-L245`}
- {Config diff: `config/model.json — n_layers: 8 → 16`}
- {Config diff: `config/trainer.json — batch_size: 48 → 32`}
- {etc.}

## Run config

- Run name: `exp_NNN_slug`
- Config snapshots: [`config/`](./config/)
- Dataset: `taiko2_v1`, split `train` / `val`

─────────────────────────────────────────────────────────────────────
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
─────────────────────────────────────────────────────────────────────

## Results summary

| Metric | Baseline (exp N) | This run | Δ | Direction |
|---|---:|---:|---:|:---:|
| val/single/hit_e1 | — | — | — | — |
| val/single/miss | — | — | — | — |
| train/overall/loss | — | — | — | — |

Final eval: eval step `{n}`, wall time `{mm:ss}`, epochs `{k}`.

Machine-readable copy: [`metrics.json`](./metrics.json).

## Visualizations

![Training loss](graphs/01_train_loss.png)
![Validation metric](graphs/02_val_metric.png)

{Add more as relevant: overfit curves, per-star-rating eval, AR
density adherence, prediction distribution, etc.}

## Vs prediction

- `val/single/hit_e1`: predicted +0.5 to +1.5pp → actual `{Δ}` → **{match / beat / miss / wrong direction}**
- `val/single/miss`: predicted ≤ +0.5pp → actual `{Δ}` → **{…}**
- `train/overall/loss`: predicted ≤ 2.35 → actual `{value}` → **{…}**

{One-paragraph summary. Reject the hypothesis here if applicable; put
the *why* in Takeaways.}

## Takeaways

- {One concrete sentence.}
- {Next.}
- {No retrofitting — label surprises as "unexpected: …".}

## Followup questions

- {Question.} — {suggested next experiment or dataset probe}
- {Question.} — {…}
