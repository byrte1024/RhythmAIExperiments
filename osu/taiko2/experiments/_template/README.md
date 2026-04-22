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

### Final vs baseline

| Metric | Baseline (exp N) | This run (final) | Δ | Direction |
|---|---:|---:|---:|:---:|
| val/single/hit_e1 | — | — | — | — |
| val/single/miss | — | — | — | — |
| train/overall/loss | — | — | — | — |

Final eval: eval step `{n}`, wall time `{hh:mm}`, epochs `{k}`.

### Per-eval progression

{One row per eval. Include **every** metric the trainer reported.
Generated from `runs/{run_name}/metrics.jsonl`.}

| Eval | Step | val/single/hit_e1 | val/single/miss | val/single/loss | train/running/loss | train/overall/loss | lr | wall_time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | — | — | — | — | — | — | — | — |
| 2 | — | — | — | — | — | — | — | — |
| … |   |   |   |   |   |   |   |   |

Machine-readable copies (both tables): [`metrics.json`](./metrics.json).

## Visualizations

![Training loss](graphs/01_train_loss.png)
*Training loss over steps (log-y).*

![Validation progression](graphs/02_val_progression.png)
*{Watched metric} across evals.*

{Add custom graphs as needed — overfit curves, per-star-rating eval,
prediction distribution, AR density adherence, attention maps, per-
kind confusion, anything else. Each gets a numbered file in `graphs/`
and a one-sentence caption here.}

## Custom analyses (optional)

{Reference anything under `custom/` — directories with their own
READMEs explaining what each artifact shows. Keep entries short;
point at the sub-README for detail.}

- [{Name}](custom/{slug}/) — {one-sentence summary}.

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
