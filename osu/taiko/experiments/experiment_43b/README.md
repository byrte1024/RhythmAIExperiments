# Experiment 43-B - AR Resilience Comparison (Diagnostic)

## Hypothesis

Exp 42-AR human evaluation showed context-dependent models (42, 35-C) don't outperform context-free (14) in real AR generation despite higher per-sample HIT. The AR cascade degradation (75% → 5% over 8 steps in light AR) may explain this.

**Compare AR resilience across models:**
- **Exp 14** (68.9% HIT, no context) — expected most resilient, no context to corrupt
- **Exp 35-C** (71.6% HIT, mel ramps) — medium context dependency
- **Exp 42** (73.2% HIT, event embeddings) — deepest context dependency, expected least resilient

### Metrics tracked
- Light AR: per-step HIT rate curve (cascade degradation)
- Light AR: unique predictions per step (metronome detection)
- Light AR: pred mean/std/range over steps (drift detection)
- Full AR: event HIT/MISS, pred HIT/hallucination, density ratio
- Ablation: metronome and no_events benchmarks for reference

## Result

*Pending*

## Lesson

*Pending*
