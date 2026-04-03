# Experiment 56-B - Density Sensitivity Test

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.

## Hypothesis

[Exp 56](../experiment_56/README.md) showed the model under-predicts density (ratio 0.83 avg) and that density conditioning has limited control over output. But does varying the conditioned density actually change the model's output at all?

**Test: run each song 3 times** with different density conditioning:
- **0.8x**: 80% of the chart's actual density
- **1.0x**: actual chart density (baseline, same as exp 56)
- **1.2x**: 120% of the chart's actual density

If the model is sensitive to density, we should see proportional changes in predicted event count. If it's largely ignoring density, all three runs will produce similar output.

### Launch

Same model as [exp 56](../experiment_56/README.md): [exp 45](../experiment_45/README.md).

```bash
cd osu/taiko
python experiments/experiment_56b/run_density_sweep.py --checkpoint runs/detect_experiment_45/checkpoints/best.pt
```

## Result

*Pending*

## Lesson

*Pending*
