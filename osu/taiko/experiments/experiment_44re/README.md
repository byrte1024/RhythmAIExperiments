# Experiment 44-RE - Reproducibility Verification

> **[Full Architecture Specification](../experiment_44/ARCHITECTURE.md)** — identical to experiment 44.

## Purpose

Exact reproduction of [experiment 44](../experiment_44/README.md) on a completely different machine to verify:

1. **Reproducibility**: Can results be reproduced on different hardware/OS?
2. **Multi-machine validation**: Establishing a second training environment for parallel experiments.

## Environment Differences

| Component | Original (exp 44) | Reproduction (44-RE) |
|---|---|---|
| GPU | NVIDIA RTX 5070 (12 GB) | NVIDIA RTX 4060 (8 GB) |
| OS | Windows 11 | CachyOS (Linux) |
| Triton | Not available (Windows) | Available (Linux) |
| torch.compile | Skipped | Enabled |
| Python | 3.13.12 | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 | 2.12.0.dev20260307+cu128 |
| CUDA | 12.8 | 12.8 |

## Expected Results

Exp 44 achieved:
- **Peak: 73.7% HIT at eval 11 (epoch 4.7)**
- Stop F1: 0.570
- Val loss: 2.480

Due to different GPU, RNG differences from torch.compile, and potential CUDA version differences, we expect **similar but not identical** results. Within ±1pp of HIT% would confirm reproducibility.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_44re --model-type event_embed --no-gap-ratios --density-jitter-rate 0.30 --density-jitter-pct 0.10 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

Critical flags to match exp 44's original config:
- `--no-gap-ratios`: exp 44 predates gap ratios (current default is ON)
- `--density-jitter-rate 0.30 --density-jitter-pct 0.10`: exp 44's hardcoded jitter (current default is 0.10/0.02)
- No `--b-pred`: defaults to B_BINS=500, N_CLASSES=501 (exp 44's config)

## Result

*Pending*

## Lesson

*Pending*
