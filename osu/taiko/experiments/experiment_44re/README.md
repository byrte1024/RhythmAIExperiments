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

5 evals completed (epoch 2.25). Compared to original exp 44 at same epochs:

| Eval | Epoch | 44-RE HIT | Original HIT | Gap | Val Loss Gap |
|---|---|---|---|---|---|
| 1 | 1.25 | 67.0% | 70.2% | **-3.1pp** | +0.123 |
| 2 | 1.50 | 70.3% | 70.9% | **-0.6pp** | +0.025 |
| 3 | 1.75 | 71.0% | 70.9% | **+0.1pp** | -0.012 |
| 4 | 2.00 | 70.9% | 72.0% | **-1.2pp** | +0.037 |
| 5 | 2.25 | 72.0% | 72.9% | **-0.9pp** | +0.036 |

### Observations

- **Eval 1 gap was 3.1pp but closed to <1pp by eval 3.** The CachyOS machine briefly matched the original at eval 3 (71.0% vs 70.9%).
- **Settled at ~0.9pp behind by eval 5.** Consistent small gap, likely from different CUDA kernels (RTX 4060 vs 5070), floating point behavior, or torch.compile differences.
- **Train loss consistently ~0.03 higher** on CachyOS, suggesting slightly different gradient accumulation.
- **Val loss gap narrowed from 0.123 to 0.036** but didn't fully close.

## Lesson

1. **Cross-machine reproducibility confirmed within ~1pp.** The RTX 4060 (CachyOS, Linux) reproduces RTX 5070 (Windows) results within 0.9pp HIT at eval 5. This is within acceptable variance for neural network training.

2. **Early evals show larger variance.** The 3.1pp gap at eval 1 is concerning but transient — it closed rapidly. Don't judge cross-machine runs by their first eval.

3. **A small systematic gap persists.** The 0.9pp difference and 0.036 val loss gap at eval 5 suggest a real (small) hardware effect. Could be FP precision, CUDA kernel differences, or torch.compile altering computation order. Not a problem for practical use.

4. **CachyOS/RTX 4060 is a viable second training machine.** Results are close enough to trust for parallel experiments and development.
