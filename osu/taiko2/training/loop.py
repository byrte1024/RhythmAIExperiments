"""Reusable training loop for any (sampler × adapter × model × loss)
combination that satisfies the framework ABCs.

Design notes:
  - Sample draw for training comes from a user-supplied `train_fetch`
    callable; default is `augment_sample` if the sampler has it, else
    `get_sample`. This lets training loops swap augmentation without
    touching the sampler.
  - Weighted sampling is opt-in via `train_weights` — a `(N,)` float
    array returned by e.g. `TaikoDetectionSampler.compute_target_weights`.
    When absent, the loop uses a seeded shuffle each epoch.
  - Checkpoints fire at eval boundaries only, via `CheckpointHook`.
  - All metrics land in `{run_dir}/metrics.jsonl` via `MetricLoggerHook`.
    Anything `print`ed by the loop must also be in the log.
  - The loop resumes automatically from `{run_dir}/checkpoints/latest.pt`
    if present. Fresh run otherwise.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from ..domain.adapter import SampleToModelAdapter
from ..domain.loss import Loss
from ..domain.metrics import MetricInput, MetricSet
from ..domain.model import Model
from ..domain.sampling import DataSample, DataSampler
from ..domain.training import RunSpec, TrainerConfig, TrainerHook, TrainingState
from ..persistence.checkpoint import load_latest_if_any
from .hooks import CheckpointHook, MetricLoggerHook


# ─────────────────────────── helpers ──────────────────────────────────

def _pick_fetch(
    sampler: DataSampler, *, augmented: bool,
) -> Callable[[int], DataSample]:
    """Prefer `augment_sample`/`raw_sample` when the concrete sampler
    exposes them; fall back to the ABC's `get_sample`."""
    preferred = "augment_sample" if augmented else "raw_sample"
    return getattr(sampler, preferred, sampler.get_sample)


def _draw_indices(
    n: int, *, weights: np.ndarray | None, rng: np.random.Generator,
) -> np.ndarray:
    if weights is None:
        return rng.permutation(n)
    probs = weights.astype(np.float64)
    probs = probs / probs.sum()
    return rng.choice(n, size=n, p=probs, replace=True)


def _batch_slices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


# ─────────────────────────── eval pass ────────────────────────────────

def _run_eval(
    *,
    model: Model,
    loss_fn: Loss,
    adapter: SampleToModelAdapter,
    sampler: DataSampler,
    fetch: Callable[[int], DataSample],
    metrics: MetricSet | None,
    artifacts: Sequence[Any],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Full pass over `sampler`; returns flat `dict[str, float]`.

    Artifacts (if any) are reset here and updated per batch; the loop
    saves them after this returns.
    """
    model.eval()
    if metrics is not None:
        metrics.reset()
    for a in artifacts:
        a.reset()
    n = sampler.count_samples()

    total_loss = 0.0
    n_seen = 0
    loss_sub: dict[str, float] = {}
    with torch.no_grad():
        for lo, hi in _batch_slices(n, batch_size):
            samples = [fetch(i) for i in range(lo, hi)]
            inp, tgt = adapter.make_batch(samples, device=device)
            out = model.predict(inp)
            result = loss_fn(out, tgt)
            batch_len = len(samples)
            total_loss += float(result.loss.detach()) * batch_len
            n_seen += batch_len
            for k, v in result.metrics.items():
                loss_sub[k] = loss_sub.get(k, 0.0) + float(v) * batch_len
            if metrics is not None or artifacts:
                batch_in = MetricInput(
                    output=out,
                    target=tgt,
                    input=inp,
                    samples=tuple(samples),
                    loss=float(result.loss.detach()),
                    loss_metrics=dict(result.metrics),
                )
                if metrics is not None:
                    metrics.update(batch_in)
                for a in artifacts:
                    a.update(batch_in)
    out_dict: dict[str, float] = {}
    denom = max(n_seen, 1)
    out_dict["loss"] = total_loss / denom
    for k, v in loss_sub.items():
        if k == "loss":
            continue
        out_dict[k] = v / denom
    if metrics is not None:
        out_dict.update(metrics.compute())
    return out_dict


# ─────────────────────────── train entry point ────────────────────────

def train(
    *,
    spec: RunSpec,
    trainer_config: TrainerConfig,
    model: Model,
    loss: Loss,
    adapter: SampleToModelAdapter,
    train_sampler: DataSampler,
    val_sampler: DataSampler,
    train_metrics: MetricSet | None = None,
    val_metrics: MetricSet | None = None,
    eval_artifacts: Sequence[Any] = (),
    train_weights: np.ndarray | None = None,
    extra_hooks: Sequence[TrainerHook] = (),
    device: torch.device | str = torch.device("cpu"),
) -> TrainingState:
    """Run training to completion (or resume if a checkpoint exists)."""
    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    loss.to(device)

    train_fetch = _pick_fetch(train_sampler, augmented=True)
    val_fetch = _pick_fetch(val_sampler, augmented=False)

    n_train = train_sampler.count_samples()
    batch_size = trainer_config.batch_size
    steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * max(1, trainer_config.epochs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps,
    )

    spec.ensure()

    # Resume if possible.
    state = TrainingState(started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    resumed = load_latest_if_any(spec)
    if resumed is not None:
        state = resumed.restore_to(
            model=model, optimizer=optimizer, scheduler=scheduler,
        )

    # Hooks: defaults + user-provided.
    hooks: list[TrainerHook] = [
        MetricLoggerHook(spec),
        CheckpointHook(
            spec=spec,
            model=model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_config=trainer_config,
        ),
    ] + list(extra_hooks)

    evals_per_epoch = max(1, trainer_config.evals_per_epoch)
    eval_every = max(1, steps_per_epoch // evals_per_epoch)

    for h in hooks:
        h.on_train_start(state, spec)

    exc: BaseException | None = None
    try:
        while state.epoch < trainer_config.epochs:
            for h in hooks:
                h.on_epoch_start(state)

            rng = np.random.default_rng(trainer_config.seed + state.epoch)
            indices = _draw_indices(n_train, weights=train_weights, rng=rng)

            model.train()
            for lo, hi in _batch_slices(n_train, batch_size):
                batch_idx = indices[lo:hi]
                samples = [train_fetch(int(i)) for i in batch_idx]
                inp, tgt = adapter.make_batch(samples, device=device)

                optimizer.zero_grad(set_to_none=True)
                out = model.predict(inp)
                result = loss(out, tgt)
                result.loss.backward()
                if trainer_config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), trainer_config.grad_clip,
                    )
                optimizer.step()
                scheduler.step()

                state.step += 1
                state.samples_seen += len(samples)

                if train_metrics is not None:
                    train_metrics.update(MetricInput(
                        output=out,
                        target=tgt,
                        input=inp,
                        samples=tuple(samples),
                        loss=float(result.loss.detach()),
                        loss_metrics=dict(result.metrics),
                    ))

                for h in hooks:
                    h.on_step_end(state, result)

                if state.step % eval_every == 0:
                    val_out = _run_eval(
                        model=model,
                        loss_fn=loss,
                        adapter=adapter,
                        sampler=val_sampler,
                        fetch=val_fetch,
                        metrics=val_metrics,
                        artifacts=eval_artifacts,
                        batch_size=batch_size,
                        device=device,
                    )
                    state.last_eval_metrics = val_out
                    # Persist artifacts under {run_dir}/eval_{step}/
                    # before any hook sees the metrics — keeps the
                    # "save before print" invariant.
                    if eval_artifacts:
                        eval_dir = spec.run_dir / f"eval_{state.step}"
                        for a in eval_artifacts:
                            try:
                                a.save(eval_dir, step=state.step)
                            except Exception:
                                pass
                    for h in hooks:
                        h.on_eval_end(state, val_out)
                    model.train()

            for h in hooks:
                h.on_epoch_end(state)
            state.epoch += 1
    except BaseException as e:
        exc = e
        raise
    finally:
        for h in hooks:
            try:
                h.on_train_end(state, exc)
            except Exception:
                pass

    return state
