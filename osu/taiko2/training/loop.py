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
from ..domain.loss import Loss, LossResult
from ..domain.metrics import MetricInput, MetricSet
from ..domain.model import Model
from ..domain.sampling import DataSample, DataSampler
from ..domain.training import RunSpec, TrainerConfig, TrainerHook, TrainingState
from ..persistence.checkpoint import (
    Checkpoint,
    find_last_eval_checkpoint,
    load_latest_if_any,
    truncate_stats_after_step,
)
from .hooks import (
    CheckpointHook,
    ConsoleLoggerHook,
    MetricCurvesHook,
    MetricLoggerHook,
    PerEvalJsonHook,
)


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


def _batch_stats(
    output: Any, target: Any, *, b_pred: int,
) -> dict[str, float]:
    """Per-batch onset stats — **full metric set**, delegating to
    `OnsetMetric` on a scratch instance so every definition lives in
    one place.

    Returned flat dict matches ``OnsetMetric.compute()`` exactly:
    ``onset/exact``, ``fhit``, ``fgood``, ``fmiss``, ``rhit``,
    ``rgood``, ``rmiss``, ``hit``, ``good``, ``miss``, ``pred_stop_rate``,
    ``n_total``, ``n_nonstop``, ``n_stop_target`` — plus ``ihit``,
    ``igood``, ``imiss``, ``n_any_future`` when `target.all_future_bins`
    is present.

    ``b_pred`` = STOP class index (n_classes - 1).
    """
    # Local imports keep startup light; cached after first call.
    from ..domain.metrics import MetricInput
    from .metrics_onset import OnsetMetric, OnsetMetricConfig

    scratch = OnsetMetric(OnsetMetricConfig(b_pred=b_pred))
    scratch.update(MetricInput(output=output, target=target))
    return scratch.compute()


def _infer_b_pred(output: Any) -> int:
    """Bin-offset-family models output `(B, n_classes)` where
    ``n_classes = b_pred + 1``. Read it off the last axis."""
    return int(output.logits.size(-1)) - 1


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
    progress: bool = False,
    indices: Sequence[int] | None = None,
    sample_mutation: "Callable[[DataSample], DataSample | None] | None" = None,
    desc: str = "eval",
) -> dict[str, float]:
    """Eval pass. Defaults to the full sampler; can be scoped to a
    subset via `indices`, and every sample can be mutated before the
    adapter via `sample_mutation` (return `None` from the mutation to
    skip that sample entirely — used by `advanced_metronome` benchmark
    to exclude uninformative cases).

    Artifacts (if any) are reset here and updated per batch; the loop
    saves them after this returns.
    """
    model.eval()
    if metrics is not None:
        metrics.reset()
    for a in artifacts:
        a.reset()

    if indices is None:
        n = sampler.count_samples()
        sample_indices = list(range(n))
    else:
        sample_indices = list(indices)

    total_loss = 0.0
    n_seen = 0
    loss_sub: dict[str, float] = {}
    eval_hit_sum = 0.0
    eval_miss_sum = 0.0
    eval_batches = 0

    # Batches are contiguous slices of `sample_indices`; mutation-
    # skipped samples drop out of each batch but the batch's remaining
    # samples are still evaluated together.
    idx_chunks: list[list[int]] = [
        sample_indices[i: i + batch_size]
        for i in range(0, len(sample_indices), batch_size)
    ]
    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(idx_chunks, desc=desc, unit="batch", leave=False)
        except ImportError:
            pass
    chunk_iter = pbar if pbar is not None else idx_chunks

    with torch.no_grad():
        for chunk in chunk_iter:
            raw = [fetch(i) for i in chunk]
            if sample_mutation is not None:
                raw = [sample_mutation(s) for s in raw]
            samples = [s for s in raw if s is not None]
            if not samples:
                continue
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

            # Postfix with per-batch + running-average onset stats.
            if pbar is not None:
                b_pred = _infer_b_pred(out)
                bs = _batch_stats(out, tgt, b_pred=b_pred)
                eval_hit_sum += bs["onset/hit"]
                eval_miss_sum += bs["onset/miss"]
                eval_batches += 1
                batch_loss = float(result.loss.detach())
                avg_loss = total_loss / max(n_seen, 1)
                pbar.set_postfix({
                    "loss": f"{batch_loss:.3f}/{avg_loss:.3f}",
                    "hit":  f"{bs['onset/hit']:.3f}/"
                            f"{eval_hit_sum / eval_batches:.3f}",
                    "miss":  f"{bs['onset/miss']:.3f}/"
                            f"{eval_miss_sum / eval_batches:.3f}",
                })
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
    pre_hooks: Sequence[TrainerHook] = (),
    train_noaug_fraction: float = 0.0,
    benchmarks: "Sequence[Any]" = (),
    benchmark_fraction: float = 0.05,
    benchmark_seed: int = 42,
    device: torch.device | str = torch.device("cpu"),
    progress: bool = True,
    resume: bool = False,
) -> TrainingState:
    """Run training to completion.

    Resume behavior:
      - ``resume=False`` (default): if ``latest.pt`` exists, auto-resume
        from it — same behavior the loop has always had.
      - ``resume=True``: resume from the last finished eval snapshot
        (``eval_{step}/checkpoint.pt`` with the largest step). Any
        metrics.jsonl rows and eval directories past that step are
        truncated so the run picks up with clean stats matching the
        checkpoint's TrainingState.
    """
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

    # Resume.
    # `_resume_skip_slices` is the number of inner-loop steps to skip
    # on the first epoch iteration when we restart mid-epoch (i.e.
    # `state.step % steps_per_epoch != 0` on the resumed checkpoint).
    # One-shot: reset to 0 after the first epoch iteration consumes it.
    state = TrainingState(started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    _resume_skip_slices = 0
    if resume:
        hit = find_last_eval_checkpoint(spec)
        if hit is None:
            raise FileNotFoundError(
                f"--resume: no eval_{{step}}/checkpoint.pt under "
                f"{spec.run_dir}. Nothing to resume from."
            )
        ckpt_path, last_step = hit
        print(
            f"[train] resuming from eval checkpoint @ step {last_step}: "
            f"{ckpt_path}"
        )
        resumed_ckpt = Checkpoint.load(ckpt_path)
        state = resumed_ckpt.restore_to(
            model=model, optimizer=optimizer, scheduler=scheduler,
        )
        # Truncate metrics.jsonl + delete later eval_{N}/ dirs so stats
        # start clean at the resumed step. Then refresh latest.pt so
        # subsequent restarts see a coherent snapshot.
        truncate_stats_after_step(spec, last_step)
        resumed_ckpt.save(spec.latest_checkpoint)
        # Mid-epoch resume: if the checkpoint was taken inside an epoch
        # (evals_per_epoch > 1), skip the slices we've already trained
        # on rather than re-running them.
        _resume_skip_slices = int(state.step) % max(steps_per_epoch, 1)
        if _resume_skip_slices:
            print(
                f"[train] mid-epoch resume: skipping first "
                f"{_resume_skip_slices} slice(s) of epoch {state.epoch}"
            )
    else:
        resumed = load_latest_if_any(spec)
        if resumed is not None:
            state = resumed.restore_to(
                model=model, optimizer=optimizer, scheduler=scheduler,
            )

    # Hooks: defaults + user-provided.
    # Order matters — MetricLogger must fire BEFORE ConsoleLogger so
    # everything the console surfaces is already persisted.
    default_hooks: list[TrainerHook] = [
        MetricLoggerHook(spec),
        CheckpointHook(
            spec=spec,
            model=model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_config=trainer_config,
        ),
    ]
    if progress:
        default_hooks.append(ConsoleLoggerHook())
    # Always install MetricCurvesHook — cheap to maintain, writes on
    # every eval and on train_end. Users who want to disable it can
    # subclass the loop; for 99% of runs this is the right default.
    default_hooks.append(MetricCurvesHook())
    # PerEvalJsonHook: one tiny JSON per eval under eval_{step}/, makes
    # it trivial to query one eval's metrics without parsing the full
    # jsonl stream.
    default_hooks.append(PerEvalJsonHook())
    # `pre_hooks` run BEFORE the defaults — used by contributors that
    # need to mutate `val_metrics` (e.g. `InferCorpusHook`) before the
    # default MetricLoggerHook / PerEvalJsonHook / MetricCurvesHook
    # capture it. `extra_hooks` still run AFTER the defaults.
    hooks: list[TrainerHook] = (
        list(pre_hooks) + default_hooks + list(extra_hooks)
    )

    evals_per_epoch = max(1, trainer_config.evals_per_epoch)
    eval_every = max(1, steps_per_epoch // evals_per_epoch)

    for h in hooks:
        h.on_train_start(state, spec)

    exc: BaseException | None = None
    try:
        while state.epoch < trainer_config.epochs:
            for h in hooks:
                h.on_epoch_start(state)

            # Reset per-epoch train running metrics so the postfix
            # shows "mean so far this epoch", not since training start.
            if train_metrics is not None:
                train_metrics.reset()

            rng = np.random.default_rng(trainer_config.seed + state.epoch)
            indices = _draw_indices(n_train, weights=train_weights, rng=rng)

            model.train()
            slices = list(_batch_slices(n_train, batch_size))
            if _resume_skip_slices:
                slices = slices[_resume_skip_slices:]
                _resume_skip_slices = 0  # one-shot — only first epoch
            pbar = None
            if progress:
                try:
                    from tqdm.auto import tqdm
                    pbar = tqdm(
                        slices,
                        desc=f"epoch {state.epoch + 1}/{trainer_config.epochs}",
                        unit="batch",
                    )
                    slices = pbar
                except ImportError:
                    pass

            # Per-epoch running averages for the progress bar.
            # Everything shown here is TRAIN-phase only — val numbers
            # land via the eval bar on its own pass.
            epoch_loss_sum = 0.0
            epoch_hit_sum = 0.0
            epoch_miss_sum = 0.0
            epoch_n = 0

            for lo, hi in slices:
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

                # Per-batch onset stats — computed inline so they can
                # be both saved (into the step log alongside the loss
                # sub-metrics) and shown in the tqdm postfix. Full set
                # from OnsetMetric, not just hit/miss/exact.
                b_pred = _infer_b_pred(out)
                batch_onset = _batch_stats(out, tgt, b_pred=b_pred)

                # Merge into a new LossResult so hooks see hit/miss/exact
                # alongside loss when MetricLoggerHook writes the step.
                step_result = LossResult(
                    loss=result.loss,
                    metrics={**result.metrics, **batch_onset},
                )
                for h in hooks:
                    h.on_step_end(state, step_result)

                batch_loss = float(result.loss.detach())
                epoch_loss_sum += batch_loss
                epoch_hit_sum += batch_onset["onset/hit"]
                epoch_miss_sum += batch_onset["onset/miss"]
                epoch_n += 1
                if pbar is not None:
                    denom = max(epoch_n, 1)
                    post = {
                        "loss": (
                            f"{batch_loss:.3f}/"
                            f"{epoch_loss_sum / denom:.3f}"
                        ),
                        "hit": (
                            f"{batch_onset['onset/hit']:.3f}/"
                            f"{epoch_hit_sum / denom:.3f}"
                        ),
                        "miss": (
                            f"{batch_onset['onset/miss']:.3f}/"
                            f"{epoch_miss_sum / denom:.3f}"
                        ),
                    }
                    pbar.set_postfix(post)

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
                        progress=progress,
                    )
                    state.last_eval_metrics = val_out

                    # train_noaug diagnostic: fresh deterministic
                    # subset of the TRAIN split, augmentations OFF.
                    # Distinguishes overfitting from data-ceiling —
                    # see #002 Followup questions.
                    #
                    # We reuse `val_metrics` across the extra passes;
                    # `_run_eval` resets it at the start of each call,
                    # and the full computed dict for the primary val
                    # pass is already copied into `val_out` above.
                    if train_noaug_fraction > 0:
                        noaug_n = max(
                            1,
                            int(round(n_train * train_noaug_fraction)),
                        )
                        noaug_rng = np.random.default_rng(
                            benchmark_seed + state.step,
                        )
                        noaug_indices = sorted(
                            noaug_rng.choice(
                                n_train, size=noaug_n, replace=False,
                            ).tolist()
                        )
                        train_noaug_raw = _pick_fetch(
                            train_sampler, augmented=False,
                        )
                        train_noaug_out = _run_eval(
                            model=model, loss_fn=loss, adapter=adapter,
                            sampler=train_sampler,
                            fetch=train_noaug_raw,
                            metrics=val_metrics,
                            artifacts=(),
                            batch_size=batch_size,
                            device=device,
                            progress=progress,
                            indices=noaug_indices,
                            desc="train_noaug",
                        )
                        for k, v in train_noaug_out.items():
                            val_out[f"train_noaug/{k}"] = v

                    # Benchmark suite: a fraction of the val split,
                    # run once per mode with the mode's transform
                    # applied. Metrics per mode land under
                    # `bench/{mode_name}/*`.
                    if benchmarks and benchmark_fraction > 0:
                        bench_n = max(
                            1,
                            int(round(
                                val_sampler.count_samples() * benchmark_fraction,
                            )),
                        )
                        bench_rng = np.random.default_rng(benchmark_seed)
                        bench_indices = sorted(
                            bench_rng.choice(
                                val_sampler.count_samples(),
                                size=bench_n, replace=False,
                            ).tolist()
                        )
                        import random as _random
                        for mode in benchmarks:
                            py_rng = _random.Random(
                                benchmark_seed ^ (hash(mode.name) & 0xFFFFFFFF),
                            )
                            def _mut(
                                s, _py_rng=py_rng, _t=mode.transform,
                            ):
                                return _t(s, _py_rng)
                            bench_out = _run_eval(
                                model=model, loss_fn=loss, adapter=adapter,
                                sampler=val_sampler,
                                fetch=val_fetch,
                                metrics=val_metrics,
                                artifacts=(),
                                batch_size=batch_size,
                                device=device,
                                progress=progress,
                                indices=bench_indices,
                                sample_mutation=_mut,
                                desc=f"bench[{mode.name}]",
                            )
                            for k, v in bench_out.items():
                                val_out[f"bench/{mode.name}/{k}"] = v

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
