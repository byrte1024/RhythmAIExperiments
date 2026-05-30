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


def _fetch_batch(
    fetch: Callable[[int], DataSample],
    indices: Sequence[int],
    pool: "ThreadPoolExecutor | None" = None,
) -> list[DataSample]:
    """Fetch a batch of samples, optionally in parallel."""
    if pool is not None:
        return list(pool.map(lambda i: fetch(int(i)), indices))
    return [fetch(int(i)) for i in indices]


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


_ONSET_STATS_KEYS: tuple[str, ...] = (
    "onset/exact", "onset/fhit", "onset/fgood", "onset/fmiss",
    "onset/rhit", "onset/rgood", "onset/rmiss",
    "onset/hit", "onset/good", "onset/miss",
    "onset/pred_stop_rate", "onset/n_total", "onset/n_nonstop",
    "onset/n_stop_target",
)


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

    Returns an all-zero placeholder when the target has no
    ``target_bin`` attribute — framewise targets (#016) carry an
    activation map rather than a single next-bin index, and the
    OnsetMetric semantics don't apply. The training loop disables
    its `hit`/`miss` tqdm postfix in that case.
    """
    if not hasattr(target, "target_bin"):
        return {k: 0.0 for k in _ONSET_STATS_KEYS}
    # Local imports keep startup light; cached after first call.
    from ..domain.metrics import MetricInput
    from .metrics_onset import OnsetMetric, OnsetMetricConfig

    scratch = OnsetMetric(OnsetMetricConfig(b_pred=b_pred))
    scratch.update(MetricInput(output=output, target=target))
    return scratch.compute()


def _framewise_batch_stats(output: Any, target: Any) -> dict[str, float]:
    """Per-batch framewise stats for #016-style activation-map targets.

    Both recall and precision are reported at threshold 0.5, exact-bin
    and loose (±2 bins) variants. Gaussian σ=2 smoothing of the GT map
    means exact-bin scoring penalizes near-miss predictions; loose
    matching mirrors the eval-time tolerance grid.

    Returns:
      - ``fw/recall`` / ``fw/recall_loose``: fraction of GT bins
        covered by a ``pred>=0.5`` (exact bin / within ±2 bins).
      - ``fw/precision`` / ``fw/precision_loose``: fraction of bins
        with ``pred>=0.5`` that land on (or within ±2 of) a GT bin.
      - ``fw/f1`` / ``fw/f1_loose``: harmonic mean of the above pair.
      - ``fw/pmean``: mean predicted activation (post-clamp).
      - ``fw/pmean_at_gt`` / ``fw/pmean_off_gt``: mean activation on
        vs off GT bins.
      - ``fw/n_pred_pos``: average #bins/sample with ``pred>=0.5``.
      - ``fw/n_gt``: average #GT onsets per sample.
    """
    conf = getattr(output, "confidence_map", None)
    if conf is not None:
        pred = conf.detach()
    else:
        pred = output.logits.detach().clamp(0.0, 1.0)    # (B, n_bins)
    gt_binary = target.target_map_binary.detach()      # (B, n_bins)
    gt_padded = target.gt_bins_padded.detach()          # (B, M)

    B, n_bins = pred.shape
    pmean = float(pred.mean())
    pred_pos = pred >= 0.5
    n_pred_pos_total = float(pred_pos.sum())
    n_pred_pos_avg = n_pred_pos_total / max(B, 1)

    gt_mask = gt_binary > 0.5
    n_gt_total = float(gt_mask.sum())

    if n_gt_total > 0:
        pmean_at_gt = float((pred * gt_binary).sum() / n_gt_total)
        n_off = float((1.0 - gt_binary).sum())
        pmean_off_gt = float(
            (pred * (1.0 - gt_binary)).sum() / max(n_off, 1.0)
        )

        # Loose-GT mask: max-pool the binary GT map with kernel 5 so any
        # bin within ±2 of a GT onset is True. Reused for both loose
        # precision and (cheap) loose recall via the gather path below.
        gt_dilated = torch.nn.functional.max_pool1d(
            gt_binary.unsqueeze(1), kernel_size=5, stride=1, padding=2,
        ).squeeze(1) > 0.5

        # Recall (exact / loose).
        recall = float((pred_pos & gt_mask).sum() / n_gt_total)
        valid = gt_padded >= 0
        if valid.any():
            idx = gt_padded.clamp(min=0, max=n_bins - 1).long()
            window_max = pred.new_zeros(idx.shape)
            for off in range(-2, 3):
                shifted = (idx + off).clamp(min=0, max=n_bins - 1)
                window_max = torch.maximum(
                    window_max, pred.gather(1, shifted),
                )
            recall_loose = float(
                ((window_max >= 0.5) & valid).sum()
                / max(valid.sum().item(), 1)
            )
        else:
            recall_loose = 0.0

        # Precision (exact / loose).
        if n_pred_pos_total > 0:
            precision = float(
                (pred_pos & gt_mask).sum() / n_pred_pos_total
            )
            precision_loose = float(
                (pred_pos & gt_dilated).sum() / n_pred_pos_total
            )
        else:
            precision = 0.0
            precision_loose = 0.0
    else:
        pmean_at_gt = 0.0
        pmean_off_gt = pmean
        recall = recall_loose = 0.0
        precision = precision_loose = 0.0

    def _f1(p: float, r: float) -> float:
        return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    return {
        "fw/recall": recall,
        "fw/recall_loose": recall_loose,
        "fw/precision": precision,
        "fw/precision_loose": precision_loose,
        "fw/f1": _f1(precision, recall),
        "fw/f1_loose": _f1(precision_loose, recall_loose),
        "fw/pmean": pmean,
        "fw/pmean_at_gt": pmean_at_gt,
        "fw/pmean_off_gt": pmean_off_gt,
        "fw/n_pred_pos": n_pred_pos_avg,
        "fw/n_gt": float(target.n_gt.float().mean()),
    }


def _is_framewise_target(target: Any) -> bool:
    return hasattr(target, "target_map_binary") and not hasattr(
        target, "target_bin",
    )


def _infer_b_pred(output: Any, model_b_pred: int | None = None) -> int:
    """Return ``b_pred`` for metric computation.

    When ``model_b_pred`` is provided (from model config), use it
    directly — this is the only reliable path for ratio-mode and MDN
    outputs whose packed tensor width doesn't equal ``b_pred + 1``.
    Falls back to ``output.logits.size(-1) - 1`` for the standard
    softmax head.
    """
    if model_b_pred is not None:
        return model_b_pred
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
    model_b_pred: int | None = None,
    worker_pool: "ThreadPoolExecutor | None" = None,
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
            pbar = tqdm(
                range(len(idx_chunks)), desc=desc, unit="batch", leave=False,
            )
        except ImportError:
            pass

    # Prefetch first chunk while setting up.
    _pf_exec = None
    _pf_future = None
    if worker_pool is not None and idx_chunks:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _pf_exec = _TPE(max_workers=1)
        _pf_future = _pf_exec.submit(
            _fetch_batch, fetch, idx_chunks[0], worker_pool,
        )

    with torch.no_grad():
        for ci, chunk in enumerate(idx_chunks):
            if _pf_future is not None:
                raw = _pf_future.result()
                if ci + 1 < len(idx_chunks):
                    _pf_future = _pf_exec.submit(
                        _fetch_batch, fetch, idx_chunks[ci + 1], worker_pool,
                    )
                else:
                    _pf_future = None
            else:
                raw = _fetch_batch(fetch, chunk, worker_pool)
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

            # Postfix with per-batch + running-average stats. Onset
            # stats for event-target mode, framewise stats for #016.
            if pbar is not None:
                eval_batches += 1
                batch_loss = float(result.loss.detach())
                avg_loss = total_loss / max(n_seen, 1)
                if _is_framewise_target(tgt):
                    fs = _framewise_batch_stats(out, tgt)
                    # Re-use the hit/miss accumulators for the two
                    # headline framewise stats: loose recall + loose
                    # precision. Avoids growing local state surface.
                    eval_hit_sum += fs["fw/recall_loose"]
                    eval_miss_sum += fs["fw/precision_loose"]
                    avg_r = eval_hit_sum / eval_batches
                    avg_p = eval_miss_sum / eval_batches
                    pbar.set_postfix({
                        "loss": f"{batch_loss:.3f}/{avg_loss:.3f}",
                        "rec":  f"{fs['fw/recall_loose']:.3f}/{avg_r:.3f}",
                        "prec": f"{fs['fw/precision_loose']:.3f}/{avg_p:.3f}",
                        "f1":   f"{fs['fw/f1_loose']:.3f}",
                        "p@gt": f"{fs['fw/pmean_at_gt']:.3f}",
                    })
                else:
                    b_pred = _infer_b_pred(out, model_b_pred)
                    bs = _batch_stats(out, tgt, b_pred=b_pred)
                    eval_hit_sum += bs["onset/hit"]
                    eval_miss_sum += bs["onset/miss"]
                    pbar.set_postfix({
                        "loss": f"{batch_loss:.3f}/{avg_loss:.3f}",
                        "hit":  f"{bs['onset/hit']:.3f}/"
                                f"{eval_hit_sum / eval_batches:.3f}",
                        "miss":  f"{bs['onset/miss']:.3f}/"
                                f"{eval_miss_sum / eval_batches:.3f}",
                    })

            if pbar is not None:
                pbar.update(1)

    if _pf_exec is not None:
        _pf_exec.shutdown(wait=False)
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
    compile: bool = False,
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

    if compile:
        model = torch.compile(model)

    # b_pred from the model config — used for per-batch metrics. Must
    # NOT be inferred from output width because ratio/MDN modes pack
    # non-softmax tensors into the logits field.
    _model_b_pred: int | None = getattr(getattr(model, "config", None), "b_pred", None)

    train_fetch = _pick_fetch(train_sampler, augmented=True)
    val_fetch = _pick_fetch(val_sampler, augmented=False)

    _worker_pool = None
    if trainer_config.num_workers > 0:
        from concurrent.futures import ThreadPoolExecutor
        _worker_pool = ThreadPoolExecutor(
            max_workers=trainer_config.num_workers,
        )

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

    evals_per_epoch = max(1e-9, float(trainer_config.evals_per_epoch))
    eval_every = max(1, int(steps_per_epoch / evals_per_epoch))

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
            # Materialize slices into a list BEFORE tqdm wrapping so
            # prefetch can index into it without consuming the iterator.
            _slices_list = list(slices) if not isinstance(slices, list) else slices

            pbar = None
            if progress:
                try:
                    from tqdm.auto import tqdm
                    pbar = tqdm(
                        range(len(_slices_list)),
                        desc=f"epoch {state.epoch + 1}/{trainer_config.epochs}",
                        unit="batch",
                    )
                except ImportError:
                    pass

            # Per-epoch running averages for the progress bar.
            # Everything shown here is TRAIN-phase only — val numbers
            # land via the eval bar on its own pass.
            epoch_loss_sum = 0.0
            epoch_hit_sum = 0.0
            epoch_miss_sum = 0.0
            epoch_n = 0
            _prefetch_future = None
            if _worker_pool is not None and _slices_list:
                from concurrent.futures import ThreadPoolExecutor as _TPE
                _prefetch_exec = _TPE(max_workers=1)
                first_lo, first_hi = _slices_list[0]
                _prefetch_future = _prefetch_exec.submit(
                    _fetch_batch, train_fetch, indices[first_lo:first_hi], _worker_pool,
                )
            else:
                _prefetch_exec = None

            for si, (lo, hi) in enumerate(_slices_list):
                batch_idx = indices[lo:hi]
                if _prefetch_future is not None:
                    samples = _prefetch_future.result()
                    if si + 1 < len(_slices_list):
                        nlo, nhi = _slices_list[si + 1]
                        _prefetch_future = _prefetch_exec.submit(
                            _fetch_batch, train_fetch, indices[nlo:nhi], _worker_pool,
                        )
                    else:
                        _prefetch_future = None
                else:
                    samples = _fetch_batch(train_fetch, batch_idx, _worker_pool)
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

                # Per-batch stats — saved into the step log alongside
                # the loss sub-metrics and shown in the tqdm postfix.
                # Onset-target mode uses the full OnsetMetric set;
                # framewise mode (#016) uses the activation-map stats.
                framewise_mode = _is_framewise_target(tgt)
                if framewise_mode:
                    batch_extras = _framewise_batch_stats(out, tgt)
                else:
                    b_pred = _infer_b_pred(out, _model_b_pred)
                    batch_extras = _batch_stats(out, tgt, b_pred=b_pred)

                # Merge into a new LossResult so hooks see per-batch
                # stats alongside loss when MetricLoggerHook writes.
                step_result = LossResult(
                    loss=result.loss,
                    metrics={**result.metrics, **batch_extras},
                )
                for h in hooks:
                    h.on_step_end(state, step_result)

                batch_loss = float(result.loss.detach())
                epoch_loss_sum += batch_loss
                epoch_n += 1
                if framewise_mode:
                    # Re-use the hit/miss accumulators to track the
                    # two headline framewise stats (loose recall +
                    # loose precision) without growing train()'s
                    # local state surface.
                    epoch_hit_sum += batch_extras["fw/recall_loose"]
                    epoch_miss_sum += batch_extras["fw/precision_loose"]
                    if pbar is not None:
                        denom = max(epoch_n, 1)
                        pbar.set_postfix({
                            "loss": (
                                f"{batch_loss:.3f}/"
                                f"{epoch_loss_sum / denom:.3f}"
                            ),
                            "rec": (
                                f"{batch_extras['fw/recall_loose']:.3f}/"
                                f"{epoch_hit_sum / denom:.3f}"
                            ),
                            "prec": (
                                f"{batch_extras['fw/precision_loose']:.3f}/"
                                f"{epoch_miss_sum / denom:.3f}"
                            ),
                            "f1": f"{batch_extras['fw/f1_loose']:.3f}",
                            "p@gt": f"{batch_extras['fw/pmean_at_gt']:.3f}",
                        })
                else:
                    epoch_hit_sum += batch_extras["onset/hit"]
                    epoch_miss_sum += batch_extras["onset/miss"]
                    if pbar is not None:
                        denom = max(epoch_n, 1)
                        pbar.set_postfix({
                            "loss": (
                                f"{batch_loss:.3f}/"
                                f"{epoch_loss_sum / denom:.3f}"
                            ),
                            "hit": (
                                f"{batch_extras['onset/hit']:.3f}/"
                                f"{epoch_hit_sum / denom:.3f}"
                            ),
                            "miss": (
                                f"{batch_extras['onset/miss']:.3f}/"
                                f"{epoch_miss_sum / denom:.3f}"
                            ),
                        })

                if pbar is not None:
                    pbar.update(1)

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
                        model_b_pred=_model_b_pred,
                        worker_pool=_worker_pool,
                    )
                    state.last_eval_metrics = val_out

                    # Save val artifacts BEFORE any extra pass can reset
                    # them. Train-noaug also gets artifacts (below) so
                    # the ratio-error heatmap / metronome / etc. can be
                    # directly compared against val to distinguish
                    # "model can solve this on memorized data" from
                    # "architecture can't solve this at all".
                    eval_dir = spec.run_dir / f"eval_{state.step}"
                    if eval_artifacts:
                        for a in eval_artifacts:
                            try:
                                a.save(eval_dir, step=state.step)
                            except Exception:
                                pass

                    # train_noaug diagnostic: fresh deterministic
                    # subset of the TRAIN split, augmentations OFF.
                    # Distinguishes overfitting from data-ceiling —
                    # see #002 Followup questions. Artifacts from the
                    # val pass have just been persisted; now we reset
                    # them via `_run_eval` and accumulate train_noaug
                    # state for a second save below.
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
                            artifacts=eval_artifacts,
                            batch_size=batch_size,
                            device=device,
                            progress=progress,
                            indices=noaug_indices,
                            desc="train_noaug",
                            model_b_pred=_model_b_pred,
                            worker_pool=_worker_pool,
                        )
                        for k, v in train_noaug_out.items():
                            val_out[f"train_noaug/{k}"] = v
                        if eval_artifacts:
                            noaug_dir = eval_dir / "train_noaug"
                            for a in eval_artifacts:
                                try:
                                    a.save(noaug_dir, step=state.step)
                                except Exception:
                                    pass

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
                                model_b_pred=_model_b_pred,
                                worker_pool=_worker_pool,
                            )
                            for k, v in bench_out.items():
                                val_out[f"bench/{mode.name}/{k}"] = v

                    # (Val + train_noaug artifacts were already saved
                    # above, before their respective passes could reset
                    # or overwrite each other.)
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
        if _worker_pool is not None:
            _worker_pool.shutdown(wait=False)
        for h in hooks:
            try:
                h.on_train_end(state, exc)
            except Exception:
                pass

    return state
