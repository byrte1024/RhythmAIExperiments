"""Multi-eval determinism: when `evals_per_epoch > 1`, every eval within
an epoch (and across epochs, before any weight change) must see the
exact same val set — same samples, same order, same targets, no
augmentation — so the metric time-series is comparable across evals.

This guards against accidental: (a) partitioned val set per eval,
(b) train-time augmentations leaking into val fetch, (c) nondeterministic
val sampler shuffling.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from osu.taiko2.domain.metrics import MetricSet
from osu.taiko2.training import OnsetMetric, OnsetMetricConfig
from osu.taiko2.training.loop import _pick_fetch, _run_eval


def _hash_tensor(t: torch.Tensor) -> str:
    arr = t.detach().cpu().numpy()
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _hash_sample(s) -> str:
    h = hashlib.sha256()
    h.update(str(s.sample_id).encode())
    # audio_past + audio_future drive mel input; past_events drive
    # event_offsets; future_events[0] drives target_bin — hash all of them
    # to catch any drift in augmentation, ordering, or target derivation.
    h.update(s.audio_past.tobytes())
    h.update(s.audio_future.tobytes())
    for o in s.past_events:
        h.update(str(o.cursor_offset).encode())
    h.update(s.past_events_mask.tobytes())
    for o in s.future_events:
        h.update(str(o.cursor_offset).encode())
    h.update(s.future_events_mask.tobytes())
    return h.hexdigest()


class TestMultiEvalDeterminism:
    @pytest.fixture
    def stack(self, tmp_path: Path):
        # Reuse the end-to-end fixture via direct construction — mirror
        # `TestEndToEndLoop._make_all` without pulling in its teardown.
        from osu.taiko2.tests.test_training import TestEndToEndLoop  # noqa: F401
        helper = TestEndToEndLoop()
        return helper._make_all(tmp_path, b_pred=40)

    def test_val_fetch_is_unaugmented(self, stack):
        _, _, val_s, _, _, _ = stack
        # The loop uses `_pick_fetch(..., augmented=False)` for val; it
        # must NOT resolve to `augment_sample` even if the sampler exposes
        # one.
        fetch = _pick_fetch(val_s, augmented=False)
        if hasattr(val_s, "augment_sample"):
            assert fetch is not val_s.augment_sample, (
                "val fetch resolved to augment_sample — augmentations "
                "would leak into val"
            )

    def test_two_evals_see_identical_samples(self, stack):
        _, _, val_s, _, _, _ = stack
        fetch = _pick_fetch(val_s, augmented=False)
        n = val_s.count_samples()

        hashes_a = [_hash_sample(fetch(i)) for i in range(n)]
        hashes_b = [_hash_sample(fetch(i)) for i in range(n)]
        assert hashes_a == hashes_b, (
            "val fetch returned different samples on repeat read — "
            "nondeterministic sampler or leaking aug state"
        )

    def test_two_evals_produce_identical_metrics(self, stack):
        _, _, val_s, model, loss, adapter = stack
        fetch = _pick_fetch(val_s, augmented=False)
        model.eval()

        metrics_a = MetricSet(OnsetMetric(OnsetMetricConfig(b_pred=40)))
        out_a = _run_eval(
            model=model, loss_fn=loss, adapter=adapter,
            sampler=val_s, fetch=fetch, metrics=metrics_a,
            artifacts=(), batch_size=4, device=torch.device("cpu"),
        )
        metrics_b = MetricSet(OnsetMetric(OnsetMetricConfig(b_pred=40)))
        out_b = _run_eval(
            model=model, loss_fn=loss, adapter=adapter,
            sampler=val_s, fetch=fetch, metrics=metrics_b,
            artifacts=(), batch_size=4, device=torch.device("cpu"),
        )
        assert set(out_a) == set(out_b)
        for k in out_a:
            assert out_a[k] == pytest.approx(out_b[k], abs=1e-9), (
                f"eval metric {k!r} differs between back-to-back evals: "
                f"{out_a[k]} vs {out_b[k]}"
            )

    def test_eval_visits_full_val_set(self, stack):
        _, _, val_s, model, loss, adapter = stack
        fetch = _pick_fetch(val_s, augmented=False)
        n = val_s.count_samples()
        metrics = MetricSet(OnsetMetric(OnsetMetricConfig(b_pred=40)))
        out = _run_eval(
            model=model, loss_fn=loss, adapter=adapter,
            sampler=val_s, fetch=fetch, metrics=metrics,
            artifacts=(), batch_size=4, device=torch.device("cpu"),
        )
        # n_total comes from OnsetMetric's denominator bookkeeping —
        # must equal the full val sample count.
        assert int(out["onset/n_total"]) == n, (
            f"eval saw {int(out['onset/n_total'])} samples but val set "
            f"has {n} — partial eval"
        )

    def test_targets_stable_across_fetches(self, stack):
        _, _, val_s, _, _, adapter = stack
        fetch = _pick_fetch(val_s, augmented=False)
        n = val_s.count_samples()
        samples_a = [fetch(i) for i in range(n)]
        samples_b = [fetch(i) for i in range(n)]
        tgt_a = adapter.make_target(samples_a, device=torch.device("cpu"))
        tgt_b = adapter.make_target(samples_b, device=torch.device("cpu"))
        assert _hash_tensor(tgt_a.target_bin) == _hash_tensor(tgt_b.target_bin)
        assert _hash_tensor(tgt_a.all_future_bins) == _hash_tensor(tgt_b.all_future_bins)
        assert _hash_tensor(tgt_a.all_future_mask) == _hash_tensor(tgt_b.all_future_mask)
