"""Tests for the onset metric, eval artifacts, and the end-to-end loop.

Uses the tiny config + synthetic dataset from existing fixtures to
keep these runnable without any ML workload. The full-loop test runs
2 epochs on ~20 samples and confirms `metrics.jsonl`, `latest.pt`,
and every artifact lands on disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from osu.taiko2.data_samplers import TaikoDetectionSampler, TaikoDetectionSamplerConfig
from osu.taiko2.dataset import _safe_filename
from osu.taiko2.domain.beatmap import OnsetBinned, OnsetKind
from osu.taiko2.domain.dataset import (
    ChartEntry,
    DatasetManifest,
    MelSamplerConfig,
)
from osu.taiko2.domain.metrics import MetricInput, MetricSet
from osu.taiko2.domain.training import RunSpec, TrainerConfig
from osu.taiko2.models import (
    EventEmbeddingConfig,
    EventEmbeddingDetector,
    EventEmbeddingOutput,
    EventEmbeddingTarget,
)
from osu.taiko2.persistence.events import save_events
from osu.taiko2.persistence.features import save_features
from osu.taiko2.persistence.manifest import save_manifest
from osu.taiko2.training import (
    DetectionSampleAdapter,
    DetectionSampleAdapterConfig,
    DistributionArtifact,
    ErrorHistogramArtifact,
    OnsetLoss,
    OnsetLossConfig,
    OnsetMetric,
    OnsetMetricConfig,
    PredictionHeatmapArtifact,
    RatioErrorHeatmapArtifact,
    train,
)


# ─────────────────────────── helpers ──────────────────────────────────

def _batch_for_metric(
    *,
    target_bins: list[int],
    pred_bins: list[int],
    b_pred: int,
    all_future: list[list[int]] | None = None,
) -> MetricInput:
    """Build a MetricInput by hand — deterministic logits that argmax
    to `pred_bins`, matching `target_bins`."""
    B = len(target_bins)
    n_classes = b_pred + 1
    logits = torch.full((B, n_classes), -10.0)
    for i, p in enumerate(pred_bins):
        logits[i, p] = 10.0
    target = torch.tensor(target_bins, dtype=torch.int64)

    if all_future is None:
        all_bins_t = None
        all_mask_t = None
    else:
        K = max(len(row) for row in all_future) or 1
        all_bins = np.zeros((B, K), dtype=np.int64)
        all_mask = np.ones((B, K), dtype=bool)
        for i, row in enumerate(all_future):
            for j, v in enumerate(row):
                all_bins[i, j] = v
                all_mask[i, j] = False
        all_bins_t = torch.from_numpy(all_bins)
        all_mask_t = torch.from_numpy(all_mask)

    return MetricInput(
        output=EventEmbeddingOutput(logits=logits),
        target=EventEmbeddingTarget(
            target_bin=target,
            all_future_bins=all_bins_t,
            all_future_mask=all_mask_t,
        ),
    )


# ─────────────────────────── OnsetMetric ──────────────────────────────

class TestOnsetMetric:
    def test_exact_and_fhit(self):
        m = OnsetMetric(OnsetMetricConfig(b_pred=100))
        # target=50, preds: 50 (EXACT), 52 (FHIT diff=2), 57 (FGOOD),
        #                  60 (FMISS via F; RGOOD via ratio).
        batch = _batch_for_metric(
            target_bins=[50, 50, 50, 50],
            pred_bins=[50, 52, 57, 60],
            b_pred=100,
        )
        m.update(batch)
        out = m.compute()
        assert out["onset/exact"] == 0.25    # 1/4
        assert out["onset/fhit"] == 0.50     # diff ≤2: 50, 52 → 2/4
        assert out["onset/fgood"] == 0.75    # diff ≤7: 50, 52, 57 → 3/4
        assert out["onset/fmiss"] == 0.25

    def test_rhit_and_rgood(self):
        m = OnsetMetric(OnsetMetricConfig(b_pred=500))
        # target=99 (so target+1=100).
        # pred=96 → ratio 97/100 ⇒ |log(97/100)| ≈ 0.030; ≈ R-HIT cutoff.
        # pred=89 → ratio 90/100 ⇒ |log(.9)| ≈ 0.105; at R-GOOD boundary.
        # pred=80 → ratio 81/100 ⇒ outside both.
        batch = _batch_for_metric(
            target_bins=[99, 99, 99, 99],
            pred_bins=[99, 96, 89, 80],
            b_pred=500,
        )
        m.update(batch)
        out = m.compute()
        # Exactly-equal gets R-HIT trivially. 96 is just below the
        # 97/100 cutoff → R-HIT.
        assert out["onset/rhit"] >= 0.5

    def test_stop_target_excluded(self):
        """STOP-target samples are excluded from the main metrics'
        denominator (n_nonstop)."""
        m = OnsetMetric(OnsetMetricConfig(b_pred=100))
        batch = _batch_for_metric(
            target_bins=[100, 100, 50],  # two STOPs, one real
            pred_bins=[100, 50, 50],
            b_pred=100,
        )
        m.update(batch)
        out = m.compute()
        assert out["onset/n_nonstop"] == 1
        assert out["onset/n_stop_target"] == 2
        # That one non-STOP sample has pred==target → EXACT, HIT, GOOD.
        assert out["onset/exact"] == 1.0
        assert out["onset/hit"] == 1.0

    def test_pred_stop_counts_as_failure(self):
        """Model predicting STOP on a non-STOP target is total fail."""
        m = OnsetMetric(OnsetMetricConfig(b_pred=100))
        batch = _batch_for_metric(
            target_bins=[50, 50],
            pred_bins=[100, 50],     # first predicts STOP
            b_pred=100,
        )
        m.update(batch)
        out = m.compute()
        assert out["onset/pred_stop_rate"] == 0.5
        # Only the second sample contributes to fhit/fgood.
        assert out["onset/fhit"] == 0.5
        assert out["onset/miss"] == 0.5

    def test_any_future_metrics(self):
        """IHIT counts a prediction that matches ANY upcoming onset."""
        m = OnsetMetric(OnsetMetricConfig(b_pred=200))
        # target is next-onset (50), but there are more onsets at 100, 150.
        # pred=100 → misses target but matches a future onset → IHIT.
        batch = _batch_for_metric(
            target_bins=[50, 50, 50],
            pred_bins=[50, 100, 180],        # exact, future-match, miss-all
            b_pred=200,
            all_future=[[50, 100, 150], [50, 100, 150], [50, 100, 150]],
        )
        m.update(batch)
        out = m.compute()
        # All three samples have valid future onsets.
        assert out["onset/n_any_future"] == 3
        # Exact hit → IHIT, future match → IHIT, miss-all → not IHIT.
        assert out["onset/ihit"] == pytest.approx(2 / 3)
        # 180 is farther than 7 frames from 150 (diff 30) and ratio 181/151
        # gives |log| ≈ 0.18 > log(100/90) → not IGOOD either.
        assert out["onset/igood"] == pytest.approx(2 / 3)
        assert out["onset/imiss"] == pytest.approx(1 / 3)

    def test_reset(self):
        m = OnsetMetric(OnsetMetricConfig(b_pred=100))
        m.update(_batch_for_metric(
            target_bins=[50], pred_bins=[50], b_pred=100,
        ))
        m.reset()
        out = m.compute()
        # n_nonstop = 0 → n floor = 1; accumulators all 0 → all rates 0.
        assert out["onset/n_nonstop"] == 0
        assert out["onset/exact"] == 0.0

    def test_name(self):
        m = OnsetMetric(OnsetMetricConfig())
        assert m.name == "onset"


# ─────────────────────────── artifacts ────────────────────────────────

class TestArtifacts:
    def _simple_batch(self, b_pred: int = 50) -> MetricInput:
        return _batch_for_metric(
            target_bins=[5, 10, 20, 50],       # last is STOP
            pred_bins=[5, 12, 22, 50],
            b_pred=b_pred,
        )

    def test_heatmap(self, tmp_path: Path):
        art = PredictionHeatmapArtifact(b_pred=50)
        art.update(self._simple_batch())
        art.save(tmp_path, step=100)
        assert (tmp_path / "heatmap.png").exists()
        assert (tmp_path / "heatmap.npy").exists()
        # Raw histogram totals match number of samples.
        hist = np.load(tmp_path / "heatmap.npy")
        assert hist.sum() == 4

    def test_distribution(self, tmp_path: Path):
        art = DistributionArtifact(b_pred=50)
        art.update(self._simple_batch())
        art.save(tmp_path, step=100)
        assert (tmp_path / "distributions.png").exists()
        data = np.load(tmp_path / "distributions.npz")
        assert data["targets"].sum() == 4
        assert data["preds"].sum() == 4

    def test_ratio_error(self, tmp_path: Path):
        art = RatioErrorHeatmapArtifact(b_pred=50)
        art.update(self._simple_batch())
        art.save(tmp_path, step=100)
        assert (tmp_path / "ratio_error.png").exists()
        data = np.load(tmp_path / "ratio_error.npz")
        # 3 non-STOP samples land in the 2-D histogram.
        assert int(data["n_seen"]) == 3
        assert data["hist"].sum() + int(data["n_oob"]) == 3

    def test_error_hist(self, tmp_path: Path):
        art = ErrorHistogramArtifact(b_pred=50)
        art.update(self._simple_batch())
        art.save(tmp_path, step=100)
        assert (tmp_path / "error_hist.png").exists()
        data = np.load(tmp_path / "error_hist.npz")
        # 3 non-STOP samples.
        assert data["errors"].size == 3
        # Errors: 5→5 (0), 10→12 (+2), 20→22 (+2). STOP sample excluded.
        assert set(data["errors"].tolist()) == {0, 2}

    def test_heatmap_reset(self, tmp_path: Path):
        art = PredictionHeatmapArtifact(b_pred=10)
        art.update(_batch_for_metric(
            target_bins=[5], pred_bins=[5], b_pred=10,
        ))
        assert art.histogram.sum() == 1
        art.reset()
        assert art.histogram.sum() == 0


# ─────────────────────────── end-to-end loop ──────────────────────────

def _build_tiny_dataset(tmp_path: Path) -> Path:
    """Small synthetic on-disk dataset: 2 songs × 2 diffs × 10 onsets."""
    ds = tmp_path / "ds"
    feat_dir = ds / "features"
    evt_dir = ds / "events"
    feat_dir.mkdir(parents=True)
    evt_dir.mkdir(parents=True)

    n_frames = 5_000
    entries = []
    for song_name, bset in [("song1", "s1"), ("song2", "s2")]:
        feat_rel = Path("features") / f"{song_name}.npy"
        save_features(
            np.zeros((16, n_frames), dtype=np.float32), ds / feat_rel,
        )
        for diff in ["Normal", "Oni"]:
            chart_id = f"{song_name} [{diff}]"
            onsets = tuple(
                OnsetBinned(time_ms=int(b * 5), kind=OnsetKind.DON, bin=int(b))
                for b in range(200, 1200, 100)
            )
            save_events(
                onsets, evt_dir / f"{_safe_filename(chart_id)}.npz",
            )
            entries.append(ChartEntry(
                chart_id=chart_id, beatmap_id=chart_id, beatmapset_id=bset,
                artist="a", title="t", difficulty_version=diff,
                overall_difficulty=5.0, star_rating=None,
                density_mean=4.3, density_peak=8, density_std=1.5,
                duration_s=60.0, total_events=10,
                audio_filename=f"{bset}.mp3",
                features_path=feat_rel, n_frames=n_frames,
            ))
    save_manifest(DatasetManifest(
        name="tiny", created_at="t",
        sampler_config=MelSamplerConfig(),
        charts=tuple(entries),
    ), ds / "manifest.json")
    return ds


class TestEndToEndLoop:
    """A minimal but full-stack loop run. Exercises adapter, loss,
    metric, hooks, artifacts, checkpoint save/resume."""

    def _make_all(self, tmp_path: Path, *, b_pred: int = 40):
        ds_root = _build_tiny_dataset(tmp_path)

        train_s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
            batch_size=4, dataset_root=ds_root,
            a_bins=40, b_bins=40, c_events=8, d_events=3,
            min_cursor_bin=0,
            allowed_overlap_forward=0, allowed_overlap_back=0,
            split_ratios=(("train", 0.5), ("val", 0.5)),
            split="train", split_seed=0,
        ))
        val_s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
            batch_size=4, dataset_root=ds_root,
            a_bins=40, b_bins=40, c_events=8, d_events=3,
            min_cursor_bin=0,
            allowed_overlap_forward=0, allowed_overlap_back=0,
            split_ratios=(("train", 0.5), ("val", 0.5)),
            split="val", split_seed=0,
        ))
        train_s.load_data()
        val_s.load_data()

        model = EventEmbeddingDetector(EventEmbeddingConfig(
            n_mels=16, d_model=32, n_layers=2, n_heads=4,
            c_events=8, a_bins=40, b_bins=40, b_pred=b_pred,
            dropout=0.0,
        ))
        loss = OnsetLoss(OnsetLossConfig(stop_weight=1.0))
        adapter = DetectionSampleAdapter(
            DetectionSampleAdapterConfig(b_pred=b_pred),
        )
        return ds_root, train_s, val_s, model, loss, adapter

    def test_full_loop_writes_everything(self, tmp_path: Path):
        b_pred = 40
        _, train_s, val_s, model, loss, adapter = self._make_all(
            tmp_path, b_pred=b_pred,
        )

        spec = RunSpec(root=tmp_path / "runs", name="r1")
        metric_key = "onset/miss"
        trainer_cfg = TrainerConfig(
            epochs=1, batch_size=4,
            learning_rate=1e-3, weight_decay=0.0,
            grad_clip=1.0, evals_per_epoch=1, amp=False,
            metric_to_watch=metric_key,
            metric_lower_is_better=True,
        )

        val_metrics = MetricSet(OnsetMetric(OnsetMetricConfig(b_pred=b_pred)))
        artifacts = [
            PredictionHeatmapArtifact(b_pred=b_pred),
            DistributionArtifact(b_pred=b_pred),
            RatioErrorHeatmapArtifact(b_pred=b_pred),
            ErrorHistogramArtifact(b_pred=b_pred),
        ]

        final_state = train(
            spec=spec,
            trainer_config=trainer_cfg,
            model=model, loss=loss, adapter=adapter,
            train_sampler=train_s, val_sampler=val_s,
            val_metrics=val_metrics,
            eval_artifacts=artifacts,
            device=torch.device("cpu"),
        )
        assert final_state.step > 0
        assert final_state.epoch == 1

        # metrics.jsonl exists and has at least one step + one eval line.
        lines = spec.metrics_path.read_text(encoding="utf-8").splitlines()
        events = [l for l in lines if l]
        assert len(events) >= 2
        import json
        event_types = {json.loads(l)["event"] for l in events}
        assert "step" in event_types
        assert "eval" in event_types
        assert "train_end" in event_types

        # Checkpoints written.
        assert spec.latest_checkpoint.exists()
        assert spec.best_checkpoint.exists()

        # Artifacts: one eval_{step} dir with all four artifact PNGs.
        eval_dirs = list(spec.run_dir.glob("eval_*"))
        assert len(eval_dirs) >= 1
        ed = eval_dirs[0]
        for stem in ("heatmap", "distributions", "ratio_error", "error_hist"):
            assert (ed / f"{stem}.png").exists(), f"missing {stem}.png"

    def test_resume_picks_up_step(self, tmp_path: Path):
        """After a run, starting a second `train()` with the same spec
        resumes at the checkpoint's step — no duplicate work."""
        b_pred = 40
        _, train_s, val_s, model, loss, adapter = self._make_all(
            tmp_path, b_pred=b_pred,
        )
        spec = RunSpec(root=tmp_path / "runs", name="r2")
        cfg = TrainerConfig(
            epochs=1, batch_size=4, learning_rate=1e-3, weight_decay=0.0,
            grad_clip=1.0, evals_per_epoch=1,
            metric_to_watch="onset/miss", metric_lower_is_better=True,
        )

        # First run.
        state_a = train(
            spec=spec, trainer_config=cfg,
            model=model, loss=loss, adapter=adapter,
            train_sampler=train_s, val_sampler=val_s,
            val_metrics=MetricSet(OnsetMetric(OnsetMetricConfig(b_pred=b_pred))),
        )
        step_a = state_a.step
        assert step_a > 0

        # Second run with a fresh model + optimizer: should resume.
        fresh_model = EventEmbeddingDetector(model.config)
        fresh_loss = OnsetLoss(loss.config)
        state_b = train(
            spec=spec, trainer_config=cfg,
            model=fresh_model, loss=fresh_loss, adapter=adapter,
            train_sampler=train_s, val_sampler=val_s,
            val_metrics=MetricSet(OnsetMetric(OnsetMetricConfig(b_pred=b_pred))),
        )
        # Resumed at step_a, then no further steps (epochs=1 → already done).
        assert state_b.step == step_a

    def test_resume_flag_drops_post_eval_stats(self, tmp_path: Path):
        """`--resume` finds the last eval's checkpoint and truncates any
        metrics.jsonl rows + eval_{N}/ directories past that step."""
        import json, shutil  # noqa: E401

        b_pred = 40
        _, train_s, val_s, model, loss, adapter = self._make_all(
            tmp_path, b_pred=b_pred,
        )
        spec = RunSpec(root=tmp_path / "runs", name="r_resume")
        cfg = TrainerConfig(
            epochs=1, batch_size=4, learning_rate=1e-3, weight_decay=0.0,
            grad_clip=1.0, evals_per_epoch=1,
            metric_to_watch="onset/miss", metric_lower_is_better=True,
        )
        state_a = train(
            spec=spec, trainer_config=cfg,
            model=model, loss=loss, adapter=adapter,
            train_sampler=train_s, val_sampler=val_s,
            val_metrics=MetricSet(OnsetMetric(OnsetMetricConfig(b_pred=b_pred))),
        )
        last_step = state_a.step
        assert last_step > 0

        # Simulate mid-epoch garbage past the eval: fake extra step rows
        # in metrics.jsonl + a bogus eval_{step+1}/ directory. `--resume`
        # should strip both.
        with spec.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": "step", "step": last_step + 7, "epoch": 1,
                "wall_time": 0, "train/batch/loss": 99.0,
            }) + "\n")
        bogus_dir = spec.run_dir / f"eval_{last_step + 11}"
        bogus_dir.mkdir()
        # No checkpoint.pt inside — find_last_eval_checkpoint should not
        # select this dir, but truncate_stats_after_step should still
        # delete it as "past the resumed step".
        (bogus_dir / "artifact.txt").write_text("leftover")

        # Resume. Per-eval checkpoints are snapshotted inside the epoch
        # (before `state.epoch += 1`) so resume restarts at epoch=0 and
        # replays the rest of epoch 0 — that's the documented semantics
        # of "state right after the last eval finished". We don't assert
        # on final step count; we assert the truncation side effects.
        fresh_model = EventEmbeddingDetector(model.config)
        fresh_loss = OnsetLoss(loss.config)
        train(
            spec=spec, trainer_config=cfg,
            model=fresh_model, loss=fresh_loss, adapter=adapter,
            train_sampler=train_s, val_sampler=val_s,
            val_metrics=MetricSet(OnsetMetric(OnsetMetricConfig(b_pred=b_pred))),
            resume=True,
        )

        # Truncation evidence:
        with spec.metrics_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert not any(
            r.get("event") == "step" and r.get("step") == last_step + 7
            for r in rows
        ), "bogus mid-epoch step row survived --resume truncation"
        assert not bogus_dir.exists(), (
            "bogus future eval_{N}/ dir survived --resume truncation"
        )
