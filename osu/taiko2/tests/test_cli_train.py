"""Smoke test for `cli.train` — builds a tiny synthetic dataset,
writes all 5 config JSONs, and runs `main(argv)` to exercise the
full wiring path (config load → sampler → model → loss → adapter →
loop → checkpoint → artifacts)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from osu.taiko2.cli.train import main as cli_main
from osu.taiko2.dataset import _safe_filename
from osu.taiko2.domain.beatmap import OnsetBinned, OnsetKind
from osu.taiko2.domain.dataset import (
    ChartEntry,
    DatasetManifest,
    MelSamplerConfig,
)
from osu.taiko2.persistence.events import save_events
from osu.taiko2.persistence.features import save_features
from osu.taiko2.persistence.manifest import save_manifest


def _build_tiny_dataset(root: Path) -> Path:
    ds = root / "ds"
    feat_dir = ds / "features"
    evt_dir = ds / "events"
    feat_dir.mkdir(parents=True)
    evt_dir.mkdir(parents=True)
    entries = []
    n_frames = 4_000
    for song, bset in [("song1", "s1"), ("song2", "s2")]:
        feat_rel = Path("features") / f"{song}.npy"
        save_features(np.zeros((16, n_frames), dtype=np.float32), ds / feat_rel)
        for diff in ("Normal", "Oni"):
            chart_id = f"{song} [{diff}]"
            onsets = tuple(
                OnsetBinned(time_ms=int(b * 5), kind=OnsetKind.DON, bin=int(b))
                for b in range(200, 1200, 100)
            )
            save_events(onsets, evt_dir / f"{_safe_filename(chart_id)}.npz")
            entries.append(ChartEntry(
                chart_id=chart_id, beatmap_id=chart_id, beatmapset_id=bset,
                artist="a", title="t", difficulty_version=diff,
                overall_difficulty=5.0, star_rating=None,
                density_mean=4.0, density_peak=8, density_std=1.0,
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


def _write_tiny_configs(cfg_dir: Path) -> None:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "model.json").write_text(json.dumps({
        "__class__": "osu.taiko2.models.event_embedding:EventEmbeddingConfig",
        "n_mels": 16, "d_model": 32, "n_layers": 2, "n_heads": 4,
        "dropout": 0.0, "c_events": 8, "gap_ratios": True, "cond_dim": 16,
        "a_bins": 40, "b_bins": 40, "b_pred": 40,
    }))
    (cfg_dir / "loss.json").write_text(json.dumps({
        "__class__": "osu.taiko2.training.losses:OnsetLossConfig",
        "hard_alpha": 0.5, "good_pct": 0.03, "fail_pct": 0.20,
        "frame_tolerance": 2, "stop_weight": 1.0,
    }))
    (cfg_dir / "trainer.json").write_text(json.dumps({
        "__class__": "osu.taiko2.domain.training:TrainerConfig",
        "epochs": 1, "batch_size": 4, "learning_rate": 1e-3,
        "weight_decay": 0.0, "grad_clip": 1.0, "evals_per_epoch": 1,
        "amp": False, "num_workers": 0, "seed": 42,
        "metric_to_watch": "onset/miss", "metric_lower_is_better": True,
    }))
    (cfg_dir / "data.json").write_text(json.dumps({
        "__class__": "osu.taiko2.data_samplers.detection:TaikoDetectionSamplerConfig",
        "batch_size": 4, "a_bins": 40, "b_bins": 40,
        "c_events": 8, "d_events": 1, "min_cursor_bin": 0,
        "allowed_overlap_forward": 0, "allowed_overlap_back": 0,
        "subsample": 2,
        "split_ratios": [["train", 0.5], ["val", 0.5]],
        "split_seed": 0,
    }))
    (cfg_dir / "adapter.json").write_text(json.dumps({
        "__class__": "osu.taiko2.training.adapters:DetectionSampleAdapterConfig",
        "b_pred": 40,
    }))


def test_cli_train_runs_end_to_end(tmp_path: Path):
    ds_root = _build_tiny_dataset(tmp_path)
    cfg_dir = tmp_path / "cfgs"
    _write_tiny_configs(cfg_dir)

    runs_dir = tmp_path / "runs"
    argv = [
        "--run-name", "smoke",
        "--runs-dir", str(runs_dir),
        "--config-dir", str(cfg_dir),
        "--dataset", "ds",
        "--datasets-dir", str(tmp_path),
        "--device", "cpu",
        # Disable aug — keeps the smoke deterministic + fast, the
        # aug-under-load path is covered by `test_ar_and_augs.py`.
        "--no-augmentation",
        "--no-weighted-sampling",
    ]
    rc = cli_main(argv)
    assert rc == 0

    run_dir = runs_dir / "smoke"
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoints" / "latest.pt").exists()
    eval_dirs = list(run_dir.glob("eval_*"))
    assert eval_dirs, "no eval_<step> directory was created"
    ed = eval_dirs[0]
    for stem in ("heatmap", "distributions", "ratio_error", "error_hist"):
        assert (ed / f"{stem}.png").exists()


def test_cli_rejects_mismatched_b_pred(tmp_path: Path):
    """Adapter b_pred ≠ model b_pred must fail before training starts."""
    _build_tiny_dataset(tmp_path)
    cfg_dir = tmp_path / "cfgs"
    _write_tiny_configs(cfg_dir)

    # Break adapter.json so b_pred mismatches the model.
    (cfg_dir / "adapter.json").write_text(json.dumps({
        "__class__": "osu.taiko2.training.adapters:DetectionSampleAdapterConfig",
        "b_pred": 999,     # doesn't match model's 40
    }))

    with pytest.raises(ValueError, match="b_pred"):
        cli_main([
            "--run-name", "bad",
            "--runs-dir", str(tmp_path / "runs"),
            "--config-dir", str(cfg_dir),
            "--dataset", "ds",
            "--datasets-dir", str(tmp_path),
            "--device", "cpu",
            "--no-augmentation",
            "--no-weighted-sampling",
        ])
