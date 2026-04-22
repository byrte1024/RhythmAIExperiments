"""Tests for the abstract model / loss / adapter / metrics / training ABCs.

These ABCs have no concrete implementations in the framework yet (the
exp 45 port comes next). Tests here exercise:
  - abstract-method enforcement (subclassing without overrides fails).
  - typed contracts (concrete subclasses with minimal fields work).
  - MetricSet's collision detection.
  - MetricsReport's key-prefixing convention.
  - RunSpec path helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from osu.taiko2.domain.adapter import SampleToModelAdapter
from osu.taiko2.domain.loss import Loss, LossConfig, LossResult
from osu.taiko2.domain.metrics import (
    Metric,
    MetricConfig,
    MetricInput,
    MetricSet,
    MetricWindow,
    MetricsReport,
)
from osu.taiko2.domain.model import (
    Model,
    ModelConfig,
    ModelInput,
    ModelOutput,
    ModelTarget,
)
from osu.taiko2.domain.sampling import DataSample
from osu.taiko2.domain.training import (
    CheckpointMeta,
    RunSpec,
    TrainerConfig,
    TrainerHook,
    TrainingState,
)


# ─────────────────────────── concrete test types ──────────────────────

@dataclass(frozen=True, slots=True)
class _DummyConfig(ModelConfig):
    d_model: int = 4


@dataclass(frozen=True, slots=True)
class _DummyInput(ModelInput):
    x: torch.Tensor = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class _DummyOutput(ModelOutput):
    y: torch.Tensor = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class _DummyTarget(ModelTarget):
    t: torch.Tensor = None  # type: ignore[assignment]


class _DummyModel(Model[_DummyConfig, _DummyInput, _DummyOutput]):
    def __init__(self, config: _DummyConfig):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.d_model, config.d_model)

    def predict(self, x: _DummyInput) -> _DummyOutput:
        return _DummyOutput(y=self.linear(x.x))


@dataclass(frozen=True, slots=True)
class _DummyLossConfig(LossConfig):
    weight: float = 1.0


class _DummyLoss(Loss[_DummyLossConfig, _DummyOutput, _DummyTarget]):
    def forward(self, output: _DummyOutput, target: _DummyTarget) -> LossResult:
        l = ((output.y - target.t) ** 2).mean() * self.config.weight
        return LossResult(loss=l, metrics={"mse": float(l.detach())})


# ─────────────────────────── Model / Loss ─────────────────────────────

class TestModel:
    def test_abstract_predict_enforced(self):
        """A subclass missing `predict` can't be instantiated."""
        class Broken(Model[_DummyConfig, _DummyInput, _DummyOutput]):
            pass
        with pytest.raises(TypeError, match="abstract"):
            Broken(_DummyConfig())  # type: ignore[abstract]

    def test_n_params(self):
        m = _DummyModel(_DummyConfig(d_model=4))
        # Linear(4, 4) = 16 weights + 4 bias = 20
        assert m.n_params == 20

    def test_config_stored(self):
        cfg = _DummyConfig(d_model=8)
        m = _DummyModel(cfg)
        assert m.config is cfg

    def test_predict_returns_typed_output(self):
        m = _DummyModel(_DummyConfig(d_model=4))
        out = m.predict(_DummyInput(x=torch.zeros(2, 4)))
        assert isinstance(out, _DummyOutput)
        assert out.y.shape == (2, 4)


class TestLoss:
    def test_abstract_forward_enforced(self):
        class Broken(Loss[_DummyLossConfig, _DummyOutput, _DummyTarget]):
            pass
        with pytest.raises(TypeError, match="abstract"):
            Broken(_DummyLossConfig())  # type: ignore[abstract]

    def test_returns_loss_result(self):
        loss = _DummyLoss(_DummyLossConfig())
        out = _DummyOutput(y=torch.ones(2, 4))
        tgt = _DummyTarget(t=torch.zeros(2, 4))
        r = loss(out, tgt)
        assert isinstance(r, LossResult)
        assert r.loss.requires_grad is False  # inputs don't require grad
        assert "mse" in r.metrics
        assert r.metrics["mse"] == pytest.approx(1.0)


# ─────────────────────────── Adapter ──────────────────────────────────

@dataclass(frozen=True, slots=True)
class _DummySample(DataSample):
    value: float = 0.0


class _DummyAdapter(SampleToModelAdapter[_DummySample, _DummyInput, _DummyTarget]):
    def make_input(self, samples, *, device):
        return _DummyInput(x=torch.tensor(
            [[s.value] * 4 for s in samples], device=device, dtype=torch.float32,
        ))

    def make_target(self, samples, *, device):
        return _DummyTarget(t=torch.tensor(
            [[s.value * 2] * 4 for s in samples], device=device, dtype=torch.float32,
        ))


class TestAdapter:
    def test_abstract_methods_enforced(self):
        class Broken(SampleToModelAdapter[_DummySample, _DummyInput, _DummyTarget]):
            pass
        with pytest.raises(TypeError, match="abstract"):
            Broken()  # type: ignore[abstract]

    def test_make_batch(self):
        adapter = _DummyAdapter()
        samples = [_DummySample(sample_id=0, value=1.0), _DummySample(sample_id=1, value=2.0)]
        inp, tgt = adapter.make_batch(samples, device=torch.device("cpu"))
        assert inp.x.shape == (2, 4)
        assert tgt.t.shape == (2, 4)
        assert inp.x[0, 0].item() == 1.0
        assert tgt.t[1, 0].item() == 4.0


# ─────────────────────────── Metrics ──────────────────────────────────

class _MeanMetric(Metric):
    """Concrete metric that accumulates a running mean of `output.y.mean()`."""
    def __init__(self, name: str = "mean"):
        self._name = name
        self._sum = 0.0
        self._count = 0

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, batch: MetricInput) -> None:
        y = batch.output.y  # type: ignore[attr-defined]
        self._sum += float(y.mean())
        self._count += 1

    def compute(self) -> dict[str, float]:
        return {f"{self._name}/mean": self._sum / max(self._count, 1)}


class TestMetric:
    def test_abstract_enforcement(self):
        class Broken(Metric):
            pass
        with pytest.raises(TypeError, match="abstract"):
            Broken()  # type: ignore[abstract]

    def test_reset_update_compute(self):
        m = _MeanMetric()
        m.update(MetricInput(
            output=_DummyOutput(y=torch.tensor([1.0, 2.0])),
            target=_DummyTarget(),
        ))
        m.update(MetricInput(
            output=_DummyOutput(y=torch.tensor([3.0, 4.0])),
            target=_DummyTarget(),
        ))
        assert m.compute() == {"mean/mean": pytest.approx(2.5)}
        m.reset()
        assert m.compute() == {"mean/mean": 0.0}


class TestMetricSet:
    def test_aggregates(self):
        a = _MeanMetric("a")
        b = _MeanMetric("b")
        s = MetricSet(a, b)
        s.reset()
        s.update(MetricInput(
            output=_DummyOutput(y=torch.tensor([1.0])),
            target=_DummyTarget(),
        ))
        out = s.compute()
        assert "a/mean" in out and "b/mean" in out

    def test_key_collision_raises(self):
        a = _MeanMetric("same")
        b = _MeanMetric("same")  # same name → same keys
        s = MetricSet(a, b)
        s.update(MetricInput(
            output=_DummyOutput(y=torch.tensor([1.0])),
            target=_DummyTarget(),
        ))
        with pytest.raises(RuntimeError, match="collision"):
            s.compute()


class TestMetricsReport:
    def test_with_values_namespaces_by_split_and_window(self):
        base = MetricsReport(event="eval", step=100, epoch=2, wall_time=1.0)
        r = base.with_values("train", MetricWindow.BATCH, {"loss": 0.5})
        r = r.with_values("train", MetricWindow.OVERALL, {"loss": 0.4})
        r = r.with_values("val",   MetricWindow.SINGLE, {"loss": 0.6, "acc": 0.9})
        assert r.values == {
            "train/batch/loss": 0.5,
            "train/overall/loss": 0.4,
            "val/single/loss": 0.6,
            "val/single/acc": 0.9,
        }
        # original untouched (frozen dataclass, immutable semantics)
        assert base.values == {}


# ─────────────────────────── Training ─────────────────────────────────

class TestRunSpec:
    def test_path_helpers(self, tmp_path: Path):
        spec = RunSpec(root=tmp_path / "runs", name="exp1")
        assert spec.run_dir == tmp_path / "runs" / "exp1"
        assert spec.checkpoints_dir == spec.run_dir / "checkpoints"
        assert spec.latest_checkpoint == spec.checkpoints_dir / "latest.pt"
        assert spec.best_checkpoint == spec.checkpoints_dir / "best.pt"
        assert spec.step_checkpoint(100) == spec.checkpoints_dir / "step_100.pt"

    def test_ensure_creates_dirs(self, tmp_path: Path):
        spec = RunSpec(root=tmp_path, name="e")
        spec.ensure()
        assert spec.checkpoints_dir.is_dir()


class TestTrainerHook:
    def test_no_op_defaults(self):
        """All hook methods are non-abstract no-ops."""
        class Custom(TrainerHook):
            pass
        h = Custom()
        state = TrainingState()
        spec = RunSpec(root=Path("/tmp"), name="x")
        # none of these should raise
        h.on_train_start(state, spec)
        h.on_epoch_start(state)
        h.on_step_end(state, LossResult(loss=torch.tensor(0.0)))
        h.on_eval_end(state, {})
        h.on_epoch_end(state)
        h.on_train_end(state, None)

    def test_hook_can_observe_state(self):
        collected: list[int] = []

        class Collector(TrainerHook):
            def on_step_end(self, state, train_loss):
                collected.append(state.step)

        h = Collector()
        state = TrainingState(step=5)
        h.on_step_end(state, LossResult(loss=torch.tensor(0.0)))
        assert collected == [5]


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.epochs == 50
        assert cfg.metric_to_watch == "val/eval/loss"
        assert cfg.metric_lower_is_better is True


# ─────────────────────────── Checkpoint ───────────────────────────────

def test_checkpoint_round_trip(tmp_path: Path):
    """Full save / load: model weights restore exactly, training state preserved."""
    from osu.taiko2.persistence.checkpoint import Checkpoint

    model = _DummyModel(_DummyConfig(d_model=4))
    loss = _DummyLoss(_DummyLossConfig(weight=0.5))
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    trainer_cfg = TrainerConfig(batch_size=16, epochs=10)
    state = TrainingState(epoch=3, step=123, samples_seen=1968)

    ckpt = Checkpoint.from_runtime(
        model=model, loss=loss, optimizer=opt, scheduler=None,
        trainer_config=trainer_cfg, training_state=state,
    )
    path = tmp_path / "latest.pt"
    ckpt.save(path)
    assert path.exists()
    # No leftover .tmp
    assert not path.with_suffix(".pt.tmp").exists()

    loaded = Checkpoint.load(path)
    assert loaded.meta.training_state.step == 123
    assert loaded.meta.training_state.epoch == 3
    assert loaded.meta.trainer_config.batch_size == 16
    assert loaded.meta.loss_config.weight == 0.5
    assert loaded.meta.model_config.d_model == 4

    # Restore into a fresh model and confirm weights match
    fresh = _DummyModel(_DummyConfig(d_model=4))
    fresh_state = loaded.restore_to(model=fresh, optimizer=None)
    assert fresh_state.step == 123
    for p1, p2 in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(p1.detach(), p2.detach())


def test_checkpoint_atomic_write_no_tmp_leftover(tmp_path: Path):
    """The temp file is cleaned up on successful writes."""
    from osu.taiko2.persistence.checkpoint import Checkpoint

    model = _DummyModel(_DummyConfig(d_model=4))
    loss = _DummyLoss(_DummyLossConfig())
    ckpt = Checkpoint.from_runtime(
        model=model, loss=loss, optimizer=None, scheduler=None,
        trainer_config=TrainerConfig(), training_state=TrainingState(),
    )
    path = tmp_path / "cp.pt"
    ckpt.save(path)
    assert path.exists()
    assert not (tmp_path / "cp.pt.tmp").exists()


def test_save_latest_also_writes_best_when_flagged(tmp_path: Path):
    from osu.taiko2.persistence.checkpoint import Checkpoint, save_latest

    spec = RunSpec(root=tmp_path, name="r")
    model = _DummyModel(_DummyConfig(d_model=4))
    loss = _DummyLoss(_DummyLossConfig())
    ckpt = Checkpoint.from_runtime(
        model=model, loss=loss, optimizer=None, scheduler=None,
        trainer_config=TrainerConfig(), training_state=TrainingState(),
    )
    save_latest(spec, ckpt, is_best=True)
    assert spec.latest_checkpoint.exists()
    assert spec.best_checkpoint.exists()


def test_load_latest_if_any_missing_returns_none(tmp_path: Path):
    from osu.taiko2.persistence.checkpoint import load_latest_if_any

    spec = RunSpec(root=tmp_path, name="empty")
    assert load_latest_if_any(spec) is None


# ─────────────────────────── CheckpointMeta ───────────────────────────

def test_checkpoint_meta_is_frozen():
    """Meta should be immutable."""
    model_cfg = _DummyConfig()
    loss_cfg = _DummyLossConfig()
    trainer_cfg = TrainerConfig()
    state = TrainingState()
    meta = CheckpointMeta(
        model_class="x:X",
        loss_class="x:Y",
        model_config=model_cfg,
        loss_config=loss_cfg,
        trainer_config=trainer_cfg,
        training_state=state,
        created_at="now",
    )
    with pytest.raises(Exception):
        meta.epoch = 99  # type: ignore[misc]
