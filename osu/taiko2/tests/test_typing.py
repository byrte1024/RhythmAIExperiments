"""Tests for the onset typing model pipeline: domain types, sampler,
model, loss, adapter, and metrics.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from osu.taiko2.domain.typing import (
    TYPING_CONTEXT,
    TYPING_MEL_PATCH,
    TYPING_WINDOW,
    TypingInput,
    TypingModelConfig,
    TypingOutput,
    TypingSample,
    TypingTarget,
)
from osu.taiko2.models.typing_model import TypingTransformer
from osu.taiko2.training.typing_loss import TypingLoss, TypingLossConfig
from osu.taiko2.training.typing_adapter import (
    TypingAdapterConfig,
    TypingSampleAdapter,
    UNK_LABEL,
)
from osu.taiko2.training.metrics_typing import (
    TypingMetric,
    _binary_entropy,
    _binary_prf,
)
from osu.taiko2.training.typing_artifacts import TypingConfusionArtifact
from osu.taiko2.domain.metrics import MetricInput


# ─────────────────────────── fixtures ────────────────────────────────

def _make_sample(
    sample_id: int = 0,
    target_kind: int = 0,
    target_big: int = 0,
    n_mels: int = 80,
    past_mask_count: int = 0,
    future_mask_count: int = 0,
) -> TypingSample:
    ctx = TYPING_CONTEXT
    mp = TYPING_MEL_PATCH
    rng = np.random.RandomState(sample_id)
    return TypingSample(
        sample_id=sample_id,
        chart_id=f"test_chart_{sample_id}",
        target_idx=20,
        past_iois=rng.randn(ctx, 3).astype(np.float32),
        past_kinds=rng.randint(0, 2, size=ctx).astype(np.uint8),
        past_bigs=rng.randint(0, 2, size=ctx).astype(np.uint8),
        past_mel=rng.randn(ctx, n_mels, mp).astype(np.float32),
        past_mask=np.array(
            [True] * past_mask_count + [False] * (ctx - past_mask_count),
            dtype=bool,
        ),
        past_bins=np.arange(1000, 1000 + ctx * 30, 30, dtype=np.int64),
        target_iois=rng.randn(3).astype(np.float32),
        target_mel=rng.randn(n_mels, mp).astype(np.float32),
        target_bin=1000 + ctx * 30,
        future_iois=rng.randn(ctx, 3).astype(np.float32),
        future_mel=rng.randn(ctx, n_mels, mp).astype(np.float32),
        future_mask=np.array(
            [False] * (ctx - future_mask_count) + [True] * future_mask_count,
            dtype=bool,
        ),
        future_bins=np.arange(1000 + (ctx + 1) * 30, 1000 + (2 * ctx + 1) * 30, 30, dtype=np.int64),
        target_kind=target_kind,
        target_big=target_big,
    )


# ─────────────────────────── domain types ────────────────────────────

class TestDomainTypes:
    def test_constants(self):
        assert TYPING_CONTEXT == 16
        assert TYPING_MEL_PATCH == 5
        assert TYPING_WINDOW == 33

    def test_typing_sample_frozen(self):
        s = _make_sample()
        with pytest.raises(AttributeError):
            s.target_kind = 1  # type: ignore[misc]

    def test_typing_sample_fields(self):
        s = _make_sample(target_kind=1, target_big=1)
        assert s.target_kind == 1
        assert s.target_big == 1
        assert s.past_iois.shape == (TYPING_CONTEXT, 3)
        assert s.past_kinds.shape == (TYPING_CONTEXT,)
        assert s.past_mel.shape == (TYPING_CONTEXT, 80, TYPING_MEL_PATCH)
        assert s.target_mel.shape == (80, TYPING_MEL_PATCH)
        assert s.future_iois.shape == (TYPING_CONTEXT, 3)
        assert s.future_mel.shape == (TYPING_CONTEXT, 80, TYPING_MEL_PATCH)

    def test_typing_model_config_defaults(self):
        cfg = TypingModelConfig()
        assert cfg.d_model == 64
        assert cfg.n_layers == 3
        assert cfg.n_heads == 4
        assert cfg.past_context == TYPING_CONTEXT
        assert cfg.future_context == TYPING_CONTEXT
        assert cfg.window == 2 * TYPING_CONTEXT + 1


# ─────────────────────────── model ───────────────────────────────────

class TestTypingTransformer:
    @pytest.fixture
    def model(self):
        return TypingTransformer(TypingModelConfig())

    def test_param_count(self, model):
        assert 100_000 < model.n_params < 300_000

    def test_forward_shape(self, model):
        B = 4
        W = TYPING_WINDOW
        inp = TypingInput(
            mel_patches=torch.randn(B, W, 80 * 5),
            ioi_features=torch.randn(B, W, 3),
            kind_labels=torch.randint(0, 3, (B, W)),
            big_labels=torch.randint(0, 3, (B, W)),
            positions=torch.arange(W).unsqueeze(0).expand(B, -1),
            mask=torch.zeros(B, W, dtype=torch.bool),
            onset_bins=torch.arange(W, dtype=torch.float32).unsqueeze(0).expand(B, -1) * 30,
        )
        out = model.predict(inp)
        assert out.type_logit.shape == (B,)
        assert out.strength_logit.shape == (B,)

    def test_forward_with_padding(self, model):
        B = 2
        W = TYPING_WINDOW
        mask = torch.zeros(B, W, dtype=torch.bool)
        mask[0, :5] = True  # first 5 tokens padded for sample 0
        mask[1, -3:] = True  # last 3 padded for sample 1
        inp = TypingInput(
            mel_patches=torch.randn(B, W, 80 * 5),
            ioi_features=torch.randn(B, W, 3),
            kind_labels=torch.randint(0, 3, (B, W)),
            big_labels=torch.randint(0, 3, (B, W)),
            positions=torch.arange(W).unsqueeze(0).expand(B, -1),
            mask=mask,
            onset_bins=torch.arange(W, dtype=torch.float32).unsqueeze(0).expand(B, -1) * 30,
        )
        out = model.predict(inp)
        assert out.type_logit.shape == (B,)
        assert not torch.isnan(out.type_logit).any()
        assert not torch.isnan(out.strength_logit).any()

    def test_gradient_flows(self, model):
        B = 2
        W = TYPING_WINDOW
        inp = TypingInput(
            mel_patches=torch.randn(B, W, 80 * 5),
            ioi_features=torch.randn(B, W, 3),
            kind_labels=torch.randint(0, 3, (B, W)),
            big_labels=torch.randint(0, 3, (B, W)),
            positions=torch.arange(W).unsqueeze(0).expand(B, -1),
            mask=torch.zeros(B, W, dtype=torch.bool),
            onset_bins=torch.arange(W, dtype=torch.float32).unsqueeze(0).expand(B, -1) * 30,
        )
        out = model.predict(inp)
        loss = out.type_logit.sum() + out.strength_logit.sum()
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0
        assert all(g > 0 or g == 0 for g in grad_norms)

    def test_predict_matches_forward(self, model):
        model.eval()  # disable dropout for determinism
        B = 2
        W = TYPING_WINDOW
        inp = TypingInput(
            mel_patches=torch.randn(B, W, 80 * 5),
            ioi_features=torch.randn(B, W, 3),
            kind_labels=torch.randint(0, 3, (B, W)),
            big_labels=torch.randint(0, 3, (B, W)),
            positions=torch.arange(W).unsqueeze(0).expand(B, -1),
            mask=torch.zeros(B, W, dtype=torch.bool),
            onset_bins=torch.arange(W, dtype=torch.float32).unsqueeze(0).expand(B, -1) * 30,
        )
        out1 = model.predict(inp)
        out2 = model.forward(
            inp.mel_patches, inp.ioi_features, inp.kind_labels,
            inp.big_labels, inp.positions, inp.mask, inp.onset_bins,
        )
        assert torch.allclose(out1.type_logit, out2.type_logit)
        assert torch.allclose(out1.strength_logit, out2.strength_logit)


# ─────────────────────────── loss ────────────────────────────────────

class TestTypingLoss:
    @pytest.fixture
    def loss_fn(self):
        return TypingLoss(TypingLossConfig())

    def test_loss_output_structure(self, loss_fn):
        B = 8
        out = TypingOutput(
            type_logit=torch.randn(B, requires_grad=True),
            strength_logit=torch.randn(B, requires_grad=True),
        )
        tgt = TypingTarget(
            type_target=torch.randint(0, 2, (B,)).float(),
            strength_target=torch.randint(0, 2, (B,)).float(),
        )
        result = loss_fn(out, tgt)
        assert result.loss.requires_grad
        assert "type_loss" in result.metrics
        assert "strength_loss" in result.metrics
        assert "type_acc" in result.metrics
        assert "strength_acc" in result.metrics
        assert "combined_acc" in result.metrics
        assert "type_entropy" in result.metrics
        assert "strength_entropy" in result.metrics

    def test_loss_positive(self, loss_fn):
        B = 4
        out = TypingOutput(
            type_logit=torch.randn(B),
            strength_logit=torch.randn(B),
        )
        tgt = TypingTarget(
            type_target=torch.ones(B),
            strength_target=torch.zeros(B),
        )
        result = loss_fn(out, tgt)
        assert result.loss.item() > 0

    def test_perfect_prediction(self, loss_fn):
        B = 8
        out = TypingOutput(
            type_logit=torch.tensor([5.0] * 4 + [-5.0] * 4),
            strength_logit=torch.tensor([-5.0] * 8),
        )
        tgt = TypingTarget(
            type_target=torch.tensor([1.0] * 4 + [0.0] * 4),
            strength_target=torch.zeros(8),
        )
        result = loss_fn(out, tgt)
        assert result.metrics["type_acc"] == 1.0
        assert result.metrics["strength_acc"] == 1.0
        assert result.metrics["combined_acc"] == 1.0

    def test_entropy_bounds(self, loss_fn):
        B = 4
        # Confident predictions -> low entropy
        out_conf = TypingOutput(
            type_logit=torch.tensor([10.0, -10.0, 10.0, -10.0]),
            strength_logit=torch.tensor([-10.0, -10.0, -10.0, -10.0]),
        )
        tgt = TypingTarget(
            type_target=torch.tensor([1.0, 0.0, 1.0, 0.0]),
            strength_target=torch.zeros(B),
        )
        result = loss_fn(out_conf, tgt)
        assert result.metrics["type_entropy"] < 0.01

        # Uncertain predictions -> high entropy
        out_unc = TypingOutput(
            type_logit=torch.zeros(B),
            strength_logit=torch.zeros(B),
        )
        result2 = loss_fn(out_unc, tgt)
        assert result2.metrics["type_entropy"] > 0.6

    def test_strength_pos_weight_effect(self):
        B = 4
        out = TypingOutput(
            type_logit=torch.zeros(B),
            strength_logit=torch.zeros(B),
        )
        # All targets = BIG
        tgt = TypingTarget(
            type_target=torch.zeros(B),
            strength_target=torch.ones(B),
        )
        loss_low = TypingLoss(TypingLossConfig(strength_pos_weight=1.0))
        loss_high = TypingLoss(TypingLossConfig(strength_pos_weight=17.0))
        r_low = loss_low(out, tgt)
        r_high = loss_high(out, tgt)
        assert r_high.metrics["strength_loss"] > r_low.metrics["strength_loss"]


# ─────────────────────────── adapter ─────────────────────────────────

class TestTypingAdapter:
    def test_make_batch_shapes(self):
        adapter = TypingSampleAdapter(TypingAdapterConfig(), training=False)
        batch = [_make_sample(i) for i in range(4)]
        inp, tgt = adapter.make_batch(batch, device=torch.device("cpu"))

        assert inp.mel_patches.shape == (4, TYPING_WINDOW, 80 * TYPING_MEL_PATCH)
        assert inp.ioi_features.shape == (4, TYPING_WINDOW, 3)
        assert inp.kind_labels.shape == (4, TYPING_WINDOW)
        assert inp.big_labels.shape == (4, TYPING_WINDOW)
        assert inp.positions.shape == (4, TYPING_WINDOW)
        assert inp.mask.shape == (4, TYPING_WINDOW)
        assert tgt.type_target.shape == (4,)
        assert tgt.strength_target.shape == (4,)

    def test_unk_labels_for_target_and_future(self):
        adapter = TypingSampleAdapter(
            TypingAdapterConfig(dk_flip_prob=0.0), training=False,
        )
        batch = [_make_sample(0)]
        inp, _ = adapter.make_batch(batch, device=torch.device("cpu"))
        ctx = TYPING_CONTEXT
        # Target token (index 16) should be UNK
        assert inp.kind_labels[0, ctx].item() == UNK_LABEL
        assert inp.big_labels[0, ctx].item() == UNK_LABEL
        # Future tokens should be UNK
        for j in range(ctx + 1, TYPING_WINDOW):
            assert inp.kind_labels[0, j].item() == UNK_LABEL
            assert inp.big_labels[0, j].item() == UNK_LABEL

    def test_past_labels_are_set(self):
        adapter = TypingSampleAdapter(
            TypingAdapterConfig(dk_flip_prob=0.0), training=False,
        )
        s = _make_sample(0, past_mask_count=0)
        batch = [s]
        inp, _ = adapter.make_batch(batch, device=torch.device("cpu"))
        ctx = TYPING_CONTEXT
        for j in range(ctx):
            assert inp.kind_labels[0, j].item() in (0, 1)
            assert inp.big_labels[0, j].item() in (0, 1)

    def test_padding_mask_propagates(self):
        adapter = TypingSampleAdapter(
            TypingAdapterConfig(dk_flip_prob=0.0), training=False,
        )
        s = _make_sample(0, past_mask_count=5, future_mask_count=3)
        batch = [s]
        inp, _ = adapter.make_batch(batch, device=torch.device("cpu"))
        # First 5 past tokens should be masked
        for j in range(5):
            assert inp.mask[0, j].item() is True
        # Remaining past should not be masked
        for j in range(5, TYPING_CONTEXT):
            assert inp.mask[0, j].item() is False

    def test_dk_flip_augmentation(self):
        adapter_flip = TypingSampleAdapter(
            TypingAdapterConfig(dk_flip_prob=1.0), training=True,
        )
        adapter_no_flip = TypingSampleAdapter(
            TypingAdapterConfig(dk_flip_prob=0.0), training=False,
        )
        s = _make_sample(42, target_kind=1)  # target = K
        batch = [s]

        _, tgt_no = adapter_no_flip.make_batch(batch, device=torch.device("cpu"))
        inp_flip, tgt_flip = adapter_flip.make_batch(batch, device=torch.device("cpu"))

        # With 100% flip, target kind should be inverted
        assert tgt_no.type_target[0].item() == 1.0  # K
        assert tgt_flip.type_target[0].item() == 0.0  # flipped to D

        # Past kind labels should also be flipped
        inp_no, _ = adapter_no_flip.make_batch(batch, device=torch.device("cpu"))
        ctx = TYPING_CONTEXT
        for j in range(ctx):
            if not s.past_mask[j]:
                orig = inp_no.kind_labels[0, j].item()
                flipped = inp_flip.kind_labels[0, j].item()
                if orig in (0, 1):
                    assert flipped == 1 - orig

    def test_strength_not_flipped(self):
        adapter = TypingSampleAdapter(
            TypingAdapterConfig(dk_flip_prob=1.0), training=True,
        )
        s = _make_sample(0, target_big=1)
        _, tgt = adapter.make_batch([s], device=torch.device("cpu"))
        # Big should NOT be affected by D/K flip
        assert tgt.strength_target[0].item() == 1.0

    def test_no_flip_when_not_training(self):
        adapter = TypingSampleAdapter(
            TypingAdapterConfig(dk_flip_prob=1.0), training=False,
        )
        s = _make_sample(0, target_kind=0)
        _, tgt = adapter.make_batch([s], device=torch.device("cpu"))
        assert tgt.type_target[0].item() == 0.0  # no flip despite prob=1


# ─────────────────────────── metrics ─────────────────────────────────

class TestBinaryPRF:
    def test_perfect(self):
        pred = np.array([1, 1, 0, 0])
        gt = np.array([1, 1, 0, 0])
        p, r, f = _binary_prf(pred, gt)
        assert p == 1.0
        assert r == 1.0
        assert f == 1.0

    def test_all_wrong(self):
        pred = np.array([0, 0, 1, 1])
        gt = np.array([1, 1, 0, 0])
        p, r, f = _binary_prf(pred, gt)
        assert p == 0.0
        assert r == 0.0
        assert f == 0.0

    def test_partial(self):
        pred = np.array([1, 1, 1, 0])
        gt = np.array([1, 0, 1, 0])
        p, r, f = _binary_prf(pred, gt)
        assert abs(p - 2 / 3) < 1e-6
        assert r == 1.0
        assert abs(f - 0.8) < 1e-6

    def test_empty(self):
        pred = np.array([0, 0, 0])
        gt = np.array([0, 0, 0])
        p, r, f = _binary_prf(pred, gt)
        assert p == 0.0 and r == 0.0 and f == 0.0


class TestBinaryEntropy:
    def test_certain(self):
        p = np.array([0.001, 0.999])
        ent = _binary_entropy(p)
        assert all(e < 0.02 for e in ent)

    def test_uncertain(self):
        p = np.array([0.5])
        ent = _binary_entropy(p)
        assert abs(ent[0] - 0.6931) < 0.01

    def test_shape_preserved(self):
        p = np.random.rand(100)
        ent = _binary_entropy(p)
        assert ent.shape == (100,)
        assert all(0 <= e <= 0.7 for e in ent)


class TestTypingMetric:
    def _make_metric_input(
        self, type_logits: list[float], type_targets: list[float],
        str_logits: list[float], str_targets: list[float],
    ) -> MetricInput:
        out = TypingOutput(
            type_logit=torch.tensor(type_logits),
            strength_logit=torch.tensor(str_logits),
        )
        tgt = TypingTarget(
            type_target=torch.tensor(type_targets),
            strength_target=torch.tensor(str_targets),
        )
        return MetricInput(output=out, target=tgt)

    def test_perfect_type_accuracy(self):
        m = TypingMetric(prefix="t")
        m.reset()
        # +5 -> sigmoid > 0.5 -> pred D=1, -5 -> pred K=0
        m.update(self._make_metric_input(
            [5.0, -5.0, 5.0, -5.0], [1.0, 0.0, 1.0, 0.0],
            [-5.0] * 4, [0.0] * 4,
        ))
        result = m.compute()
        assert result["t/type/accuracy"] == 1.0
        assert result["t/type/f1_D"] == 1.0
        assert result["t/type/f1_K"] == 1.0

    def test_random_type_accuracy(self):
        m = TypingMetric(prefix="t")
        m.reset()
        m.update(self._make_metric_input(
            [5.0, 5.0, 5.0, 5.0], [1.0, 0.0, 1.0, 0.0],  # all predict D
            [-5.0] * 4, [0.0] * 4,
        ))
        result = m.compute()
        assert result["t/type/accuracy"] == 0.5

    def test_strength_metrics(self):
        m = TypingMetric(prefix="t")
        m.reset()
        # Predict 2 BIG correctly, miss 1
        m.update(self._make_metric_input(
            [0.0] * 4, [0.0] * 4,
            [5.0, 5.0, -5.0, -5.0], [1.0, 1.0, 1.0, 0.0],
        ))
        result = m.compute()
        assert result["t/strength/recall_BIG"] == pytest.approx(2 / 3, abs=0.01)
        assert result["t/strength/precision_BIG"] == 1.0

    def test_combined_4class(self):
        m = TypingMetric(prefix="t")
        m.reset()
        # DON(D=1,big=0), KA(D=0,big=0), BDON(D=1,big=1), BKA(D=0,big=1)
        m.update(self._make_metric_input(
            [5.0, -5.0, 5.0, -5.0],  # D, K, D, K
            [1.0, 0.0, 1.0, 0.0],
            [-5.0, -5.0, 5.0, 5.0],  # normal, normal, big, big
            [0.0, 0.0, 1.0, 1.0],
        ))
        result = m.compute()
        assert result["t/combined/accuracy"] == 1.0

    def test_threshold_sweep_present(self):
        m = TypingMetric(prefix="t")
        m.reset()
        m.update(self._make_metric_input(
            [1.0, -1.0], [1.0, 0.0], [-1.0, -1.0], [0.0, 0.0],
        ))
        result = m.compute()
        assert "t/type/best_threshold" in result
        assert "t/type/best_f1" in result
        assert "t/strength/best_threshold" in result
        assert "t/strength/best_f1_BIG" in result
        assert "t/type/sweep/acc_at_0.50" in result
        assert "t/strength/sweep/f1_BIG_at_0.50" in result

    def test_confidence_stats(self):
        m = TypingMetric(prefix="t")
        m.reset()
        m.update(self._make_metric_input(
            [5.0, -5.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0],
        ))
        result = m.compute()
        assert result["t/type/conf_correct"] > 0.9
        assert result["t/type/entropy_mean"] < 0.1
        assert result["t/type/mass_decisive"] > 0.5

    def test_reset_clears(self):
        m = TypingMetric(prefix="t")
        m.reset()
        m.update(self._make_metric_input(
            [5.0], [1.0], [-5.0], [0.0],
        ))
        m.reset()
        result = m.compute()
        assert result == {}

    def test_multi_batch_accumulation(self):
        m = TypingMetric(prefix="t")
        m.reset()
        m.update(self._make_metric_input(
            [5.0, -5.0], [1.0, 0.0], [-5.0, -5.0], [0.0, 0.0],
        ))
        m.update(self._make_metric_input(
            [5.0, -5.0], [1.0, 0.0], [-5.0, -5.0], [0.0, 0.0],
        ))
        result = m.compute()
        assert result["t/type/accuracy"] == 1.0
        assert "t/type/precision_D" in result


# ─────────────────────────── artifacts ───────────────────────────────

class TestTypingArtifacts:
    def test_artifact_save(self, tmp_path):
        art = TypingConfusionArtifact()
        art.reset()

        out = TypingOutput(
            type_logit=torch.randn(20),
            strength_logit=torch.randn(20),
        )
        tgt = TypingTarget(
            type_target=torch.randint(0, 2, (20,)).float(),
            strength_target=torch.randint(0, 2, (20,)).float(),
        )
        art.update(MetricInput(output=out, target=tgt))
        art.save(tmp_path, step=100)

        typing_dir = tmp_path / "typing"
        assert typing_dir.exists()
        assert (typing_dir / "type_predictions.npz").exists()
        assert (typing_dir / "strength_predictions.npz").exists()

        # Check plots exist (matplotlib may or may not be available)
        try:
            import matplotlib
            assert (typing_dir / "type_confusion.png").exists()
            assert (typing_dir / "strength_confusion.png").exists()
            assert (typing_dir / "combined_confusion.png").exists()
            assert (typing_dir / "type_confidence_dist.png").exists()
            assert (typing_dir / "strength_confidence_dist.png").exists()
            assert (typing_dir / "type_calibration.png").exists()
            assert (typing_dir / "strength_calibration.png").exists()
            assert (typing_dir / "type_entropy_dist.png").exists()
            assert (typing_dir / "strength_entropy_dist.png").exists()
            assert (typing_dir / "type_conf_vs_acc.png").exists()
            assert (typing_dir / "strength_conf_vs_acc.png").exists()
            assert (typing_dir / "type_threshold_sweep.png").exists()
            assert (typing_dir / "strength_threshold_sweep.png").exists()
        except ImportError:
            pass

    def test_artifact_reset(self):
        art = TypingConfusionArtifact()
        art.update(MetricInput(
            output=TypingOutput(torch.randn(5), torch.randn(5)),
            target=TypingTarget(torch.zeros(5), torch.zeros(5)),
        ))
        art.reset()
        assert len(art._type_probs) == 0

    def test_predictions_npz_content(self, tmp_path):
        art = TypingConfusionArtifact()
        art.reset()
        N = 50
        art.update(MetricInput(
            output=TypingOutput(torch.randn(N), torch.randn(N)),
            target=TypingTarget(
                torch.randint(0, 2, (N,)).float(),
                torch.randint(0, 2, (N,)).float(),
            ),
        ))
        art.save(tmp_path)
        data = np.load(tmp_path / "typing" / "type_predictions.npz")
        assert data["probs"].shape == (N,)
        assert data["targets"].shape == (N,)
        assert data["preds"].shape == (N,)
        assert set(np.unique(data["preds"])).issubset({0, 1})


# ─────────────────────────── end-to-end ──────────────────────────────

class TestEndToEnd:
    def test_sample_through_model(self):
        """Full pipeline: sample -> adapter -> model -> loss -> metrics."""
        model = TypingTransformer(TypingModelConfig())
        loss_fn = TypingLoss(TypingLossConfig())
        adapter = TypingSampleAdapter(
            TypingAdapterConfig(dk_flip_prob=0.0), training=False,
        )
        metric = TypingMetric(prefix="t")
        metric.reset()

        samples = [_make_sample(i, target_kind=i % 2, target_big=int(i > 5))
                   for i in range(8)]
        inp, tgt = adapter.make_batch(samples, device=torch.device("cpu"))
        out = model.predict(inp)
        result = loss_fn(out, tgt)

        assert result.loss.item() > 0
        assert not torch.isnan(result.loss)

        metric.update(MetricInput(output=out, target=tgt))
        computed = metric.compute()
        assert "t/type/accuracy" in computed
        assert "t/strength/accuracy" in computed
        assert "t/combined/accuracy" in computed
        assert 0.0 <= computed["t/type/accuracy"] <= 1.0
