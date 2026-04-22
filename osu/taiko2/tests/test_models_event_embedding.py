"""Tests for the exp 45 port: common primitives + EventEmbeddingDetector."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from osu.taiko2.domain.loss import Loss, LossConfig, LossResult
from osu.taiko2.domain.model import Model, ModelInput, ModelOutput
from osu.taiko2.domain.training import TrainerConfig, TrainingState
from osu.taiko2.models import (
    AudioConvStem,
    EventEmbeddingConfig,
    EventEmbeddingDetector,
    EventEmbeddingInput,
    EventEmbeddingOutput,
    FiLM,
    SinusoidalPosEmb,
)
from osu.taiko2.persistence.checkpoint import Checkpoint


# ─────────────────────────── primitives ───────────────────────────────

class TestSinusoidalPosEmb:
    def test_shape(self):
        pe = SinusoidalPosEmb(d_model=16)
        positions = torch.arange(10).unsqueeze(0).expand(4, -1)
        out = pe(positions)
        assert out.shape == (4, 10, 16)

    def test_position_0_is_sin0_cos0(self):
        """Position 0 produces sin(0)=0 for the first half and
        cos(0)=1 for the second half."""
        pe = SinusoidalPosEmb(d_model=8)
        out = pe(torch.zeros(1, 1, dtype=torch.long))
        assert torch.allclose(out[0, 0, :4], torch.zeros(4), atol=1e-6)
        assert torch.allclose(out[0, 0, 4:], torch.ones(4), atol=1e-6)

    def test_different_positions_differ(self):
        pe = SinusoidalPosEmb(d_model=16)
        out = pe(torch.arange(5).unsqueeze(0))
        # Consecutive rows should differ
        for i in range(4):
            assert not torch.allclose(out[0, i], out[0, i + 1])

    def test_odd_d_model_rejected(self):
        with pytest.raises(ValueError, match="even"):
            SinusoidalPosEmb(d_model=15)


class TestFiLM:
    def test_starts_as_identity(self):
        """Zero-init means `(x, cond) → x` regardless of `cond`."""
        film = FiLM(cond_dim=8, feat_dim=16)
        x = torch.randn(3, 4, 16)
        cond = torch.randn(3, 8)
        out = film(x, cond)
        assert torch.allclose(out, x)

    def test_modulates_after_training(self):
        """After loading non-zero weights, FiLM becomes non-trivial."""
        film = FiLM(cond_dim=8, feat_dim=16)
        with torch.no_grad():
            film.fc.weight.normal_(0, 0.1)
            film.fc.bias.normal_(0, 0.1)
        x = torch.randn(3, 4, 16)
        cond = torch.randn(3, 8)
        out = film(x, cond)
        assert not torch.allclose(out, x)
        assert out.shape == x.shape


class TestAudioConvStem:
    def test_4x_downsample(self):
        stem = AudioConvStem(n_mels=80, d_model=128)
        mel = torch.randn(2, 80, 1000)
        out = stem(mel)
        assert out.shape == (2, 250, 128)

    def test_output_is_normalized(self):
        """LayerNorm on output → per-token (approx) zero-mean unit-var."""
        stem = AudioConvStem(n_mels=80, d_model=64)
        # Large-magnitude input; output should still have bounded stats.
        mel = torch.randn(2, 80, 400) * 100
        out = stem(mel)
        # Per-token std ≈ 1 after the final LayerNorm.
        per_token_std = out.std(dim=-1)
        assert per_token_std.max() < 5  # loose bound; LayerNorm normalizes


# ─────────────────────────── config ───────────────────────────────────

class TestEventEmbeddingConfig:
    def test_defaults_match_exp45(self):
        c = EventEmbeddingConfig()
        assert c.d_model == 384
        assert c.n_layers == 8
        assert c.n_heads == 8
        assert c.c_events == 128
        assert c.gap_ratios is True
        assert c.a_bins == 500
        assert c.b_bins == 500
        assert c.b_pred == 500
        # Derived
        assert c.n_classes == 501
        assert c.n_audio_tokens == 250
        assert c.cursor_token == 125

    def test_b_pred_less_than_b_bins(self):
        """Model can see further than it predicts into."""
        c = EventEmbeddingConfig(a_bins=500, b_bins=1000, b_pred=500)
        assert c.n_classes == 501
        assert c.n_audio_tokens == 375
        assert c.cursor_token == 125

    def test_b_pred_exceeds_b_bins_rejected(self):
        with pytest.raises(ValueError, match="b_pred"):
            EventEmbeddingConfig(b_bins=300, b_pred=500)

    def test_a_bins_not_multiple_of_4_rejected(self):
        with pytest.raises(ValueError, match="a_bins"):
            EventEmbeddingConfig(a_bins=501, b_bins=500)

    def test_total_window_not_multiple_of_4_rejected(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            EventEmbeddingConfig(a_bins=500, b_bins=502, b_pred=500)

    def test_head_dim_divisibility_enforced(self):
        with pytest.raises(ValueError, match="n_heads"):
            EventEmbeddingConfig(d_model=384, n_heads=7)


# ─────────────────────────── forward pass ─────────────────────────────

def _tiny_config() -> EventEmbeddingConfig:
    """Small config for fast tests."""
    return EventEmbeddingConfig(
        n_mels=16, d_model=32, n_layers=2, n_heads=4,
        c_events=16, a_bins=100, b_bins=100, b_pred=100,
        dropout=0.0,
    )


def _make_batch(cfg: EventEmbeddingConfig, batch_size: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return EventEmbeddingInput(
        mel=torch.randn(batch_size, cfg.n_mels, cfg.a_bins + cfg.b_bins, generator=g),
        event_offsets=torch.zeros(batch_size, cfg.c_events, dtype=torch.int64),
        event_mask=torch.ones(batch_size, cfg.c_events, dtype=torch.bool),
        conditioning=torch.tensor(
            [[4.0, 8.0, 1.5]] * batch_size, dtype=torch.float32,
        ),
    )


class TestEventEmbeddingDetector:
    def test_output_shape(self):
        cfg = _tiny_config()
        model = EventEmbeddingDetector(cfg).eval()
        batch = _make_batch(cfg)
        out = model.predict(batch)
        assert isinstance(out, EventEmbeddingOutput)
        assert out.logits.shape == (3, cfg.n_classes)

    def test_predict_wraps_forward(self):
        """`predict(input)` must match `forward(*tensors)` numerically."""
        cfg = _tiny_config()
        model = EventEmbeddingDetector(cfg).eval()
        batch = _make_batch(cfg)
        with torch.no_grad():
            via_predict = model.predict(batch).logits
            via_forward = model.forward(
                batch.mel, batch.event_offsets,
                batch.event_mask, batch.conditioning,
            )
        assert torch.allclose(via_predict, via_forward)

    def test_gap_ratios_toggle(self):
        """Both gap_ratios=True and False produce valid outputs."""
        for gap_ratios in (False, True):
            cfg = replace(_tiny_config(), gap_ratios=gap_ratios)
            model = EventEmbeddingDetector(cfg).eval()
            out = model.predict(_make_batch(cfg))
            assert out.logits.shape == (3, cfg.n_classes)

    def test_larger_b_bins_than_b_pred(self):
        """Model sees more audio than it predicts; should still work."""
        cfg = replace(
            _tiny_config(),
            a_bins=100, b_bins=200, b_pred=100,
        )
        model = EventEmbeddingDetector(cfg).eval()
        batch = _make_batch(cfg)
        out = model.predict(batch)
        # n_classes = b_pred + 1 = 101
        assert out.logits.shape == (3, 101)

    def test_real_past_events_affect_output(self):
        """Toggle event_mask from all-padding to one real event; the
        output should change (proves event embeddings actually flow)."""
        cfg = _tiny_config()
        model = EventEmbeddingDetector(cfg).eval()
        no_events = _make_batch(cfg)

        # Build a batch with one real past event at offset -40.
        with_event = EventEmbeddingInput(
            mel=no_events.mel.clone(),
            event_offsets=no_events.event_offsets.clone(),
            event_mask=no_events.event_mask.clone(),
            conditioning=no_events.conditioning.clone(),
        )
        with_event.event_offsets[:, -1] = -40
        with_event.event_mask[:, -1] = False

        with torch.no_grad():
            a = model.predict(no_events).logits
            b = model.predict(with_event).logits
        assert not torch.allclose(a, b)

    def test_no_nan_inf(self):
        """Finite logits for a plausible input."""
        cfg = _tiny_config()
        model = EventEmbeddingDetector(cfg).eval()
        batch = _make_batch(cfg)
        with torch.no_grad():
            out = model.predict(batch).logits
        assert torch.isfinite(out).all()

    def test_n_params_in_range_for_full_config(self):
        """Full exp 45 config should land near 16-17M params."""
        model = EventEmbeddingDetector(EventEmbeddingConfig())
        assert 15_000_000 <= model.n_params <= 18_000_000, (
            f"unexpected param count: {model.n_params:,}"
        )

    def test_config_stored(self):
        cfg = _tiny_config()
        model = EventEmbeddingDetector(cfg)
        assert model.config is cfg


# ─────────────────────────── checkpoint round-trip ────────────────────

# A minimal Loss is needed for `Checkpoint.from_runtime`.
@dataclass(frozen=True, slots=True)
class _NoopLossConfig(LossConfig):
    weight: float = 1.0


class _NoopLoss(Loss[_NoopLossConfig, EventEmbeddingOutput, ModelInput]):  # type: ignore[type-var]
    def forward(self, output, target):
        return LossResult(loss=torch.tensor(0.0), metrics={})


def test_event_embedding_checkpoint_round_trip(tmp_path: Path):
    """Save an initialized EventEmbeddingDetector, reload, confirm
    weights + config round-trip bit-exact."""
    cfg = _tiny_config()
    model = EventEmbeddingDetector(cfg)
    loss = _NoopLoss(_NoopLossConfig())

    ckpt = Checkpoint.from_runtime(
        model=model, loss=loss,
        optimizer=None, scheduler=None,
        trainer_config=TrainerConfig(),
        training_state=TrainingState(),
    )
    path = tmp_path / "event_embed.pt"
    ckpt.save(path)

    loaded = Checkpoint.load(path)
    fresh = EventEmbeddingDetector(loaded.meta.model_config)
    loaded.restore_to(model=fresh, optimizer=None, restore_rng=False)

    for p_src, p_dst in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(p_src.detach(), p_dst.detach())

    # Config values survived
    assert fresh.config.n_layers == cfg.n_layers
    assert fresh.config.c_events == cfg.c_events
    assert fresh.config.gap_ratios == cfg.gap_ratios
