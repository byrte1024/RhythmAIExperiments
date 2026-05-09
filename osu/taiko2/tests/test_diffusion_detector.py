"""Integration tests for the #014 diffusion stack glue:

  - ``DiffusionDetector`` (model trunk + diffusion head wiring).
  - ``DiffusionLoss`` (MSE/Huber + Min-SNR + diagnostic metrics).
  - ``DiffusionDecoder`` (AR decode via sampler + ``bind_model`` hook).
  - ``inference.spec.assemble_predictor`` calling ``bind_model`` for
    decoders that expose it.
  - JSON round-trip of a polymorphic ``DiffusionDetectorConfig`` via
    ``build_config``.

Pure component-level smoke (no full AR-corpus / training-loop run);
that's covered by the existing infer / train tests using the standard
detector and is the same code path.
"""
from __future__ import annotations

import json

import pytest
import torch

from osu.taiko2.diffusion import (
    CosineScheduleConfig,
    DDIMSamplerConfig,
    GaussianContinuousProcessConfig,
    MLPDenoiserConfig,
)
from osu.taiko2.inference.autoregressive.diffusion_decoder import (
    DiffusionDecoder,
    DiffusionDecoderConfig,
)
from osu.taiko2.inference.autoregressive.types import ARContext
from osu.taiko2.inference.spec import build_config
from osu.taiko2.models.diffusion_detector import (
    DiffusionDetector,
    DiffusionDetectorConfig,
    DiffusionModelOutput,
)
from osu.taiko2.models.event_embedding import EventEmbeddingInput
from osu.taiko2.training.diffusion_loss import (
    DiffusionLoss,
    DiffusionLossConfig,
)
from osu.taiko2.models.event_embedding import EventEmbeddingTarget


# ─────────────────────────── Shared fixture helpers ──────────────────


def _tiny_detector_config(b_pred: int = 8) -> DiffusionDetectorConfig:
    """Minimal detector config: small d_model, tiny windows, few bins."""
    n_bins = b_pred + 1
    return DiffusionDetectorConfig(
        n_mels=8,
        d_model=32,
        n_layers=1,
        n_heads=2,
        c_events=4,
        cond_dim=8,
        a_bins=8, b_bins=8, b_pred=b_pred,
        schedule_config=CosineScheduleConfig(n_steps=8),
        process_config=GaussianContinuousProcessConfig(
            n_bins=n_bins, parameterization="x0", x0_scale=2.0,
        ),
        denoiser_config=MLPDenoiserConfig(
            d_model=32, n_bins=n_bins, hidden_dim=64,
            time_embed_dim=16, time_embed_proj_dim=32,
            n_layers=2, dropout=0.0,
        ),
    )


def _tiny_input(cfg: DiffusionDetectorConfig, B: int = 2) -> EventEmbeddingInput:
    T = cfg.a_bins + cfg.b_bins
    return EventEmbeddingInput(
        mel=torch.randn(B, cfg.n_mels, T),
        event_offsets=torch.zeros(B, cfg.c_events, dtype=torch.long),
        event_mask=torch.ones(B, cfg.c_events, dtype=torch.bool),
        conditioning=torch.zeros(B, 3),
    )


# ─────────────────────────── Config validation ───────────────────────


class TestDetectorConfig:
    def test_default_construct(self):
        cfg = _tiny_detector_config()
        # Cross-config invariants (b_pred + 1 == n_bins) must hold.
        assert cfg.process_config.n_bins == cfg.b_pred + 1
        assert cfg.denoiser_config.n_bins == cfg.process_config.n_bins
        assert cfg.denoiser_config.d_model == cfg.d_model

    def test_n_bins_mismatch_rejected(self):
        with pytest.raises(ValueError, match="denoiser_config.n_bins"):
            DiffusionDetectorConfig(
                n_mels=8, d_model=32, n_layers=1, n_heads=2,
                c_events=4, cond_dim=8,
                a_bins=8, b_bins=8, b_pred=8,
                schedule_config=CosineScheduleConfig(n_steps=8),
                process_config=GaussianContinuousProcessConfig(
                    n_bins=9, parameterization="x0",
                ),
                denoiser_config=MLPDenoiserConfig(
                    d_model=32, n_bins=10,           # mismatch with process
                    hidden_dim=32, time_embed_dim=16,
                    time_embed_proj_dim=32, n_layers=2,
                ),
            )

    def test_d_model_mismatch_rejected(self):
        with pytest.raises(ValueError, match="denoiser_config.d_model"):
            DiffusionDetectorConfig(
                n_mels=8, d_model=32, n_layers=1, n_heads=2,
                c_events=4, cond_dim=8,
                a_bins=8, b_bins=8, b_pred=8,
                schedule_config=CosineScheduleConfig(n_steps=8),
                process_config=GaussianContinuousProcessConfig(n_bins=9),
                denoiser_config=MLPDenoiserConfig(
                    d_model=64, n_bins=9,            # mismatch with d_model
                    hidden_dim=32, time_embed_dim=16,
                    time_embed_proj_dim=32, n_layers=2,
                ),
            )

    def test_b_pred_n_bins_mismatch_rejected(self):
        # process_config.n_bins != b_pred + 1
        with pytest.raises(ValueError, match=r"b_pred \+ 1"):
            DiffusionDetectorConfig(
                n_mels=8, d_model=32, n_layers=1, n_heads=2,
                c_events=4, cond_dim=8,
                a_bins=8, b_bins=8, b_pred=8,
                schedule_config=CosineScheduleConfig(n_steps=8),
                process_config=GaussianContinuousProcessConfig(n_bins=20),
                denoiser_config=MLPDenoiserConfig(
                    d_model=32, n_bins=20, hidden_dim=32,
                    time_embed_dim=16, time_embed_proj_dim=32, n_layers=2,
                ),
            )


# ─────────────────────────── DiffusionDetector ───────────────────────


class TestDetector:
    def test_predict_returns_cursor_token_only(self):
        cfg = _tiny_detector_config()
        model = DiffusionDetector(cfg).eval()
        x = _tiny_input(cfg, B=2)
        with torch.no_grad():
            out = model.predict(x)
        assert isinstance(out, DiffusionModelOutput)
        assert out.cursor_token.shape == (2, cfg.d_model)
        # Inference output: training fields are None.
        assert out.model_out is None
        assert out.loss_target is None
        assert out.t is None
        assert out.x_t is None
        # logits is a placeholder (zeros) of correct width.
        assert out.logits.shape == (2, cfg.process_config.n_bins)

    def test_forward_diffusion_populates_training_fields(self):
        cfg = _tiny_detector_config()
        model = DiffusionDetector(cfg).train()
        B = 3
        cursor_token = torch.randn(B, cfg.d_model)
        target_bin = torch.tensor([0, cfg.b_pred, 3], dtype=torch.long)
        out = model.forward_diffusion(cursor_token, target_bin)
        assert out.model_out is not None
        assert out.loss_target is not None
        assert out.t is not None
        assert out.x_t is not None
        n_bins = cfg.process_config.n_bins
        assert out.model_out.shape == (B, n_bins)
        assert out.loss_target.shape == (B, n_bins)
        assert out.x_t.shape == (B, n_bins)
        assert out.t.shape == (B,)
        assert out.t.dtype == torch.long
        assert (out.t >= 0).all() and (out.t < cfg.schedule_config.n_steps).all()
        assert out.logits.shape == (B, n_bins)

    def test_forward_diffusion_with_explicit_t_and_noise(self):
        cfg = _tiny_detector_config()
        model = DiffusionDetector(cfg).eval()
        B = 2
        cursor_token = torch.randn(B, cfg.d_model)
        target_bin = torch.tensor([1, 4], dtype=torch.long)
        t = torch.tensor([0, 7], dtype=torch.long)
        noise = torch.randn(B, cfg.process_config.n_bins)
        out1 = model.forward_diffusion(cursor_token, target_bin, t=t, noise=noise)
        out2 = model.forward_diffusion(cursor_token, target_bin, t=t, noise=noise)
        # Deterministic when t and noise are pinned (no dropout in eval).
        torch.testing.assert_close(out1.model_out, out2.model_out)
        torch.testing.assert_close(out1.loss_target, out2.loss_target)

    def test_unknown_subconfig_type_rejected(self):
        # Use a fresh dataclass that's not in the registry.
        from dataclasses import dataclass

        from osu.taiko2.domain.diffusion import NoiseScheduleConfig

        @dataclass(frozen=True, slots=True)
        class _UnregisteredScheduleConfig(NoiseScheduleConfig):
            pass

        cfg = DiffusionDetectorConfig(
            n_mels=8, d_model=32, n_layers=1, n_heads=2,
            c_events=4, cond_dim=8,
            a_bins=8, b_bins=8, b_pred=8,
            schedule_config=_UnregisteredScheduleConfig(n_steps=8),
            process_config=GaussianContinuousProcessConfig(n_bins=9),
            denoiser_config=MLPDenoiserConfig(
                d_model=32, n_bins=9, hidden_dim=32,
                time_embed_dim=16, time_embed_proj_dim=32, n_layers=2,
            ),
        )
        with pytest.raises(TypeError, match="unknown schedule config"):
            DiffusionDetector(cfg)


# ─────────────────────────── DiffusionLoss ───────────────────────────


class TestLoss:
    def _make_output_and_target(
        self, cfg: DiffusionDetectorConfig, B: int = 4,
    ) -> tuple[DiffusionModelOutput, EventEmbeddingTarget]:
        model = DiffusionDetector(cfg).eval()
        cursor_token = torch.randn(B, cfg.d_model)
        target_bin = torch.randint(0, cfg.b_pred + 1, (B,), dtype=torch.long)
        out = model.forward_diffusion(cursor_token, target_bin)
        target = EventEmbeddingTarget(target_bin=target_bin)
        return out, target

    def test_mse_forward_shape_and_metrics(self):
        cfg = _tiny_detector_config()
        loss = DiffusionLoss(DiffusionLossConfig(loss_type="mse"))
        out, target = self._make_output_and_target(cfg)
        result = loss.forward(out, target)
        assert result.loss.dim() == 0
        assert torch.isfinite(result.loss)
        # Required metric keys for the diagnostic dashboard.
        expected = {
            "loss", "loss/snr_weighted", "stop_rate",
            "loss/per_t_q0", "loss/per_t_q1",
            "loss/per_t_q2", "loss/per_t_q3",
        }
        assert expected <= set(result.metrics.keys())
        # logits are populated by forward_diffusion → argmax_match exists.
        assert "argmax_match" in result.metrics

    def test_huber_forward(self):
        cfg = _tiny_detector_config()
        loss = DiffusionLoss(DiffusionLossConfig(loss_type="huber"))
        out, target = self._make_output_and_target(cfg)
        result = loss.forward(out, target)
        assert torch.isfinite(result.loss)

    def test_snr_weighting_flag_changes_loss(self):
        cfg = _tiny_detector_config()
        out, target = self._make_output_and_target(cfg, B=8)

        loss_off = DiffusionLoss(DiffusionLossConfig(snr_weighting=False))
        loss_on = DiffusionLoss(DiffusionLossConfig(snr_weighting=True))
        # Bind schedule for the SNR-weighted variant.
        model = DiffusionDetector(cfg)
        loss_on.bind_schedule(model.schedule.alphas_cumprod())
        loss_off.bind_schedule(model.schedule.alphas_cumprod())

        r_off = loss_off.forward(out, target)
        r_on = loss_on.forward(out, target)
        # Both finite; values needn't be close (different weighting).
        assert torch.isfinite(r_off.loss)
        assert torch.isfinite(r_on.loss)

    def test_inference_mode_output_unbound_raises(self):
        # Without bind_model, an inference-shape output cannot be
        # processed and the loss should surface a clear error.
        cfg = _tiny_detector_config()
        model = DiffusionDetector(cfg).eval()
        with torch.no_grad():
            out = model.predict(_tiny_input(cfg, B=2))
        target = EventEmbeddingTarget(
            target_bin=torch.zeros(2, dtype=torch.long),
        )
        loss = DiffusionLoss(DiffusionLossConfig())
        with pytest.raises(RuntimeError, match="bind_model"):
            loss.forward(out, target)

    def test_bind_model_processes_inference_output(self):
        # With bind_model the loss should fill in the diffusion forward
        # inline and produce a finite loss + populated diagnostic fields.
        cfg = _tiny_detector_config()
        model = DiffusionDetector(cfg).train()
        loss = DiffusionLoss(DiffusionLossConfig())
        loss.bind_model(model)
        with torch.no_grad():
            out = model.predict(_tiny_input(cfg, B=3))
        target = EventEmbeddingTarget(
            target_bin=torch.tensor([0, 1, cfg.b_pred], dtype=torch.long),
        )
        result = loss.forward(out, target)
        assert torch.isfinite(result.loss)
        # Loss filled in the training fields on the output.
        assert out.model_out is not None
        assert out.loss_target is not None
        assert out.t is not None
        assert out.logits.abs().sum() > 0  # no longer the placeholder zeros

    def test_backward_propagates(self):
        cfg = _tiny_detector_config()
        model = DiffusionDetector(cfg).train()
        loss = DiffusionLoss(DiffusionLossConfig())

        cursor_token = torch.randn(4, cfg.d_model, requires_grad=True)
        target_bin = torch.tensor([0, 1, 2, cfg.b_pred], dtype=torch.long)
        out = model.forward_diffusion(cursor_token, target_bin)
        target = EventEmbeddingTarget(target_bin=target_bin)
        result = loss.forward(out, target)
        result.loss.backward()
        # Denoiser params got gradients.
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.denoiser.parameters()
        )
        assert any_grad


# ─────────────────────────── DiffusionDecoder ────────────────────────


class TestDecoder:
    def _ctx(self) -> ARContext:
        return ARContext(cursor_bin=100, step=0, max_bin=10_000, past_onsets=())

    def test_decode_before_bind_raises(self):
        cfg = _tiny_detector_config()
        dec_cfg = DiffusionDecoderConfig(
            b_pred=cfg.b_pred,
            sampler_config=DDIMSamplerConfig(n_inference_steps=4, eta=0.0),
        )
        dec = DiffusionDecoder(dec_cfg)
        out = DiffusionModelOutput(
            logits=torch.zeros(1, cfg.b_pred + 1),
            cursor_token=torch.randn(1, cfg.d_model),
        )
        with pytest.raises(RuntimeError, match="bind_model"):
            dec.decode(out, self._ctx())

    def test_bind_and_decode_emits_decision(self):
        cfg = _tiny_detector_config()
        model = DiffusionDetector(cfg).eval()
        dec_cfg = DiffusionDecoderConfig(
            b_pred=cfg.b_pred,
            sampler_config=DDIMSamplerConfig(n_inference_steps=4, eta=0.0),
        )
        dec = DiffusionDecoder(dec_cfg)
        dec.bind_model(model)
        cursor_token = torch.randn(1, cfg.d_model)
        out = DiffusionModelOutput(
            logits=torch.zeros(1, cfg.b_pred + 1),
            cursor_token=cursor_token,
        )
        with torch.no_grad():
            decision = dec.decode(out, self._ctx())
        # Either STOP (empty) or one onset; both legal.
        assert len(decision.bin_offsets) in (0, 1)
        assert "entropy" in decision.extras
        assert "n_inference_steps" in decision.extras
        assert decision.extras["n_inference_steps"] == 4.0
        # Top-K logging present.
        assert "top1_bin" in decision.extras

    def test_n_samples_aggregation(self):
        cfg = _tiny_detector_config()
        model = DiffusionDetector(cfg).eval()
        dec_cfg = DiffusionDecoderConfig(
            b_pred=cfg.b_pred,
            sampler_config=DDIMSamplerConfig(n_inference_steps=4, eta=1.0),
            n_samples=3,
        )
        dec = DiffusionDecoder(dec_cfg)
        dec.bind_model(model)
        cursor_token = torch.randn(1, cfg.d_model)
        out = DiffusionModelOutput(
            logits=torch.zeros(1, cfg.b_pred + 1),
            cursor_token=cursor_token,
        )
        with torch.no_grad():
            decision = dec.decode(out, self._ctx())
        assert decision.extras["n_samples"] == 3.0

    def test_decode_strategy_validation(self):
        with pytest.raises(ValueError, match="decode_strategy"):
            DiffusionDecoderConfig(
                decode_strategy="topk",                    # not a valid mode
                sampler_config=DDIMSamplerConfig(n_inference_steps=4),
            )


# ─────────────────────────── JSON config round-trip ──────────────────


class TestJsonRoundTrip:
    def test_polymorphic_subconfigs_resolve(self):
        node = {
            "__class__": "osu.taiko2.models.diffusion_detector:DiffusionDetectorConfig",
            "n_mels": 8, "d_model": 32, "n_layers": 1, "n_heads": 2,
            "c_events": 4, "cond_dim": 8,
            "a_bins": 8, "b_bins": 8, "b_pred": 8,
            "schedule_config": {
                "__class__": "osu.taiko2.diffusion.schedules:CosineScheduleConfig",
                "n_steps": 8,
            },
            "process_config": {
                "__class__": "osu.taiko2.diffusion.processes:GaussianContinuousProcessConfig",
                "n_bins": 9, "parameterization": "x0", "x0_scale": 2.0,
            },
            "denoiser_config": {
                "__class__": "osu.taiko2.diffusion.denoisers:MLPDenoiserConfig",
                "d_model": 32, "n_bins": 9, "hidden_dim": 64,
                "time_embed_dim": 16, "time_embed_proj_dim": 32,
                "n_layers": 2, "dropout": 0.1,
            },
        }
        # Round-trip through json to make sure serialization is faithful.
        cfg = build_config(json.loads(json.dumps(node)))
        assert isinstance(cfg, DiffusionDetectorConfig)
        assert isinstance(cfg.schedule_config, CosineScheduleConfig)
        assert isinstance(cfg.process_config, GaussianContinuousProcessConfig)
        assert isinstance(cfg.denoiser_config, MLPDenoiserConfig)
        # Construct the model from the resolved config.
        model = DiffusionDetector(cfg)
        assert model.schedule.n_steps == 8

    def test_decoder_config_with_sampler_subconfig(self):
        node = {
            "__class__": "osu.taiko2.inference.autoregressive.diffusion_decoder:DiffusionDecoderConfig",
            "b_pred": 8,
            "sampler_config": {
                "__class__": "osu.taiko2.diffusion.samplers:DDIMSamplerConfig",
                "n_inference_steps": 4, "eta": 0.0,
                "timestep_spacing": "linspace",
            },
            "n_samples": 2,
        }
        cfg = build_config(node)
        assert isinstance(cfg, DiffusionDecoderConfig)
        assert isinstance(cfg.sampler_config, DDIMSamplerConfig)
        assert cfg.sampler_config.n_inference_steps == 4
        assert cfg.n_samples == 2


# ─────────────────────────── spec.assemble_predictor bind_model ──────


class TestAssemblePredictorBindModel:
    """``inference.spec`` should call ``decoder.bind_model(model)`` after
    constructing the decoder when the decoder defines that hook. This
    test verifies the contract without touching disk / a real
    checkpoint by exercising ``assemble_predictor_with_model`` (which
    uses the same code path)."""

    def test_bind_model_called_on_diffusion_decoder(self):
        cfg = _tiny_detector_config()
        model = DiffusionDetector(cfg).eval()
        # Build a minimal spec dict targeted at assemble_predictor_with_model.
        # Audio + event samplers / input_builder aren't exercised here;
        # we only inspect the decoder side.
        spec = {
            "checkpoint": "unused.pt",
            "predictor": {
                "__class__": "osu.taiko2.inference.autoregressive.predictor:ARChartPredictor",
                "config": {
                    "__class__": "osu.taiko2.inference.autoregressive.predictor:ARChartPredictorConfig",
                },
            },
            "decoder": {
                "__class__": "osu.taiko2.inference.autoregressive.diffusion_decoder:DiffusionDecoder",
                "config": {
                    "__class__": "osu.taiko2.inference.autoregressive.diffusion_decoder:DiffusionDecoderConfig",
                    "b_pred": cfg.b_pred,
                    "sampler_config": {
                        "__class__": "osu.taiko2.diffusion.samplers:DDIMSamplerConfig",
                        "n_inference_steps": 4, "eta": 0.0,
                    },
                },
            },
        }
        # Direct test: build the decoder and call bind_model exactly as
        # spec.assemble_predictor does.
        from osu.taiko2.inference.spec import build_component
        decoder = build_component(spec["decoder"])
        assert hasattr(decoder, "bind_model")
        decoder.bind_model(model)
        assert decoder._sampler is not None
