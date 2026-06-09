"""Second-pass typing: run the typing model over a chart's onsets to
assign D/K and normal/big labels.

Works on any Chart that already has onset positions placed (e.g. from
the onset detector AR loop). Replaces the default `kind=DON` with
proper types predicted autoregressively.

Spec JSON shape (mirrors the onset predictor spec pattern)::

    {
      "checkpoint": "runs/.../checkpoints/best.pt",
      "config": {
        "__class__": "osu.taiko2.inference.typing_pass:TypingInferConfig",
        "strength_threshold": 0.8,
        "bin_ms": 5.0
      }
    }

Entry points:
  - ``load_typing_spec``: load spec JSON -> (model, config).
  - ``type_chart``: takes a live model + features, returns a new Chart.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import json
import numpy as np
import torch

from ..domain.beatmap import OnsetBinned, OnsetKind
from ..domain.chart import Chart
from ..domain.typing import (
    TYPING_CONTEXT,
    TYPING_MEL_PATCH,
    TYPING_WINDOW,
    TypingInput,
    TypingModelConfig,
)
from ..models.typing_model import TypingTransformer
from ..inference.loader import load_model_from_checkpoint


_HALF_MEL = TYPING_MEL_PATCH // 2

_DK_BIG_TO_KIND = {
    (0, 0): OnsetKind.DON,
    (1, 0): OnsetKind.KA,
    (0, 1): OnsetKind.BIG_DON,
    (1, 1): OnsetKind.BIG_KA,
}


# ─────────────────────────── config ──────────────────────────────────

@dataclass(frozen=True, slots=True)
class TypingInferConfig:
    strength_threshold: float = 0.8
    bin_ms: float = 5.0


# ─────────────────────────── spec loading ────────────────────────────

def load_typing_spec(
    spec_path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[TypingTransformer, TypingInferConfig]:
    """Load a typing spec JSON -> (model on device, config)."""
    with open(Path(spec_path), "r", encoding="utf-8") as f:
        spec = json.load(f)
    return assemble_typing_from_spec(spec, device=device)


def assemble_typing_from_spec(
    spec: dict[str, Any],
    device: torch.device | str = "cpu",
) -> tuple[TypingTransformer, TypingInferConfig]:
    """Build typing model + config from a parsed spec dict."""
    ckpt_path = Path(spec["checkpoint"])
    model, _loss, _meta = load_model_from_checkpoint(ckpt_path, device=device)
    model.eval()

    cfg_node = spec.get("config", {})
    cfg = TypingInferConfig(
        strength_threshold=cfg_node.get("strength_threshold", 0.8),
        bin_ms=cfg_node.get("bin_ms", 5.0),
    )
    return model, cfg


def assemble_typing_with_model(
    spec: dict[str, Any],
    model: TypingTransformer,
) -> TypingInferConfig:
    """Build config from spec, using an already-live model (for hooks)."""
    cfg_node = spec.get("config", {})
    return TypingInferConfig(
        strength_threshold=cfg_node.get("strength_threshold", 0.8),
        bin_ms=cfg_node.get("bin_ms", 5.0),
    )


def _extract_mel_patch(features: np.ndarray, b: int) -> np.ndarray:
    n_mels, n_frames = features.shape
    lo = b - _HALF_MEL
    hi = b + _HALF_MEL + 1
    if lo < 0 or hi > n_frames:
        patch = np.zeros((n_mels, TYPING_MEL_PATCH), dtype=np.float32)
        src_lo = max(0, lo)
        src_hi = min(n_frames, hi)
        dst_lo = src_lo - lo
        dst_hi = dst_lo + (src_hi - src_lo)
        if src_hi > src_lo:
            patch[:, dst_lo:dst_hi] = features[:, src_lo:src_hi]
        return patch
    return np.asarray(features[:, lo:hi], dtype=np.float32)


def _compute_iois(bins: np.ndarray, idx: int) -> np.ndarray:
    n = len(bins)
    ioi_before = float(np.log1p(abs(bins[idx] - bins[idx - 1]))) if idx > 0 else 0.0
    ioi_after = float(np.log1p(abs(bins[idx + 1] - bins[idx]))) if idx < n - 1 else 0.0
    ratio = (ioi_before - ioi_after) if (ioi_before > 0 and ioi_after > 0) else 0.0
    return np.array([ioi_before, ioi_after, ratio], dtype=np.float32)


@torch.no_grad()
def type_chart(
    model: TypingTransformer,
    chart: Chart,
    features: np.ndarray,
    *,
    device: torch.device | str = "cpu",
    config: TypingInferConfig | None = None,
    strength_threshold: float | None = None,
    bin_ms: float | None = None,
) -> Chart:
    """Run AR typing over a chart's onsets and return a new Chart with
    corrected kinds.

    Pass ``config`` (from a spec JSON) or individual overrides.
    ``features`` is the (n_mels, T) mel spectrogram.
    """
    if config is None:
        config = TypingInferConfig()
    _str_thr = strength_threshold if strength_threshold is not None else config.strength_threshold
    _bin_ms = bin_ms if bin_ms is not None else config.bin_ms
    device = torch.device(device) if isinstance(device, str) else device
    onsets = chart.track.onsets
    if not onsets:
        return chart

    # Extract bins from onsets
    bins = np.array([o.bin if hasattr(o, "bin") else round(o.time_ms / _bin_ms)
                     for o in onsets], dtype=np.int64)

    ctx = TYPING_CONTEXT
    W = TYPING_WINDOW
    n_hits = len(bins)
    n_mels = features.shape[0]
    mel_dim = n_mels * TYPING_MEL_PATCH

    pred_dk = np.zeros(n_hits, dtype=np.int64)
    pred_big = np.zeros(n_hits, dtype=np.int64)

    for i in range(n_hits):
        mel_all = np.zeros((1, W, mel_dim), dtype=np.float32)
        ioi_all = np.zeros((1, W, 3), dtype=np.float32)
        kind_all = np.full((1, W), 2, dtype=np.int64)
        big_all = np.full((1, W), 2, dtype=np.int64)
        pos_all = np.arange(W, dtype=np.int64).reshape(1, W)
        mask_all = np.ones((1, W), dtype=bool)

        # Past (own predictions)
        past_start = max(0, i - ctx)
        past_count = i - past_start
        for j in range(past_count):
            src = past_start + j
            dst = ctx - past_count + j
            mel_all[0, dst] = _extract_mel_patch(features, int(bins[src])).ravel()
            ioi_all[0, dst] = _compute_iois(bins, src)
            kind_all[0, dst] = int(pred_dk[src])
            big_all[0, dst] = int(pred_big[src])
            mask_all[0, dst] = False

        # Target
        mel_all[0, ctx] = _extract_mel_patch(features, int(bins[i])).ravel()
        ioi_all[0, ctx] = _compute_iois(bins, i)
        mask_all[0, ctx] = False

        # Future
        future_end = min(n_hits, i + 1 + ctx)
        for j in range(future_end - (i + 1)):
            fi = ctx + 1 + j
            src = i + 1 + j
            mel_all[0, fi] = _extract_mel_patch(features, int(bins[src])).ravel()
            ioi_all[0, fi] = _compute_iois(bins, src)
            mask_all[0, fi] = False

        inp = TypingInput(
            mel_patches=torch.from_numpy(mel_all).to(device),
            ioi_features=torch.from_numpy(ioi_all).to(device),
            kind_labels=torch.from_numpy(kind_all).to(device),
            big_labels=torch.from_numpy(big_all).to(device),
            positions=torch.from_numpy(pos_all).to(device),
            mask=torch.from_numpy(mask_all).to(device),
        )
        out = model.predict(inp)
        type_prob = torch.sigmoid(out.type_logit).item()
        str_prob = torch.sigmoid(out.strength_logit).item()

        pred_dk[i] = 1 if type_prob > 0.5 else 0
        pred_big[i] = 1 if str_prob > _str_thr else 0

    # Build new onsets with predicted kinds
    new_onsets: list[OnsetBinned] = []
    for idx, onset in enumerate(onsets):
        dk = int(pred_dk[idx])
        big = int(pred_big[idx])
        kind = _DK_BIG_TO_KIND[(dk, big)]
        if isinstance(onset, OnsetBinned):
            new_onsets.append(OnsetBinned(
                time_ms=onset.time_ms, kind=kind, bin=onset.bin,
            ))
        else:
            new_onsets.append(OnsetBinned(
                time_ms=onset.time_ms, kind=kind,
                bin=round(onset.time_ms / bin_ms),
            ))

    from ..parsing.osu import compute_density
    new_track = replace(chart.track, onsets=tuple(new_onsets),
                        density=compute_density(tuple(new_onsets)))
    return Chart(track=new_track, audio=chart.audio)
