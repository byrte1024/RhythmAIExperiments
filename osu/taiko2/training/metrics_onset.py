"""Onset-prediction metrics — the 13 reported fractions from the
exp 45 spec, computed in one vectorized pass per batch.

Definitions (all conditional on target != STOP and prediction != STOP
unless stated otherwise):

  EXACT:  pred == target
  FHIT:   |pred - target| <= fhit_frames           (default 2)
  FGOOD:  |pred - target| <= fgood_frames          (default 7)
  FMISS:  not FGOOD

  RHIT:   |log((pred+1)/(target+1))| < log(100 / (100 - rhit_pct))
  RGOOD:  same with rgood_pct
  RMISS:  not RGOOD

  HIT:    FHIT or RHIT
  GOOD:   FGOOD or RGOOD
  BAD:    not GOOD

  I* variants — same comparisons against ANY onset in
  `target.all_future_bins` (not just the next one). Denominator is
  the number of samples that have at least one real future bin.

A prediction of STOP for a non-STOP target counts as FMISS / RMISS /
BAD / IBAD (total fail); a STOP target is excluded from all of the
above denominators.

Ratio thresholds are expressed as percent offsets (``rhit_pct=3``
means "within 3 %" of target, symmetric in log-ratio space).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..domain.metrics import Metric, MetricConfig, MetricInput


@dataclass(frozen=True, slots=True)
class OnsetMetricConfig(MetricConfig):
    b_pred: int = 500             # STOP class index
    fhit_frames: int = 2
    fgood_frames: int = 7
    rhit_pct: float = 3.0         # percent tolerance for R-HIT
    rgood_pct: float = 10.0       # percent tolerance for R-GOOD


class OnsetMetric(Metric):
    """All 13 onset metrics + a few bookkeeping counts in one pass."""

    def __init__(self, config: OnsetMetricConfig):
        self.config = config
        self._log_rhit = math.log(100.0 / (100.0 - config.rhit_pct))
        self._log_rgood = math.log(100.0 / (100.0 - config.rgood_pct))
        self.reset()

    @property
    def name(self) -> str:
        return "onset"

    # ── accumulators ──────────────────────────────────────────────────

    def reset(self) -> None:
        self._n_total = 0
        self._n_nonstop = 0
        self._n_any_future = 0
        self._exact = 0
        self._fhit = 0
        self._fgood = 0
        self._rhit = 0
        self._rgood = 0
        self._hit = 0
        self._good = 0
        self._ihit = 0
        self._igood = 0
        self._pred_stop = 0            # pred==STOP while target!=STOP (pure failure)
        self._stop_target = 0          # target==STOP (excluded from main metrics)

    # ── update ────────────────────────────────────────────────────────

    def update(self, batch: MetricInput) -> None:
        cfg = self.config
        logits = batch.output.logits               # (B, n_classes)
        target = batch.target.target_bin           # (B,)
        pred = logits.argmax(dim=-1)               # (B,)
        stop_idx = cfg.b_pred

        # Masks.
        target_stop = target == stop_idx
        pred_stop = pred == stop_idx
        nonstop_target = ~target_stop
        eval_mask = nonstop_target & (~pred_stop)  # rows we actually evaluate

        # ── frame-based ──
        diff = (pred - target).abs()
        fhit = eval_mask & (diff <= cfg.fhit_frames)
        fgood = eval_mask & (diff <= cfg.fgood_frames)
        exact = eval_mask & (pred == target)

        # ── ratio-based (avoid log(0) via +1) ──
        log_ratio = torch.log(
            (pred.float() + 1.0) / (target.float() + 1.0).clamp(min=1.0)
        ).abs()
        rhit = eval_mask & (log_ratio < self._log_rhit)
        rgood = eval_mask & (log_ratio < self._log_rgood)

        # ── composites ──
        hit = eval_mask & (fhit | rhit)
        good = eval_mask & (fgood | rgood)

        # ── any-future variants (IHIT / IGOOD) ──
        ihit = torch.zeros_like(eval_mask)
        igood = torch.zeros_like(eval_mask)
        any_future_available = torch.zeros_like(eval_mask)

        all_future = batch.target.all_future_bins
        all_future_mask = batch.target.all_future_mask
        if all_future is not None and all_future_mask is not None:
            fb = all_future.float()                                # (B, K)
            fb_valid = ~all_future_mask                            # (B, K)
            p_expanded = pred.float().unsqueeze(1)                 # (B, 1)

            fdiff = (p_expanded - fb).abs()                        # (B, K)
            f_any_hit = ((fdiff <= cfg.fhit_frames) & fb_valid).any(dim=1)
            f_any_good = ((fdiff <= cfg.fgood_frames) & fb_valid).any(dim=1)

            log_r = torch.log(
                (p_expanded + 1.0) / (fb + 1.0).clamp(min=1.0)
            ).abs()
            r_any_hit = ((log_r < self._log_rhit) & fb_valid).any(dim=1)
            r_any_good = ((log_r < self._log_rgood) & fb_valid).any(dim=1)

            any_future_available = nonstop_target & fb_valid.any(dim=1)
            ihit = any_future_available & (f_any_hit | r_any_hit)
            igood = any_future_available & (f_any_good | r_any_good)

        # ── accumulate ──
        self._n_total += int(target.size(0))
        self._n_nonstop += int(nonstop_target.sum().item())
        self._stop_target += int(target_stop.sum().item())
        self._pred_stop += int((nonstop_target & pred_stop).sum().item())
        self._exact += int(exact.sum().item())
        self._fhit += int(fhit.sum().item())
        self._fgood += int(fgood.sum().item())
        self._rhit += int(rhit.sum().item())
        self._rgood += int(rgood.sum().item())
        self._hit += int(hit.sum().item())
        self._good += int(good.sum().item())
        self._n_any_future += int(any_future_available.sum().item())
        self._ihit += int(ihit.sum().item())
        self._igood += int(igood.sum().item())

    # ── compute ───────────────────────────────────────────────────────

    def compute(self) -> dict[str, float]:
        # Denominator for F/R/composite metrics: samples with non-STOP
        # target. STOP-target samples are excluded from the main
        # success rates (they'd skew the definitions). Pred-STOP-while-
        # target-non-STOP samples stay in the denominator — they're
        # pure misses.
        n = max(self._n_nonstop, 1)
        n_any = max(self._n_any_future, 1)

        out: dict[str, float] = {
            "onset/exact":          self._exact / n,
            "onset/fhit":           self._fhit / n,
            "onset/fgood":          self._fgood / n,
            "onset/fmiss":          1.0 - (self._fgood / n),
            "onset/rhit":           self._rhit / n,
            "onset/rgood":          self._rgood / n,
            "onset/rmiss":          1.0 - (self._rgood / n),
            "onset/hit":            self._hit / n,
            "onset/good":           self._good / n,
            "onset/bad":            1.0 - (self._good / n),
            "onset/pred_stop_rate": self._pred_stop / n,
            "onset/n_total":        float(self._n_total),
            "onset/n_nonstop":      float(self._n_nonstop),
            "onset/n_stop_target":  float(self._stop_target),
        }
        if self._n_any_future > 0:
            out["onset/ihit"] = self._ihit / n_any
            out["onset/igood"] = self._igood / n_any
            out["onset/ibad"] = 1.0 - (self._igood / n_any)
            out["onset/n_any_future"] = float(self._n_any_future)
        return out
