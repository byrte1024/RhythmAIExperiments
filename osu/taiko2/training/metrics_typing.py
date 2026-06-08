"""Typing model metric: per-class P/R/F1, confidence, entropy, threshold sweep."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.metrics import Metric, MetricConfig, MetricInput
from ..domain.typing import TypingOutput, TypingTarget


TYPE_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
STRENGTH_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)


def _binary_prf(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall, F1 for positive class (1)."""
    tp = float(np.sum((pred == 1) & (gt == 1)))
    fp = float(np.sum((pred == 1) & (gt == 0)))
    fn = float(np.sum((pred == 0) & (gt == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _binary_entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@dataclass(frozen=True, slots=True)
class TypingMetricConfig(MetricConfig):
    pass


class TypingMetric(Metric):
    """Accumulates typing predictions across batches for full eval stats."""

    def __init__(self, prefix: str = "typing"):
        self._prefix = prefix
        self._type_probs: list[np.ndarray] = []
        self._type_targets: list[np.ndarray] = []
        self._str_probs: list[np.ndarray] = []
        self._str_targets: list[np.ndarray] = []

    @property
    def name(self) -> str:
        return self._prefix

    def reset(self) -> None:
        self._type_probs.clear()
        self._type_targets.clear()
        self._str_probs.clear()
        self._str_targets.clear()

    def update(self, batch: MetricInput) -> None:
        out: TypingOutput = batch.output  # type: ignore[assignment]
        tgt: TypingTarget = batch.target  # type: ignore[assignment]
        self._type_probs.append(out.type_logit.detach().sigmoid().cpu().numpy())
        self._type_targets.append(tgt.type_target.detach().cpu().numpy())
        self._str_probs.append(out.strength_logit.detach().sigmoid().cpu().numpy())
        self._str_targets.append(tgt.strength_target.detach().cpu().numpy())

    def compute(self) -> dict[str, float]:
        if not self._type_probs:
            return {}

        tp = np.concatenate(self._type_probs)
        tt = np.concatenate(self._type_targets)
        sp = np.concatenate(self._str_probs)
        st = np.concatenate(self._str_targets)

        out: dict[str, float] = {}
        pfx = self._prefix

        # ── Type head (D/K) ──
        out.update(self._type_metrics(tp, tt, pfx))

        # ── Strength head (normal/big) ──
        out.update(self._strength_metrics(sp, st, pfx))

        # ── Combined 4-class ──
        type_pred = (tp > 0.5).astype(np.int64)
        str_pred = (sp > 0.5).astype(np.int64)
        type_gt = tt.astype(np.int64)
        str_gt = st.astype(np.int64)
        pred_4 = type_pred + str_pred * 2
        gt_4 = type_gt + str_gt * 2
        out[f"{pfx}/combined/accuracy"] = float(np.mean(pred_4 == gt_4))

        labels_4 = ["DON", "KA", "BDON", "BKA"]
        for cls_idx, cls_name in enumerate(labels_4):
            p_cls = (pred_4 == cls_idx).astype(np.int64)
            g_cls = (gt_4 == cls_idx).astype(np.int64)
            prec, rec, f1 = _binary_prf(p_cls, g_cls)
            out[f"{pfx}/combined/precision_{cls_name}"] = prec
            out[f"{pfx}/combined/recall_{cls_name}"] = rec
            out[f"{pfx}/combined/f1_{cls_name}"] = f1

        return out

    def _type_metrics(
        self, probs: np.ndarray, targets: np.ndarray, pfx: str,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        pred_05 = (probs > 0.5).astype(np.int64)
        gt = targets.astype(np.int64)

        out[f"{pfx}/type/accuracy"] = float(np.mean(pred_05 == gt))

        # D = 1 (positive), K = 0
        prec_d, rec_d, f1_d = _binary_prf(pred_05, gt)
        out[f"{pfx}/type/precision_D"] = prec_d
        out[f"{pfx}/type/recall_D"] = rec_d
        out[f"{pfx}/type/f1_D"] = f1_d
        # K = flip
        prec_k, rec_k, f1_k = _binary_prf(1 - pred_05, 1 - gt)
        out[f"{pfx}/type/precision_K"] = prec_k
        out[f"{pfx}/type/recall_K"] = rec_k
        out[f"{pfx}/type/f1_K"] = f1_k

        # Confidence
        correct = (pred_05 == gt)
        out[f"{pfx}/type/conf_correct"] = float(np.mean(
            np.where(correct, np.maximum(probs, 1 - probs), 0.0)
        )) / max(float(np.mean(correct)), 1e-8) if np.any(correct) else 0.0
        out[f"{pfx}/type/conf_wrong"] = float(np.mean(
            np.where(~correct, np.maximum(probs, 1 - probs), 0.0)
        )) / max(float(np.mean(~correct)), 1e-8) if np.any(~correct) else 0.0
        out[f"{pfx}/type/conf_mean"] = float(np.mean(np.maximum(probs, 1 - probs)))
        out[f"{pfx}/type/conf_std"] = float(np.std(np.maximum(probs, 1 - probs)))

        # Entropy
        ent = _binary_entropy(probs)
        out[f"{pfx}/type/entropy_mean"] = float(np.mean(ent))
        out[f"{pfx}/type/entropy_std"] = float(np.std(ent))

        # Decisive/conflicted mass
        max_conf = np.maximum(probs, 1 - probs)
        out[f"{pfx}/type/mass_decisive"] = float(np.mean(max_conf > 0.9))
        out[f"{pfx}/type/mass_conflicted"] = float(np.mean(
            (max_conf >= 0.4) & (max_conf <= 0.6)
        ))

        # Threshold sweep
        best_f1 = 0.0
        best_thr = 0.5
        for thr in TYPE_THRESHOLDS:
            pred_t = (probs > thr).astype(np.int64)
            acc = float(np.mean(pred_t == gt))
            _, _, f1_d_t = _binary_prf(pred_t, gt)
            _, _, f1_k_t = _binary_prf(1 - pred_t, 1 - gt)
            macro_f1 = (f1_d_t + f1_k_t) / 2
            out[f"{pfx}/type/sweep/acc_at_{thr:.2f}"] = acc
            out[f"{pfx}/type/sweep/f1_D_at_{thr:.2f}"] = f1_d_t
            out[f"{pfx}/type/sweep/f1_K_at_{thr:.2f}"] = f1_k_t
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_thr = thr
        out[f"{pfx}/type/best_threshold"] = best_thr
        out[f"{pfx}/type/best_f1"] = best_f1

        return out

    def _strength_metrics(
        self, probs: np.ndarray, targets: np.ndarray, pfx: str,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        pred_05 = (probs > 0.5).astype(np.int64)
        gt = targets.astype(np.int64)

        out[f"{pfx}/strength/accuracy"] = float(np.mean(pred_05 == gt))

        # BIG = 1 (positive), NORMAL = 0
        prec_b, rec_b, f1_b = _binary_prf(pred_05, gt)
        out[f"{pfx}/strength/precision_BIG"] = prec_b
        out[f"{pfx}/strength/recall_BIG"] = rec_b
        out[f"{pfx}/strength/f1_BIG"] = f1_b
        prec_n, rec_n, f1_n = _binary_prf(1 - pred_05, 1 - gt)
        out[f"{pfx}/strength/precision_NORMAL"] = prec_n
        out[f"{pfx}/strength/recall_NORMAL"] = rec_n
        out[f"{pfx}/strength/f1_NORMAL"] = f1_n

        # Confidence
        correct = (pred_05 == gt)
        max_conf = np.maximum(probs, 1 - probs)
        out[f"{pfx}/strength/conf_correct"] = (
            float(np.mean(max_conf[correct])) if np.any(correct) else 0.0
        )
        out[f"{pfx}/strength/conf_wrong"] = (
            float(np.mean(max_conf[~correct])) if np.any(~correct) else 0.0
        )
        out[f"{pfx}/strength/conf_mean"] = float(np.mean(max_conf))
        out[f"{pfx}/strength/conf_std"] = float(np.std(max_conf))

        # Entropy
        ent = _binary_entropy(probs)
        out[f"{pfx}/strength/entropy_mean"] = float(np.mean(ent))
        out[f"{pfx}/strength/entropy_std"] = float(np.std(ent))

        # Decisive/conflicted
        out[f"{pfx}/strength/mass_decisive"] = float(np.mean(max_conf > 0.9))
        out[f"{pfx}/strength/mass_conflicted"] = float(np.mean(
            (max_conf >= 0.4) & (max_conf <= 0.6)
        ))

        # Threshold sweep
        best_f1 = 0.0
        best_thr = 0.5
        for thr in STRENGTH_THRESHOLDS:
            pred_t = (probs > thr).astype(np.int64)
            acc = float(np.mean(pred_t == gt))
            prec_b_t, rec_b_t, f1_b_t = _binary_prf(pred_t, gt)
            out[f"{pfx}/strength/sweep/acc_at_{thr:.2f}"] = acc
            out[f"{pfx}/strength/sweep/precision_BIG_at_{thr:.2f}"] = prec_b_t
            out[f"{pfx}/strength/sweep/recall_BIG_at_{thr:.2f}"] = rec_b_t
            out[f"{pfx}/strength/sweep/f1_BIG_at_{thr:.2f}"] = f1_b_t
            if f1_b_t > best_f1:
                best_f1 = f1_b_t
                best_thr = thr
        out[f"{pfx}/strength/best_threshold"] = best_thr
        out[f"{pfx}/strength/best_f1_BIG"] = best_f1

        return out
