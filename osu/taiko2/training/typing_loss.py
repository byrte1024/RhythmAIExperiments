"""Loss for the typing model: BCE on type (D/K) + weighted BCE on strength."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..domain.loss import Loss, LossConfig, LossResult
from ..domain.typing import TypingOutput, TypingTarget


@dataclass(frozen=True, slots=True)
class TypingLossConfig(LossConfig):
    strength_pos_weight: float = 17.0


class TypingLoss(Loss[TypingLossConfig, TypingOutput, TypingTarget]):

    def __init__(self, config: TypingLossConfig):
        super().__init__(config)
        self.register_buffer(
            "_strength_pw",
            torch.tensor([config.strength_pos_weight]),
        )

    def forward(
        self, output: TypingOutput, target: TypingTarget,
    ) -> LossResult:
        type_loss = F.binary_cross_entropy_with_logits(
            output.type_logit, target.type_target,
        )
        strength_loss = F.binary_cross_entropy_with_logits(
            output.strength_logit, target.strength_target,
            pos_weight=self._strength_pw,
        )
        total = type_loss + strength_loss

        # Per-batch metrics (detached, for train/batch/* logging)
        with torch.no_grad():
            type_prob = torch.sigmoid(output.type_logit)
            type_pred = (type_prob > 0.5).float()
            type_acc = float((type_pred == target.type_target).float().mean())

            str_prob = torch.sigmoid(output.strength_logit)
            str_pred = (str_prob > 0.5).float()
            str_acc = float((str_pred == target.strength_target).float().mean())

            # Combined 4-class
            pred_kind = type_pred.long()           # 0=D, 1=K
            pred_big = str_pred.long()             # 0=normal, 1=big
            pred_4 = pred_kind + pred_big * 2      # 0=D, 1=K, 2=BD, 3=BK
            gt_kind = target.type_target.long()
            gt_big = target.strength_target.long()
            gt_4 = gt_kind + gt_big * 2
            combined_acc = float((pred_4 == gt_4).float().mean())

            # Entropy
            eps = 1e-7
            type_ent = -(
                type_prob * torch.log(type_prob + eps)
                + (1 - type_prob) * torch.log(1 - type_prob + eps)
            )
            str_ent = -(
                str_prob * torch.log(str_prob + eps)
                + (1 - str_prob) * torch.log(1 - str_prob + eps)
            )

        return LossResult(
            loss=total,
            metrics={
                "loss": float(total.detach()),
                "type_loss": float(type_loss.detach()),
                "strength_loss": float(strength_loss.detach()),
                "type_acc": type_acc,
                "strength_acc": str_acc,
                "combined_acc": combined_acc,
                "type_entropy_mean": float(type_ent.mean()),
                "strength_entropy_mean": float(str_ent.mean()),
            },
        )
