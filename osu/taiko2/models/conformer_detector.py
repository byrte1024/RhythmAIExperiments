"""EventEmbeddingDetector with the transformer trunk swapped for
Conformer blocks.

Architecturally identical to ``EventEmbeddingDetector`` everywhere
except the trunk:

  - Conv stem: same (Conv1d 80→192→384, k=7, s=2 twice).
  - Audio + event token mixer: same (event embeddings scatter-added
    into audio tokens).
  - FiLM conditioning: applied per block, after the block's final
    LayerNorm — same placement as the parent's per-layer FiLM.
  - Output head: same (cursor token → LN → Linear → Conv1d smooth).

Only ``self.layers`` is replaced: ``nn.TransformerEncoderLayer`` × N
becomes ``ConformerBlock`` × N. Because ``ConformerBlock.forward(x)``
matches the call shape of ``nn.TransformerEncoderLayer.forward(x)``,
the parent's ``get_cursor_token`` loop runs unchanged.

Hyperparameters added by ``ConformerDetectorConfig``:
- ``ffn_dim``: macaron FFN hidden dim. Canonical = 4 × d_model.
- ``depthwise_conv_kernel_size``: depthwise conv kernel along time.
  Canonical = 31 (odd).
- ``use_group_norm``: GroupNorm vs BatchNorm in the conv module.
  Default True (BeatThis convention).

Inherits ``EventEmbeddingConfig`` for backbone geometry; adds the
three Conformer-specific fields above.

This file exists for two reasons:

1. **Explicit semantics in checkpoint metadata.** A saved
   ``EventEmbeddingConfig`` doesn't carry the Conformer hyperparams.
   ``ConformerDetectorConfig`` does, so a future loader can
   reconstruct the trunk exactly.

2. **Future architectural divergence.** If we want to ablate
   Conformer-specific knobs (kernel size, FFN expansion factor,
   conv-first vs attention-first ordering), this file is the home
   for those experiments.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .conformer_block import ConformerBlock
from .event_embedding import EventEmbeddingConfig, EventEmbeddingDetector


@dataclass(frozen=True, slots=True)
class ConformerDetectorConfig(EventEmbeddingConfig):
    """``EventEmbeddingConfig`` with Conformer-specific fields.

    The parent's ``n_layers`` controls the Conformer block count;
    ``d_model``, ``n_heads``, ``dropout`` carry over unchanged.
    """
    ffn_dim: int = 1536                                     # = 4 * d_model at d=384
    depthwise_conv_kernel_size: int = 31
    use_group_norm: bool = True

    def __post_init__(self):
        # slots=True + frozen dataclass inheritance breaks super() in
        # __post_init__ on some CPython versions (see OnsetAugmentedConfig
        # for the same workaround).
        EventEmbeddingConfig.__post_init__(self)
        if self.depthwise_conv_kernel_size <= 0:
            raise ValueError(
                f"depthwise_conv_kernel_size must be > 0 "
                f"(got {self.depthwise_conv_kernel_size})"
            )
        if self.depthwise_conv_kernel_size % 2 == 0:
            raise ValueError(
                f"depthwise_conv_kernel_size must be odd for centered "
                f"conv (got {self.depthwise_conv_kernel_size})"
            )
        if self.ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be > 0 (got {self.ffn_dim})")


class ConformerDetector(EventEmbeddingDetector):
    """``EventEmbeddingDetector`` with Conformer trunk.

    Constructs the parent fully, then replaces ``self.layers`` (a
    ``ModuleList`` of ``nn.TransformerEncoderLayer``) with a
    ``ModuleList`` of ``ConformerBlock``. ``self.film_layers`` is
    inherited unchanged — FiLM is applied per block exactly as the
    parent applies it per transformer layer.
    """

    config: ConformerDetectorConfig

    def __init__(self, config: ConformerDetectorConfig):
        super().__init__(config)
        c = config
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=c.d_model,
                n_heads=c.n_heads,
                ffn_dim=c.ffn_dim,
                depthwise_kernel_size=c.depthwise_conv_kernel_size,
                dropout=c.dropout,
                use_group_norm=c.use_group_norm,
            )
            for _ in range(c.n_layers)
        ])
