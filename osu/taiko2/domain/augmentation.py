"""Augmentation primitives: abstract pre- and post-sample hooks plus a
composable pipeline.

Two phases exist because some perturbations have to happen *before* data
is collected (shifting the sample cursor, changing which event indices
get pulled) while others happen *after* (masking an already-extracted
sample, adding mel noise, dropping event tokens).

Both phase ABCs are generic. Concrete samplers pick the context / sample
types they hand to augmentations:

    TaikoPreAug  = PreSampleAugmentation[TaikoDetectionPreContext]
    TaikoPostAug = PostSampleAugmentation[TaikoDetectionSample]
    TaikoPipe    = AugmentationPipeline[TaikoDetectionPreContext,
                                        TaikoDetectionSample]

Order of application is the tuple order in the pipeline — callers fully
control the sequence by arranging the tuple.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

Ctx = TypeVar("Ctx")
Smp = TypeVar("Smp")


class PreSampleAugmentation(ABC, Generic[Ctx]):
    """Runs before sample extraction. Given a context describing what
    would be sampled (cursor, event indices, chart arrays), returns a
    possibly-modified context.

    Subclasses may freely return a new context object (preferred for
    frozen/immutable context types) or mutate + return the same one.
    """

    @abstractmethod
    def apply(self, context: Ctx) -> Ctx:
        ...


class PostSampleAugmentation(ABC, Generic[Smp]):
    """Runs after sample extraction. Given a fully-built sample, returns
    a possibly-modified sample.

    Subclasses may return a new sample (preferred for frozen dataclasses
    with `dataclasses.replace`) or mutate arrays in place.
    """

    @abstractmethod
    def apply(self, sample: Smp) -> Smp:
        ...


@dataclass(frozen=True, slots=True)
class AugmentationPipeline(Generic[Ctx, Smp]):
    """Ordered composition of pre- and post-sample augmentations.

    `pre` and `post` are separate because their inputs differ. Within
    each tuple, augmentations run in order (index 0 first). An empty
    pipeline is a no-op; it's the default for training loops that don't
    want augmentation yet.
    """
    pre: tuple[PreSampleAugmentation[Ctx], ...] = field(default_factory=tuple)
    post: tuple[PostSampleAugmentation[Smp], ...] = field(default_factory=tuple)

    def apply_pre(self, context: Ctx) -> Ctx:
        for aug in self.pre:
            context = aug.apply(context)
        return context

    def apply_post(self, sample: Smp) -> Smp:
        for aug in self.post:
            sample = aug.apply(sample)
        return sample

    @property
    def is_empty(self) -> bool:
        return not self.pre and not self.post
