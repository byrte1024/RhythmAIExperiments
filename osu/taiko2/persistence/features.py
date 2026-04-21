"""Save/load (F, T) audio feature arrays as .npy.

float16 on disk (matches old `create_dataset.py`) to halve storage; caller
gets float32 back for numerical work.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_features(features: np.ndarray, path: Path) -> None:
    """Write a (F, T) feature array to `path` as float16 .npy."""
    if features.ndim != 2:
        raise ValueError(f"features must be 2D (F, T), got shape {features.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features.astype(np.float16))


def load_features(path: Path, mmap: bool = False) -> np.ndarray:
    """Read a features .npy. `mmap=True` for lazy, OS-page-cached reads."""
    arr = np.load(Path(path), mmap_mode="r" if mmap else None)
    return arr
