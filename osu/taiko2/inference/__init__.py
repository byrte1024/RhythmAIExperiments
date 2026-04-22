"""Inference package. Public entry points: `ChartPredictor` (from
`osu.taiko2.domain.inference`) and the concrete predictors here."""
from .loader import load_model_from_checkpoint

__all__ = ["load_model_from_checkpoint"]
