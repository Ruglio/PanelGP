"""The PanelGP model, its training loop and the simulation DGP.

This subpackage does not depend on ``panelgp.experiments``, which applies it
to the paper's S&P 500 panel.
"""

from .data import PanelSchema, PanelTensors, build_time_grid, to_tensors
from .panel_gp import (
    CrossSectionMoments,
    ModelSettings,
    PanelGP,
    PredictiveCumulants,
    PredictiveMoments,
    RowMoments,
)
from .training import TRAINING_DEFAULTS, FitResult, fit_model, predict_rows

__all__ = [
    "TRAINING_DEFAULTS",
    "CrossSectionMoments",
    "FitResult",
    "ModelSettings",
    "PanelGP",
    "PanelSchema",
    "PanelTensors",
    "PredictiveCumulants",
    "PredictiveMoments",
    "RowMoments",
    "build_time_grid",
    "fit_model",
    "predict_rows",
    "to_tensors",
]
