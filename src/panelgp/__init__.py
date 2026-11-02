"""PanelGP: a probabilistic factor model for asset-return panels.

``panelgp.model`` contains the model and its estimation (``fit_model``,
``predict_rows``); ``panelgp.experiments`` reproduces the paper's empirical
application.
"""

from .model import (
    CrossSectionMoments,
    FitResult,
    ModelSettings,
    PanelGP,
    PanelSchema,
    PredictiveCumulants,
    PredictiveMoments,
    fit_model,
    predict_rows,
)

__all__ = [
    "CrossSectionMoments",
    "FitResult",
    "ModelSettings",
    "PanelGP",
    "PanelSchema",
    "PredictiveCumulants",
    "PredictiveMoments",
    "fit_model",
    "predict_rows",
]
__version__ = "1.0.0"
