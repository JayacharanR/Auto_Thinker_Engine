"""
JEPA (Joint Embedding Predictive Architecture) components.

Includes context encoder, EMA target encoder, predictor,
masking strategies, and loss functions for self-supervised
video pretraining on driving data.
"""

from src.jepa.encoder import ViTEncoder
from src.jepa.target_encoder import EMATargetEncoder
from src.jepa.predictor import JEPAPredictor
from src.jepa.masking import MaskGenerator, verify_no_leak
from src.jepa.losses import JEPALoss, CollapseMonitor

__all__ = [
    "ViTEncoder",
    "EMATargetEncoder",
    "JEPAPredictor",
    "MaskGenerator",
    "verify_no_leak",
    "JEPALoss",
    "CollapseMonitor",
]
