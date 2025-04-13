"""
Feature extraction module.
Converts raw GameState objects into numerical representations suitable for NN input.
"""

from .extractor import extract_state_features, GameStateFeatures
from . import grid_features

__all__ = [
    "extract_state_features",
    "GameStateFeatures",
    "grid_features",
]
