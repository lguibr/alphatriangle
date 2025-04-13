"""
Neural Network module for the AlphaTriangle agent.
Contains the model definition and a wrapper for inference and training interface.
"""

from .model import AlphaTriangleNet
from .network import NeuralNetwork

__all__ = [
    "AlphaTriangleNet",
    "NeuralNetwork",
]
