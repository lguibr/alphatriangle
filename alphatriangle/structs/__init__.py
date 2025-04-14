"""
Module for core data structures used across different parts of the application,
like environment, visualization, and features. Helps avoid circular dependencies.
"""

from .constants import SHAPE_COLORS
from .shape import Shape
from .triangle import Triangle

__all__ = [
    "Triangle",
    "Shape",
    "SHAPE_COLORS",
]
