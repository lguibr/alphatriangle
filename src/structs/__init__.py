# File: src/structs/__init__.py
"""
Module for core data structures used across different parts of the application,
like environment, visualization, and features. Helps avoid circular dependencies.
"""
from .triangle import Triangle
from .shape import Shape
from .constants import SHAPE_COLORS

__all__ = [
    "Triangle",
    "Shape",
    "SHAPE_COLORS",
]
