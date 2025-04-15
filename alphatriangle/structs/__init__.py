# File: alphatriangle/structs/__init__.py
"""
Module for core data structures used across different parts of the application,
like environment, visualization, and features. Helps avoid circular dependencies.
"""

# Correctly export constants from the constants submodule
from .constants import (
    COLOR_ID_MAP,
    COLOR_TO_ID_MAP,
    DEBUG_COLOR_ID,
    NO_COLOR_ID,
    SHAPE_COLORS,
)
from .shape import Shape
from .triangle import Triangle

__all__ = [
    "Triangle",
    "Shape",
    # Exported Constants
    "SHAPE_COLORS",
    "NO_COLOR_ID",
    "DEBUG_COLOR_ID",
    "COLOR_ID_MAP",
    "COLOR_TO_ID_MAP",
]
