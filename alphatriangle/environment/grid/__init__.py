# File: alphatriangle/environment/grid/__init__.py
"""
Grid submodule handling the triangular grid structure, data, and logic.
"""

# Removed: from .triangle import Triangle
from . import logic
from .grid_data import GridData

# DO NOT import grid_features here. It has been moved up one level
# to alphatriangle/environment/grid_features.py to break circular dependencies.

__all__ = [
    "GridData",
    "logic",
]
