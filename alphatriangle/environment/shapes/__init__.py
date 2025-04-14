# File: src/environment/shapes/__init__.py
# File: src/environment/shapes/__init__.py
"""
Shapes submodule handling shape generation and management.
"""

# Import logic functions directly into the package namespace
from .logic import (
    generate_random_shape,
    get_neighbors,
    is_shape_connected,
    refill_shape_slots,
)

# Import the constant from the templates file
from .templates import PREDEFINED_SHAPE_TEMPLATES

__all__ = [
    "generate_random_shape",
    "refill_shape_slots",  # Signature changed internally, but name is the same
    "is_shape_connected",
    "get_neighbors",
    "PREDEFINED_SHAPE_TEMPLATES",  # Export the constant
]
