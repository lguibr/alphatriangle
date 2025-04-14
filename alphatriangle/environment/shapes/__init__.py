"""
Shapes submodule handling shape generation and management.
"""

from .logic import (
    generate_random_shape,
    get_neighbors,
    is_shape_connected,
    refill_shape_slots,
)
from .templates import PREDEFINED_SHAPE_TEMPLATES

__all__ = [
    "generate_random_shape",
    "refill_shape_slots",
    "is_shape_connected",
    "get_neighbors",
    "PREDEFINED_SHAPE_TEMPLATES",
]
