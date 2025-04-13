# File: src/environment/__init__.py
"""
Environment module defining the game rules, state, actions, and logic.
This module is now independent of feature extraction for the NN.
"""
# Core components
from .core.game_state import GameState
from .core.action_codec import encode_action, decode_action

# Grid related components
from .grid.grid_data import GridData

# Import Triangle and Shape from the new structs module
# These are implicitly used via GridData and GameState, no need to export directly
# from src.structs import Triangle, Shape

from .grid import logic as GridLogic  # Expose grid logic functions via a namespace

# Shape related components
from .shapes import logic as ShapeLogic  # Expose shape logic functions via a namespace

# Game Logic components (Actions, Step)
from .logic.actions import get_valid_actions
from .logic.step import execute_placement, calculate_reward

# Configuration (often needed alongside environment components)
from src.config import EnvConfig


__all__ = [
    # Core
    "GameState",
    "encode_action",
    "decode_action",
    # Grid
    "GridData",
    "GridLogic",
    # Shapes
    "ShapeLogic",
    # Logic
    "get_valid_actions",
    "execute_placement",
    "calculate_reward",
    # Config
    "EnvConfig",
]
