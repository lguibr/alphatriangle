"""
Environment module defining the game rules, state, actions, and logic.
This module is now independent of feature extraction for the NN.
"""

from alphatriangle.config import EnvConfig

from .core.action_codec import decode_action, encode_action
from .core.game_state import GameState
from .grid import logic as GridLogic
from .grid.grid_data import GridData
from .logic.actions import get_valid_actions
from .logic.step import calculate_reward, execute_placement
from .shapes import logic as ShapeLogic

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
