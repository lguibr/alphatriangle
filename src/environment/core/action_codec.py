# File: src/environment/core/action_codec.py

from src.config import EnvConfig
from src.utils.types import ActionType


def encode_action(shape_idx: int, r: int, c: int, config: EnvConfig) -> ActionType:
    """Encodes a (shape_idx, r, c) action into a single integer."""
    if not (0 <= shape_idx < config.NUM_SHAPE_SLOTS):
        raise ValueError(
            f"Invalid shape index: {shape_idx}, must be < {config.NUM_SHAPE_SLOTS}"
        )
    if not (0 <= r < config.ROWS):
        raise ValueError(f"Invalid row index: {r}, must be < {config.ROWS}")
    if not (0 <= c < config.COLS):
        raise ValueError(f"Invalid column index: {c}, must be < {config.COLS}")

    action_index = shape_idx * (config.ROWS * config.COLS) + r * config.COLS + c
    return action_index


def decode_action(action_index: ActionType, config: EnvConfig) -> tuple[int, int, int]:
    """Decodes an integer action into (shape_idx, r, c)."""
    # Cast ACTION_DIM to int for comparison
    if not (0 <= action_index < int(config.ACTION_DIM)):
        raise ValueError(
            f"Invalid action index: {action_index}, must be < {config.ACTION_DIM}"
        )

    grid_size = config.ROWS * config.COLS
    shape_idx = action_index // grid_size
    remainder = action_index % grid_size
    r = remainder // config.COLS
    c = remainder % config.COLS

    return shape_idx, r, c
