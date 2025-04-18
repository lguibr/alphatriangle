# File: alphatriangle/environment/shapes/logic.py
import logging
import random
from typing import TYPE_CHECKING

from ...structs import SHAPE_COLORS, Shape
from .templates import PREDEFINED_SHAPE_TEMPLATES

if TYPE_CHECKING:
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def generate_random_shape(rng: random.Random) -> Shape:
    """Generates a random shape from predefined templates and colors."""
    template = rng.choice(PREDEFINED_SHAPE_TEMPLATES)
    color = rng.choice(SHAPE_COLORS)
    return Shape(template, color)


def refill_shape_slots(game_state: "GameState", rng: random.Random) -> None:
    """
    Refills ALL empty shape slots in the GameState, but ONLY if ALL slots are currently empty.
    This implements batch refilling.
    """
    # --- CHANGED: Check if ALL slots are None ---
    if all(shape is None for shape in game_state.shapes):
        logger.debug("All shape slots are empty. Refilling all slots.")
        for i in range(game_state.env_config.NUM_SHAPE_SLOTS):
            game_state.shapes[i] = generate_random_shape(rng)
            logger.debug(f"Refilled slot {i} with {game_state.shapes[i]}")
    else:
        logger.debug("Not all shape slots are empty. Skipping refill.")
    # --- END CHANGED ---


def get_neighbors(r: int, c: int, is_up: bool) -> list[tuple[int, int]]:
    """Gets potential neighbor coordinates for connectivity check."""
    if is_up:
        # Up triangle neighbors: (r, c-1), (r, c+1), (r+1, c)
        return [(r, c - 1), (r, c + 1), (r + 1, c)]
    else:
        # Down triangle neighbors: (r, c-1), (r, c+1), (r-1, c)
        return [(r, c - 1), (r, c + 1), (r - 1, c)]


def is_shape_connected(shape: Shape) -> bool:
    """Checks if all triangles in a shape are connected."""
    if not shape.triangles or len(shape.triangles) <= 1:
        return True

    coords_set = {(r, c) for r, c, _ in shape.triangles}
    start_node = shape.triangles[0][:2]  # (r, c) of the first triangle
    visited: set[tuple[int, int]] = set()
    queue = [start_node]
    visited.add(start_node)

    while queue:
        current_r, current_c = queue.pop(0)
        # Find the orientation of the current triangle in the shape list
        current_is_up = False
        for r, c, is_up in shape.triangles:
            if r == current_r and c == current_c:
                current_is_up = is_up
                break

        for nr, nc in get_neighbors(current_r, current_c, current_is_up):
            neighbor_coord = (nr, nc)
            if neighbor_coord in coords_set and neighbor_coord not in visited:
                visited.add(neighbor_coord)
                queue.append(neighbor_coord)

    return len(visited) == len(coords_set)
