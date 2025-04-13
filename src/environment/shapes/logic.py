# File: src/environment/shapes/logic.py
import logging
import random
from typing import TYPE_CHECKING

from src.structs import SHAPE_COLORS, Shape

from .templates import PREDEFINED_SHAPE_TEMPLATES

if TYPE_CHECKING:
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def generate_random_shape(rng: random.Random) -> Shape:
    """Generates a random shape from the predefined templates."""
    template = rng.choice(PREDEFINED_SHAPE_TEMPLATES)
    color = rng.choice(SHAPE_COLORS)
    return Shape(template, color)


def refill_shape_slots(game_state: "GameState", rng: random.Random):
    """Refills ALL empty shape slots in the game state."""
    refilled_count = 0
    for i in range(len(game_state.shapes)):
        if game_state.shapes[i] is None:
            game_state.shapes[i] = generate_random_shape(rng)
            refilled_count += 1
    if refilled_count > 0:
        # Changed log level from INFO to DEBUG
        logger.debug(f"Refilled {refilled_count} shape slots.")


def get_neighbors(r: int, c: int, is_up: bool) -> list[tuple[int, int]]:
    """Gets potential neighbor coordinates for connectivity check."""
    neighbors = []
    # Horizontal neighbors
    neighbors.append((r, c - 1))
    neighbors.append((r, c + 1))
    vertical_neighbors = (r + 1, c) if is_up else (r - 1, c)
    neighbors.append(vertical_neighbors)
    return neighbors


def is_shape_connected(triangles: list[tuple[int, int, bool]]) -> bool:
    """Checks if all triangles in a shape definition are connected."""
    if not triangles or len(triangles) <= 1:
        return True

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    triangle_coords = set((r, c) for r, c, _ in triangles)

    for r, c, is_up in triangles:
        coord = (r, c)
        if coord not in adj:
            adj[coord] = []
        for nr, nc in get_neighbors(r, c, is_up):
            neighbor_coord = (nr, nc)
            if neighbor_coord in triangle_coords:
                if neighbor_coord not in adj:
                    adj[neighbor_coord] = []
                if neighbor_coord not in adj[coord]:
                    adj[coord].append(neighbor_coord)
                if coord not in adj[neighbor_coord]:
                    adj[neighbor_coord].append(coord)

    # Perform BFS or DFS to check connectivity
    start_node = next(iter(triangle_coords))
    visited = {start_node}
    queue = [start_node]
    while queue:
        u = queue.pop(0)
        if u in adj:
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

    return len(visited) == len(triangle_coords)
