import logging
import random
from typing import TYPE_CHECKING

from alphatriangle.structs import SHAPE_COLORS, Shape

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
    """
    Refills ALL empty shape slots in the game state with new random shapes.
    This is typically called only when all slots are empty.
    """
    refilled_count = 0
    for i in range(game_state.env_config.NUM_SHAPE_SLOTS):
        if game_state.shapes[i] is None:
            game_state.shapes[i] = generate_random_shape(rng)
            refilled_count += 1
    if refilled_count > 0:
        logger.debug(f"Refilled {refilled_count} shape slots.")


def get_neighbors(r: int, c: int, is_up: bool) -> list[tuple[int, int]]:
    """Gets potential neighbor coordinates for a triangle."""
    if is_up:
        # Up-pointing triangle neighbors: Left, Right, Below
        return [(r, c - 1), (r, c + 1), (r + 1, c)]
    else:
        # Down-pointing triangle neighbors: Left, Right, Above
        return [(r, c - 1), (r, c + 1), (r - 1, c)]


def is_shape_connected(triangles: list[tuple[int, int, bool]]) -> bool:
    """Checks if all triangles in a shape definition are connected."""
    if not triangles or len(triangles) == 1:
        return True

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    triangle_coords = {(r, c) for r, c, _ in triangles}

    for r, c, is_up in triangles:
        pos = (r, c)
        if pos not in adj:
            adj[pos] = []
        for nr, nc in get_neighbors(r, c, is_up):
            neighbor_pos = (nr, nc)
            if neighbor_pos in triangle_coords:
                if neighbor_pos not in adj:
                    adj[neighbor_pos] = []
                if neighbor_pos not in adj[pos]:
                    adj[pos].append(neighbor_pos)
                if pos not in adj[neighbor_pos]:
                    adj[neighbor_pos].append(pos)

    # Perform BFS or DFS to check connectivity
    start_node = (triangles[0][0], triangles[0][1])
    visited = {start_node}
    queue = [start_node]
    while queue:
        node = queue.pop(0)
        if node in adj:
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return len(visited) == len(triangle_coords)
