import logging
import random
from typing import TYPE_CHECKING

from src.structs import Triangle

from .. import shapes
from ..grid import logic as GridLogic

if TYPE_CHECKING:
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def calculate_reward(
    placed_count: int, cleared_lines_set: set[frozenset[Triangle]]
) -> float:
    """
    Calculates the reward based on placed triangles and cleared lines.
    +1 per placed triangle.
    +2 per unique triangle in cleared lines.
    """
    placement_reward = float(placed_count)
    line_clear_reward = 0.0
    unique_triangles_cleared: set[Triangle] = set()
    for line in cleared_lines_set:
        unique_triangles_cleared.update(line)
    line_clear_reward = len(unique_triangles_cleared) * 2.0
    total_reward = placement_reward + line_clear_reward
    logger.debug(
        f"Reward calculated: Placement={placement_reward}, LineClear={line_clear_reward} ({len(unique_triangles_cleared)} tris), Total={total_reward}"
    )
    return total_reward


def execute_placement(
    game_state: "GameState", shape_idx: int, r: int, c: int, rng: random.Random
) -> float:
    """
    Places the shape, updates the grid, calculates reward, clears lines,
    and handles shape refilling.
    Returns the calculated reward for the step.
    """
    shape = game_state.shapes[shape_idx]
    if not shape:
        logger.error(f"Attempted to place None shape at index {shape_idx}.")
        return 0.0

    if not GridLogic.can_place(game_state.grid_data, shape, r, c):
        logger.error(
            f"Invalid placement attempted in execute_placement for shape {shape_idx} at ({r},{c})."
        )
        # Should not happen if called after valid_actions check, but good safeguard.
        game_state.game_over = True
        return 0.0

    # Place the shape
    newly_occupied_triangles: set[Triangle] = set()
    placed_count = 0
    for dr, dc, _is_up in shape.triangles:  # Rename is_up to _is_up
        tri_r, tri_c = r + dr, c + dc
        if game_state.grid_data.valid(tri_r, tri_c):
            tri = game_state.grid_data.triangles[tri_r][tri_c]
            if not tri.is_death and not tri.is_occupied:
                tri.is_occupied = True
                tri.color = shape.color
                game_state.grid_data._occupied_np[tri_r, tri_c] = True
                newly_occupied_triangles.add(tri)
                placed_count += 1
            else:
                # This case should ideally be caught by can_place, but log if it occurs
                logger.warning(
                    f"Overlap detected during placement at ({tri_r},{tri_c}) despite can_place=True."
                )
        else:
            logger.error(
                f"Invalid coordinates ({tri_r},{tri_c}) derived during placement."
            )

    # Clear the placed shape slot
    game_state.shapes[shape_idx] = None
    game_state.pieces_placed_this_episode += 1

    # Check for line clears
    (
        lines_cleared_count,
        unique_triangles_cleared,
        cleared_lines_set,
    ) = GridLogic.check_and_clear_lines(game_state.grid_data, newly_occupied_triangles)
    game_state.triangles_cleared_this_episode += len(unique_triangles_cleared)

    # Calculate reward
    reward = calculate_reward(placed_count, cleared_lines_set)
    game_state.game_score += reward

    # Check if all shape slots are now empty and refill if necessary
    if all(s is None for s in game_state.shapes):
        logger.debug("All shape slots empty, triggering batch refill.")
        shapes.refill_shape_slots(game_state, rng)
        # Check if game is over *after* refill
        if not game_state.valid_actions():
            logger.info("Game over: No valid moves after shape refill.")
            game_state.game_over = True

    return reward
