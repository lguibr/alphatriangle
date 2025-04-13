# File: src/environment/logic/step.py
import logging
import random
from typing import TYPE_CHECKING, Set, Tuple

from ..grid import logic as GridLogic
from ..shapes import logic as ShapeLogic  # Import ShapeLogic namespace
from src.structs import Triangle

if TYPE_CHECKING:
    from ..core.game_state import GameState

logger = logging.getLogger(__name__)


def calculate_reward(
    placed_count: int, cleared_lines_set: Set[frozenset[Triangle]]
) -> float:
    """
    Calculates the reward based on placed triangles and cleared lines.
    - +1 point per triangle placed.
    - +2 points per triangle in each cleared line (triangles in multiple cleared lines count multiple times).
    """
    placement_reward = float(placed_count)
    line_clear_reward = 0.0
    unique_triangles_cleared = set()

    if cleared_lines_set:
        for line in cleared_lines_set:
            line_clear_reward += len(line) * 2.0
            unique_triangles_cleared.update(line)

    total_reward = placement_reward + line_clear_reward
    logger.debug(
        f"Reward calculated: Placement={placement_reward}, LineClear={line_clear_reward} (Lines: {len(cleared_lines_set)}, Unique Tris: {len(unique_triangles_cleared)}), Total={total_reward}"
    )
    return total_reward


def execute_placement(
    game_state: "GameState", shape_idx: int, r: int, c: int, rng: random.Random
) -> float:
    """
    Executes placing a shape on the grid, updates state, checks for line clears,
    and potentially refills shape slots *if all are empty*.

    Args:
        game_state: The current game state (will be modified).
        shape_idx: Index of the shape to place.
        r: Target row for the shape's origin.
        c: Target column for the shape's origin.
        rng: Random number generator instance.

    Returns:
        The reward obtained from this placement.
    """
    shape = game_state.shapes[shape_idx]
    if not shape:
        logger.error(f"Attempted to place an empty shape slot: {shape_idx}")
        return 0.0

    if not GridLogic.can_place(game_state.grid_data, shape, r, c):
        logger.error(
            f"Invalid placement attempted in execute_placement for shape {shape_idx} at ({r},{c}). Should be checked before calling."
        )
        # Optionally, could set game_over here or return a large negative reward
        game_state.game_over = True  # Penalize invalid move attempt
        return -10.0  # Negative reward for trying invalid move

    # Place the shape
    newly_occupied_triangles: Set[Triangle] = set()
    placed_count = 0
    for dr, dc, is_up in shape.triangles:
        tri_r, tri_c = r + dr, c + dc
        if game_state.grid_data.valid(tri_r, tri_c):
            tri = game_state.grid_data.triangles[tri_r][tri_c]
            if not tri.is_occupied and not tri.is_death:
                tri.is_occupied = True
                tri.color = shape.color
                game_state.grid_data._occupied_np[tri_r, tri_c] = True
                newly_occupied_triangles.add(tri)
                placed_count += 1
            else:
                # This case should ideally not happen if can_place was checked correctly
                logger.warning(
                    f"Overlap detected during placement at ({tri_r},{tri_c}) even after can_place check."
                )
        else:
            logger.warning(
                f"Triangle ({tri_r},{tri_c}) out of bounds during placement."
            )

    # Check for line clears
    lines_cleared_count, unique_tris_cleared, cleared_lines_set = (
        GridLogic.check_and_clear_lines(game_state.grid_data, newly_occupied_triangles)
    )

    # Calculate reward
    reward = calculate_reward(placed_count, cleared_lines_set)
    game_state.game_score += reward
    game_state.pieces_placed_this_episode += 1
    game_state.triangles_cleared_this_episode += len(unique_tris_cleared)

    # Clear the used shape slot
    game_state.shapes[shape_idx] = None
    logger.debug(f"Cleared shape slot {shape_idx} after placement.")

    # --- Refill Logic Change ---
    # Check if ALL shape slots are now empty
    if all(s is None for s in game_state.shapes):
        logger.info("All shape slots are empty. Refilling all slots.")
        # Call the modified refill function (no index needed)
        ShapeLogic.refill_shape_slots(game_state, rng)
        # After refilling, check if the game is now over (no valid moves with new shapes)
        if not game_state.valid_actions():
            logger.info("Game over: No valid moves after refilling shape slots.")
            game_state.game_over = True
    # --- End Refill Logic Change ---

    return reward
