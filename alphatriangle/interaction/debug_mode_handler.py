# File: src/interaction/debug_mode_handler.py
# File: src/interaction/debug_mode_handler.py
import logging
from typing import TYPE_CHECKING

import pygame

# Use relative imports
from ..environment import grid as env_grid
from ..visualization import core as vis_core

# Move Triangle import into TYPE_CHECKING
if TYPE_CHECKING:
    from ..structs import Triangle
    from .input_handler import InputHandler  # Import InputHandler type

logger = logging.getLogger(__name__)


def handle_debug_click(event: pygame.event.Event, handler: "InputHandler") -> None:
    """Handles mouse clicks in debug mode (toggle triangle state). Modifies handler state."""
    if not (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
        return

    game_state = handler.game_state
    visualizer = handler.visualizer
    mouse_pos = handler.mouse_pos

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    if not grid_rect:
        logger.error("Grid layout rectangle not available for debug click.")
        return

    grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
        mouse_pos, grid_rect, game_state.env_config
    )
    if not grid_coords:
        return

    r, c = grid_coords
    if game_state.grid_data.valid(r, c):
        # Use TYPE_CHECKING import for Triangle type hint
        tri: Triangle = game_state.grid_data.triangles[r][c]
        if not tri.is_death:
            # Toggle state directly on the GameState's GridData
            tri.is_occupied = not tri.is_occupied
            game_state.grid_data._occupied_np[r, c] = (
                tri.is_occupied
            )  # Update numpy cache
            tri.color = vis_core.colors.DEBUG_TOGGLE_COLOR if tri.is_occupied else None
            logger.info(
                f"DEBUG: Toggled triangle ({r},{c}) -> {'Occupied' if tri.is_occupied else 'Empty'}"
            )

            # Check for line clears after toggle (optional, but good for consistency)
            if tri.is_occupied:
                # Pass the single toggled triangle
                lines_cleared, unique_tris, _ = env_grid.logic.check_and_clear_lines(
                    game_state.grid_data, newly_occupied_triangles={tri}
                )
                if lines_cleared > 0:
                    logger.info(
                        f"DEBUG: Cleared {lines_cleared} lines ({len(unique_tris)} tris) after toggle."
                    )
        else:
            logger.info(f"Clicked on death cell ({r},{c}). No action.")


def update_debug_hover(handler: "InputHandler") -> None:
    """Updates the debug highlight position within the InputHandler."""
    # Reset highlight state first
    handler.debug_highlight_coord = None

    game_state = handler.game_state
    visualizer = handler.visualizer
    mouse_pos = handler.mouse_pos

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    if not grid_rect or not grid_rect.collidepoint(mouse_pos):
        return  # Not hovering over grid

    grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
        mouse_pos, grid_rect, game_state.env_config
    )

    if grid_coords:
        r, c = grid_coords
        # Highlight only valid, non-death cells
        if (
            game_state.grid_data.valid(r, c)
            and not game_state.grid_data.triangles[r][c].is_death
        ):
            handler.debug_highlight_coord = grid_coords  # Update handler state
