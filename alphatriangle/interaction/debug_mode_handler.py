# File: alphatriangle/interaction/debug_mode_handler.py
import logging
from typing import TYPE_CHECKING

import pygame

from ..environment import grid as env_grid

# Import constants from the structs package directly
from ..structs import DEBUG_COLOR_ID, NO_COLOR_ID
from ..visualization import core as vis_core

if TYPE_CHECKING:
    # Keep Triangle for type hinting if GridLogic still uses it temporarily
    from .input_handler import InputHandler

logger = logging.getLogger(__name__)


def handle_debug_click(event: pygame.event.Event, handler: "InputHandler") -> None:
    """Handles mouse clicks in debug mode (toggle triangle state using NumPy arrays)."""
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
        # Check death zone first
        if not game_state.grid_data._death_np[r, c]:
            # Toggle occupancy state in NumPy array
            current_occupied_state = game_state.grid_data._occupied_np[r, c]
            new_occupied_state = not current_occupied_state
            game_state.grid_data._occupied_np[r, c] = new_occupied_state

            # Update color ID based on new state
            new_color_id = DEBUG_COLOR_ID if new_occupied_state else NO_COLOR_ID
            game_state.grid_data._color_id_np[r, c] = new_color_id

            logger.debug(
                f": Toggled triangle ({r},{c}) -> {'Occupied' if new_occupied_state else 'Empty'}"
            )

            # Check for line clears if the cell became occupied
            if new_occupied_state:
                # Pass the coordinate tuple in a set
                lines_cleared, unique_tris_coords, _ = (
                    env_grid.logic.check_and_clear_lines(
                        game_state.grid_data, newly_occupied_coords={(r, c)}
                    )
                )
                if lines_cleared > 0:
                    logger.debug(
                        f"Cleared {lines_cleared} lines ({len(unique_tris_coords)} coords) after toggle."
                    )
        else:
            logger.info(f"Clicked on death cell ({r},{c}). No action.")


def update_debug_hover(handler: "InputHandler") -> None:
    """Updates the debug highlight position within the InputHandler."""
    handler.debug_highlight_coord = None  # Reset hover state

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
        if game_state.grid_data.valid(r, c) and not game_state.grid_data.is_death(r, c):
            handler.debug_highlight_coord = grid_coords
