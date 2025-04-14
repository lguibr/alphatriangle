import logging
from typing import TYPE_CHECKING

import pygame

from ..environment import core as env_core
from ..environment import grid as env_grid
from ..visualization import core as vis_core

if TYPE_CHECKING:
    from ..structs import Shape
    from .input_handler import InputHandler

logger = logging.getLogger(__name__)


def handle_play_click(event: pygame.event.Event, handler: "InputHandler") -> None:
    """Handles mouse clicks in play mode (select preview, place shape). Modifies handler state."""
    if not (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
        return

    game_state = handler.game_state
    visualizer = handler.visualizer
    mouse_pos = handler.mouse_pos

    if game_state.is_over():
        logger.info("Game is over, ignoring click.")
        return

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    # Get preview rects from visualizer cache
    preview_rects = visualizer.preview_rects

    # 1. Check for clicks on shape previews
    preview_idx = vis_core.coord_mapper.get_preview_index_from_screen(
        mouse_pos, preview_rects
    )
    if preview_idx is not None:
        if handler.selected_shape_idx == preview_idx:
            # Clicked selected shape again: deselect
            handler.selected_shape_idx = -1
            handler.hover_grid_coord = None  # Clear hover state on deselect
            handler.hover_shape = None
            logger.info("Deselected shape.")
        elif (
            0 <= preview_idx < len(game_state.shapes) and game_state.shapes[preview_idx]
        ):
            # Clicked a valid, available shape: select it
            handler.selected_shape_idx = preview_idx
            logger.info(f"Selected shape index: {preview_idx}")
            # Immediately update hover based on current mouse pos after selection
            update_play_hover(handler)  # Update hover state within handler
        else:
            # Clicked an empty or invalid slot
            logger.info(f"Clicked empty/invalid preview slot: {preview_idx}")
            # Deselect if clicking an empty slot while another is selected
            if handler.selected_shape_idx != -1:
                handler.selected_shape_idx = -1
                handler.hover_grid_coord = None
                handler.hover_shape = None
        return  # Handled preview click

    # 2. Check for clicks on the grid (if a shape is selected)
    selected_idx = handler.selected_shape_idx
    if selected_idx != -1 and grid_rect and grid_rect.collidepoint(mouse_pos):
        # A shape is selected, and the click is within the grid area.
        grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
            mouse_pos, grid_rect, game_state.env_config
        )
        # Use TYPE_CHECKING import for Shape type hint
        shape_to_place: Shape | None = game_state.shapes[selected_idx]

        # Check if the placement is valid *at the clicked location*
        if (
            grid_coords
            and shape_to_place
            and env_grid.logic.can_place(
                game_state.grid_data, shape_to_place, grid_coords[0], grid_coords[1]
            )
        ):
            # Valid placement click!
            r, c = grid_coords
            action = env_core.action_codec.encode_action(
                selected_idx, r, c, game_state.env_config
            )
            # Execute the step using the game state's method
            reward, done = game_state.step(action)  # Now returns (reward, done)
            logger.info(
                f"Placed shape {selected_idx} at {grid_coords}. R={reward:.1f}, Done={done}"
            )
            # Deselect shape after successful placement
            handler.selected_shape_idx = -1
            handler.hover_grid_coord = None  # Clear hover state
            handler.hover_shape = None
        else:
            # Clicked grid, shape selected, but not a valid placement spot for the click
            logger.info(f"Clicked grid at {grid_coords}, but placement invalid.")


def update_play_hover(handler: "InputHandler") -> None:
    """Updates the hover state within the InputHandler."""
    # Reset hover state first
    handler.hover_grid_coord = None
    handler.hover_is_valid = False
    handler.hover_shape = None

    game_state = handler.game_state
    visualizer = handler.visualizer
    mouse_pos = handler.mouse_pos

    if game_state.is_over() or handler.selected_shape_idx == -1:
        return  # No hover if game over or no shape selected

    layout_rects = visualizer.ensure_layout()
    grid_rect = layout_rects.get("grid")
    if not grid_rect or not grid_rect.collidepoint(mouse_pos):
        return  # Not hovering over grid

    shape_idx = handler.selected_shape_idx
    if not (0 <= shape_idx < len(game_state.shapes)):
        return
    shape: Shape | None = game_state.shapes[shape_idx]
    if not shape:
        return

    # Get grid coordinates under mouse
    grid_coords = vis_core.coord_mapper.get_grid_coords_from_screen(
        mouse_pos, grid_rect, game_state.env_config
    )

    if grid_coords:
        # Check if placement is valid at these coordinates
        is_valid = env_grid.logic.can_place(
            game_state.grid_data, shape, grid_coords[0], grid_coords[1]
        )
        # Update handler's hover state
        handler.hover_grid_coord = grid_coords
        handler.hover_is_valid = is_valid
        handler.hover_shape = shape  # Store the shape being hovered
    else:
        handler.hover_shape = shape  # Store shape for floating preview
