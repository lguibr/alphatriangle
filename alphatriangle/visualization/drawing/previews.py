import logging
from typing import TYPE_CHECKING

import pygame

from ...structs import Shape, Triangle
from ..core import colors, coord_mapper
from .shapes import draw_shape

if TYPE_CHECKING:
    from ...config import EnvConfig, VisConfig
    from ...environment import GameState

logger = logging.getLogger(__name__)


def render_previews(
    surface: pygame.Surface,
    game_state: "GameState",
    area_topleft: tuple[int, int],
    _mode: str,
    env_config: "EnvConfig",
    vis_config: "VisConfig",
    selected_shape_idx: int = -1,
) -> dict[int, pygame.Rect]:
    """Renders shape previews in their area. Returns dict {index: screen_rect}."""
    surface.fill(colors.PREVIEW_BG)
    preview_rects_screen: dict[int, pygame.Rect] = {}
    num_slots = env_config.NUM_SHAPE_SLOTS
    pad = vis_config.PREVIEW_PADDING
    inner_pad = vis_config.PREVIEW_INNER_PADDING
    border = vis_config.PREVIEW_BORDER_WIDTH
    selected_border = vis_config.PREVIEW_SELECTED_BORDER_WIDTH

    if num_slots <= 0:
        return {}

    # Calculate dimensions for each slot
    total_pad_h = (num_slots + 1) * pad
    available_h = surface.get_height() - total_pad_h
    slot_h = available_h / num_slots if num_slots > 0 else 0
    slot_w = surface.get_width() - 2 * pad

    current_y = float(pad)  # Start y position as float

    for i in range(num_slots):
        # Calculate local rectangle for the slot within the preview surface
        slot_rect_local = pygame.Rect(pad, int(current_y), int(slot_w), int(slot_h))
        # Calculate screen rectangle by offsetting local rect
        slot_rect_screen = slot_rect_local.move(area_topleft)
        preview_rects_screen[i] = (
            slot_rect_screen  # Store screen rect for interaction mapping
        )

        shape: Shape | None = game_state.shapes[i]
        # Use the passed selected_shape_idx for highlighting
        is_selected = selected_shape_idx == i

        # Determine border style based on selection
        border_width = selected_border if is_selected else border
        border_color = (
            colors.PREVIEW_SELECTED_BORDER if is_selected else colors.PREVIEW_BORDER
        )
        # Draw the border rectangle onto the local preview surface
        pygame.draw.rect(surface, border_color, slot_rect_local, border_width)

        # Draw the shape if it exists
        if shape:
            # Calculate drawing area inside the border and padding
            draw_area_w = slot_w - 2 * (border_width + inner_pad)
            draw_area_h = slot_h - 2 * (border_width + inner_pad)

            if draw_area_w > 0 and draw_area_h > 0:
                # Calculate shape bounding box and required cell size
                min_r, min_c, max_r, max_c = shape.bbox()
                shape_rows = max_r - min_r + 1
                # Effective width considering triangle geometry (0.75 factor)
                shape_cols_eff = (
                    (max_c - min_c + 1) * 0.75 + 0.25 if shape.triangles else 1
                )

                # Determine cell size based on available space and shape dimensions
                scale_w = (
                    draw_area_w / shape_cols_eff if shape_cols_eff > 0 else draw_area_w
                )
                scale_h = draw_area_h / shape_rows if shape_rows > 0 else draw_area_h
                cell_size = max(1.0, min(scale_w, scale_h))  # Use the smaller scale

                # Calculate centered top-left position for drawing the shape
                shape_render_w = shape_cols_eff * cell_size
                shape_render_h = shape_rows * cell_size
                draw_topleft_x = (
                    slot_rect_local.left
                    + border_width
                    + inner_pad
                    + (draw_area_w - shape_render_w) / 2
                )
                draw_topleft_y = (
                    slot_rect_local.top
                    + border_width
                    + inner_pad
                    + (draw_area_h - shape_render_h) / 2
                )

                # Draw the shape onto the local preview surface
                # Cast float coordinates to int for draw_shape
                # Use _is_selected to match the function signature
                draw_shape(
                    surface,
                    shape,
                    (int(draw_topleft_x), int(draw_topleft_y)),
                    cell_size,
                    _is_selected=is_selected,
                    origin_offset=(
                        -min_r,
                        -min_c,
                    ),  # Adjust drawing origin based on bbox
                )

        # Move to the next slot position
        current_y += slot_h + pad

    return preview_rects_screen


def draw_placement_preview(
    surface: pygame.Surface,
    shape: "Shape",
    r: int,
    c: int,
    is_valid: bool,
    config: "EnvConfig",
) -> None:
    """Draws a semi-transparent shape snapped to the grid."""
    if not shape or not shape.triangles:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    # Use valid/invalid colors (could be passed in or defined here)
    base_color = (
        colors.PLACEMENT_VALID_COLOR[:3]
        if is_valid
        else colors.PLACEMENT_INVALID_COLOR[:3]
    )
    alpha = (
        colors.PLACEMENT_VALID_COLOR[3]
        if is_valid
        else colors.PLACEMENT_INVALID_COLOR[3]
    )
    color = list(base_color) + [alpha]  # Combine RGB and Alpha

    # Use a temporary surface for transparency
    temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    temp_surface.fill((0, 0, 0, 0))  # Fully transparent background

    for dr, dc, is_up in shape.triangles:
        tri_r, tri_c = r + dr, c + dc
        # Create a temporary Triangle to get points easily
        temp_tri = Triangle(tri_r, tri_c, is_up)
        pts = temp_tri.get_points(ox, oy, cw, ch)
        pygame.draw.polygon(temp_surface, color, pts)

    # Blit the transparent preview onto the main grid surface
    surface.blit(temp_surface, (0, 0))


def draw_floating_preview(
    surface: pygame.Surface,
    shape: "Shape",
    screen_pos: tuple[int, int],  # Position relative to the surface being drawn on
    _config: "EnvConfig",  # Mark config as unused
) -> None:
    """Draws a semi-transparent shape floating at the screen position."""
    if not shape or not shape.triangles:
        return

    cell_size = 20.0  # Fixed size for floating preview? Or scale based on config?
    color = list(shape.color) + [100]  # Base color with fixed alpha

    # Use a temporary surface for transparency
    temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    temp_surface.fill((0, 0, 0, 0))

    # Center the shape around the screen_pos
    min_r, min_c, max_r, max_c = shape.bbox()
    center_r = (min_r + max_r) / 2.0
    center_c = (min_c + max_c) / 2.0

    for dr, dc, is_up in shape.triangles:
        # Calculate position relative to shape center and screen_pos
        pt_x = screen_pos[0] + (dc - center_c) * (cell_size * 0.75)
        pt_y = screen_pos[1] + (dr - center_r) * cell_size

        # Create a temporary Triangle at origin to get relative points
        temp_tri = Triangle(0, 0, is_up)
        # Get points relative to 0,0 and scale
        rel_pts = temp_tri.get_points(0, 0, cell_size, cell_size)
        # Translate points to the calculated screen position
        pts = [(px + pt_x, py + pt_y) for px, py in rel_pts]
        pygame.draw.polygon(temp_surface, color, pts)

    # Blit the transparent preview onto the target surface
    surface.blit(temp_surface, (0, 0))
