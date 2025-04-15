# File: alphatriangle/visualization/drawing/grid.py
from typing import TYPE_CHECKING

import pygame

# Import constants from the structs package directly
from ...structs import COLOR_ID_MAP, DEBUG_COLOR_ID, NO_COLOR_ID, Triangle
from ..core import colors, coord_mapper

if TYPE_CHECKING:
    from ...config import EnvConfig
    from ...environment import GridData


def draw_grid_background(surface: pygame.Surface, bg_color: tuple) -> None:
    """Fills the grid area surface with a background color."""
    surface.fill(bg_color)


def draw_grid_triangles(
    surface: pygame.Surface, grid_data: "GridData", config: "EnvConfig"
) -> None:
    """Draws all triangles (empty, occupied, death) on the grid surface using NumPy state."""
    if surface.get_width() <= 0 or surface.get_height() <= 0:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    # Get direct references to NumPy arrays
    occupied_np = grid_data._occupied_np
    death_np = grid_data._death_np
    color_id_np = grid_data._color_id_np

    for r in range(grid_data.rows):
        for c in range(grid_data.cols):
            is_death = death_np[r, c]
            is_occupied = occupied_np[r, c]
            color_id = color_id_np[r, c]
            is_up = (r + c) % 2 != 0  # Calculate orientation

            color: tuple[int, int, int] | None = None
            border_color = colors.GRID_LINE_COLOR
            border_width = 1

            if is_death:
                color = colors.DARK_GRAY
                border_color = colors.RED
            elif is_occupied:
                if color_id == DEBUG_COLOR_ID:
                    color = colors.DEBUG_TOGGLE_COLOR  # Special debug color
                elif color_id != NO_COLOR_ID and 0 <= color_id < len(COLOR_ID_MAP):
                    color = COLOR_ID_MAP[color_id]
                else:
                    # Fallback if occupied but no valid color ID (shouldn't happen)
                    color = colors.PURPLE  # Error color
            else:  # Empty playable cell
                color = colors.TRIANGLE_EMPTY_COLOR

            # Create temporary Triangle only for geometry calculation
            temp_tri = Triangle(r, c, is_up)
            pts = temp_tri.get_points(ox, oy, cw, ch)

            if color:  # Should always be true unless error
                pygame.draw.polygon(surface, color, pts)
            pygame.draw.polygon(surface, border_color, pts, border_width)


def draw_grid_indices(
    surface: pygame.Surface,
    grid_data: "GridData",
    config: "EnvConfig",
    fonts: dict[str, pygame.font.Font | None],
) -> None:
    """Draws the index number inside each triangle, including death cells."""
    if surface.get_width() <= 0 or surface.get_height() <= 0:
        return

    font = fonts.get("help")
    if not font:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    # Get direct references to NumPy arrays
    occupied_np = grid_data._occupied_np
    death_np = grid_data._death_np
    color_id_np = grid_data._color_id_np

    for r in range(grid_data.rows):
        for c in range(grid_data.cols):
            is_death = death_np[r, c]
            is_occupied = occupied_np[r, c]
            color_id = color_id_np[r, c]
            is_up = (r + c) % 2 != 0  # Calculate orientation

            # Create temporary Triangle only for geometry calculation
            temp_tri = Triangle(r, c, is_up)
            pts = temp_tri.get_points(ox, oy, cw, ch)
            center_x = sum(p[0] for p in pts) / 3
            center_y = sum(p[1] for p in pts) / 3

            text_color = colors.WHITE  # Default

            if is_death:
                text_color = colors.LIGHT_GRAY
            elif is_occupied:
                bg_color: tuple[int, int, int] | None = None
                if color_id == DEBUG_COLOR_ID:
                    bg_color = colors.DEBUG_TOGGLE_COLOR
                elif color_id != NO_COLOR_ID and 0 <= color_id < len(COLOR_ID_MAP):
                    bg_color = COLOR_ID_MAP[color_id]

                if bg_color:
                    brightness = sum(bg_color) / 3
                    text_color = colors.WHITE if brightness < 128 else colors.BLACK
                else:  # Fallback if color missing
                    text_color = colors.RED
            else:  # Empty playable
                bg_color = colors.TRIANGLE_EMPTY_COLOR
                brightness = sum(bg_color) / 3
                text_color = colors.WHITE if brightness < 128 else colors.BLACK

            index = r * config.COLS + c
            text_surf = font.render(str(index), True, text_color)
            text_rect = text_surf.get_rect(center=(center_x, center_y))
            surface.blit(text_surf, text_rect)
