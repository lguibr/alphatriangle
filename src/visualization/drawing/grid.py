import pygame
from typing import TYPE_CHECKING, Dict, Optional

from ..core import colors, coord_mapper

from src.structs import Triangle

if TYPE_CHECKING:
    from src.environment import GridData
    from src.config import EnvConfig


def draw_grid_background(surface: pygame.Surface, bg_color: tuple) -> None:
    """Fills the grid area surface with a background color."""
    surface.fill(bg_color)


def draw_grid_triangles(
    surface: pygame.Surface, grid_data: "GridData", config: "EnvConfig"
) -> None:
    """Draws all triangles (empty, occupied, death) on the grid surface."""
    if surface.get_width() <= 0 or surface.get_height() <= 0:
        return

    cw, ch, ox, oy = coord_mapper._calculate_render_params(
        surface.get_width(), surface.get_height(), config
    )
    if cw <= 0 or ch <= 0:
        return

    for r in range(grid_data.rows):
        for c in range(grid_data.cols):
            tri: Triangle = grid_data.triangles[r][c]

            if tri.is_death:
                color = colors.DARK_GRAY
                border_color = colors.RED
                border_width = 1
            elif tri.is_occupied:
                color = tri.color if tri.color else colors.DEBUG_TOGGLE_COLOR
                border_color = colors.GRID_LINE_COLOR
                border_width = 1
            else:
                color = colors.TRIANGLE_EMPTY_COLOR
                border_color = colors.GRID_LINE_COLOR
                border_width = 1

            pts = tri.get_points(ox, oy, cw, ch)

            pygame.draw.polygon(surface, color, pts)
            pygame.draw.polygon(surface, border_color, pts, border_width)


def draw_grid_indices(
    surface: pygame.Surface,
    grid_data: "GridData",
    config: "EnvConfig",
    fonts: Dict[str, Optional[pygame.font.Font]],
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

    for r in range(grid_data.rows):
        for c in range(grid_data.cols):
            tri: Triangle = grid_data.triangles[r][c]
            pts = tri.get_points(ox, oy, cw, ch)
            center_x = sum(p[0] for p in pts) / 3
            center_y = sum(p[1] for p in pts) / 3

            if tri.is_death:
                bg_color = colors.DARK_GRAY
                text_color = colors.LIGHT_GRAY
            elif tri.is_occupied:
                bg_color = tri.color if tri.color else colors.DEBUG_TOGGLE_COLOR
                brightness = sum(bg_color) / 3
                text_color = colors.WHITE if brightness < 128 else colors.BLACK
            else:
                bg_color = colors.TRIANGLE_EMPTY_COLOR
                brightness = sum(bg_color) / 3
                text_color = colors.WHITE if brightness < 128 else colors.BLACK

            index = r * config.COLS + c
            text_surf = font.render(str(index), True, text_color)
            text_rect = text_surf.get_rect(center=(center_x, center_y))
            surface.blit(text_surf, text_rect)
