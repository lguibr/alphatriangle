import pygame

from ...structs import Shape, Triangle
from ..core import colors


def draw_shape(
    surface: pygame.Surface,
    shape: Shape,
    topleft: tuple[int, int],
    cell_size: float,
    _is_selected: bool = False,
    origin_offset: tuple[int, int] = (0, 0),
) -> None:
    """Draws a single shape onto a surface."""
    if not shape or not shape.triangles or cell_size <= 0:
        return

    shape_color = shape.color
    border_color = colors.GRAY

    cw = cell_size
    ch = cell_size

    for dr, dc, is_up in shape.triangles:
        adj_r, adj_c = dr + origin_offset[0], dc + origin_offset[1]

        tri_x = topleft[0] + adj_c * (cw * 0.75)
        tri_y = topleft[1] + adj_r * ch

        temp_tri = Triangle(0, 0, is_up)
        pts = [(px + tri_x, py + tri_y) for px, py in temp_tri.get_points(0, 0, cw, ch)]

        pygame.draw.polygon(surface, shape_color, pts)
        pygame.draw.polygon(surface, border_color, pts, 1)
