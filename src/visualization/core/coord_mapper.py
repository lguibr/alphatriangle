import pygame

from ...config import EnvConfig
from ...structs import Triangle
from ...utils.geometry import is_point_in_polygon


def _calculate_render_params(
    width: int, height: int, config: EnvConfig
) -> tuple[float, float, float, float]:
    """Calculates scale (cw, ch) and offset (ox, oy) for rendering the grid."""
    rows, cols = config.ROWS, config.COLS
    cols_eff = cols * 0.75 + 0.25 if cols > 0 else 1
    scale_w = width / cols_eff if cols_eff > 0 else 1
    scale_h = height / rows if rows > 0 else 1
    scale = max(1.0, min(scale_w, scale_h))
    cell_size = scale
    grid_w_px = cols_eff * cell_size
    grid_h_px = rows * cell_size
    offset_x = (width - grid_w_px) / 2
    offset_y = (height - grid_h_px) / 2
    return cell_size, cell_size, offset_x, offset_y


def get_grid_coords_from_screen(
    screen_pos: tuple[int, int], grid_area_rect: pygame.Rect, config: EnvConfig
) -> tuple[int, int] | None:
    """Maps screen coordinates (relative to screen) to grid row/column."""
    if not grid_area_rect or not grid_area_rect.collidepoint(screen_pos):
        return None

    local_x = screen_pos[0] - grid_area_rect.left
    local_y = screen_pos[1] - grid_area_rect.top
    cw, ch, ox, oy = _calculate_render_params(
        grid_area_rect.width, grid_area_rect.height, config
    )
    if cw <= 0 or ch <= 0:
        return None

    row = int((local_y - oy) / ch) if ch > 0 else -1
    approx_col_center_index = (local_x - ox - cw / 4) / (cw * 0.75) if cw > 0 else -1
    col = int(round(approx_col_center_index))

    for r_check in [row, row - 1, row + 1]:
        if not (0 <= r_check < config.ROWS):
            continue
        for c_check in [col, col - 1, col + 1]:
            if not (0 <= c_check < config.COLS):
                continue
            # Use corrected orientation check
            is_up = (r_check + c_check) % 2 != 0
            temp_tri = Triangle(r_check, c_check, is_up)
            pts = temp_tri.get_points(ox, oy, cw, ch)
            if is_point_in_polygon((local_x, local_y), pts):
                return r_check, c_check

    if 0 <= row < config.ROWS and 0 <= col < config.COLS:
        return row, col
    return None


def get_preview_index_from_screen(
    screen_pos: tuple[int, int], preview_rects: dict[int, pygame.Rect]
) -> int | None:
    """Maps screen coordinates to a shape preview index."""
    if not preview_rects:
        return None
    for idx, rect in preview_rects.items():
        if rect and rect.collidepoint(screen_pos):
            return idx
    return None
