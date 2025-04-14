# File: src/environment/grid/logic.py
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...structs import Shape, Triangle
    from .grid_data import GridData

logger = logging.getLogger(__name__)


def link_neighbors(grid_data: "GridData"):
    """Links adjacent triangles (left, right, vertical) on the grid."""
    for r in range(grid_data.rows):
        for c in range(grid_data.cols):
            tri = grid_data.triangles[r][c]
            if tri is None:
                continue

            # Horizontal Neighbors (Left/Right)
            left_c, right_c = c - 1, c + 1
            if grid_data.valid(r, left_c):
                tri.neighbor_left = grid_data.triangles[r][left_c]
            if grid_data.valid(r, right_c):
                tri.neighbor_right = grid_data.triangles[r][right_c]

            # Vertical Neighbors (Up/Down - shares a full edge)
            # Only link if they share a horizontal edge.
            # This happens when an UP triangle is above a DOWN triangle.
            if tri.is_up:
                # Up-pointing triangle: Its vertical neighbor is below it (r+1, c)
                down_r = r + 1
                if grid_data.valid(down_r, c):
                    neighbor_below = grid_data.triangles[down_r][c]
                    # Check if the triangle below is DOWN-pointing
                    if neighbor_below and not neighbor_below.is_up:
                        tri.neighbor_vert = neighbor_below
                        # Also link the neighbor back
                        if neighbor_below.neighbor_vert is None:
                            neighbor_below.neighbor_vert = tri
                        elif neighbor_below.neighbor_vert != tri:
                            logger.warning(
                                f"Mismatch vertical link assignment at ({r},{c}) and ({down_r},{c})"
                            )

            else:  # Down-pointing triangle
                # Down-pointing triangle: Its vertical neighbor is above it (r-1, c)
                up_r = r - 1
                if grid_data.valid(up_r, c):
                    neighbor_above = grid_data.triangles[up_r][c]
                    # Check if the triangle above is UP-pointing
                    if neighbor_above and neighbor_above.is_up:
                        tri.neighbor_vert = neighbor_above
                        # Also link the neighbor back
                        if neighbor_above.neighbor_vert is None:
                            neighbor_above.neighbor_vert = tri
                        elif neighbor_above.neighbor_vert != tri:
                            logger.warning(
                                f"Mismatch vertical link assignment at ({r},{c}) and ({up_r},{c})"
                            )


def can_place(grid_data: "GridData", shape: "Shape", r: int, c: int) -> bool:
    """Checks if a shape can be placed at the specified (r, c) grid position."""
    for dr, dc, shape_is_up in shape.triangles:
        grid_r, grid_c = r + dr, c + dc

        if not grid_data.valid(grid_r, grid_c):
            return False

        target_tri = grid_data.triangles[grid_r][grid_c]

        if target_tri.is_death or target_tri.is_occupied:
            return False

        # Check for orientation match
        if target_tri.is_up != shape_is_up:
            return False  # Orientation mismatch

    return True


def check_and_clear_lines(
    grid_data: "GridData", newly_occupied_triangles: set["Triangle"]
) -> tuple[int, set["Triangle"], set[frozenset["Triangle"]]]:
    """
    Checks for completed lines involving the newly occupied triangles and clears them.

    Returns:
        - Number of lines cleared.
        - Set of unique triangles that were cleared.
        - Set of the actual cleared lines (as frozensets of triangles).
    """
    lines_to_check: set[frozenset[Triangle]] = set()
    for tri in newly_occupied_triangles:
        if tri in grid_data._triangle_to_lines_map:
            lines_to_check.update(grid_data._triangle_to_lines_map[tri])

    cleared_lines_set: set[frozenset[Triangle]] = set()
    triangles_cleared: set[Triangle] = set()

    for line in lines_to_check:
        if all(tri.is_occupied for tri in line):
            cleared_lines_set.add(line)
            triangles_cleared.update(line)

    if triangles_cleared:
        logger.info(
            f"Clearing {len(cleared_lines_set)} lines involving {len(triangles_cleared)} triangles."
        )
        for tri in triangles_cleared:
            tri.is_occupied = False
            tri.color = None
            grid_data._occupied_np[tri.row, tri.col] = False

    return len(cleared_lines_set), triangles_cleared, cleared_lines_set
