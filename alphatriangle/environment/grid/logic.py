# File: alphatriangle/environment/grid/logic.py
import logging
from typing import TYPE_CHECKING

# Import NO_COLOR_ID from the structs package directly
from ...structs import NO_COLOR_ID

if TYPE_CHECKING:
    from ...structs import Shape
    from .grid_data import GridData

logger = logging.getLogger(__name__)


# Removed link_neighbors function as it's no longer needed


def can_place(grid_data: "GridData", shape: "Shape", r: int, c: int) -> bool:
    """
    Checks if a shape can be placed at the specified (r, c) top-left position
    on the grid, considering occupancy, death zones, and triangle orientation.
    Reads state from GridData's NumPy arrays.
    """
    if not shape or not shape.triangles:
        return False

    for dr, dc, is_up_shape in shape.triangles:
        tri_r, tri_c = r + dr, c + dc

        # Check bounds and death zone first
        if not grid_data.valid(tri_r, tri_c) or grid_data._death_np[tri_r, tri_c]:
            return False

        # Check occupancy
        if grid_data._occupied_np[tri_r, tri_c]:
            return False

        # Check orientation match
        is_up_grid = (tri_r + tri_c) % 2 != 0
        if is_up_grid != is_up_shape:
            # Log the mismatch for debugging the test failure
            logger.debug(
                f"Orientation mismatch at ({tri_r},{tri_c}): Grid is {'Up' if is_up_grid else 'Down'}, Shape requires {'Up' if is_up_shape else 'Down'}"
            )
            return False

    return True


def check_and_clear_lines(
    grid_data: "GridData", newly_occupied_coords: set[tuple[int, int]]
) -> tuple[int, set[tuple[int, int]], set[frozenset[tuple[int, int]]]]:
    """
    Checks for completed lines involving the newly occupied coordinates and clears them.
    Operates on GridData's NumPy arrays.

    Args:
        grid_data: The GridData object (will be modified).
        newly_occupied_coords: A set of (r, c) tuples that were just occupied.

    Returns:
        Tuple containing:
            - int: Number of lines cleared.
            - set[tuple[int, int]]: Set of unique (r, c) coordinates cleared.
            - set[frozenset[tuple[int, int]]]: Set containing the frozenset representations
                                                of the actual lines that were cleared.
    """
    lines_to_check: set[frozenset[tuple[int, int]]] = set()
    for coord in newly_occupied_coords:
        if coord in grid_data._coord_to_lines_map:
            lines_to_check.update(grid_data._coord_to_lines_map[coord])

    cleared_lines_set: set[frozenset[tuple[int, int]]] = set()
    unique_coords_cleared: set[tuple[int, int]] = set()

    if not lines_to_check:
        return 0, unique_coords_cleared, cleared_lines_set

    logger.debug(f"Checking {len(lines_to_check)} potential lines for completion.")

    for line_coords_fs in lines_to_check:
        is_complete = True
        for r_line, c_line in line_coords_fs:
            # Check occupancy directly from the NumPy array
            if not grid_data._occupied_np[r_line, c_line]:
                is_complete = False
                break

        if is_complete:
            logger.debug(f"Line completed: {line_coords_fs}")
            cleared_lines_set.add(line_coords_fs)
            # Add coordinates from this cleared line to the set of unique cleared coordinates
            unique_coords_cleared.update(line_coords_fs)

    if unique_coords_cleared:
        logger.info(
            f"Clearing {len(cleared_lines_set)} lines involving {len(unique_coords_cleared)} unique coordinates."
        )
        # Update NumPy arrays for cleared coordinates
        # Convert set to tuple of arrays for advanced indexing
        if unique_coords_cleared:  # Ensure set is not empty
            rows_idx, cols_idx = zip(*unique_coords_cleared, strict=False)
            grid_data._occupied_np[rows_idx, cols_idx] = False
            grid_data._color_id_np[rows_idx, cols_idx] = NO_COLOR_ID

    return len(cleared_lines_set), unique_coords_cleared, cleared_lines_set
