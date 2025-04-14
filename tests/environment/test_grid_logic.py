# File: tests/environment/test_grid_logic.py
# File: tests/environment/test_grid_logic.py

import pytest

from alphatriangle.config import EnvConfig
from alphatriangle.environment.core.game_state import GameState
from alphatriangle.environment.grid import logic as GridLogic
from alphatriangle.environment.grid.grid_data import GridData
from alphatriangle.structs import Shape, Triangle


# --- Test Helpers ---
def occupy_line(
    grid_data: GridData, line_indices: list[int], config: EnvConfig
) -> set[Triangle]:
    """Helper to occupy triangles for a given line index list."""
    occupied_tris: set[Triangle] = set()
    for idx in line_indices:
        r, c = divmod(idx, config.COLS)
        if grid_data.valid(r, c):
            tri = grid_data.triangles[r][c]
            if not tri.is_death:
                tri.is_occupied = True
                grid_data._occupied_np[r, c] = True
                occupied_tris.add(tri)
    return occupied_tris


def get_line_indices(grid_data: GridData, line_fset: frozenset[Triangle]) -> list[int]:
    """Helper to get grid indices from a frozenset of triangles."""
    indices = []
    config = grid_data.config
    for tri in line_fset:
        indices.append(tri.row * config.COLS + tri.col)
    return sorted(indices)


# --- Precomputed Correct Lines (DO NOT CHANGE THESE LISTS) ---
expected_horizontal_indices = [
    [3, 4, 5, 6, 7, 8, 9, 10, 11],
    [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74],
    [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88],
    [92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102],
    [108, 109, 110, 111, 112, 113, 114, 115, 116],
]  # Manually evaluated it is correct dont change

expected_diag_tlbr_indices = [
    [45, 60, 61, 76, 77, 92, 93, 108, 109],
    [31, 46, 47, 62, 63, 78, 79, 94, 95, 110, 111],
    [17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113],
    [3, 18, 19, 34, 35, 50, 51, 66, 67, 82, 83, 98, 99, 114, 115],
    [4, 5, 20, 21, 36, 37, 52, 53, 68, 69, 84, 85, 100, 101, 116],
    [6, 7, 22, 23, 38, 39, 54, 55, 70, 71, 86, 87, 102],
    [8, 9, 24, 25, 40, 41, 56, 57, 72, 73, 88],
    [10, 11, 26, 27, 42, 43, 58, 59, 74],
]  # Manually evaluated it is correct dont change

expected_diag_bltr_indices = [
    [60, 45, 46, 31, 32, 17, 18, 3, 4],
    [76, 61, 62, 47, 48, 33, 34, 19, 20, 5, 6],
    [92, 77, 78, 63, 64, 49, 50, 35, 36, 21, 22, 7, 8],
    [108, 93, 94, 79, 80, 65, 66, 51, 52, 37, 38, 23, 24, 9, 10],
    [109, 110, 95, 96, 81, 82, 67, 68, 53, 54, 39, 40, 25, 26, 11],
    [111, 112, 97, 98, 83, 84, 69, 70, 55, 56, 41, 42, 27],
    [113, 114, 99, 100, 85, 86, 71, 72, 57, 58, 43],
    [115, 116, 101, 102, 87, 88, 73, 74, 59],
]  # Manually evaluated it is correct dont change


# --- Test Grid Line Initialization ---
def test_line_precomputation(game_state: GameState):
    """Verify that the precomputed lines match the manually verified lists."""
    grid_data = game_state.grid_data
    assert len(grid_data.potential_lines) > 0

    precomputed_lines_indices: list[list[int]] = []
    for line_fset in grid_data.potential_lines:
        precomputed_lines_indices.append(get_line_indices(grid_data, line_fset))

    all_expected_indices = (
        expected_horizontal_indices
        + expected_diag_tlbr_indices
        + expected_diag_bltr_indices
    )

    # Convert lists to sets of tuples for comparison
    precomputed_set = {tuple(sorted(line)) for line in precomputed_lines_indices}
    expected_set = {tuple(sorted(line)) for line in all_expected_indices}

    assert len(precomputed_set) == len(all_expected_indices), (
        f"Expected {len(all_expected_indices)} lines, but found {len(precomputed_set)}"
    )
    assert precomputed_set == expected_set, (
        "Precomputed lines do not match expected lines"
    )


# --- Test Line Clearing Logic ---
def test_check_and_clear_lines_single_horizontal(game_state: GameState):
    """Test clearing a single horizontal line."""
    grid_data = game_state.grid_data  # Get grid_data from game_state
    config = game_state.env_config
    target_line_indices = expected_horizontal_indices[0]  # Indices 3 to 11
    occupy_line(grid_data, target_line_indices, config)

    # Simulate the last placed piece was index 11
    last_placed_r, last_placed_c = divmod(target_line_indices[-1], config.COLS)
    newly_occupied = {grid_data.triangles[last_placed_r][last_placed_c]}

    # Check and clear
    lines_cleared, unique_tris_cleared, cleared_lines_set = (
        GridLogic.check_and_clear_lines(grid_data, newly_occupied)
    )

    assert lines_cleared == 1
    assert len(unique_tris_cleared) == len(target_line_indices)
    cleared_indices = sorted(
        [tri.row * config.COLS + tri.col for tri in unique_tris_cleared]
    )
    assert cleared_indices == sorted(target_line_indices)
    assert len(cleared_lines_set) == 1
    assert get_line_indices(grid_data, list(cleared_lines_set)[0]) == sorted(
        target_line_indices
    )

    # Verify grid is updated
    for idx in target_line_indices:
        r, c = divmod(idx, config.COLS)
        assert not grid_data.triangles[r][c].is_occupied
        assert not grid_data._occupied_np[r, c]


def test_check_and_clear_lines_multiple(
    game_state: GameState,
):
    """Test clearing multiple lines simultaneously by placing the intersecting piece."""
    grid_data = game_state.grid_data
    config = game_state.env_config

    # Choose two intersecting lines (e.g., horiz[0] and diag_tlbr[3])
    h_line_indices = expected_horizontal_indices[0]  # [3, 4, 5, 6, 7, 8, 9, 10, 11]
    d_line_indices = expected_diag_tlbr_indices[3]  # [3, 18, 19, 34, ..., 115]
    intersection_index = 3  # Index where they intersect

    # Occupy all triangles *except* the intersection
    indices_to_occupy = (set(h_line_indices) | set(d_line_indices)) - {
        intersection_index
    }
    occupy_line(grid_data, list(indices_to_occupy), config)

    # Simulate placing the intersecting triangle (index 3)
    intersect_r, intersect_c = divmod(intersection_index, config.COLS)
    intersect_tri = grid_data.triangles[intersect_r][intersect_c]
    assert not intersect_tri.is_occupied  # Verify it wasn't occupied
    intersect_tri.is_occupied = True  # Occupy it now
    grid_data._occupied_np[intersect_r][intersect_c] = True
    newly_occupied = {intersect_tri}

    # Identify expected cleared triangles and lines
    h_line_tris = {
        grid_data.triangles[divmod(idx, config.COLS)[0]][divmod(idx, config.COLS)[1]]
        for idx in h_line_indices
    }
    d_line_tris = {
        grid_data.triangles[divmod(idx, config.COLS)[0]][divmod(idx, config.COLS)[1]]
        for idx in d_line_indices
    }
    expected_cleared_tris = h_line_tris | d_line_tris
    expected_cleared_lines_set = {frozenset(h_line_tris), frozenset(d_line_tris)}
    expected_cleared_lines_count = 2  # Expecting 2 lines now

    # Check and clear
    lines_cleared, unique_tris_cleared, cleared_lines_set = (
        GridLogic.check_and_clear_lines(grid_data, newly_occupied)
    )

    assert lines_cleared == expected_cleared_lines_count, (
        f"Expected {expected_cleared_lines_count} lines cleared, got {lines_cleared}"
    )
    assert unique_tris_cleared == expected_cleared_tris
    assert cleared_lines_set == expected_cleared_lines_set

    # Verify grid is updated
    for tri in expected_cleared_tris:
        assert not tri.is_occupied
        assert not grid_data._occupied_np[tri.row, tri.col]


def test_check_and_clear_lines_no_clear(game_state: GameState):
    """Test when no lines are cleared."""
    grid_data = game_state.grid_data
    config = game_state.env_config
    # Occupy some triangles, but not a full line
    partial_line = [3, 4, 5]
    newly_occupied = occupy_line(grid_data, partial_line, config)

    # Check and clear
    lines_cleared, unique_tris_cleared, cleared_lines_set = (
        GridLogic.check_and_clear_lines(grid_data, newly_occupied)
    )

    assert lines_cleared == 0
    assert len(unique_tris_cleared) == 0
    assert len(cleared_lines_set) == 0
    # Verify grid is NOT updated (except for the occupied ones)
    for idx in partial_line:
        r, c = divmod(idx, config.COLS)
        assert grid_data.triangles[r][c].is_occupied
        assert grid_data._occupied_np[r, c]


# --- Test Placement Logic ---
@pytest.fixture
def simple_shape() -> Shape:
    """Provides a simple 3-triangle shape (Down, Up, Down)."""
    # Example: L-shape (Down at 0,0; Up at 1,0; Down at 1,1 relative)
    triangles = [(0, 0, False), (1, 0, True), (1, 1, False)]
    color = (255, 0, 0)
    return Shape(triangles, color)


def test_can_place_empty_grid(game_state: GameState, simple_shape: Shape):
    """Test placing a shape on an empty grid."""
    grid_data = game_state.grid_data
    # Place at (2,2). Grid(2,2) is Down (2+2=4, even). Shape(0,0) is Down. OK.
    # Grid(3,2) is Up (3+2=5, odd). Shape(1,0) is Up. OK.
    # Grid(3,3) is Down (3+3=6, even). Shape(1,1) is Down. OK.
    r, c = 2, 2
    assert GridLogic.can_place(grid_data, simple_shape, r, c)


def test_can_place_overlap(game_state: GameState, simple_shape: Shape):
    """Test placing where it overlaps an occupied cell."""
    grid_data = game_state.grid_data
    r, c = 2, 2
    # Occupy one cell where the shape would go: (2,2) + (0,0) = (2,2)
    grid_data.triangles[2][2].is_occupied = True
    grid_data._occupied_np[2, 2] = True
    assert not GridLogic.can_place(grid_data, simple_shape, r, c)


def test_can_place_death_zone(game_state: GameState, simple_shape: Shape):
    """Test placing where it overlaps a death cell."""
    grid_data = game_state.grid_data
    # Find a death cell and try to place the shape origin there
    death_r, death_c = -1, -1
    for r_d in range(grid_data.rows):
        for c_d in range(grid_data.cols):
            if grid_data.triangles[r_d][c_d].is_death:
                death_r, death_c = r_d, c_d
                break
        if death_r != -1:
            break
    assert death_r != -1, "Could not find a death cell in default grid"
    # Place shape origin such that its (0,0) triangle hits the death cell
    # Adjust r, c based on shape's first triangle (0,0)
    r, c = death_r, death_c
    assert not GridLogic.can_place(grid_data, simple_shape, r, c)


def test_can_place_out_of_bounds(game_state: GameState, simple_shape: Shape):
    """Test placing the shape out of grid bounds."""
    grid_data = game_state.grid_data
    # Try placing too far right
    r, c = 2, grid_data.cols
    assert not GridLogic.can_place(grid_data, simple_shape, r, c)
    # Try placing too far down
    r, c = grid_data.rows, 2
    assert not GridLogic.can_place(grid_data, simple_shape, r, c)


def test_can_place_orientation_match(game_state: GameState):
    """Test can_place returns True if shape orientation matches grid."""
    grid_data = game_state.grid_data
    # Shape: single DOWN triangle
    shape_down = Shape([(0, 0, False)], (0, 0, 255))
    # Target location: (0, 0), which is a DOWN triangle on the grid
    r, c = 0, 4

    # Pre-assertions to verify cell state
    target_cell = grid_data.triangles[r][c]
    assert not target_cell.is_occupied, f"Cell ({r},{c}) is unexpectedly occupied."
    assert not target_cell.is_death, f"Cell ({r},{c}) is unexpectedly a death cell."
    assert not target_cell.is_up, f"Cell ({r},{c}) is unexpectedly Up."
    assert not shape_down.triangles[0][2], "Shape triangle is unexpectedly Up."

    # The actual test
    assert GridLogic.can_place(grid_data, shape_down, r, c)


def test_can_place_orientation_mismatch(game_state: GameState):
    """Test can_place returns False if shape orientation mismatches grid."""
    grid_data = game_state.grid_data
    # Shape: single UP triangle
    shape_up = Shape([(0, 0, True)], (0, 0, 255))
    # Target location: (0, 0), which is a DOWN triangle on the grid
    r, c = 0, 0
    assert not grid_data.triangles[r][c].is_up  # Grid is Down
    assert shape_up.triangles[0][2]  # Shape is Up
    assert not GridLogic.can_place(grid_data, shape_up, r, c)

    # Shape: single DOWN triangle
    shape_down = Shape([(0, 0, False)], (0, 0, 255))
    # Target location: (0, 1), which is an UP triangle on the grid
    r, c = 0, 1
    assert grid_data.triangles[r][c].is_up  # Grid is Up
    assert not shape_down.triangles[0][2]  # Shape is Down
    assert not GridLogic.can_place(grid_data, shape_down, r, c)
