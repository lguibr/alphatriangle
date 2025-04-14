# File: tests/environment/test_step.py
import random
from time import sleep
from typing import Set  # Import Set

import pytest

from src.config import EnvConfig
from src.environment.core.game_state import GameState
from src.environment.grid import logic as GridLogic  # Import GridLogic for setup
from src.environment.grid.grid_data import GridData
from src.environment.logic.step import calculate_reward, execute_placement
from src.structs import Shape, Triangle

# Fixtures are now implicitly injected from tests/environment/conftest.py


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


def test_calculate_reward_placement_only(simple_shape: Shape):
    """Test reward calculation when only placing a shape (no lines cleared)."""
    placed_count = len(simple_shape.triangles)
    cleared_lines_set: set[frozenset[Triangle]] = set()
    reward = calculate_reward(placed_count, cleared_lines_set)
    assert reward == pytest.approx(float(placed_count))  # +1 per triangle


def test_calculate_reward_single_line_clear(simple_shape: Shape):
    """Test reward calculation when placing a shape clears one line."""
    placed_count = len(simple_shape.triangles)
    # Simulate a cleared line of 9 triangles
    # Create dummy triangles for the set structure
    line_tris = {Triangle(0, i, False) for i in range(9)}
    cleared_lines_set: set[frozenset[Triangle]] = {frozenset(line_tris)}
    expected_line_reward = len(line_tris) * 2.0
    expected_total_reward = float(placed_count) + expected_line_reward

    reward = calculate_reward(placed_count, cleared_lines_set)
    assert reward == pytest.approx(expected_total_reward)


def test_calculate_reward_multi_line_clear(simple_shape: Shape):
    """Test reward calculation when placing a shape clears multiple lines."""
    placed_count = len(simple_shape.triangles)
    # Simulate two cleared lines sharing some triangles
    line1_tris = {Triangle(0, i, False) for i in range(9)}  # 9 tris
    line2_tris = {Triangle(i, 0, True) for i in range(5)}  # 5 tris
    # Assume Triangle(0,0,F) is shared if line2 was defined differently,
    # but with this setup, they are disjoint. Let's make them share.
    line2_tris = {Triangle(0, 0, False)} | {
        Triangle(i, 0, True) for i in range(1, 5)
    }  # Share (0,0,F)

    cleared_lines_set: set[frozenset[Triangle]] = {
        frozenset(line1_tris),
        frozenset(line2_tris),
    }
    expected_line1_reward = len(line1_tris) * 2.0  # 9 * 2 = 18
    expected_line2_reward = len(line2_tris) * 2.0  # 5 * 2 = 10
    expected_total_reward = (
        float(placed_count) + expected_line1_reward + expected_line2_reward
    )

    reward = calculate_reward(placed_count, cleared_lines_set)
    assert reward == pytest.approx(expected_total_reward)  # 3 + 18 + 10 = 31


def test_execute_placement_simple_no_refill(
    game_state_with_fixed_shapes: GameState,
    # Removed: simple_shape: Shape # Not used directly
):
    """Test placing a shape without clearing lines, verify NO immediate refill."""
    gs = game_state_with_fixed_shapes  # Uses 3 slots, initially filled
    shape_idx = 0
    original_shape_in_slot_1 = gs.shapes[1]  # Store for later check
    original_shape_in_slot_2 = gs.shapes[2]  # Store for later check
    shape_to_place = gs.shapes[shape_idx]
    assert shape_to_place is not None

    # Place at (2,2). Grid(2,2) is Down (2+2=4, even). Shape(0,0) is Down. OK.
    # Grid(3,2) is Up (3+2=5, odd). Shape(1,0) is Up. OK.
    # Grid(3,3) is Down (3+3=6, even). Shape(1,1) is Down. OK.
    r, c = 2, 2
    assert GridLogic.can_place(gs.grid_data, shape_to_place, r, c)

    # Mock random for predictable refill (though refill shouldn't happen here)
    mock_rng = random.Random(42)

    reward = execute_placement(gs, shape_idx, r, c, mock_rng)

    # Verify reward (only placement points)
    assert reward == pytest.approx(float(len(shape_to_place.triangles)))
    assert gs.game_score == reward

    # Verify grid state
    for dr, dc, _ in shape_to_place.triangles:
        tri_r, tri_c = r + dr, c + dc
        assert gs.grid_data.triangles[tri_r][tri_c].is_occupied
        assert gs.grid_data.triangles[tri_r][tri_c].color == shape_to_place.color
        assert gs.grid_data._occupied_np[tri_r, tri_c]

    # Verify shape slot is now EMPTY
    assert gs.shapes[shape_idx] is None

    # --- Verify NO REFILL ---
    assert gs.shapes[1] is original_shape_in_slot_1  # Slot 1 should be unchanged
    assert gs.shapes[2] is original_shape_in_slot_2  # Slot 2 should be unchanged

    assert gs.pieces_placed_this_episode == 1
    assert gs.triangles_cleared_this_episode == 0
    assert not gs.is_over()


def test_execute_placement_clear_line_no_refill(
    game_state_with_fixed_shapes: GameState,
):
    """Test placing a shape that clears a line, verify NO immediate refill."""
    gs = game_state_with_fixed_shapes  # Uses 3 slots, initially filled
    config = gs.env_config
    shape_idx = 0  # Use the single Down triangle from the fixture
    shape_single_down = gs.shapes[shape_idx]
    assert (
        shape_single_down is not None
        and len(shape_single_down.triangles) == 1
        and not shape_single_down.triangles[0][2]
    )
    original_shape_in_slot_1 = gs.shapes[1]
    original_shape_in_slot_2 = gs.shapes[2]

    # Pre-occupy line [3..11] except index 4 (r=0, c=4)
    pre_occupied_indices = [3, 5, 6, 7, 8, 9, 10, 11]
    occupy_line(gs.grid_data, pre_occupied_indices, config)

    r, c = 0, 4  # Placement position for the single down triangle
    assert GridLogic.can_place(gs.grid_data, shape_single_down, r, c)

    mock_rng = random.Random(42)
    reward = execute_placement(gs, shape_idx, r, c, mock_rng)

    # Verify reward (placement + line clear)
    expected_placement_reward = float(len(shape_single_down.triangles))  # 1
    expected_line_reward = 9 * 2.0  # Line [3..11] has 9 triangles
    expected_total_reward = expected_placement_reward + expected_line_reward
    assert reward == pytest.approx(expected_total_reward)  # 1 + 18 = 19
    assert gs.game_score == reward

    # Verify line is cleared
    cleared_line_indices = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    for idx in cleared_line_indices:
        row, col = divmod(idx, config.COLS)
        assert not gs.grid_data.triangles[row][col].is_occupied
        assert not gs.grid_data._occupied_np[row][col]

    # Verify shape slot is now EMPTY
    assert gs.shapes[shape_idx] is None

    # --- Verify NO REFILL ---
    assert gs.shapes[1] is original_shape_in_slot_1  # Slot 1 should be unchanged
    assert gs.shapes[2] is original_shape_in_slot_2  # Slot 2 should be unchanged

    assert gs.pieces_placed_this_episode == 1
    assert gs.triangles_cleared_this_episode == 9  # 9 unique triangles cleared
    assert not gs.is_over()


def test_execute_placement_batch_refill(game_state_with_fixed_shapes: GameState):
    """Test that placing the last shape triggers a refill of all slots."""
    gs = game_state_with_fixed_shapes  # Uses 3 slots, initially filled
    mock_rng = random.Random(123)

    print(f"Initial shapes: {gs.shapes}")
    shape_1_coordinates = (0, 4)
    # Check shape exists before calling can_place
    assert gs.shapes[0] is not None
    assert GridLogic.can_place(gs.grid_data, gs.shapes[0], *shape_1_coordinates)
    _ = execute_placement(gs, 0, 0, 4, mock_rng)

    assert gs.shapes[0] is None
    assert gs.shapes[1] is not None
    assert gs.shapes[2] is not None

    shape_2_coordinates = (0, 3)
    # Check shape exists before calling can_place
    assert gs.shapes[1] is not None
    assert GridLogic.can_place(gs.grid_data, gs.shapes[1], *shape_2_coordinates)
    _ = execute_placement(gs, 1, 0, 3, mock_rng)
    assert gs.shapes[0] is None
    assert gs.shapes[1] is None
    assert gs.shapes[2] is not None

    shape_3_coordinates = (2, 2)
    # Check shape exists before calling can_place
    assert gs.shapes[2] is not None
    assert GridLogic.can_place(gs.grid_data, gs.shapes[2], *shape_3_coordinates)
    _ = execute_placement(gs, 2, 2, 2, mock_rng)
    sleep(0.01)  # Allow time for refill to happen (though it should be synchronous)
    assert gs.shapes[0] is not None
    assert gs.shapes[1] is not None
    assert gs.shapes[2] is not None

    # --- Verify REFILL happened ---
    # All slots should now contain *new* shapes
    assert all(s is not None for s in gs.shapes), "Not all slots were refilled"
    # Check they are likely different from originals (hard to guarantee with random)
    # This check assumes the fixed shapes are not regenerated by chance immediately
    assert gs.shapes[0] != Shape([(0, 0, False)], (255, 0, 0))
    assert gs.shapes[1] != Shape([(0, 0, True)], (0, 255, 0))
    assert gs.shapes[2] != Shape([(0, 0, False), (0, 1, True)], (0, 0, 255))

    assert gs.pieces_placed_this_episode == 3
    assert not gs.is_over()
