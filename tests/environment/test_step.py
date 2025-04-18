# File: tests/environment/test_step.py
import random
from time import sleep

import pytest

# Import mocker fixture from pytest-mock
from pytest_mock import MockerFixture

from alphatriangle.config import EnvConfig
from alphatriangle.environment.core.game_state import GameState
from alphatriangle.environment.grid import (
    logic as GridLogic,
)
from alphatriangle.environment.grid.grid_data import GridData
from alphatriangle.environment.logic.step import calculate_reward, execute_placement
from alphatriangle.structs import Shape, Triangle

# Fixtures are now implicitly injected from tests/environment/conftest.py


def occupy_line(
    grid_data: GridData, line_indices: list[int], config: EnvConfig
) -> set[Triangle]:
    """Helper to occupy triangles for a given line index list."""
    # occupied_tris: set[Triangle] = set() # Removed unused variable
    for idx in line_indices:
        r, c = divmod(idx, config.COLS)
        # Combine nested if using 'and'
        if grid_data.valid(r, c) and not grid_data.is_death(r, c):
            grid_data._occupied_np[r, c] = True
            # Cannot easily return Triangle objects anymore
    # Return empty set as Triangle objects are not the primary state
    return set()


def occupy_coords(grid_data: GridData, coords: set[tuple[int, int]]):
    """Helper to occupy specific coordinates."""
    for r, c in coords:
        if grid_data.valid(r, c) and not grid_data.is_death(r, c):
            grid_data._occupied_np[r, c] = True


# --- New Reward Calculation Tests (v3) ---


def test_calculate_reward_v3_placement_only(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: only placement, game not over."""
    placed_count = len(simple_shape.triangles)
    unique_coords_cleared: set[tuple[int, int]] = set()
    is_game_over = False
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + default_env_config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)


def test_calculate_reward_v3_single_line_clear(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: placement + line clear, game not over."""
    placed_count = len(simple_shape.triangles)
    # Simulate a cleared line of 9 unique coordinates
    unique_coords_cleared: set[tuple[int, int]] = {(0, i) for i in range(9)}
    is_game_over = False
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + len(unique_coords_cleared) * default_env_config.REWARD_PER_CLEARED_TRIANGLE
        + default_env_config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)


def test_calculate_reward_v3_multi_line_clear(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: placement + multi-line clear (overlapping coords), game not over."""
    placed_count = len(simple_shape.triangles)
    # Simulate two lines sharing coordinate (0,0)
    line1_coords = {(0, i) for i in range(9)}
    line2_coords = {(i, 0) for i in range(5)}
    unique_coords_cleared = line1_coords.union(line2_coords)  # Union handles uniqueness
    is_game_over = False
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + len(unique_coords_cleared) * default_env_config.REWARD_PER_CLEARED_TRIANGLE
        + default_env_config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)


def test_calculate_reward_v3_game_over(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: placement, no line clear, game IS over."""
    placed_count = len(simple_shape.triangles)
    unique_coords_cleared: set[tuple[int, int]] = set()
    is_game_over = True
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + default_env_config.PENALTY_GAME_OVER
    )
    assert reward == pytest.approx(expected_reward)


def test_calculate_reward_v3_game_over_with_clear(
    simple_shape: Shape, default_env_config: EnvConfig
):
    """Test reward: placement + line clear, game IS over."""
    placed_count = len(simple_shape.triangles)
    unique_coords_cleared: set[tuple[int, int]] = {(0, i) for i in range(9)}
    is_game_over = True
    reward = calculate_reward(
        placed_count, unique_coords_cleared, is_game_over, default_env_config
    )
    expected_reward = (
        placed_count * default_env_config.REWARD_PER_PLACED_TRIANGLE
        + len(unique_coords_cleared) * default_env_config.REWARD_PER_CLEARED_TRIANGLE
        + default_env_config.PENALTY_GAME_OVER
    )
    assert reward == pytest.approx(expected_reward)


# --- Test execute_placement with new reward ---


def test_execute_placement_simple_no_refill_v3(
    game_state_with_fixed_shapes: GameState,
):
    """Test placing a shape without clearing lines, verify reward and NO immediate refill."""
    gs = game_state_with_fixed_shapes  # Uses 3 slots, initially filled
    config = gs.env_config
    shape_idx = 0
    original_shape_in_slot_1 = gs.shapes[1]
    original_shape_in_slot_2 = gs.shapes[2]
    shape_to_place = gs.shapes[shape_idx]
    assert shape_to_place is not None
    placed_count = len(shape_to_place.triangles)

    r, c = 2, 2
    assert GridLogic.can_place(gs.grid_data, shape_to_place, r, c)
    mock_rng = random.Random(42)

    reward = execute_placement(gs, shape_idx, r, c, mock_rng)

    # Verify reward (placement + survival)
    expected_reward = (
        placed_count * config.REWARD_PER_PLACED_TRIANGLE + config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)
    # Score is still tracked separately
    assert gs.game_score == placed_count

    # Verify grid state using NumPy arrays
    for dr, dc, _ in shape_to_place.triangles:
        tri_r, tri_c = r + dr, c + dc
        assert gs.grid_data._occupied_np[tri_r, tri_c]
        # Cannot easily check color ID without map here, trust placement logic

    # Verify shape slot is now EMPTY
    assert gs.shapes[shape_idx] is None

    # --- Verify NO REFILL ---
    assert gs.shapes[1] is original_shape_in_slot_1
    assert gs.shapes[2] is original_shape_in_slot_2

    assert gs.pieces_placed_this_episode == 1
    assert gs.triangles_cleared_this_episode == 0
    assert not gs.is_over()


def test_execute_placement_clear_line_no_refill_v3(
    game_state_with_fixed_shapes: GameState,
):
    """Test placing a shape that clears a line, verify reward and NO immediate refill."""
    gs = game_state_with_fixed_shapes
    config = gs.env_config
    shape_idx = 0
    shape_single_down = gs.shapes[shape_idx]
    assert (
        shape_single_down is not None
        and len(shape_single_down.triangles) == 1
        and not shape_single_down.triangles[0][2]
    )
    placed_count = len(shape_single_down.triangles)
    original_shape_in_slot_1 = gs.shapes[1]
    original_shape_in_slot_2 = gs.shapes[2]

    # Pre-occupy line using coordinates
    # Line indices [3..11] correspond to r=0, c=3 to c=11
    line_coords_to_occupy = {(0, i) for i in range(3, 12) if i != 4}
    occupy_coords(gs.grid_data, line_coords_to_occupy)
    cleared_line_coords = {(0, i) for i in range(3, 12)}  # Coords (0,3) to (0,11)

    r, c = 0, 4  # Placement position
    assert GridLogic.can_place(gs.grid_data, shape_single_down, r, c)
    mock_rng = random.Random(42)

    reward = execute_placement(gs, shape_idx, r, c, mock_rng)

    # Verify reward (placement + line clear + survival)
    expected_reward = (
        placed_count * config.REWARD_PER_PLACED_TRIANGLE
        + len(cleared_line_coords) * config.REWARD_PER_CLEARED_TRIANGLE
        + config.REWARD_PER_STEP_ALIVE
    )
    assert reward == pytest.approx(expected_reward)
    # Score is still tracked separately
    assert gs.game_score == placed_count + len(cleared_line_coords) * 2

    # Verify line is cleared using NumPy array
    for row, col in cleared_line_coords:
        assert not gs.grid_data._occupied_np[row, col]

    # Verify shape slot is now EMPTY
    assert gs.shapes[shape_idx] is None

    # --- Verify NO REFILL ---
    assert gs.shapes[1] is original_shape_in_slot_1
    assert gs.shapes[2] is original_shape_in_slot_2

    assert gs.pieces_placed_this_episode == 1
    assert gs.triangles_cleared_this_episode == len(cleared_line_coords)
    assert not gs.is_over()


def test_execute_placement_batch_refill_v3(game_state_with_fixed_shapes: GameState):
    """Test that placing the last shape triggers a refill and correct reward."""
    gs = game_state_with_fixed_shapes
    config = gs.env_config
    mock_rng = random.Random(123)

    # Place first shape
    shape_1_coords = (0, 4)
    assert gs.shapes[0] is not None
    placed_count_1 = len(gs.shapes[0].triangles)
    assert GridLogic.can_place(gs.grid_data, gs.shapes[0], *shape_1_coords)
    reward1 = execute_placement(gs, 0, 0, 4, mock_rng)
    expected_reward1 = (
        placed_count_1 * config.REWARD_PER_PLACED_TRIANGLE
        + config.REWARD_PER_STEP_ALIVE
    )
    assert reward1 == pytest.approx(expected_reward1)
    assert gs.shapes[0] is None
    assert gs.shapes[1] is not None
    assert gs.shapes[2] is not None

    # Place second shape
    shape_2_coords = (0, 3)
    assert gs.shapes[1] is not None
    placed_count_2 = len(gs.shapes[1].triangles)
    assert GridLogic.can_place(gs.grid_data, gs.shapes[1], *shape_2_coords)
    reward2 = execute_placement(gs, 1, 0, 3, mock_rng)
    expected_reward2 = (
        placed_count_2 * config.REWARD_PER_PLACED_TRIANGLE
        + config.REWARD_PER_STEP_ALIVE
    )
    assert reward2 == pytest.approx(expected_reward2)
    assert gs.shapes[0] is None
    assert gs.shapes[1] is None
    assert gs.shapes[2] is not None

    # Place third shape (triggers refill)
    shape_3_coords = (2, 2)
    assert gs.shapes[2] is not None
    placed_count_3 = len(gs.shapes[2].triangles)
    assert GridLogic.can_place(gs.grid_data, gs.shapes[2], *shape_3_coords)
    reward3 = execute_placement(gs, 2, 2, 2, mock_rng)
    expected_reward3 = (
        placed_count_3 * config.REWARD_PER_PLACED_TRIANGLE
        + config.REWARD_PER_STEP_ALIVE
    )  # Game not over yet
    assert reward3 == pytest.approx(expected_reward3)
    sleep(0.01)  # Allow time for refill to happen (though it should be synchronous)

    # --- Verify REFILL happened ---
    assert all(s is not None for s in gs.shapes), "Not all slots were refilled"
    assert gs.shapes[0] != Shape([(0, 0, False)], (255, 0, 0))
    assert gs.shapes[1] != Shape([(0, 0, True)], (0, 255, 0))
    assert gs.shapes[2] != Shape([(0, 0, False), (0, 1, True)], (0, 0, 255))

    assert gs.pieces_placed_this_episode == 3
    assert not gs.is_over()


# Add mocker fixture to the test signature
def test_execute_placement_game_over_v3(game_state: GameState, mocker: MockerFixture):
    """Test reward when placement leads to game over, mocking line clears."""
    config = game_state.env_config
    # Fill grid almost completely using NumPy arrays
    playable_mask = ~game_state.grid_data._death_np
    game_state.grid_data._occupied_np[playable_mask] = True

    # Make one spot empty
    empty_r, empty_c = 0, 4
    if not game_state.grid_data.is_death(empty_r, empty_c):  # Ensure it's playable
        game_state.grid_data._occupied_np[empty_r, empty_c] = False

    # Provide a shape that fits the empty spot
    shape_to_place = Shape([(0, 0, False)], (255, 0, 0))  # Single down triangle
    placed_count = len(shape_to_place.triangles)

    # --- Modify setup to prevent refill ---
    unplaceable_shape = Shape([(0, 0, False), (1, 0, False), (2, 0, False)], (1, 1, 1))
    game_state.shapes = [None] * config.NUM_SHAPE_SLOTS
    game_state.shapes[0] = shape_to_place
    if config.NUM_SHAPE_SLOTS > 1:
        game_state.shapes[1] = unplaceable_shape
    # --- End modification ---

    assert GridLogic.can_place(game_state.grid_data, shape_to_place, empty_r, empty_c)
    mock_rng = random.Random(999)

    # --- Mock check_and_clear_lines ---
    # Patch the function within the logic module where execute_placement imports it from
    mock_clear = mocker.patch(
        "alphatriangle.environment.grid.logic.check_and_clear_lines",
        return_value=(0, set(), set()),  # Simulate no lines cleared
    )
    # --- End Mock ---

    # Execute placement - this should fill the last spot and trigger game over
    reward = execute_placement(game_state, 0, empty_r, empty_c, mock_rng)

    # Verify the mock was called (optional but good practice)
    mock_clear.assert_called_once()

    # Verify game is over
    assert game_state.is_over(), (
        "Game should be over after placing the final piece with no other valid moves"
    )

    # Verify reward (placement + game over penalty)
    expected_reward = (
        placed_count * config.REWARD_PER_PLACED_TRIANGLE + config.PENALTY_GAME_OVER
    )
    # Use a slightly larger tolerance if needed, but approx should work
    assert reward == pytest.approx(expected_reward)
