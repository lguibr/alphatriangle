# File: tests/environment/test_actions.py
import pytest

from alphatriangle.config import EnvConfig
from alphatriangle.environment.core.action_codec import decode_action
from alphatriangle.environment.core.game_state import GameState

# Import GridLogic correctly
from alphatriangle.environment.grid import logic as GridLogic
from alphatriangle.environment.logic import actions as ActionLogic
from alphatriangle.structs import Shape

# Fixtures are now implicitly injected from tests/environment/conftest.py


@pytest.fixture
def game_state_almost_full(default_env_config: EnvConfig) -> GameState:
    """
    Provides a game state where only a few placements are possible.
    Grid is filled completely, then specific spots are made empty.
    """
    # Use a fresh GameState to avoid side effects from other tests using the shared 'game_state' fixture
    gs = GameState(config=default_env_config, initial_seed=987)
    # Fill the entire playable grid first using NumPy arrays
    playable_mask = ~gs.grid_data._death_np
    gs.grid_data._occupied_np[playable_mask] = True

    # Explicitly make specific spots empty: (0,4) [Down] and (0,5) [Up]
    empty_spots = [(0, 4), (0, 5)]
    for r_empty, c_empty in empty_spots:
        if gs.grid_data.valid(r_empty, c_empty):
            gs.grid_data._occupied_np[r_empty, c_empty] = False
            # Reset color ID as well
            gs.grid_data._color_id_np[
                r_empty, c_empty
            ] = -1  # Assuming -1 is NO_COLOR_ID
    return gs


# --- Test Action Logic ---
def test_get_valid_actions_initial(game_state: GameState):
    """Test valid actions in an initial empty state."""
    valid_actions = ActionLogic.get_valid_actions(game_state)
    assert isinstance(valid_actions, list)
    assert len(valid_actions) > 0  # Should be many valid actions initially

    # Check if decoded actions are valid placements
    for action_index in valid_actions[:10]:  # Check a sample
        shape_idx, r, c = decode_action(action_index, game_state.env_config)
        shape = game_state.shapes[shape_idx]
        assert shape is not None
        assert GridLogic.can_place(game_state.grid_data, shape, r, c)


def test_get_valid_actions_almost_full(game_state_almost_full: GameState):
    """Test valid actions in a nearly full state with only (0,4) and (0,5) free."""
    gs = game_state_almost_full
    # Ensure cells (0,4) and (0,5) are indeed empty using NumPy array
    assert not gs.grid_data._occupied_np[0, 4], "Cell (0,4) should be empty"
    assert not gs.grid_data._occupied_np[0, 5], "Cell (0,5) should be empty"
    # Check orientation (calculated) - Apply Ruff suggestion
    assert (0 + 4) % 2 == 0, "Cell (0,4) should be Down"  # Changed from not (... != 0)
    assert (0 + 5) % 2 != 0, "Cell (0,5) should be Up"

    # Single down triangle fits at (0,4) [which is Down]
    gs.shapes[0] = Shape([(0, 0, False)], (255, 0, 0))
    # Single up triangle fits at (0,5) [which is Up]
    gs.shapes[1] = Shape([(0, 0, True)], (0, 255, 0))
    # Make other slots empty or contain unfittable shapes
    if len(gs.shapes) > 2:
        gs.shapes[2] = Shape([(0, 0, False), (1, 0, False)], (0, 0, 255))  # Unfittable
    if len(gs.shapes) > 3:
        gs.shapes[3] = None

    valid_actions = ActionLogic.get_valid_actions(gs)

    # Expect fewer valid actions
    assert isinstance(valid_actions, list)
    # The setup should allow placing shape 0 at (0,4) and shape 1 at (0,5)
    assert len(valid_actions) == 2, (
        f"Expected 2 valid actions, found {len(valid_actions)}. Actions: {valid_actions}"
    )

    expected_placements = {(0, 0, 4), (1, 0, 5)}  # (shape_idx, r, c)
    found_placements = set()

    # Check if decoded actions are valid placements in the few remaining spots
    for action_index in valid_actions:
        shape_idx, r, c = decode_action(action_index, gs.env_config)
        shape = gs.shapes[shape_idx]
        assert shape is not None, f"Shape at index {shape_idx} is None"
        assert GridLogic.can_place(gs.grid_data, shape, r, c), (
            f"can_place returned False for action {action_index} -> shape_idx={shape_idx}, r={r}, c={c}"
        )
        # Check if placement is in the expected empty area
        is_expected_placement = (r == 0 and c == 4 and shape_idx == 0) or (
            r == 0 and c == 5 and shape_idx == 1
        )
        assert is_expected_placement, (
            f"Action {action_index} -> {(shape_idx, r, c)} is not one of the expected placements (0,0,4) or (1,0,5)"
        )
        found_placements.add((shape_idx, r, c))

    assert found_placements == expected_placements


def test_get_valid_actions_no_shapes(game_state: GameState):
    """Test valid actions when no shapes are available."""
    game_state.shapes = [None] * game_state.env_config.NUM_SHAPE_SLOTS
    valid_actions = ActionLogic.get_valid_actions(game_state)
    assert valid_actions == []


def test_get_valid_actions_no_space(game_state: GameState):
    """Test valid actions when the grid is completely full (or no space for any shape)."""
    # Fill the entire playable grid using NumPy arrays
    playable_mask = ~game_state.grid_data._death_np
    game_state.grid_data._occupied_np[playable_mask] = True

    valid_actions = ActionLogic.get_valid_actions(game_state)
    assert valid_actions == []
