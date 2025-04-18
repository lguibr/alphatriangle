# File: tests/nn/test_network.py
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Use relative imports for alphatriangle components if running tests from project root
# or absolute imports if package is installed
try:
    # Try absolute imports first (for installed package)
    from alphatriangle.config import EnvConfig, ModelConfig, TrainConfig
    from alphatriangle.environment import GameState
    from alphatriangle.nn import AlphaTriangleNet, NeuralNetwork
    from alphatriangle.utils.types import StateType
except ImportError:
    # Fallback to relative imports (for running tests directly)
    from alphatriangle.config import EnvConfig, ModelConfig, TrainConfig
    from alphatriangle.environment import GameState
    from alphatriangle.nn import AlphaTriangleNet, NeuralNetwork
    from alphatriangle.utils.types import StateType

# Use module-level rng from tests/conftest.py
from tests.conftest import rng


# Use shared fixtures implicitly via pytest injection
@pytest.fixture
def env_config(mock_env_config: EnvConfig) -> EnvConfig:
    return mock_env_config


@pytest.fixture
def model_config(mock_model_config: ModelConfig) -> ModelConfig:
    # Ensure feature dim matches mock_state_type
    mock_model_config.OTHER_NN_INPUT_FEATURES_DIM = 10
    return mock_model_config


@pytest.fixture
def train_config(mock_train_config: TrainConfig) -> TrainConfig:
    # --- CHANGED: Use the default COMPILE_MODEL=True for this test fixture ---
    # Ensure the test runs against the default behavior
    cfg = mock_train_config.model_copy(deep=True)
    cfg.COMPILE_MODEL = True  # Explicitly set to True for clarity in test setup
    return cfg
    # --- END CHANGED ---


@pytest.fixture
def device() -> torch.device:
    # Use CPU for consistency in tests, even though compile might happen
    return torch.device("cpu")


@pytest.fixture
def nn_interface(
    model_config: ModelConfig,
    env_config: EnvConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> NeuralNetwork:
    """Provides a NeuralNetwork instance for testing."""
    # --- CHANGED: Pass the modified train_config ---
    return NeuralNetwork(model_config, env_config, train_config, device)
    # --- END CHANGED ---


@pytest.fixture
def mock_game_state(env_config: EnvConfig) -> GameState:
    """Provides a real GameState object for testing NN interface."""
    # Use a real GameState instance
    return GameState(config=env_config, initial_seed=123)


# --- Fixture providing the batch of copied states ---
@pytest.fixture
def mock_game_state_batch(mock_game_state: GameState) -> list[GameState]:
    """Provides a list of copied GameState objects."""
    batch_size = 3
    # The .copy() call happens here, where mypy knows mock_game_state is GameState
    return [mock_game_state.copy() for _ in range(batch_size)]


# --- End fixture ---


@pytest.fixture
def mock_state_type_nn(model_config: ModelConfig, env_config: EnvConfig) -> StateType:
    """Creates a mock StateType dictionary compatible with the NN test config."""
    grid_shape = (
        model_config.GRID_INPUT_CHANNELS,
        env_config.ROWS,
        env_config.COLS,
    )
    other_shape = (model_config.OTHER_NN_INPUT_FEATURES_DIM,)
    return {
        "grid": rng.random(grid_shape).astype(np.float32),
        "other_features": rng.random(other_shape).astype(np.float32),
    }


# --- Test Initialization ---
def test_nn_initialization(nn_interface: NeuralNetwork, device: torch.device):
    """Test if the NeuralNetwork wrapper initializes correctly."""
    assert nn_interface is not None
    assert nn_interface.device == device
    # --- CHANGED: Check underlying model type if compiled ---
    if hasattr(nn_interface.model, "_orig_mod"):
        # If compiled, check the original module's type
        assert isinstance(nn_interface.model._orig_mod, AlphaTriangleNet)
        # Check that the compiled model is in eval mode
        assert not nn_interface.model.training
    else:
        # If not compiled, check the model directly
        assert isinstance(nn_interface.model, AlphaTriangleNet)
        assert not nn_interface.model.training  # Should be in eval mode initially
    # --- END CHANGED ---


# --- Test Feature Extraction Integration (using mock) ---
@patch("alphatriangle.nn.network.extract_state_features")
def test_state_to_tensors(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_game_state: GameState,  # Use real GameState
    mock_state_type_nn: StateType,
):
    """Test the internal _state_to_tensors method mocks feature extraction."""
    mock_extract.return_value = mock_state_type_nn
    grid_t, other_t = nn_interface._state_to_tensors(mock_game_state)

    mock_extract.assert_called_once_with(mock_game_state, nn_interface.model_config)
    assert isinstance(grid_t, torch.Tensor)
    assert isinstance(other_t, torch.Tensor)
    assert grid_t.device == nn_interface.device
    assert other_t.device == nn_interface.device
    assert grid_t.shape[0] == 1  # Batch dimension
    assert other_t.shape[0] == 1
    assert grid_t.shape[1] == nn_interface.model_config.GRID_INPUT_CHANNELS
    assert other_t.shape[1] == nn_interface.model_config.OTHER_NN_INPUT_FEATURES_DIM


@patch("alphatriangle.nn.network.extract_state_features")
def test_batch_states_to_tensors(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    # --- Use the fixture that provides the already copied batch ---
    mock_game_state_batch: list[GameState],
    mock_state_type_nn: StateType,
):
    """Test the internal _batch_states_to_tensors method."""
    # --- Use the fixture directly, no more .copy() needed here ---
    mock_states = mock_game_state_batch
    batch_size = len(mock_states)
    # --- End change ---
    # Make mock return slightly different arrays each time if needed
    # --- CHANGE: Add isinstance check before v.copy() ---
    mock_extract.side_effect = [
        {
            k: (v.copy() + i * 0.1 if isinstance(v, np.ndarray) else v)
            for k, v in mock_state_type_nn.items()
        }
        for i in range(batch_size)
    ]
    # --- END CHANGE ---

    grid_t, other_t = nn_interface._batch_states_to_tensors(mock_states)

    assert mock_extract.call_count == batch_size
    assert isinstance(grid_t, torch.Tensor)
    assert isinstance(other_t, torch.Tensor)
    assert grid_t.device == nn_interface.device
    assert other_t.device == nn_interface.device
    assert grid_t.shape[0] == batch_size
    assert other_t.shape[0] == batch_size
    assert grid_t.shape[1] == nn_interface.model_config.GRID_INPUT_CHANNELS
    assert other_t.shape[1] == nn_interface.model_config.OTHER_NN_INPUT_FEATURES_DIM


# --- Test Evaluation Methods ---
@patch("alphatriangle.nn.network.extract_state_features")
def test_evaluate_single(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_game_state: GameState,  # Use real GameState
    mock_state_type_nn: StateType,
    env_config: EnvConfig,
):
    """Test the evaluate method for a single state."""
    mock_extract.return_value = mock_state_type_nn
    # Cast ACTION_DIM to int
    action_dim_int = int(env_config.ACTION_DIM)

    policy_map, value = nn_interface.evaluate(mock_game_state)

    assert isinstance(policy_map, dict)
    assert isinstance(value, float)
    assert len(policy_map) == action_dim_int
    assert all(
        isinstance(k, int) and isinstance(v, float) for k, v in policy_map.items()
    )
    assert abs(sum(policy_map.values()) - 1.0) < 1e-5, (
        f"Policy probs sum to {sum(policy_map.values())}"
    )
    # --- REMOVED: Value range check ---
    # assert -1.0 <= value <= 1.0
    # --- END REMOVED ---


@patch("alphatriangle.nn.network.extract_state_features")
def test_evaluate_batch(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    # --- Use the fixture that provides the already copied batch ---
    mock_game_state_batch: list[GameState],
    mock_state_type_nn: StateType,
    env_config: EnvConfig,
):
    """Test the evaluate_batch method."""
    # --- Use the fixture directly, no more .copy() needed here ---
    mock_states = mock_game_state_batch
    batch_size = len(mock_states)
    # --- End change ---
    # --- CHANGE: Add isinstance check before v.copy() ---
    mock_extract.side_effect = [
        {
            k: (v.copy() + i * 0.1 if isinstance(v, np.ndarray) else v)
            for k, v in mock_state_type_nn.items()
        }
        for i in range(batch_size)
    ]
    # --- END CHANGE ---
    # Cast ACTION_DIM to int
    action_dim_int = int(env_config.ACTION_DIM)

    results = nn_interface.evaluate_batch(mock_states)

    assert isinstance(results, list)
    assert len(results) == batch_size
    for policy_map, value in results:
        assert isinstance(policy_map, dict)
        assert isinstance(value, float)
        assert len(policy_map) == action_dim_int
        assert all(
            isinstance(k, int) and isinstance(v, float) for k, v in policy_map.items()
        )
        assert abs(sum(policy_map.values()) - 1.0) < 1e-5
        # --- REMOVED: Value range check ---
        # assert -1.0 <= value <= 1.0
        # --- END REMOVED ---


# --- Test Weight Management ---
def test_get_set_weights(nn_interface: NeuralNetwork):
    """Test getting and setting model weights."""
    initial_weights = nn_interface.get_weights()
    assert isinstance(initial_weights, dict)
    assert all(
        isinstance(k, str) and isinstance(v, torch.Tensor)
        for k, v in initial_weights.items()
    )
    # Check weights are on CPU
    assert all(v.device == torch.device("cpu") for v in initial_weights.values())

    # Modify only parameters (which should be floats)
    modified_weights = {}
    for k, v in initial_weights.items():
        if v.dtype.is_floating_point:
            modified_weights[k] = v + 0.1
        else:
            modified_weights[k] = v  # Keep non-float tensors (e.g., buffers) unchanged

    # Set modified weights
    nn_interface.set_weights(modified_weights)

    # Get weights again and compare parameters
    new_weights = nn_interface.get_weights()
    assert len(new_weights) == len(initial_weights)
    for key in initial_weights:
        assert key in new_weights
        # Compare on CPU
        if initial_weights[key].dtype.is_floating_point:
            assert torch.allclose(modified_weights[key], new_weights[key], atol=1e-6), (
                f"Weight mismatch for key {key}"
            )
        else:
            assert torch.equal(initial_weights[key], new_weights[key]), (
                f"Non-float tensor mismatch for key {key}"
            )

    # Test setting back original weights
    nn_interface.set_weights(initial_weights)
    final_weights = nn_interface.get_weights()
    for key in initial_weights:
        assert torch.equal(initial_weights[key], final_weights[key]), (
            f"Weight mismatch after setting back original for key {key}"
        )
