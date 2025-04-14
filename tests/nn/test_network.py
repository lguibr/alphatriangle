# File: tests/nn/test_network.py
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.config import EnvConfig, ModelConfig, TrainConfig
from src.environment import GameState  # Import real GameState
from src.nn import AlphaTriangleNet, NeuralNetwork
from src.utils.types import StateType

# REMOVED: from ..mcts.conftest import mock_env_config, mock_model_config, mock_train_config, MockGameState # Import shared fixtures
# from tests.mcts.conftest import MockGameState  # Import only MockGameState


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
    return mock_train_config


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def nn_interface(
    model_config: ModelConfig,
    env_config: EnvConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> NeuralNetwork:
    """Provides a NeuralNetwork instance for testing."""
    return NeuralNetwork(model_config, env_config, train_config, device)


@pytest.fixture
def mock_game_state(env_config: EnvConfig) -> GameState:
    """Provides a real GameState object for testing NN interface."""
    # Use a real GameState instance
    return GameState(config=env_config, initial_seed=123)


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
        "grid": np.random.rand(*grid_shape).astype(np.float32),
        "other_features": np.random.rand(*other_shape).astype(np.float32),
    }


# --- Test Initialization ---
def test_nn_initialization(nn_interface: NeuralNetwork, device: torch.device):
    """Test if the NeuralNetwork wrapper initializes correctly."""
    assert nn_interface is not None
    assert nn_interface.device == device
    assert isinstance(nn_interface.model, AlphaTriangleNet)
    assert nn_interface.model.training is False  # Should be in eval mode initially


# --- Test Feature Extraction Integration (using mock) ---
@patch("src.nn.network.extract_state_features")
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


@patch("src.nn.network.extract_state_features")
def test_batch_states_to_tensors(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_game_state: GameState,  # Use real GameState
    mock_state_type_nn: StateType,
):
    """Test the internal _batch_states_to_tensors method."""
    batch_size = 3
    mock_states = [mock_game_state.copy() for _ in range(batch_size)]
    # Make mock return slightly different arrays each time if needed
    mock_extract.side_effect = [
        # Ensure features are copied correctly
        {k: v.copy() + i * 0.1 for k, v in mock_state_type_nn.items()}
        for i in range(batch_size)
    ]

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
@patch("src.nn.network.extract_state_features")
def test_evaluate_single(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_game_state: GameState,  # Use real GameState
    mock_state_type_nn: StateType,
    env_config: EnvConfig,
):
    """Test the evaluate method for a single state."""
    mock_extract.return_value = mock_state_type_nn

    policy_map, value = nn_interface.evaluate(mock_game_state)

    assert isinstance(policy_map, dict)
    assert isinstance(value, float)
    assert len(policy_map) == env_config.ACTION_DIM
    assert all(
        isinstance(k, int) and isinstance(v, float) for k, v in policy_map.items()
    )
    assert (
        abs(sum(policy_map.values()) - 1.0) < 1e-5
    ), f"Policy probs sum to {sum(policy_map.values())}"
    assert -1.0 <= value <= 1.0


@patch("src.nn.network.extract_state_features")
def test_evaluate_batch(
    mock_extract: MagicMock,
    nn_interface: NeuralNetwork,
    mock_game_state: GameState,  # Use real GameState
    mock_state_type_nn: StateType,
    env_config: EnvConfig,
):
    """Test the evaluate_batch method."""
    batch_size = 3
    mock_states = [mock_game_state.copy() for _ in range(batch_size)]
    mock_extract.side_effect = [
        # Ensure features are copied correctly
        {k: v.copy() + i * 0.1 for k, v in mock_state_type_nn.items()}
        for i in range(batch_size)
    ]

    results = nn_interface.evaluate_batch(mock_states)

    assert isinstance(results, list)
    assert len(results) == batch_size
    for policy_map, value in results:
        assert isinstance(policy_map, dict)
        assert isinstance(value, float)
        assert len(policy_map) == env_config.ACTION_DIM
        assert all(
            isinstance(k, int) and isinstance(v, float) for k, v in policy_map.items()
        )
        assert abs(sum(policy_map.values()) - 1.0) < 1e-5
        assert -1.0 <= value <= 1.0


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
            assert torch.allclose(
                modified_weights[key], new_weights[key], atol=1e-6
            ), f"Weight mismatch for key {key}"
        else:
            assert torch.equal(
                initial_weights[key], new_weights[key]
            ), f"Non-float tensor mismatch for key {key}"

    # Test setting back original weights
    nn_interface.set_weights(initial_weights)
    final_weights = nn_interface.get_weights()
    for key in initial_weights:
        assert torch.equal(
            initial_weights[key], final_weights[key]
        ), f"Weight mismatch after setting back original for key {key}"
